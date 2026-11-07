"""Skin-tone matching for the swapped face.

inswapper returns a face that carries the *source photo's* skin colour and
lighting. Pasted onto the live frame it can look like a mask: a different
tint than your neck, ears and the rest of your body, and a visible seam at
the blend edge.

This module re-colours the swapped face (in the 128x128 aligned space, so it
is cheap) so its colour statistics match the *target* face's real skin under
the current lighting. It works in LAB:

* L (lightness) and a/b (colour) means are shifted to the target's;
* contrast (std) is matched only partially and clamped, so the source's
  facial detail/identity is kept;
* statistics come from a cheek/forehead region (no eyes, brows, mouth,
  hair or background), so dark features don't skew the result;
* the correction is smoothed over time per face so it doesn't flicker as
  you move, blink or turn.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

import cv2
import numpy as np

_MASK_CACHE: Dict[int, np.ndarray] = {}
_STATE_LOCK = threading.Lock()
# key -> {center, scale, shift, ratio, t_mean, t_std}
_STATE: Dict[int, dict] = {}
_next_key = 0


def _skin_sample_mask(size: int) -> np.ndarray:
    """Boolean mask of skin-only pixels in arcface-aligned face space."""
    m = _MASK_CACHE.get(size)
    if m is not None:
        return m
    s = size / 128.0
    mask = np.zeros((size, size), np.uint8)
    # Cheeks + nose bridge + lower forehead in the standard 128 template
    # (eyes at y~52, mouth at y~92).
    cv2.ellipse(mask, (int(64 * s), int(66 * s)), (int(40 * s), int(44 * s)), 0, 0, 360, 255, -1)
    # cut out eyes / brows
    cv2.ellipse(mask, (int(42 * s), int(48 * s)), (int(15 * s), int(10 * s)), 0, 0, 360, 0, -1)
    cv2.ellipse(mask, (int(86 * s), int(48 * s)), (int(15 * s), int(10 * s)), 0, 0, 360, 0, -1)
    # cut out mouth
    cv2.ellipse(mask, (int(64 * s), int(92 * s)), (int(20 * s), int(9 * s)), 0, 0, 360, 0, -1)
    m = mask > 0
    _MASK_CACHE[size] = m
    return m


def _stats(lab: np.ndarray, sel: np.ndarray, valid: Optional[np.ndarray] = None):
    if valid is not None:
        sel = sel & valid
    px = lab[sel]
    if px.shape[0] < 50:
        return None
    return px.mean(axis=0), px.std(axis=0)


def _lookup_state(center: np.ndarray, scale: float):
    """Find the smoothing state of the same face from previous frames."""
    best_key, best_d = None, None
    for k, st in _STATE.items():
        d = float(np.linalg.norm(st["center"] - center))
        if d < max(st["scale"], scale) * 0.6 and (best_d is None or d < best_d):
            best_key, best_d = k, d
    return best_key


def reset_state() -> None:
    with _STATE_LOCK:
        _STATE.clear()


def match_skin_tone(bgr_fake: np.ndarray, target_aligned: np.ndarray,
                    face_center: Optional[np.ndarray] = None,
                    face_scale: float = 0.0, strength: float = 1.0,
                    smoothing: float = 0.8) -> np.ndarray:
    """Return ``bgr_fake`` recoloured to match ``target_aligned``'s skin.

    Args:
        bgr_fake: swapped face, aligned (HxWx3 uint8).
        target_aligned: the real frame warped with the same affine (HxWx3).
        face_center / face_scale: face position/size in the frame, used to
            keep per-face temporal state (None disables smoothing and
            occlusion handling).
        strength: 0..1 how much of the correction to apply.
        smoothing: 0..1 EMA weight of the previous frame's correction.

    Hands and other things in front of the face: once a face has a skin
    model, target pixels that don't fit it (a hand, a phone, a shadow) are
    ignored, and if most of the cheeks/forehead are covered the previous
    correction is kept unchanged, so the face colour doesn't jump when you
    lift your hand.
    """
    global _next_key
    if bgr_fake is None or target_aligned is None or strength <= 0:
        return bgr_fake
    if bgr_fake.shape != target_aligned.shape or bgr_fake.ndim != 3:
        return bgr_fake
    size = bgr_fake.shape[0]
    sel = _skin_sample_mask(size)

    src_lab = cv2.cvtColor(bgr_fake, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
    s = _stats(src_lab, sel)
    if s is None:
        return bgr_fake
    s_mean, s_std = s

    # Ignore target pixels that fell outside the camera frame (black border
    # from warpAffine) or are blown-out/very dark.
    valid = sel & (tgt_lab[..., 0] > 15) & (tgt_lab[..., 0] < 250)
    n_sample = int(sel.sum())

    use_state = face_center is not None and smoothing > 0
    center = np.asarray(face_center, np.float32) if face_center is not None else None
    prev = None
    key = None
    if use_state:
        with _STATE_LOCK:
            key = _lookup_state(center, face_scale)
            prev = dict(_STATE[key]) if key is not None else None

    occluded = False
    occ_count = prev.get("occ_count", 0) if prev is not None else 0
    if occ_count > 45:
        # "Covered" for ~1.5 s: more likely the lighting changed than a
        # hand still in front of the face -- re-learn the skin model.
        prev = dict(prev, t_mean=None)
        occ_count = 0
    if prev is not None and prev.get("t_mean") is not None:
        # Keep only pixels that look like this person's skin as seen
        # recently; a hand/object in front of the face is rejected.
        dev = np.abs(tgt_lab - prev["t_mean"]) / np.maximum(prev["t_std"], 4.0)
        fits = (dev[..., 0] < 3.0) & (dev[..., 1] < 3.0) & (dev[..., 2] < 3.0)
        valid &= fits
        occluded = valid.sum() < 0.5 * n_sample

    if occluded:
        occ_count += 1
        shift, ratio = prev["shift"], prev["ratio"]
        t_mean, t_std = prev["t_mean"], prev["t_std"]
    else:
        px = tgt_lab[valid]
        if px.shape[0] < 50:
            if prev is None:
                return bgr_fake
            shift, ratio = prev["shift"], prev["ratio"]
            t_mean, t_std = prev["t_mean"], prev["t_std"]
            occ_count += 1
        else:
            occ_count = 0
            t_mean = np.median(px, axis=0)
            t_std = px.std(axis=0)
            # Contrast is adapted on lightness only (half-way, clamped); the
            # colour channels keep the picture's own variation, so lips,
            # makeup, freckles and blush stay as in the picture.
            r_l = float(np.clip(t_std[0] / max(s_std[0], 1.0), 0.75, 1.33))
            ratio = np.array([1.0 + (r_l - 1.0) * 0.5, 1.0, 1.0], np.float32)
            shift = (t_mean - s_mean).astype(np.float32)
            if prev is not None:
                shift = smoothing * prev["shift"] + (1 - smoothing) * shift
                ratio = smoothing * prev["ratio"] + (1 - smoothing) * ratio
                if prev.get("t_mean") is not None:
                    t_mean = 0.9 * prev["t_mean"] + 0.1 * t_mean
                    t_std = 0.9 * prev["t_std"] + 0.1 * t_std

    if use_state:
        with _STATE_LOCK:
            if key is None:
                key = _next_key
                _next_key += 1
                if len(_STATE) > 16:
                    _STATE.clear()
            _STATE[key] = {"center": center, "scale": float(face_scale),
                           "shift": shift, "ratio": ratio,
                           "t_mean": t_mean, "t_std": t_std,
                           "occ_count": occ_count}

    out = (src_lab - s_mean) * ratio + s_mean + shift
    if strength < 1.0:
        out = src_lab + (out - src_lab) * strength
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
