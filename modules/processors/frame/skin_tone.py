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
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

_MASK_CACHE: Dict[int, np.ndarray] = {}
_STATE_LOCK = threading.Lock()
# key -> (center_xy, face_scale, mean_shift(3,), std_ratio(3,))
_STATE: Dict[int, Tuple[np.ndarray, float, np.ndarray, np.ndarray]] = {}
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
    for k, (c, sc, _, _) in _STATE.items():
        d = float(np.linalg.norm(c - center))
        if d < max(sc, scale) * 0.6 and (best_d is None or d < best_d):
            best_key, best_d = k, d
    return best_key


def reset_state() -> None:
    with _STATE_LOCK:
        _STATE.clear()


def match_skin_tone(bgr_fake: np.ndarray, target_aligned: np.ndarray,
                    face_center: Optional[np.ndarray] = None,
                    face_scale: float = 0.0, strength: float = 1.0,
                    smoothing: float = 0.7) -> np.ndarray:
    """Return ``bgr_fake`` recoloured to match ``target_aligned``'s skin.

    Args:
        bgr_fake: swapped face, aligned (HxWx3 uint8).
        target_aligned: the real frame warped with the same affine (HxWx3).
        face_center / face_scale: face position/size in the frame, used to
            keep per-face temporal smoothing state (None disables smoothing).
        strength: 0..1 how much of the correction to apply.
        smoothing: 0..1 EMA weight of the previous frame's correction.
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

    # Ignore target pixels that fell outside the camera frame (black border
    # from warpAffine) or are blown-out/very dark.
    valid = (tgt_lab[..., 0] > 15) & (tgt_lab[..., 0] < 250)
    s = _stats(src_lab, sel)
    t = _stats(tgt_lab, sel, valid)
    if s is None or t is None:
        return bgr_fake
    s_mean, s_std = s
    t_mean, t_std = t

    ratio = np.clip(t_std / np.maximum(s_std, 1.0), 0.75, 1.33)
    # Keep most of the source's contrast (identity detail); adapt fully on
    # colour (a/b) means and lightness mean.
    ratio = 1.0 + (ratio - 1.0) * 0.5
    shift = t_mean - s_mean

    if face_center is not None and smoothing > 0:
        center = np.asarray(face_center, np.float32)
        with _STATE_LOCK:
            key = _lookup_state(center, face_scale)
            if key is not None:
                _, _, p_shift, p_ratio = _STATE[key]
                shift = smoothing * p_shift + (1 - smoothing) * shift
                ratio = smoothing * p_ratio + (1 - smoothing) * ratio
            else:
                key = _next_key
                _next_key += 1
                if len(_STATE) > 16:
                    _STATE.clear()
            _STATE[key] = (center, float(face_scale), shift, ratio)

    out = (src_lab - s_mean) * ratio + s_mean + shift
    if strength < 1.0:
        out = src_lab + (out - src_lab) * strength
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
