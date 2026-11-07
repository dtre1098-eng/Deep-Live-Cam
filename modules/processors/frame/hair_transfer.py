"""Hair transfer: show the source picture's hair on the swapped face.

inswapper only replaces the inner face (roughly hairline to chin), so the
hair in the output has always been the person in front of the camera's own
hair. This module adds the source photo's hair on top:

1. Once per source picture, the hair is cut out of the photo
   (GrabCut seeded from the face position: the region around/above the face
   is "probably hair", the face skin and far background are "not hair").
   You can override the automatic cut-out by putting a mask next to the
   photo: ``<photo name>.hair.png`` (white = hair, black = not hair).
2. Every frame, the hair layer is placed with a similarity transform
   (position, size, rotation) fitted from the source face's 5 keypoints to
   the live face's keypoints. In live mode those keypoints come from the
   smoothing tracker, so the hair follows you without jitter, scales as you
   move closer/farther and is held through brief detection drop-outs.

Limitations: it's a 2D layer, so it follows tilt but not turning the head
far to the side; it cannot erase your real hair where the source hair does
not reach; hands passing in front of the hair are drawn behind it.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

import cv2
import numpy as np

_LOCK = threading.Lock()
_CACHE: dict = {"key": None, "layer": None}


class HairLayer:
    __slots__ = ("bgr", "alpha", "kps", "levels", "ears", "skin_lab", "_shift")

    def __init__(self, bgr: np.ndarray, alpha: np.ndarray, kps: np.ndarray):
        self.bgr = bgr      # HxWx3 uint8, cropped around the hair
        self.alpha = alpha  # HxW uint8, feathered hair mask
        self.kps = kps      # 5x2 float32 source keypoints in crop coords
        self.levels = {}    # n -> (bgr, alpha) shrunk by 2**-n (cached)
        self.ears = None    # optional HairLayer with the picture's ears
        self.skin_lab = None  # source cheek colour (Lab), for recolouring ears
        self._shift = None  # smoothed per-frame ear colour shift

    def level(self, n: int):
        if n <= 0:
            return self.bgr, self.alpha
        if n not in self.levels:
            f = 0.5 ** n
            size = (max(1, int(round(self.bgr.shape[1] * f))), max(1, int(round(self.bgr.shape[0] * f))))
            self.levels[n] = (cv2.resize(self.bgr, size, interpolation=cv2.INTER_AREA),
                              cv2.resize(self.alpha, size, interpolation=cv2.INTER_AREA))
        return self.levels[n]


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (2x3) mapping src -> dst."""
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    var_s = (s_c ** 2).sum() / len(src)
    scale = (S * np.diag(D)).sum() / max(var_s, 1e-9)
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]]).astype(np.float32)


def _head_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Similarity (2x3) that keeps the picture's hair/ears locked to the skull.

    Anchored on the eyes (they move with the skull; the mouth moves when
    you talk or smile). Size comes from the eye distance, or from the
    eye-to-nose-tip distance when the eyes look closer together because the
    head is turned, so the hair doesn't pump in size when you talk, smile,
    nod or turn.
    """
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    se, de = (src[0] + src[1]) / 2, (dst[0] + dst[1]) / 2
    sv, dv = src[1] - src[0], dst[1] - dst[0]
    s_eye = np.linalg.norm(dv) / max(np.linalg.norm(sv), 1e-6)
    s_nose = np.linalg.norm(dst[2] - de) / max(np.linalg.norm(src[2] - se), 1e-6)
    scale = max(s_eye, s_nose / 1.05)
    ang = np.arctan2(dv[1], dv[0]) - np.arctan2(sv[1], sv[0])
    R = scale * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    t = de - R @ se
    return np.hstack([R, t[:, None]]).astype(np.float32)


def _normaliser(kps: np.ndarray, eye_px: float = 64.0) -> np.ndarray:
    """Affine taking the photo to upright face space (eye centre (192,160))."""
    kps = np.asarray(kps, np.float32)
    eye_c = (kps[0] + kps[1]) / 2
    eye_d = float(np.linalg.norm(kps[1] - kps[0]))
    angle = np.degrees(np.arctan2(kps[1][1] - kps[0][1], kps[1][0] - kps[0][0]))
    M = cv2.getRotationMatrix2D((float(eye_c[0]), float(eye_c[1])), angle, eye_px / max(eye_d, 1e-6))
    M[:, 2] += np.array([192.0, 160.0]) - eye_c
    return M


def _cheek_skin(norm_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Median skin colour (YCrCb) from the cheeks in upright face space."""
    ycc = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2YCrCb)
    px = np.concatenate([ycc[200:226, 140:166].reshape(-1, 3), ycc[200:226, 218:244].reshape(-1, 3)])
    return np.median(px, axis=0) if len(px) else None


def segment_ears(image: np.ndarray, kps: np.ndarray,
                 hair_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """uint8 mask (0/255) of the picture's visible ears (skin beside the face)."""
    kps = np.asarray(kps, np.float32)
    if float(np.linalg.norm(kps[1] - kps[0])) < 8:
        return np.zeros(image.shape[:2], np.uint8)
    M = _normaliser(kps)
    N = 384
    norm = cv2.warpAffine(image, M, (N, N), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    inside = cv2.warpAffine(np.full(image.shape[:2], 255, np.uint8), M, (N, N),
                            flags=cv2.INTER_NEAREST, borderValue=0) > 0
    skin0 = _cheek_skin(norm)
    if skin0 is None:
        return np.zeros(image.shape[:2], np.uint8)
    ycc = cv2.cvtColor(norm, cv2.COLOR_BGR2YCrCb).astype(np.int16)
    skin = ((np.abs(ycc[..., 1] - skin0[1]) < 12) & (np.abs(ycc[..., 2] - skin0[2]) < 14)
            & (np.abs(ycc[..., 0] - skin0[0]) < 70) & inside)
    zone = np.zeros((N, N), np.uint8)
    # Beside the face (beyond ~1.2 eye distances from the middle), between
    # eyebrow and mouth height: where ears are.
    cv2.rectangle(zone, (192 - 125, 140), (192 - 70, 250), 255, -1)
    cv2.rectangle(zone, (192 + 70, 140), (192 + 125, 250), 255, -1)
    ears = (skin & (zone > 0)).astype(np.uint8) * 255
    if hair_mask is not None:
        hm = cv2.warpAffine(hair_mask, M, (N, N), flags=cv2.INTER_NEAREST, borderValue=0)
        ears[hm > 127] = 0
    ears = cv2.morphologyEx(ears, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ears = cv2.morphologyEx(ears, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ears)
    keep = np.zeros_like(ears)
    for i in range(1, n):
        x, w_ = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_WIDTH]
        # an ear is attached to the side of the head: the blob must reach the
        # zone's inner edge (next to the face), not float in the background
        if x < 192:
            touches_face_side = x + w_ >= 192 - 70 - 2
        else:
            touches_face_side = x <= 192 + 70 + 2
        if stats[i, cv2.CC_STAT_AREA] >= 120 and touches_face_side:
            keep[lab == i] = 255
    return cv2.warpAffine(keep, cv2.invertAffineTransform(M), (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def segment_hair(image: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Return a uint8 hair mask (0/255) for the face with keypoints ``kps``.

    Works in an upright, face-normalised frame so tilted photos work too.
    """
    kps = np.asarray(kps, np.float32)
    eye_c = (kps[0] + kps[1]) / 2
    mouth_c = (kps[3] + kps[4]) / 2
    eye_d = float(np.linalg.norm(kps[1] - kps[0]))
    if eye_d < 4:
        return np.zeros(image.shape[:2], np.uint8)

    # Normalise: eyes horizontal, eye distance = 64px, eye centre at (192, 160).
    S = 64.0 / eye_d
    angle = np.degrees(np.arctan2(kps[1][1] - kps[0][1], kps[1][0] - kps[0][0]))
    M = cv2.getRotationMatrix2D((float(eye_c[0]), float(eye_c[1])), angle, S)
    M[:, 2] += np.array([192.0, 160.0]) - eye_c
    N = 384
    norm = cv2.warpAffine(image, M, (N, N), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    inside = cv2.warpAffine(np.full(image.shape[:2], 255, np.uint8), M, (N, N),
                            flags=cv2.INTER_NEAREST, borderValue=0) > 0
    mouth_y = 160 + float(np.linalg.norm(mouth_c - eye_c)) * S  # ~ 160 + 70

    # Seeds in normalised space (eye distance 64).
    gc = np.full((N, N), cv2.GC_BGD, np.uint8)
    cv2.ellipse(gc, (192, 150), (150, 175), 0, 0, 360, cv2.GC_PR_BGD, -1)
    cv2.ellipse(gc, (192, 125), (118, 120), 0, 180, 360, cv2.GC_PR_FGD, -1)   # crown
    cv2.rectangle(gc, (192 - 118, 125), (192 - 68, int(mouth_y) + 40), cv2.GC_PR_FGD, -1)  # sides
    cv2.rectangle(gc, (192 + 68, 125), (192 + 118, int(mouth_y) + 40), cv2.GC_PR_FGD, -1)
    cv2.ellipse(gc, (192, 72), (55, 22), 0, 0, 360, cv2.GC_FGD, -1)  # top of head
    # The forehead may be skin or a fringe/bangs: let GrabCut decide there
    # (skin is removed again by the colour check). Only the face below the
    # eyebrows is definitely not hair (also teaches GrabCut the skin colour).
    cv2.ellipse(gc, (192, int((160 + mouth_y) / 2)), (58, int(mouth_y - 160) + 30),
                0, 0, 360, cv2.GC_PR_BGD, -1)
    cv2.ellipse(gc, (192, int((160 + mouth_y) / 2) + 8), (54, int(mouth_y - 160) + 6),
                0, 0, 360, cv2.GC_BGD, -1)
    brow = np.zeros((N, N), np.uint8)
    cv2.ellipse(brow, (192, 160), (62, 70), 0, 180, 360, 255, -1)   # forehead band
    gc[(brow > 0) & (gc == cv2.GC_BGD)] = cv2.GC_PR_BGD
    # Neck/shoulders straight below the chin: probably not hair.
    cv2.rectangle(gc, (192 - 50, int(mouth_y) + 20), (192 + 50, N), cv2.GC_PR_BGD, -1)
    gc[~inside] = cv2.GC_BGD

    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(norm, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros(image.shape[:2], np.uint8)
    fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # Background colours come from a zoomed-out view (eye distance 24px) so
    # the whole picture away from the person is sampled, not just the
    # pixels right around the head.
    Sw = 24.0 / eye_d
    Mw = cv2.getRotationMatrix2D((float(eye_c[0]), float(eye_c[1])), angle, Sw)
    Mw[:, 2] += np.array([192.0, 160.0]) - eye_c
    wide = cv2.warpAffine(image, Mw, (N, N), flags=cv2.INTER_AREA, borderValue=0)
    wide_in = cv2.warpAffine(np.full(image.shape[:2], 255, np.uint8), Mw, (N, N),
                             flags=cv2.INTER_NEAREST, borderValue=0) > 0
    far = np.full((N, N), 255, np.uint8)
    cv2.ellipse(far, (192, 150), (80, 92), 0, 0, 360, 0, -1)          # head + hair (~3.5 eye dist)
    cv2.rectangle(far, (192 - 85, 195), (192 + 85, N), 0, -1)          # shoulders / long hair
    bg_px = cv2.cvtColor(wide, cv2.COLOR_BGR2LAB)[(far > 0) & wide_in]
    fg = _drop_background_colours(norm, fg, bg_px)

    # Clean up: close small gaps, drop specks, keep the blob(s) touching the head.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    head = np.zeros_like(fg)
    cv2.ellipse(head, (192, 125), (125, 125), 0, 0, 360, 255, -1)
    keep = np.zeros_like(fg)
    for i in range(1, n):
        comp = lab == i
        if stats[i, cv2.CC_STAT_AREA] > 150 and (head[comp] > 0).any():
            keep[comp] = 255

    Minv = cv2.invertAffineTransform(M)
    return cv2.warpAffine(keep, Minv, (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def _kmeans_centres(px: np.ndarray, k: int, min_share: float = 0.0) -> np.ndarray:
    px = px.astype(np.float32)
    if len(px) > 6000:
        px = px[np.random.default_rng(0).choice(len(px), 6000, replace=False)]
    k = max(1, min(k, len(px)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(0)
    _, labels, centres = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    if min_share > 0:
        share = np.bincount(labels.ravel(), minlength=k) / len(labels)
        major = share >= min_share
        if major.any():
            centres = centres[major]
    return centres


def _drop_background_colours(norm: np.ndarray, fg: np.ndarray, bg_px: np.ndarray) -> np.ndarray:
    """Remove pixels from the hair cut-out that look like the photo's background.

    GrabCut alone happily takes in background patches next to the hair
    (dark walls next to dark hair, objects behind the shoulders). Here the
    hair colours are learnt from the crown, the background colours from the
    area well away from the person, and every cut-out pixel that is closer
    to a background colour than to a hair colour is dropped, so none of the
    picture's background comes along with the hair.
    """
    N = norm.shape[0]
    lab = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB).astype(np.float32)
    crown = np.zeros((N, N), np.uint8)
    cv2.ellipse(crown, (192, 68), (58, 22), 0, 0, 360, 255, -1)  # top of the head, above any hairline
    hair_seed = (crown > 0) & (cv2.erode(fg, np.ones((9, 9), np.uint8)) > 0)
    if hair_seed.sum() < 80 or len(bg_px) < 200:
        return fg
    # Main hair colours only: a background object touching the top of the
    # head would otherwise sneak in as a minor "hair" colour.
    hair_c = _kmeans_centres(lab[hair_seed], 4, min_share=0.2)
    # Face skin (cheeks) counts as "not hair" too, so the forehead isn't
    # taken as hair now that fringes/bangs are allowed there.
    cheeks = np.concatenate([lab[200:226, 140:166].reshape(-1, 3), lab[200:226, 218:244].reshape(-1, 3)])
    bg_c = np.concatenate([_kmeans_centres(bg_px, 12), _kmeans_centres(cheeks, 2)])
    cand = fg > 0
    px = lab[cand]
    d_h = np.min(np.linalg.norm(px[:, None, :] - hair_c[None], axis=2), axis=1)
    d_b = np.min(np.linalg.norm(px[:, None, :] - bg_c[None], axis=2), axis=1)
    # Keep only clearly-hair pixels: nearer to hair than to background (with
    # a margin) and not wildly off every hair colour.
    keep = (d_h < d_b * 0.85) & (d_h < 40)
    out = np.zeros_like(fg)
    out[cand] = np.where(keep, 255, 0)
    return out


def build_layer(image: np.ndarray, kps: np.ndarray,
                mask: Optional[np.ndarray] = None, with_ears: bool = True) -> Optional[HairLayer]:
    kps = np.asarray(kps, np.float32)
    if mask is None:
        mask = segment_hair(image, kps)
    if mask is None or mask.max() == 0:
        return None
    ys, xs = np.nonzero(mask > 127)
    if len(xs) < 50:
        return None
    eye_d = float(np.linalg.norm(kps[1] - kps[0]))
    pad = int(eye_d * 0.2) + 2
    h, w = mask.shape[:2]
    x1, y1 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x2, y2 = min(w, xs.max() + pad + 1), min(h, ys.max() + pad + 1)
    alpha = mask[y1:y2, x1:x2].copy()
    # Feather proportionally to face size so the edge looks the same at any distance.
    f = max(3, int(eye_d * 0.08) | 1)
    alpha = cv2.GaussianBlur(alpha, (f, f), 0)
    layer = HairLayer(_decontaminate(image[y1:y2, x1:x2], mask[y1:y2, x1:x2]), alpha,
                      kps - np.array([x1, y1], np.float32))
    if with_ears:
        try:
            ear_mask = segment_ears(image, kps, mask)
            if ear_mask.max() > 0:
                layer.ears = build_layer(image, kps, ear_mask, with_ears=False)
                norm = cv2.warpAffine(image, _normaliser(kps), (384, 384), borderMode=cv2.BORDER_REPLICATE)
                skin = _cheek_skin(norm)
                if skin is not None and layer.ears is not None:
                    skin_bgr = cv2.cvtColor(np.uint8([[skin]]), cv2.COLOR_YCrCb2BGR)
                    layer.skin_lab = cv2.cvtColor(skin_bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        except Exception as e:  # ears are optional
            print(f"[DLC.HAIR] Ear extraction skipped: {e}")
    return layer


def _decontaminate(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace non-hair pixels with the nearest hair pixel's colour.

    The feathered alpha edge would otherwise blend in the source photo's
    background, leaving a light/dark halo around the hair.
    """
    hair = mask > 127
    if hair.all() or not hair.any():
        return bgr.copy()
    # Shrink a little so the colours come from solidly-hair pixels.
    core = cv2.erode(hair.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if not core.any():
        core = hair
    _, labels = cv2.distanceTransformWithLabels(
        (~core).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(core)
    lut = np.zeros((labels.max() + 1, 3), np.uint8)
    lut[labels[ys, xs]] = bgr[ys, xs]
    out = lut[labels]
    out[core] = bgr[core]
    return out


def _load_mask_override(path: str, shape) -> Optional[np.ndarray]:
    base, _ = os.path.splitext(path)
    for cand in (base + ".hair.png", path + ".hair.png"):
        if os.path.exists(cand):
            m = cv2.imdecode(np.fromfile(cand, np.uint8), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape[:2] != shape[:2]:
                    m = cv2.resize(m, (shape[1], shape[0]))
                return m
    return None


def get_layer(source_path: Optional[str], source_face: Any) -> Optional[HairLayer]:
    """Hair layer for the current source picture (built once, cached)."""
    kps = getattr(source_face, "kps", None)
    if not source_path or kps is None or not os.path.exists(source_path):
        return None
    try:
        mtime = os.path.getmtime(source_path)
    except OSError:
        return None
    mask_mtimes = tuple(
        os.path.getmtime(c) if os.path.exists(c) else 0.0
        for c in (os.path.splitext(source_path)[0] + ".hair.png", source_path + ".hair.png")
    )
    # Re-checked every frame (two cheap stat calls) so a mask edited with
    # tools/make_hair_mask.py is picked up without restarting the app.
    key = (source_path, mtime, mask_mtimes,
           np.round(np.asarray(kps, np.float32), 1).tobytes())
    with _LOCK:
        if _CACHE["key"] == key:
            return _CACHE["layer"]
        layer = None
        try:
            img = cv2.imdecode(np.fromfile(source_path, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                layer = build_layer(img, kps, _load_mask_override(source_path, img.shape))
        except Exception as e:  # never break the swap
            print(f"[DLC.HAIR] Could not extract hair from source picture: {e}")
        _CACHE["key"], _CACHE["layer"] = key, layer
        if layer is None:
            print("[DLC.HAIR] No hair found in the source picture; hair transfer skipped.")
        return layer


def _ear_colour_shift(frame: np.ndarray, layer: HairLayer, tk: np.ndarray) -> Optional[np.ndarray]:
    """Lab shift so the picture's ears match the (swapped) face's skin now."""
    if layer.skin_lab is None:
        return None
    M = _normaliser(tk, 32.0)  # small, cheap view of the face
    view = cv2.warpAffine(frame, M, (384, 384), borderMode=cv2.BORDER_REPLICATE)
    # cheeks at half scale: eye centre (192,160), eye distance 32
    lab = cv2.cvtColor(view, cv2.COLOR_BGR2LAB)
    px = np.concatenate([lab[180:193, 166:179].reshape(-1, 3), lab[180:193, 205:218].reshape(-1, 3)])
    shift = np.median(px.astype(np.float32), axis=0) - layer.skin_lab
    if layer._shift is not None:
        # smooth, and ignore sudden jumps (a hand passing over a cheek)
        jump = np.abs(shift - layer._shift).max()
        w = 0.15 if jump < 20 else 0.02
        shift = (1 - w) * layer._shift + w * shift
    layer._shift = shift
    return shift


def apply_hair(frame: np.ndarray, layer: Optional[HairLayer], target_kps,
               _recolour: Optional[np.ndarray] = None) -> np.ndarray:
    """Composite ``layer`` (ears, then hair) onto ``frame`` aligned to ``target_kps``."""
    if layer is None or target_kps is None:
        return frame
    tk = np.asarray(target_kps, np.float32)
    if tk.shape != (5, 2) or not np.all(np.isfinite(tk)):
        return frame
    if layer.ears is not None:
        shift = _ear_colour_shift(frame, layer, tk)
        apply_hair(frame, layer.ears, tk, _recolour=shift if shift is not None else np.zeros(3, np.float32))
    A = _head_transform(layer.kps, tk)
    lh, lw = layer.alpha.shape[:2]
    corners = np.array([[0, 0], [lw, 0], [lw, lh], [0, lh]], np.float32)
    t = corners @ A[:, :2].T + A[:, 2]
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, int(np.floor(t[:, 0].min()))), max(0, int(np.floor(t[:, 1].min())))
    x2, y2 = min(fw, int(np.ceil(t[:, 0].max())) + 1), min(fh, int(np.ceil(t[:, 1].max())) + 1)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return frame
    A_roi = A.copy()
    A_roi[0, 2] -= x1
    A_roi[1, 2] -= y1
    size = (x2 - x1, y2 - y1)
    # Shrinking a high-res picture a lot: use a cached, properly
    # down-sampled copy so fine strands don't shimmer (alias) as you move;
    # enlarging: bicubic to keep strands sharp.
    scale = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
    n = int(np.floor(np.log2(0.6 / scale))) + 1 if scale < 0.6 else 0
    src_bgr, src_a = layer.level(n)
    if n:
        f_x = src_bgr.shape[1] / layer.bgr.shape[1]
        f_y = src_bgr.shape[0] / layer.bgr.shape[0]
        A_roi = A_roi.copy()
        A_roi[:, 0] /= f_x
        A_roi[:, 1] /= f_y
    interp = cv2.INTER_CUBIC if scale > 1.05 else cv2.INTER_LINEAR
    hair = cv2.warpAffine(src_bgr, A_roi, size, flags=interp, borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.warpAffine(src_a, A_roi, size, flags=cv2.INTER_LINEAR, borderValue=0)
    if alpha.max() == 0:
        return frame
    if _recolour is not None and np.any(_recolour):
        lab = cv2.cvtColor(hair, cv2.COLOR_BGR2LAB).astype(np.float32) + _recolour
        hair = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    roi = frame[y1:y2, x1:x2]
    a3 = cv2.merge([alpha, alpha, alpha])
    frame[y1:y2, x1:x2] = cv2.add(cv2.multiply(hair, a3, scale=1 / 255.0),
                                  cv2.multiply(roi, 255 - a3, scale=1 / 255.0))
    return frame


def reset_cache() -> None:
    with _LOCK:
        _CACHE["key"], _CACHE["layer"] = None, None


# ---------------------------------------------------------------------------
# Hiding the live person's own hair
# ---------------------------------------------------------------------------
# The source hair is pasted on top, but wherever your real hair reaches
# beyond it (longer, bigger, different parting) it still shows, so the hair
# never looks identical to the picture. For a webcam the background is
# static, so we learn it from pixels away from the person and, around the
# head (not on the face), paint the learned background over anything that
# differs from it -- i.e. your real hair. The source hair then goes on top.
#
# Background behind your head is learned as you move around; for a clean
# result step out of the frame (or lean far to the side) for a second when
# you start. Where the background has never been seen nothing is changed.

# Canonical keypoints (arcface template) and the zone, in a 384x384
# face-normalised space: eye centre (192,160), eye distance 64.
_CANON_KPS = (np.array([[38.29, 51.70], [73.53, 51.50], [56.02, 71.74],
                        [41.55, 92.37], [70.73, 92.20]], np.float32)
              - np.array([55.91, 51.60], np.float32)) * (64.0 / 35.24) + np.array([192.0, 160.0], np.float32)
_ZONE_CACHE: dict = {}


def _canonical_ear_zone() -> np.ndarray:
    z = _ZONE_CACHE.get("ears")
    if z is None:
        z = np.zeros((384, 384), np.uint8)
        cv2.rectangle(z, (192 - 135, 135), (192 - 72, 255), 255, -1)
        cv2.rectangle(z, (192 + 72, 135), (192 + 135, 255), 255, -1)
        z = cv2.GaussianBlur(z, (15, 15), 0)
        _ZONE_CACHE["ears"] = z
    return z


def _canonical_hair_zone() -> np.ndarray:
    z = _ZONE_CACHE.get("zone")
    if z is not None:
        return z
    N = 384
    zone = np.zeros((N, N), np.uint8)
    cv2.ellipse(zone, (192, 140), (150, 150), 0, 180, 360, 255, -1)        # above/around crown
    cv2.rectangle(zone, (192 - 150, 140), (192 + 150, 280), 255, -1)        # beside the face to the jaw
    # keep the face itself (forehead to chin, ears inward) out of the zone
    cv2.ellipse(zone, (192, 190), (70, 105), 0, 0, 360, 0, -1)  # ears/neck are kept by the skin test
    # neck/shoulders below the jaw are not touched
    cv2.rectangle(zone, (192 - 40, 250), (192 + 40, N), 0, -1)
    zone = cv2.GaussianBlur(zone, (21, 21), 0)
    _ZONE_CACHE["zone"] = zone
    return zone


class BackgroundPlate:
    """Running model of the static camera background (kept cheap: cv2 ops)."""

    def __init__(self, rate: float = 0.05, min_seen: int = 4, every: int = 2):
        self.rate = rate
        self.min_seen = min_seen
        self.every = every
        self.bg: Optional[np.ndarray] = None     # float32 HxWx3
        self.bg8: Optional[np.ndarray] = None    # uint8 copy used for painting
        self.seen: Optional[np.ndarray] = None   # uint8 HxW, saturating count
        self._n = 0

    def reset(self) -> None:
        self.bg = self.bg8 = self.seen = None
        self._n = 0

    @staticmethod
    def _free_mask(shape, faces_kps) -> np.ndarray:
        """255 where it's safe to learn background (away from any person)."""
        h, w = shape[:2]
        m = np.full((h, w), 255, np.uint8)
        for k in faces_kps:
            k = np.asarray(k, np.float32)
            ed = float(np.linalg.norm(k[1] - k[0]))
            c = k.mean(axis=0)
            cv2.circle(m, (int(c[0]), int(c[1] - ed * 0.4)), int(ed * 3.6), 0, -1)
            cv2.rectangle(m, (int(c[0] - ed * 5), int(c[1])), (int(c[0] + ed * 5), h), 0, -1)
        return m

    def update(self, frame: np.ndarray, faces_kps) -> None:
        h, w = frame.shape[:2]
        if self.bg is None or self.bg.shape[:2] != (h, w):
            self.bg = frame.astype(np.float32)
            self.bg8 = frame.copy()
            self.seen = np.zeros((h, w), np.uint8)
            self._n = 0
        self._n += 1
        if self._n % self.every:
            return
        free = self._free_mask(frame.shape, faces_kps or [])
        # Camera moved / scene changed: compare on a small copy, every ~0.5s.
        if self._n % 16 == 0:
            sm = (w // 8, h // 8)
            fs, bs = cv2.resize(frame, sm), cv2.resize(self.bg8, sm)
            known = (cv2.resize(free, sm, interpolation=cv2.INTER_NEAREST) > 0) & \
                    (cv2.resize(self.seen, sm, interpolation=cv2.INTER_NEAREST) >= self.min_seen)
            if known.sum() > 200 and (cv2.absdiff(fs, bs).max(axis=2)[known] > 45).mean() > 0.5:
                self.bg[:] = frame
                self.seen[:] = 0
        # Pixels never seen before: take the frame value directly.
        newly = cv2.bitwise_and(free, cv2.compare(self.seen, 0, cv2.CMP_EQ))
        cv2.copyTo(frame.astype(np.float32), newly, self.bg)
        cv2.accumulateWeighted(frame, self.bg, self.rate, mask=free)
        cv2.add(self.seen, 1, dst=self.seen, mask=free)
        self.bg8 = cv2.convertScaleAbs(self.bg)

    def hide_hair(self, frame: np.ndarray, kps, threshold: int = 28,
                  hide_ears: bool = False) -> np.ndarray:
        """Paint the learned background over non-face, non-skin pixels around the head.

        With ``hide_ears`` your own ears (skin beside the face) are removed
        too, so the picture's ears can be drawn in their place.
        """
        if self.bg8 is None or kps is None:
            return frame
        k = np.asarray(kps, np.float32)
        if k.shape != (5, 2) or not np.all(np.isfinite(k)):
            return frame
        A = _umeyama(_CANON_KPS, k)
        zone_c = _canonical_hair_zone()
        corners = np.array([[0, 0], [384, 0], [384, 384], [0, 384]], np.float32) @ A[:, :2].T + A[:, 2]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, int(corners[:, 0].min())), max(0, int(corners[:, 1].min()))
        x2, y2 = min(fw, int(corners[:, 0].max()) + 1), min(fh, int(corners[:, 1].max()) + 1)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return frame
        A_roi = A.copy()
        A_roi[0, 2] -= x1
        A_roi[1, 2] -= y1
        zone = cv2.warpAffine(zone_c, A_roi, (x2 - x1, y2 - y1), borderValue=0)
        ear_zone = (cv2.warpAffine(_canonical_ear_zone(), A_roi, (x2 - x1, y2 - y1), borderValue=0)
                    if hide_ears else None)
        if ear_zone is not None:
            zone = cv2.max(zone, ear_zone)
        roi = frame[y1:y2, x1:x2]
        bg = self.bg8[y1:y2, x1:x2]
        known = self.seen[y1:y2, x1:x2] >= self.min_seen
        person = (cv2.absdiff(roi, bg).max(axis=2) > threshold) & known

        # Keep skin (ears, neck, hands): chroma close to this face's cheeks.
        ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        cheeks = np.array([[192 - 40, 215], [192 + 40, 215], [192, 190]], np.float32) @ A_roi[:, :2].T + A_roi[:, 2]
        ed = float(np.linalg.norm(k[1] - k[0]))
        r = max(2, int(ed * 0.12))
        samples = [ycc[max(0, int(cy) - r):int(cy) + r, max(0, int(cx) - r):int(cx) + r].reshape(-1, 3)
                   for cx, cy in cheeks]
        samples = np.concatenate([p for p in samples if p.size] or [np.zeros((0, 3), np.uint8)])
        if len(samples) >= 10:
            cr0, cb0 = np.median(samples[:, 1]), np.median(samples[:, 2])
            skin = (np.abs(ycc[..., 1].astype(np.int16) - cr0) < 10) & \
                   (np.abs(ycc[..., 2].astype(np.int16) - cb0) < 10)
            if ear_zone is not None:
                skin &= ear_zone < 128   # skin in the ear zones = your ears: hide them
            person &= ~skin

        kk = max(3, int(ed * 0.06) | 1)
        person = person.astype(np.uint8) * 255
        person = cv2.morphologyEx(person, cv2.MORPH_CLOSE, np.ones((kk, kk), np.uint8))
        person = cv2.dilate(person, np.ones((kk, kk), np.uint8))   # catch fine strands at the edge
        person = cv2.GaussianBlur(person, (kk * 2 + 1, kk * 2 + 1), 0)
        person[~known] = 0
        a = cv2.multiply(person, zone, scale=1 / 255.0)
        if a.max() == 0:
            return frame
        a3 = cv2.merge([a, a, a])
        frame[y1:y2, x1:x2] = cv2.add(cv2.multiply(bg, a3, scale=1 / 255.0),
                                      cv2.multiply(roi, 255 - a3, scale=1 / 255.0))
        return frame
