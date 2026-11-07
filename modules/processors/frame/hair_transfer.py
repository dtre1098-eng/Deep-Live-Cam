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
    __slots__ = ("bgr", "alpha", "kps")

    def __init__(self, bgr: np.ndarray, alpha: np.ndarray, kps: np.ndarray):
        self.bgr = bgr      # HxWx3 uint8, cropped around the hair
        self.alpha = alpha  # HxW uint8, feathered hair mask
        self.kps = kps      # 5x2 float32 source keypoints in crop coords


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
    # Face skin is definitely not hair (also teaches GrabCut the skin colour).
    cv2.ellipse(gc, (192, int((160 + mouth_y) / 2)), (58, int(mouth_y - 160) + 30),
                0, 0, 360, cv2.GC_BGD, -1)
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
    cv2.ellipse(crown, (192, 92), (60, 32), 0, 0, 360, 255, -1)  # just above the forehead
    hair_seed = (crown > 0) & (cv2.erode(fg, np.ones((9, 9), np.uint8)) > 0)
    if hair_seed.sum() < 80 or len(bg_px) < 200:
        return fg
    # Main hair colours only: a background object touching the top of the
    # head would otherwise sneak in as a minor "hair" colour.
    hair_c = _kmeans_centres(lab[hair_seed], 4, min_share=0.2)
    bg_c = _kmeans_centres(bg_px, 12)
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
                mask: Optional[np.ndarray] = None) -> Optional[HairLayer]:
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
    return HairLayer(_decontaminate(image[y1:y2, x1:x2], mask[y1:y2, x1:x2]), alpha,
                     kps - np.array([x1, y1], np.float32))


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


def apply_hair(frame: np.ndarray, layer: Optional[HairLayer], target_kps) -> np.ndarray:
    """Composite ``layer`` onto ``frame`` aligned to ``target_kps`` (in place)."""
    if layer is None or target_kps is None:
        return frame
    tk = np.asarray(target_kps, np.float32)
    if tk.shape != (5, 2) or not np.all(np.isfinite(tk)):
        return frame
    A = _umeyama(layer.kps, tk)
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
    hair = cv2.warpAffine(layer.bgr, A_roi, size, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.warpAffine(layer.alpha, A_roi, size, flags=cv2.INTER_LINEAR, borderValue=0)
    if alpha.max() == 0:
        return frame
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


def _canonical_hair_zone() -> np.ndarray:
    z = _ZONE_CACHE.get("zone")
    if z is not None:
        return z
    N = 384
    zone = np.zeros((N, N), np.uint8)
    cv2.ellipse(zone, (192, 140), (150, 150), 0, 180, 360, 255, -1)        # above/around crown
    cv2.rectangle(zone, (192 - 150, 140), (192 + 150, 300), 255, -1)        # beside the face to the jaw
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

    def hide_hair(self, frame: np.ndarray, kps, threshold: int = 28) -> np.ndarray:
        """Paint the learned background over non-face, non-skin pixels around the head."""
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
