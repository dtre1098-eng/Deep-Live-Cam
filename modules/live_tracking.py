"""Temporal face tracking for the live webcam path.

Why this exists
---------------
The live loop only runs face detection every N frames and, between
detections, re-used the *stale* face. The swap is placed from ``face.kps``
(5 keypoints), so while you move the swapped face slid behind your real one
and then snapped back on the next detection. When detection missed a frame
(motion blur, face partly out of frame when you lean in close, turning
away) the cached face became ``None`` and the swap vanished for that frame,
which shows up as flicker / the original face and hair popping through.

``FaceTracker`` fixes all three:

* Keypoints are smoothed with a One-Euro filter (low jitter when still, low
  lag when moving fast). Speeds are measured in *face widths per second*, so
  the filter behaves the same whether you are close to or far from the camera.
* On frames where detection is skipped, keypoints are extrapolated with the
  filtered velocity, so the swap follows the face between detections.
* When detection misses, the last face is held (and coasts) for a short
  grace period instead of dropping the swap immediately.
* The tracked face is matched to the nearest detection, so the swap does not
  jump to another person who walks into the frame.

Pure numpy, no model or UI dependencies.
"""

from __future__ import annotations

import math
import time
from typing import Any, List, Optional

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One-Euro filter over an array of values (e.g. 5x2 keypoints).

    ``scale`` normalises the derivative so ``beta`` is size-independent.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.6, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x: Optional[np.ndarray] = None
        self.dx: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.x = None
        self.dx = None

    def __call__(self, x: np.ndarray, dt: float, scale: float = 1.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        dt = max(dt, 1e-3)
        if self.x is None:
            self.x = x.copy()
            self.dx = np.zeros_like(x)
            return self.x.copy()
        raw_dx = (x - self.x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self.dx = a_d * raw_dx + (1.0 - a_d) * self.dx
        speed = float(np.abs(self.dx).mean()) / max(scale, 1e-3)
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha(cutoff, dt)
        self.x = a * x + (1.0 - a) * self.x
        return self.x.copy()

    def predict(self, dt: float) -> Optional[np.ndarray]:
        """Constant-velocity extrapolation from the last filtered state."""
        if self.x is None:
            return None
        return self.x + self.dx * dt


def _face_width(kps: np.ndarray) -> float:
    # Distance between the eyes is stable across expressions and
    # proportional to face size.
    return float(max(np.linalg.norm(kps[1] - kps[0]), 1.0)) * 2.5


def _clone_face(face: Any, kps: np.ndarray) -> Any:
    """Copy an insightface Face (a dict subclass) with new kps/bbox."""
    try:
        new = face.__class__(face)
    except Exception:
        new = face
    shift = kps.mean(axis=0) - np.asarray(face.kps, dtype=np.float32).mean(axis=0)
    new.kps = kps.astype(np.float32)
    bbox = getattr(face, "bbox", None)
    if bbox is not None:
        b = np.asarray(bbox, dtype=np.float32).copy()
        b[0:4:2] += shift[0]
        b[1:4:2] += shift[1]
        new.bbox = b
    lmk = getattr(face, "landmark_2d_106", None)
    if lmk is not None:
        new.landmark_2d_106 = np.asarray(lmk, dtype=np.float32) + shift
    return new


class _Track:
    def __init__(self, face: Any, now: float, min_cutoff: float, beta: float):
        self.filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.face = face
        self.last_seen = now
        self.last_update = now
        kps = np.asarray(face.kps, dtype=np.float32)
        self.kps = self.filter(kps, 1.0 / 30.0, _face_width(kps))

    def center(self) -> np.ndarray:
        return self.kps.mean(axis=0)

    def update(self, face: Any, now: float) -> None:
        kps = np.asarray(face.kps, dtype=np.float32)
        dt = now - self.last_update
        self.kps = self.filter(kps, dt, _face_width(kps))
        self.face = face
        self.last_seen = now
        self.last_update = now

    def coast(self, now: float) -> None:
        dt = min(now - self.last_update, 0.15)  # never extrapolate far
        pred = self.filter.predict(dt)
        if pred is not None:
            self.kps = pred

    def output(self) -> Any:
        return _clone_face(self.face, self.kps)


class FaceTracker:
    """Keeps swaps glued to faces between/through detections."""

    def __init__(self, max_faces: int = 1, hold_seconds: float = 0.4,
                 min_cutoff: float = 1.5, beta: float = 20.0):
        self.max_faces = max_faces
        self.hold_seconds = hold_seconds
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.tracks: List[_Track] = []

    def reset(self) -> None:
        self.tracks = []

    def update(self, detections: Optional[List[Any]], now: Optional[float] = None) -> List[Any]:
        """Feed fresh detections (may be empty/None when detection missed)."""
        now = time.monotonic() if now is None else now
        dets = [d for d in (detections or []) if d is not None and getattr(d, "kps", None) is not None]

        unmatched = list(range(len(dets)))
        for tr in self.tracks:
            best, best_d = None, None
            c = tr.center()
            limit = _face_width(tr.kps) * 0.9
            for j in unmatched:
                d = float(np.linalg.norm(np.asarray(dets[j].kps, np.float32).mean(axis=0) - c))
                if d < limit and (best_d is None or d < best_d):
                    best, best_d = j, d
            if best is not None:
                tr.update(dets[best], now)
                unmatched.remove(best)
            else:
                tr.coast(now)

        self.tracks = [t for t in self.tracks if now - t.last_seen <= self.hold_seconds]

        # New faces: biggest first, so the main (closest) person wins.
        unmatched.sort(key=lambda j: -_face_width(np.asarray(dets[j].kps, np.float32)))
        for j in unmatched:
            if len(self.tracks) >= self.max_faces:
                break
            self.tracks.append(_Track(dets[j], now, self.min_cutoff, self.beta))
        return [t.output() for t in self.tracks]

    def predict(self, now: Optional[float] = None) -> List[Any]:
        """Frames where detection was skipped: extrapolate tracked faces."""
        now = time.monotonic() if now is None else now
        self.tracks = [t for t in self.tracks if now - t.last_seen <= self.hold_seconds]
        out = []
        for t in self.tracks:
            t.coast(now)
            out.append(t.output())
        return out
