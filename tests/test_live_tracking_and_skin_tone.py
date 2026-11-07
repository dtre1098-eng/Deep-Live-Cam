"""Tests for the live face tracker and skin-tone matcher.

Both modules are pure numpy/cv2, so they are loaded by path to avoid
importing the model-heavy ``modules`` package.
"""
import importlib.util
import os
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lt = _load("live_tracking", "modules/live_tracking.py")
st = _load("skin_tone", "modules/processors/frame/skin_tone.py")


class Face(dict):
    def __getattr__(self, k):
        return self.get(k)

    def __setattr__(self, k, v):
        self[k] = v


BASE = np.array([[-1, -1], [1, -1], [0, 0], [-.8, 1], [.8, 1]], np.float32)


def make_face(cx, cy, scale=40.0, noise=None):
    kps = BASE * scale + np.array([cx, cy], np.float32)
    if noise is not None:
        kps = kps + noise
    return Face(kps=kps.astype(np.float32),
                bbox=np.array([cx - scale, cy - scale, cx + scale, cy + scale], np.float32))


class FaceTrackerTests(unittest.TestCase):
    def test_holds_face_through_missed_detection(self):
        tr = lt.FaceTracker(hold_seconds=0.4)
        tr.update([make_face(100, 100)], now=0.0)
        self.assertEqual(len(tr.update([], now=0.1)), 1)
        self.assertEqual(len(tr.update([], now=0.6)), 0)

    def test_reduces_jitter_when_still(self):
        rng = np.random.default_rng(0)
        tr = lt.FaceTracker()
        outs = []
        for i in range(60):
            f = make_face(200, 200, noise=rng.normal(0, 2, (5, 2)))
            outs.append(tr.update([f], now=i / 30.0)[0].kps)
        raw_jitter = 2.0 * np.sqrt(2) * np.sqrt(2 / np.pi)
        jitter = np.abs(np.diff(np.array(outs[10:]), axis=0)).mean()
        self.assertLess(jitter, raw_jitter * 0.5)

    def test_prediction_follows_motion(self):
        tr = lt.FaceTracker()
        for i in range(20):
            tr.update([make_face(100 + 5 * i, 100)], now=i / 30.0)
        pred = tr.predict(now=20 / 30.0)[0].kps.mean(axis=0)[0]
        last = tr.tracks[0].filter.x.mean(axis=0)[0]
        self.assertGreater(pred, last)

    def test_sticks_to_tracked_person_and_prefers_largest(self):
        tr = lt.FaceTracker(max_faces=1)
        out = tr.update([make_face(100, 100, 20), make_face(400, 100, 50)], now=0.0)
        self.assertGreater(out[0].kps.mean(axis=0)[0], 300)
        # a larger face appears elsewhere; we stay on the tracked one
        out = tr.update([make_face(402, 100, 50), make_face(100, 100, 80)], now=0.033)
        self.assertGreater(out[0].kps.mean(axis=0)[0], 300)

    def test_clone_does_not_mutate_detection(self):
        tr = lt.FaceTracker()
        f = make_face(100, 100)
        orig = f.kps.copy()
        tr.update([f], now=0.0)
        tr.update([make_face(130, 100)], now=0.033)
        np.testing.assert_array_equal(f.kps, orig)


class SkinToneTests(unittest.TestCase):
    def setUp(self):
        st.reset_state()

    @staticmethod
    def _face(color, seed=0):
        rng = np.random.default_rng(seed)
        img = np.zeros((128, 128, 3), np.float32) + color
        img += rng.normal(0, 6, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _mean_lab(self, img):
        m = st._skin_sample_mask(128)
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[m].astype(float).mean(0)

    def test_matches_target_skin(self):
        src, tgt = self._face((150, 170, 220)), self._face((70, 110, 160), 1)
        out = st.match_skin_tone(src, tgt)
        before = np.abs(self._mean_lab(src) - self._mean_lab(tgt)).max()
        after = np.abs(self._mean_lab(out) - self._mean_lab(tgt)).max()
        self.assertGreater(before, 20)
        self.assertLess(after, 3)

    def test_zero_strength_is_noop(self):
        src, tgt = self._face((150, 170, 220)), self._face((70, 110, 160))
        self.assertIs(st.match_skin_tone(src, tgt, strength=0), src)

    def test_temporal_smoothing_damps_flicker(self):
        src = self._face((150, 170, 220))
        c = np.array([300.0, 200.0])
        a = st.match_skin_tone(src, self._face((70, 110, 160)), c, 200)
        b = st.match_skin_tone(src, self._face((120, 160, 210)), c, 200)
        jump = np.abs(self._mean_lab(self._face((70, 110, 160))) - self._mean_lab(self._face((120, 160, 210)))).max()
        moved = np.abs(self._mean_lab(a) - self._mean_lab(b)).max()
        self.assertLess(moved, jump * 0.5)

    def test_lifting_hand_does_not_change_face_colour(self):
        src = self._face((150, 170, 220))
        c = np.array([300.0, 200.0])
        for i in range(30):
            ref = st.match_skin_tone(src, self._face((70, 110, 160), i), c, 200)
        base = self._mean_lab(ref)
        for cover in (0.3, 0.6, 0.9):
            tgt = self._face((70, 110, 160), 77)
            h = int(128 * cover)
            tgt[128 - h:] = self._face((140, 160, 225), 78)[128 - h:]   # hand coming up
            for _ in range(10):
                out = st.match_skin_tone(src, tgt, c, 200)
            self.assertLess(np.abs(self._mean_lab(out) - base).max(), 3, f"hand covering {cover:.0%}")

    def test_relearns_after_lighting_change(self):
        src = self._face((150, 170, 220))
        c = np.array([300.0, 200.0])
        for i in range(30):
            st.match_skin_tone(src, self._face((70, 110, 160), i), c, 200)
        for i in range(80):
            out = st.match_skin_tone(src, self._face((110, 150, 200), 200 + i), c, 200)
        want = st.match_skin_tone(src, self._face((110, 150, 200), 999))
        self.assertLess(np.abs(self._mean_lab(out) - self._mean_lab(want)).max(), 3)

    def test_keeps_picture_colour_detail(self):
        # lips/makeup: colour variation in the picture must survive
        src = self._face((150, 170, 220))
        src[60:70, 30:100] = (60, 60, 200)       # red stripe across the cheeks
        tgt = self._face((70, 110, 160), 1)
        out = st.match_skin_tone(src, tgt)
        lab_in = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(float)
        lab_out = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(float)
        contrast_in = lab_in[60:70, 30:100, 1].mean() - lab_in[80:90, 30:100, 1].mean()
        contrast_out = lab_out[60:70, 30:100, 1].mean() - lab_out[80:90, 30:100, 1].mean()
        self.assertGreater(contrast_out, contrast_in * 0.9)

    def test_shape_mismatch_returns_input(self):
        src = self._face((150, 170, 220))
        self.assertIs(st.match_skin_tone(src, np.zeros((64, 64, 3), np.uint8)), src)


if __name__ == "__main__":
    unittest.main()
