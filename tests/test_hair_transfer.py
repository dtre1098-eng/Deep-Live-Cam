"""Tests for source-hair extraction and placement (pure numpy/cv2)."""
import importlib.util
import os
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "hair_transfer", os.path.join(ROOT, "modules/processors/frame/hair_transfer.py"))
ht = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ht)


def portrait(cx=320, cy=260, ed=60, bg=(200, 190, 170)):
    """Synthetic head: textured hair, skin face, neck, shirt. Returns img, kps, hair GT."""
    rng = np.random.default_rng(0)
    img = np.zeros((560, 640, 3), np.uint8)
    img[:] = bg
    hair = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(hair, (cx, cy - ed // 2), (int(ed * 1.6), int(ed * 1.9)), 0, 0, 360, 255, -1)
    cv2.rectangle(hair, (cx - int(ed * 1.6), cy - ed // 2), (cx + int(ed * 1.6), cy + int(ed * 2.2)), 255, -1)
    cv2.rectangle(img, (cx - int(ed * 2.2), cy + int(ed * 1.9)), (cx + int(ed * 2.2), 560), (140, 60, 40), -1)
    hc = np.clip(np.array([30, 45, 80]) + rng.normal(0, 12, img.shape), 0, 255)
    img = np.where(hair[..., None] > 0, hc, img).astype(np.uint8)
    face = np.zeros_like(hair)
    cv2.ellipse(face, (cx, cy + int(ed * 0.35)), (int(ed * 1.05), int(ed * 1.4)), 0, 0, 360, 255, -1)
    img[face > 0] = (120, 150, 205)
    kps = np.array([[cx - ed / 2, cy], [cx + ed / 2, cy], [cx, cy + ed * 0.6],
                    [cx - ed * 0.4, cy + ed * 1.15], [cx + ed * 0.4, cy + ed * 1.15]], np.float32)
    return img, kps, (hair > 0) & (face == 0)


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


class HairTransferTests(unittest.TestCase):
    def test_segments_hair_not_face_or_background(self):
        img, kps, gt = portrait()
        self.assertGreater(iou(ht.segment_hair(img, kps) > 127, gt), 0.9)

    def test_umeyama_recovers_similarity(self):
        src = np.random.default_rng(1).random((5, 2)).astype(np.float32) * 100
        a = np.radians(20)
        R = 1.7 * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        dst = src @ R.T + [30, -12]
        A = ht._umeyama(src, dst)
        np.testing.assert_allclose(src @ A[:, :2].T + A[:, 2], dst, atol=1e-3)

    def test_hair_follows_head_at_any_distance(self):
        img, kps, _ = portrait()
        layer = ht.build_layer(img, kps)
        self.assertIsNotNone(layer)
        for cx, cy, ed in [(200, 200, 30), (320, 300, 110), (420, 250, 60)]:
            _, tk, gt = portrait(cx, cy, ed)
            out = ht.apply_hair(np.zeros((560, 640, 3), np.uint8), layer, tk)
            self.assertGreater(iou(out.max(axis=2) > 20, gt), 0.85, (cx, cy, ed))

    def test_no_background_halo(self):
        img, kps, _ = portrait(bg=(255, 255, 255))
        layer = ht.build_layer(img, kps)
        edge = (layer.alpha > 5) & (layer.alpha < 250)
        # edge colours come from hair, not the white background
        self.assertLess(layer.bgr[edge].mean(), 120)

    def test_bad_inputs_leave_frame_untouched(self):
        frame = np.full((50, 50, 3), 7, np.uint8)
        self.assertIs(ht.apply_hair(frame, None, np.zeros((5, 2))), frame)
        img, kps, _ = portrait()
        layer = ht.build_layer(img, kps)
        out = ht.apply_hair(frame.copy(), layer, np.full((5, 2), np.nan))
        np.testing.assert_array_equal(out, frame)


if __name__ == "__main__":
    unittest.main()
