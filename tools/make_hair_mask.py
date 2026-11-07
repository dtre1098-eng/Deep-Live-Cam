"""Make (and touch up) the hair mask used by "Use Source Hair".

Deep-Live-Cam cuts the hair out of your source picture automatically. On
some photos (busy background, hair a similar colour to the background or
skin) the automatic cut-out misses bits or grabs background. This script
lets you check it and fix it once; the app then always uses your mask.

Usage (from the Deep-Live-Cam folder):

    python tools/make_hair_mask.py path/to/picture.jpg

It saves ``picture.hair.png`` next to the picture (white = hair) and
``picture.hair-preview.jpg`` so you can see what will be used.

Editor controls (window opens unless you pass --no-gui):
    left mouse drag   paint hair (add)
    right mouse drag  erase (not hair)
    [  /  ]           smaller / bigger brush
    a                 redo the automatic cut-out
    v                 toggle view: overlay / mask only
    s                 save and quit
    q / Esc           quit without saving
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules.processors.frame import hair_transfer  # noqa: E402


def mask_path_for(picture: str) -> str:
    return os.path.splitext(picture)[0] + ".hair.png"


def preview_path_for(picture: str) -> str:
    return os.path.splitext(picture)[0] + ".hair-preview.jpg"


def read_image(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read picture: {path}")
    return img


def write_image(path: str, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if not ok:
        raise SystemExit(f"Could not write {path}")
    buf.tofile(path)


def detect_kps(img: np.ndarray) -> np.ndarray:
    """5 face keypoints of the main face, using the app's own detector."""
    import modules.globals
    if not modules.globals.execution_providers:
        modules.globals.execution_providers = ["CPUExecutionProvider"]
    from modules.face_analyser import get_many_faces
    faces = get_many_faces(img) or []
    faces = [f for f in faces if getattr(f, "kps", None) is not None]
    if not faces:
        raise SystemExit("No face found in the picture.")
    # biggest face = the person the picture is of
    best = max(faces, key=lambda f: float(np.linalg.norm(f.kps[1] - f.kps[0])))
    return np.asarray(best.kps, np.float32)


def overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    tint = img.copy()
    tint[mask > 127] = (0.5 * tint[mask > 127] + [0, 0, 127]).astype(np.uint8)
    edges = cv2.Canny(mask, 50, 150)
    tint[edges > 0] = (0, 255, 255)
    return tint


def edit(img: np.ndarray, mask: np.ndarray, kps: np.ndarray):
    """Interactive brush editor. Returns the edited mask, or None if cancelled."""
    h, w = img.shape[:2]
    scale = min(1.0, 900 / max(h, w))
    state = {"mask": mask.copy(), "brush": max(4, int(np.linalg.norm(kps[1] - kps[0]) * 0.15)),
             "paint": None, "mask_view": False, "cursor": (0, 0)}
    win = "Hair mask - left: add, right: erase, [ ]: brush, a: auto, v: view, s: save, q: quit"

    def on_mouse(event, x, y, flags, _):
        p = (int(x / scale), int(y / scale))
        state["cursor"] = p
        if event == cv2.EVENT_LBUTTONDOWN:
            state["paint"] = 255
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["paint"] = 0
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            state["paint"] = None
        if state["paint"] is not None:
            cv2.circle(state["mask"], p, state["brush"], state["paint"], -1)

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = (cv2.cvtColor(state["mask"], cv2.COLOR_GRAY2BGR)
                if state["mask_view"] else overlay(img, state["mask"]))
        cv2.circle(view, state["cursor"], state["brush"], (255, 255, 255), 1)
        if scale != 1.0:
            view = cv2.resize(view, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imshow(win, view)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            cv2.destroyAllWindows()
            return None
        if k == ord("s"):
            cv2.destroyAllWindows()
            return state["mask"]
        if k == ord("["):
            state["brush"] = max(1, int(state["brush"] / 1.3))
        if k == ord("]"):
            state["brush"] = int(state["brush"] * 1.3) + 1
        if k == ord("v"):
            state["mask_view"] = not state["mask_view"]
        if k == ord("a"):
            state["mask"] = hair_transfer.segment_hair(img, kps)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("picture", help="your source picture")
    ap.add_argument("--no-gui", action="store_true", help="just save the automatic mask, no editor")
    ap.add_argument("--redo", action="store_true", help="ignore an existing .hair.png and start from the automatic cut-out")
    args = ap.parse_args(argv)

    img = read_image(args.picture)
    kps = detect_kps(img)
    out = mask_path_for(args.picture)

    if os.path.exists(out) and not args.redo:
        print(f"Editing existing mask {out} (use --redo to start over)")
        mask = cv2.imdecode(np.fromfile(out, np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape[:2] != img.shape[:2]:
            mask = hair_transfer.segment_hair(img, kps)
    else:
        print("Cutting out the hair automatically...")
        mask = hair_transfer.segment_hair(img, kps)

    if not args.no_gui:
        try:
            edited = edit(img, mask, kps)
        except cv2.error as e:
            print(f"Can't open a window ({e.err if hasattr(e, 'err') else e}); saving the automatic mask.")
            edited = mask
        if edited is None:
            print("Cancelled, nothing saved.")
            return 1
        mask = edited

    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    write_image(out, mask)
    write_image(preview_path_for(args.picture), overlay(img, mask))
    hair_transfer.reset_cache()
    print(f"Saved {out}")
    print(f"Preview: {preview_path_for(args.picture)}")
    print("Deep-Live-Cam will use this mask automatically for this picture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
