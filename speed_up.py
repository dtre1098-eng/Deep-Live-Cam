"""Deep-Live-Cam: switch on the fastest live settings for a computer without a
usable graphics card. Double-click speed_up.bat in your Deep-Live-Cam folder.

Sets (in switch_states.json, which Deep-Live-Cam reads at start-up):
  * Det size 320 x 320       -- faster face search
  * Resolution 640 x 480     -- cheapest camera mode
  * Async swap ON            -- smooth preview on a slow processor
  * Face enhancer None       -- enhancers are very slow live
A copy of your old settings is kept as switch_states.before_speed_up.json.
Run with --undo to put the old settings back. Close Deep-Live-Cam first.
"""

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "switch_states.json")
BACKUP = os.path.join(HERE, "switch_states.before_speed_up.json")


def main() -> int:
    if not os.path.isfile(os.path.join(HERE, "run.py")):
        print("Put this file in your Deep-Live-Cam folder (next to run.py) and run it there.")
        return 1
    if "--undo" in sys.argv:
        if not os.path.exists(BACKUP):
            print("No saved settings to restore.")
            return 1
        shutil.copy2(BACKUP, STATE)
        print("Your previous settings are back.")
        return 0
    state = {}
    if os.path.exists(STATE):
        try:
            with open(STATE, "r") as f:
                state = json.load(f)
        except Exception:
            state = {}
        if not os.path.exists(BACKUP):
            shutil.copy2(STATE, BACKUP)
    state["det_size"] = 320
    state["capture_resolution"] = [640, 480]
    state["live_async_swap"] = True
    state["fp_ui"] = {"face_enhancer": False, "face_enhancer_gpen256": False,
                      "face_enhancer_gpen512": False}
    with open(STATE, "w") as f:
        json.dump(state, f)
    print("Done. Fast live settings are on:")
    print("  Det size 320 x 320, Resolution 640 x 480, Async swap ON, Face enhancer None.")
    print("Start Deep-Live-Cam now (close it first if it was open).")
    print("To undo: run  python speed_up.py --undo")
    return 0


if __name__ == "__main__":
    code = main()
    if os.name == "nt" and sys.stdin and sys.stdin.isatty():
        input("\nPress Enter to close...")
    sys.exit(code)
