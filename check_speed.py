"""Deep-Live-Cam speed check-up.

Double-click check_speed.bat (in your Deep-Live-Cam folder). It tells you:
  * which Python is used and whether the GPU build of onnxruntime is installed
  * which processors onnxruntime can use (graphics card or only the CPU)
  * your graphics card
  * how long one face swap really takes on each available processor
and saves everything to speed_report.txt so you can send it.
Changes nothing on your computer.
"""

import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LINES = []


def out(msg=""):
    print(msg, flush=True)
    LINES.append(msg)


def pip_has(name):
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        return None


def gpu_name():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            return "NVIDIA: " + r.stdout.strip()
    except Exception:
        pass
    if os.name == "nt":
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "(Get-CimInstance Win32_VideoController).Name"],
                               capture_output=True, text=True, timeout=20)
            if r.stdout.strip():
                return r.stdout.strip().replace("\n", " | ")
        except Exception:
            pass
    return "unknown"


def time_swap(model_path, provider):
    import numpy as np
    import onnxruntime as ort
    opts = ort.SessionOptions()
    sess = ort.InferenceSession(model_path, sess_options=opts, providers=[provider] if provider == "CPUExecutionProvider" else [provider, "CPUExecutionProvider"])
    used = sess.get_providers()[0]
    inputs = sess.get_inputs()
    feed = {}
    for inp in inputs:
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        dtype = np.float16 if "float16" in inp.type else np.float32
        feed[inp.name] = np.random.rand(*shape).astype(dtype)
    for _ in range(3):
        sess.run(None, feed)  # warm-up
    n, t0 = 20, time.perf_counter()
    for _ in range(n):
        sess.run(None, feed)
    ms = (time.perf_counter() - t0) * 1000 / n
    return used, ms


def main():
    out("Deep-Live-Cam speed check-up")
    out("=" * 40)
    out(f"Folder:  {HERE}")
    out(f"Python:  {sys.executable}  ({platform.python_version()})")
    out(f"System:  {platform.platform()}")
    out(f"Graphics card: {gpu_name()}")
    out("")
    ort_cpu, ort_gpu, ort_dml = pip_has("onnxruntime"), pip_has("onnxruntime-gpu"), pip_has("onnxruntime-directml")
    out(f"onnxruntime (CPU only) installed: {ort_cpu or 'no'}")
    out(f"onnxruntime-gpu (NVIDIA)  installed: {ort_gpu or 'no'}")
    out(f"onnxruntime-directml      installed: {ort_dml or 'no'}")
    if ort_cpu and (ort_gpu or ort_dml):
        out("  PROBLEM: both the CPU-only and a GPU onnxruntime are installed. They clash and")
        out("  usually leave you on the CPU. Fix: uninstall both, then install only the GPU one.")
    try:
        import onnxruntime as ort
        provs = ort.get_available_providers()
    except Exception as e:
        out(f"Could not load onnxruntime: {e}")
        return
    out(f"Processors onnxruntime can use: {', '.join(provs)}")
    torch_v = pip_has("torch")
    if torch_v:
        try:
            import torch
            out(f"PyTorch {torch_v}: CUDA available = {torch.cuda.is_available()}")
        except Exception as e:
            out(f"PyTorch {torch_v}: could not load ({e})")
    out("")

    model = None
    for name in ("inswapper_128_fp16.onnx", "inswapper_128.onnx"):
        p = os.path.join(HERE, "models", name)
        if os.path.exists(p):
            model = p
            break
    if model is None:
        out("No inswapper model found in the models folder -- can't time the swap.")
    else:
        out(f"Timing the face-swap model ({os.path.basename(model)}) -- takes a few seconds...")
        results = {}
        for p in [x for x in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider") if x in provs]:
            try:
                used, ms = time_swap(model, p)
                if used != p:
                    out(f"  {p}: could not start, fell back to {used}")
                    continue
                results[p] = ms
                out(f"  {p:24s} {ms:7.1f} ms per face  (~{1000 / ms:5.1f} swaps per second)")
            except Exception as e:
                out(f"  {p}: failed ({str(e).splitlines()[0][:150]})")
        out("")
        gpu = {k: v for k, v in results.items() if k != "CPUExecutionProvider"}
        if not gpu:
            out("RESULT: only the CPU works. That is why Live is slow.")
            out("  - NVIDIA card: install the GPU build (see README), then start with run-cuda.bat")
            out("  - AMD/Intel card: install onnxruntime-directml, then start with run-directml.bat")
            out("  - Or turn on 'Async swap' so the preview stays smooth on the CPU.")
        else:
            best = min(gpu, key=gpu.get)
            bat = "run-cuda.bat" if best.startswith("CUDA") else "run-directml.bat"
            out(f"RESULT: your graphics card works ({best}, {gpu[best]:.0f} ms per face).")
            out(f"  Start Deep-Live-Cam with {bat} so it uses it.")
            if gpu[best] > 40:
                out("  The card is fairly slow for this model; 'Async swap' will keep the preview smooth.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        out(f"Unexpected error: {e}")
    try:
        with open(os.path.join(HERE, "speed_report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(LINES) + "\n")
        print("\nSaved to speed_report.txt -- send this file (or a screenshot) to Claude.")
    except Exception:
        pass
    if os.name == "nt" and sys.stdin and sys.stdin.isatty():
        input("\nPress Enter to close...")
