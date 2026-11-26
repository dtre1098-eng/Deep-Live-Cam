#!/usr/bin/env bash
# Run to install the smoother live swap update into Deep-Live-Cam (macOS/Linux).
cd "$(dirname "$0")"
for p in venv/bin/python .venv/bin/python ../venv/bin/python ../.venv/bin/python; do
  [ -x "$p" ] && exec "$p" install_smoother_live_swap.py "$@"
done
exec python3 install_smoother_live_swap.py "$@"
