@echo off
rem ===========================================================
rem  Deep-Live-Cam: make it use your NVIDIA graphics card.
rem  Put this file in your Deep-Live-Cam folder (next to run.py)
rem  together with install_smoother_live_swap.py, then double-click.
rem ===========================================================
cd /d "%~dp0"
if not exist run.py (
  echo This file must be in your Deep-Live-Cam folder, next to run.py.
  pause & exit /b 1
)
set "PY="
for %%P in ("venv\Scripts\python.exe" ".venv\Scripts\python.exe") do if not defined PY if exist %%P set "PY=%%~P"
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY (where py >nul 2>nul && set "PY=py -3")
if not defined PY (
  echo Python was not found. Install Deep-Live-Cam's Python first.
  pause & exit /b 1
)

echo.
echo [1/4] Checking for an NVIDIA graphics card...
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
if errorlevel 1 (
  echo.
  echo No NVIDIA card or driver found ^(nvidia-smi failed^).
  echo If you have an NVIDIA card, install its driver from nvidia.com and run this again.
  echo Otherwise send this window to Claude.
  pause & exit /b 1
)

echo.
echo [2/4] Installing the NVIDIA CUDA 12 files ^(large download, please wait^)...
%PY% -m pip install "onnxruntime-gpu[cuda,cudnn]==1.26.0" nvidia-cublas-cu12
if errorlevel 1 (
  echo.
  echo Installing failed. Send this window to Claude.
  pause & exit /b 1
)

echo.
echo [3/4] Installing the Deep-Live-Cam update...
if exist install_smoother_live_swap.py (
  %PY% install_smoother_live_swap.py --no-pause
) else (
  echo install_smoother_live_swap.py is not in this folder - skipping. Put it here and run again.
)

echo.
echo [4/4] Starting Deep-Live-Cam on the graphics card...
echo Look for "CUDAExecutionProvider" in the messages below.
echo.
%PY% run.py --execution-provider cuda
pause
