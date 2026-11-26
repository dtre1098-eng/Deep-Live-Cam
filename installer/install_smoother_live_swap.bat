@echo off
rem Double-click to install the smoother live swap update into Deep-Live-Cam.
cd /d "%~dp0"
set "PY="
for %%P in ("venv\Scripts\python.exe" ".venv\Scripts\python.exe" "..\venv\Scripts\python.exe" "..\.venv\Scripts\python.exe") do if not defined PY if exist %%P set "PY=%%~P"
if not defined PY (where py >nul 2>nul && set "PY=py -3")
if not defined PY set "PY=python"
%PY% install_smoother_live_swap.py %*
