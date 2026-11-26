@echo off
rem Double-click: checks whether Deep-Live-Cam uses your graphics card and how fast the swap runs.
cd /d "%~dp0"
set "PY="
for %%P in ("venv\Scripts\python.exe" ".venv\Scripts\python.exe") do if not defined PY if exist %%P set "PY=%%~P"
if not defined PY (where py >nul 2>nul && set "PY=py -3")
if not defined PY set "PY=python"
%PY% check_speed.py
