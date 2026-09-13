@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run INSTALL.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "scripts\run_station.py"
if errorlevel 1 pause
