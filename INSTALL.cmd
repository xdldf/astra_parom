@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1"
if errorlevel 1 (
  echo Installation failed. See the message above.
  pause
  exit /b 1
)
echo Installation complete. Run START.cmd.
pause
