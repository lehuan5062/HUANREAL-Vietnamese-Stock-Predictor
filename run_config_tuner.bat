@echo off
title Rebound Strategy - CONFIG TUNER (looping until stopped)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: virtual environment not found at .venv
  pause
  exit /b 1
)

echo.
echo === Randomized config trials for rebound_sim_include_held ===
echo Runs continuously: each loop picks one random config, runs the sim,
echo appends the result, then reverts config.yaml.
echo Press Ctrl+C to stop (then press Y to confirm).
echo.

set /a TRIAL=0

:loop
set /a TRIAL+=1
echo.
echo ==================== Trial %TRIAL% ====================
start "" /low /wait /b .venv\Scripts\python.exe -m scripts.rebound_config_tuner
goto :loop
