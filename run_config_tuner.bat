@echo off
title Rebound Strategy - CONFIG TUNER (1 random trial)
cd /d "%~dp0"

echo.
echo === Randomized config trial for rebound_sim_include_held ===
echo Picks one random combination of backtest/recovery settings, runs the
echo simulation once, records the result, then reverts config.yaml.
echo Run this again (double-click) to add more trials to the results file.
echo.

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: virtual environment not found at .venv
  pause
  exit /b 1
)

.venv\Scripts\python.exe -m scripts.rebound_config_tuner

echo.
echo === Done. Result appended to reports\tuning\rebound_include_held_search.jsonl ===
pause
