@echo off
title Rebound Strategy - CONFIG SUGGEST (analyze tuner history)
cd /d "%~dp0"

echo.
echo === Analyze accumulated config-tuner trials ===
echo Reads reports\tuning\rebound_include_held_search.jsonl and prints which
echo knob values correlate with higher annualized_IRR. Read-only — writes
echo nothing, does not touch config.yaml.
echo.

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: virtual environment not found at .venv
  pause
  exit /b 1
)

.venv\Scripts\python.exe -m scripts.rebound_config_suggest

echo.
pause
