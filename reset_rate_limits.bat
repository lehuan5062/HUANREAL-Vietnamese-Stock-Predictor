@echo off
title Vietnamese T+2 Predictor - RESET RATE LIMITS
cd /d "%~dp0"

echo.
echo === Reset per-source rate limits ===
echo Clears cache\source_rate.json (the persisted rate ratchet and cooldown
echo growth from past 429s) so VCI / KBS / MSN all restart at their
echo configured starting rate and cooldown (api_per_min / api_per_min_overrides
echo / cooldown_start_seconds in config.yaml).
echo.

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: virtual environment not found at .venv
  pause
  exit /b 1
)

.venv\Scripts\python.exe -c "from stockpredict.data.source_rate import reset_rates; reset_rates(); print('Rate limits and cooldowns reset to config defaults.')"

echo.
pause
