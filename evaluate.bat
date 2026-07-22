@echo off
title Vietnamese T+2 Predictor - EVALUATE
cd /d "%~dp0"

echo.
echo === Evaluate past predictions ===
echo Refreshes data and scores any picks whose T+2 has now elapsed.
echo Updates the predictions ledger that Claude consults for feedback.
echo.

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: virtual environment not found at .venv
  pause
  exit /b 1
)

.venv\Scripts\python.exe -m stockpredict.cli evaluate

echo.
echo === Done. Recent performance shown above. ===
pause
