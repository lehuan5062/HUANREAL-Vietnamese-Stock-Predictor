@echo off
title Vietnamese T+2 Predictor - BASE mode
cd /d "%~dp0"

echo.
echo === Vietnamese T+2 Stock Predictor (BASE mode) ===
echo Pure ML + technical filter. No news, no LLM.
echo.

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: virtual environment not found at .venv
  echo Run setup first: py -3.13 -m venv .venv ^&^& .venv\Scripts\pip install -e ".[dev,llm]"
  pause
  exit /b 1
)

rem Horizon is always T+2 (Vietnamese settlement). Choose how many picks.
set /p PICKS=Number of picks to return [1]:
if "%PICKS%"=="" set PICKS=1

rem Pricing is per share; position sizing is left to the user.

set HOSE_FLAG=
set /p HOSE=HOSE-only (exclude HNX/UPCOM)? y/n [n]:
if /I "%HOSE%"=="y" set HOSE_FLAG=--hose-only
if "%HOSE%"=="" set HOSE=n

set ETF_FLAG=
set /p ETFS=Include HOSE ETFs (FUEVFVND, E1VFVN30, ...)? y/n [y]:
if /I "%ETFS%"=="n" set ETF_FLAG=--no-etfs
if "%ETFS%"=="" set ETFS=y

rem Per-session ticker blacklist. Comma-separated (e.g. ACB,HPG). Empty = none.
rem Excluded tickers are stripped from every universe layer + the prediction
rem panel. Picks JSON name gets a _xACB-HPG suffix so it doesn't collide with
rem a same-day full run.
set EXCLUDE_FLAG=
set EXCLUDE=
set /p EXCLUDE=Exclude tickers? comma-separated, empty for none []:
if not "%EXCLUDE%"=="" set EXCLUDE_FLAG=--exclude %EXCLUDE%

rem Warm-only -- three modes:
rem   y (default) = smart lazy fetch (skip warm, fetch stale + cold)
rem   a / always  = pure offline (use cached only, no API calls EVER)
rem   n           = force full re-fetch of every symbol (slow, rare)
set WARM_VALUE=yes
set /p WARM=Warm-only? [y]es lazy fetch / [a]lways offline / [n]o full refetch [y]:
if /I "%WARM%"=="n" set WARM_VALUE=no
if /I "%WARM%"=="a" set WARM_VALUE=always
if /I "%WARM%"=="always" set WARM_VALUE=always
if "%WARM%"=="" set WARM=y
set WARM_FLAG=--warm-only %WARM_VALUE%

echo.
echo Running: picks=%PICKS%  hose-only=%HOSE%  etfs=%ETFS%  exclude=%EXCLUDE%  warm-only=%WARM_VALUE%  mode=base
echo.

.venv\Scripts\python.exe -m stockpredict.cli run --picks %PICKS% %HOSE_FLAG% %ETF_FLAG% %EXCLUDE_FLAG% %WARM_FLAG% --mode base

echo.
echo === Done. Picks saved to reports\ ===
pause
