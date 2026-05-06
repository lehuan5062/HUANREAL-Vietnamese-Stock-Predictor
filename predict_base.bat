@echo off
title Vietnamese T+N Predictor - BASE mode
cd /d "%~dp0"

echo.
echo === Vietnamese T+N Stock Predictor (BASE mode) ===
echo Pure ML + technical filter. No news, no LLM.
echo.

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: virtual environment not found at .venv
  echo Run setup first: py -3.13 -m venv .venv ^&^& .venv\Scripts\pip install -e ".[dev,llm]"
  pause
  exit /b 1
)

set /p DURATION=Time budget in minutes, or 'full' for entire universe [full]:
if "%DURATION%"=="" set DURATION=full

set /p DAYS=Exit horizon: integer T+N (min 2) / 'end' (last trading day of month) / 'earliest' (smallest horizon with an actionable pick -- runs until found, no upper cap) [2]:
if "%DAYS%"=="" set DAYS=2

rem When days=earliest, ask for the starting T+N of the search.
rem Ignored otherwise.
set EARLIEST_START_FLAG=
if /I "%DAYS%"=="earliest" goto ask_earliest_start
goto skip_earliest_start

:ask_earliest_start
set /p EARLIEST_START=Earliest-start: T+N to begin the search (min 2) [2]:
if "%EARLIEST_START%"=="" set EARLIEST_START=2
set EARLIEST_START_FLAG=--earliest-start %EARLIEST_START%

:skip_earliest_start
set /p UNITS=Units per pick (min 100, multiple of 100) [100]:
if "%UNITS%"=="" set UNITS=100

set HOSE_FLAG=
set /p HOSE=HOSE-only (exclude HNX/UPCOM)? y/n [n]:
if /I "%HOSE%"=="y" set HOSE_FLAG=--hose-only
if "%HOSE%"=="" set HOSE=n

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
echo Running: duration=%DURATION%  days=%DAYS%  units=%UNITS%  hose-only=%HOSE%  warm-only=%WARM_VALUE%  mode=base
echo.

.venv\Scripts\python.exe -m stockpredict.cli run --duration %DURATION% --days %DAYS% %EARLIEST_START_FLAG% --units %UNITS% %HOSE_FLAG% %WARM_FLAG% --mode base

echo.
echo === Done. Picks saved to reports\ ===
pause
