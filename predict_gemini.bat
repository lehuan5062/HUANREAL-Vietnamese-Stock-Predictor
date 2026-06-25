@echo off
title Vietnamese T+2 Predictor - GEMINI mode
cd /d "%~dp0"

echo.
echo === Vietnamese T+2 Stock Predictor (GEMINI mode) ===
echo Step 1: ML stage runs locally and writes a prompt file.
echo Step 2: You paste the prompt into Gemini Chat (web, with browsing).
echo Step 3: Save Gemini's JSON response, then re-run this .bat to finalize.
echo.

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: virtual environment not found at .venv
  echo Run setup first: py -3.13 -m venv .venv ^&^& .venv\Scripts\pip install -e ".[dev,llm]"
  pause
  exit /b 1
)

:menu
echo Choose:
echo   1) Run ML stage and emit Gemini prompt (step 1)
echo   2) Finalize with Gemini's saved response (step 3)
set /p CHOICE=Enter 1 or 2:
if "%CHOICE%"=="1" goto step1
if "%CHOICE%"=="2" goto step3
echo Invalid choice.
goto menu

:step1
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
rem panel. Prompt + picks JSON name gets a _xACB-HPG suffix so it doesn't
rem collide with a same-day full run.
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

rem Missed-winners variant: ON by default. UNIONs the variant's top picks into
rem the candidates Gemini researches (flagged in the prompt) so it judges both
rem rankings. Answer n to research only the standard candidates.
set MISSED_FLAG=
set /p MISSED=Include the missed-winners variant candidates (union)? y/n [y]:
if /I "%MISSED%"=="n" set MISSED_FLAG=--no-missed

rem Standard-vs-missed A/B backtest: ON by default for Gemini mode, so the prompt
rem embeds the verdict for Gemini to weigh. SLOW (~10 min). Answer n to skip;
rem the prompt then uses the most recent prior A/B report if one exists.
set AB_FLAG=--ab
set /p AB=Run the standard-vs-missed A/B backtest (verdict goes into the prompt)? y/n [y]:
if /I "%AB%"=="n" set AB_FLAG=--no-ab

echo.
echo Running ML stage: picks=%PICKS%  hose-only=%HOSE%  etfs=%ETFS%  exclude=%EXCLUDE%  warm-only=%WARM_VALUE%  missed=%MISSED%  ab=%AB%
echo.

.venv\Scripts\python.exe -m stockpredict.cli run --picks %PICKS% %HOSE_FLAG% %ETF_FLAG% %EXCLUDE_FLAG% %WARM_FLAG% %MISSED_FLAG% %AB_FLAG% --mode gemini

echo.
echo Opening today's prompt file in Notepad.
echo.
echo NEXT:
echo   1. Copy the prompt's contents into Gemini Chat (gemini.google.com).
echo   2. Make sure Gemini has browsing enabled so it can search the web.
echo   3. Save Gemini's JSON response to:
echo      reports\gemini_response_YYYY-MM-DD.json   (same date as the prompt)
echo   4. Re-run this .bat and choose option 2 to finalize.
echo.

for /f %%f in ('dir /b /o:-d reports\gemini_prompt_*.txt 2^>nul') do (
  start notepad "reports\%%f"
  goto step1_done
)
:step1_done
pause
exit /b 0

:step3
echo.
for /f %%f in ('dir /b /o:-d reports\gemini_prompt_*.txt 2^>nul') do (
  set "PROMPT_FILE=reports\%%f"
  goto found_prompt
)
echo No prompt file found in reports\ -- run step 1 first.
pause
exit /b 1

:found_prompt
echo Using prompt: %PROMPT_FILE%
.venv\Scripts\python.exe -m stockpredict.cli gemini-finalize "%PROMPT_FILE%"
echo.
echo === Done. Final picks (with explanations) shown above. ===
pause
