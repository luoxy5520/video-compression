@echo off
setlocal
cd /d "%~dp0"
set "SAMPLE_ID=%~1"
if "%SAMPLE_ID%"=="" set "SAMPLE_ID=sample_0000"
echo Running ideal-label to H.264 chain test for %SAMPLE_ID%...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_ideal_chain_test.ps1" -SampleId "%SAMPLE_ID%"
if errorlevel 1 (
  echo.
  echo Ideal chain test failed. Read the error above.
  pause
  exit /b 1
)
echo.
echo Ideal chain test completed. Results are under chain_tests\ideal\%SAMPLE_ID%\
pause
exit /b 0
