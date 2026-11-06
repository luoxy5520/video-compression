@echo off
setlocal
cd /d "%~dp0"
set "PREDICTION=%~1"
if "%PREDICTION%"=="" set /p "PREDICTION=Enter the full path to prediction.npz: "
set "SAMPLE_ID=%~2"
if "%SAMPLE_ID%"=="" set "SAMPLE_ID=sample_0000"
echo Running prediction to H.264 for %SAMPLE_ID%...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_prediction_to_h264.ps1" -PredictionNpz "%PREDICTION%" -SampleId "%SAMPLE_ID%"
if errorlevel 1 (
  echo.
  echo Prediction-to-H.264 test failed. Read the error above.
  pause
  exit /b 1
)
echo.
echo Prediction-to-H.264 completed. Results are under chain_tests\inferred\%SAMPLE_ID%\
pause
exit /b 0
