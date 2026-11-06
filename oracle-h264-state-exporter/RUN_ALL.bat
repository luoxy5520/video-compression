@echo off
setlocal
cd /d "%~dp0"
echo Building Oracle states and the training dataset for all records...
call "%~dp0run_windows.bat" "%~dp0data\paired_data.local.json" "%~dp0output" "%~dp0JM-backend"
if errorlevel 1 (
  echo.
  echo Full pipeline failed. Read the error above. Run RUN_ALL.bat again to resume.
  pause
  exit /b 1
)
echo.
echo Full dataset completed.
echo Dataset: %~dp0output\training_dataset
pause
exit /b 0
