@echo off
setlocal
cd /d "%~dp0"
echo Running one record as a smoke test...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_from_paired_data.ps1" -PairedDataJson "%~dp0data\paired_data.local.json" -OutputRoot "%~dp0output_smoke" -JmPackageRoot "%~dp0JM-backend" -Steps 20 -MaxRecords 1 -Resume
if errorlevel 1 (
  echo.
  echo Smoke test failed. Read the error above.
  pause
  exit /b 1
)
echo.
echo Smoke test completed.
echo Dataset summary: %~dp0output_smoke\training_dataset\dataset_summary.json
pause
exit /b 0
