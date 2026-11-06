@echo off
setlocal
if "%~1"=="" goto :usage
set "PAIRED_JSON=%~f1"
if "%~2"=="" (
  set "OUTPUT_ROOT=%~dp1oracle_dataset_output"
) else (
  set "OUTPUT_ROOT=%~f2"
)
if not "%~3"=="" goto :explicit_jm
if exist "%~dp0JM-backend\scripts\run_b1_closure.ps1" goto :bundled_jm
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_from_paired_data.ps1" -PairedDataJson "%PAIRED_JSON%" -OutputRoot "%OUTPUT_ROOT%" -Resume
goto :done
:explicit_jm
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_from_paired_data.ps1" -PairedDataJson "%PAIRED_JSON%" -OutputRoot "%OUTPUT_ROOT%" -JmPackageRoot "%~f3" -Resume
goto :done
:bundled_jm
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_from_paired_data.ps1" -PairedDataJson "%PAIRED_JSON%" -OutputRoot "%OUTPUT_ROOT%" -JmPackageRoot "%~dp0JM-backend" -Resume
goto :done
:usage
echo Usage:
echo   run_windows.bat "D:\data\paired_data.local.json" "D:\output" "D:\JM-backend"
echo.
echo The second argument is optional. The third argument is optional only after
echo setup_local.ps1 was run, or when JM-backend is placed beside this BAT file.
exit /b 2
:done
if errorlevel 1 (
  echo.
  echo Pipeline failed. Read the error above; completed records can be resumed.
  exit /b 1
)
echo.
echo Pipeline completed: %OUTPUT_ROOT%
exit /b 0
