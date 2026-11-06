@echo off
setlocal
cd /d "%~dp0"
set "FAILED=0"
where python >nul 2>nul
if errorlevel 1 (echo [MISSING] Python is not in PATH.& set "FAILED=1") else (echo [OK] Python)
python -c "import numpy; print('[OK] NumPy ' + numpy.__version__)" 2>nul
if errorlevel 1 (echo [MISSING] NumPy. Run: python -m pip install numpy& set "FAILED=1")
where ffmpeg >nul 2>nul
if errorlevel 1 (echo [MISSING] FFmpeg is not in PATH.& set "FAILED=1") else (echo [OK] FFmpeg)
where ffprobe >nul 2>nul
if errorlevel 1 (echo [MISSING] FFprobe is not in PATH.& set "FAILED=1") else (echo [OK] FFprobe)
if exist "%~dp0JM-backend\bin\lencod_state_b1.exe" (echo [OK] JM encoder) else (echo [MISSING] JM encoder& set "FAILED=1")
if exist "%~dp0JM-backend\bin\ldecod_ref.exe" (echo [OK] JM decoder) else (echo [MISSING] JM decoder& set "FAILED=1")
if exist "%~dp0data\paired_data.local.json" (echo [OK] paired_data.local.json) else (echo [MISSING] paired_data.local.json& set "FAILED=1")
if "%FAILED%"=="0" (echo.& echo Environment check passed.) else (echo.& echo Environment check failed.)
pause
exit /b %FAILED%
