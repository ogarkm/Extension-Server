@echo off
REM ==========================================================
REM Animex Extension Server - Windows Launcher
REM No ExecutionPolicy setup required
REM ==========================================================

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*

IF ERRORLEVEL 1 (
    echo.
    echo ❌ Animex failed to start
    pause
    exit /b 1
)

exit /b 0
