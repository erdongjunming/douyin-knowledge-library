@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\create-desktop-shortcut.ps1"
if errorlevel 1 (
    echo.
    echo Unable to create the desktop shortcut. See the error above.
    pause
    exit /b 1
)

echo.
echo The desktop shortcut is ready.
pause
