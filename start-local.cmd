@echo off
setlocal
cd /d "%~dp0"
set "DKL_STOP_BACKEND_ON_EXIT=1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\dev.ps1"
if errorlevel 1 pause
