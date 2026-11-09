@echo off
REM Open the ESP32 serial monitor only (no rebuild). Board on its "USB" port. Type s + Enter to rerun the self-test.
REM After it opens, press RESET on the board to see the self-test.
setlocal
set PIO=%USERPROFILE%\.platformio\penv\Scripts\pio.exe
if not exist "%PIO%" set PIO=C:\glucose_venv\Scripts\pio.exe
cd /d "%~dp0"
"%PIO%" device list
echo.
echo Opening monitor. Press RESET on the board now. Ctrl+C to quit.
"%PIO%" device monitor -d firmware -b 921600
