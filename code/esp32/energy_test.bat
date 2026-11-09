@echo off
REM Energy per estimate with a multimeter. Usage: energy_test.bat COM5 921600 5.0 "USB 5 V"
REM   ESP32-S3: baud 921600 | STM32 boards: baud 460800 | volts: 5.0 at USB, 3.3 at the WB55 JP2 jumper
setlocal
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
python -m pip install --quiet pyserial
python energy_bench.py --port %1 --baud %2 --volts %3 --where %4
pause
