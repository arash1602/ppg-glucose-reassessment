@echo off
REM Stream 60 real VitalDB windows to the ESP32 and compare with Python. Double-click (uses COM3) or: hil_test.bat COM5
REM Close any open serial monitor first.
setlocal
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
python -m pip install --quiet pyserial
set PORT=%1
if "%PORT%"=="" set PORT=COM3
echo Using port %PORT% (close the serial monitor first)
python stream_test.py --port %PORT%
pause
