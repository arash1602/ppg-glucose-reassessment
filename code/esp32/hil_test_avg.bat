@echo off
REM Hardware-in-the-loop test of the K-minute averaged device model (after run_device_avg.bat and reflashing).
REM Usage: hil_test_avg.bat [COM port, default COM3]. Close any open serial monitor first.
setlocal
set PORT=%1
if "%PORT%"=="" set PORT=COM3
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
python -m pip install --quiet pyserial
python stream_test.py --port %PORT% --avg
pause
