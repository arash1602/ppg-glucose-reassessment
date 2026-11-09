@echo off
REM Build + flash the ESP32 firmware (device model + CNN comparison) and open the monitor.
REM Plug the ESP32-S3 into its "USB" port first.
setlocal
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
set PIO=%USERPROFILE%\.platformio\penv\Scripts\pio.exe
if not exist "%PIO%" set PIO=C:\glucose_venv\Scripts\pio.exe
if not exist "%PIO%" (python -m pip install --quiet platformio || goto :err)
if not exist firmware\include\device_model.h (echo device_model.h missing - run 13_device_train.py first & goto :err)
python export_device_test.py || goto :err
"%PIO%" run -d firmware -e esp32-s3-devkitc-1 -t upload || goto :err
echo.
echo Flashed. Opening serial monitor (Ctrl+C to quit). Type s + Enter to rerun the self-test.
"%PIO%" device monitor -d firmware -b 921600
exit /b 0
:err
echo FAILED - see the error text above.
pause
exit /b 1
