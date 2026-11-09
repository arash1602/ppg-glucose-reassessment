@echo off
REM Build and flash the glucose firmware to an STM32 board through its on-board ST-LINK.
REM Usage: flash_stm32.bat f746     (STM32F746G-DISCO)
REM        flash_stm32.bat wb55     (NUCLEO-WB55)
REM Connect the board's ST-LINK USB port first. The ST-LINK appears as a USB drive; the .bin is copied to it.
setlocal
set ENV=
if /I "%1"=="f746" set ENV=disco_f746ng
if /I "%1"=="wb55" set ENV=nucleo_wb55rg_p
if "%ENV%"=="" (echo Usage: flash_stm32.bat f746  or  flash_stm32.bat wb55 & pause & exit /b 1)
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
python -m pip show platformio >nul 2>&1 || python -m pip install -U platformio
python export_device_test.py || goto :err
cd firmware
echo Building and flashing %ENV% (the first build downloads the STM32 toolchain, ~5-10 min)...
pio run -e %ENV% -t upload || goto :err
echo.
echo Flashed. Press the black RESET button, then watch the self-test (Ctrl+C to quit the monitor):
pio device monitor -e %ENV%
exit /b 0
:err
echo FAILED - see the lines above.
pause
exit /b 1
