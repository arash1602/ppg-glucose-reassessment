@echo off
REM Hardware-in-the-loop test on an STM32 board (same 30 VitalDB recordings as the ESP32).
REM Usage: hil_test_stm32.bat COM7      (close any serial monitor first)
setlocal
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
python -m pip install --quiet pyserial
python stream_test.py --port %1 --baud 460800 --avg --out esp32_hil_stm32_%1.csv
pause
