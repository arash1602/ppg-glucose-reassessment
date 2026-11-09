@echo off
REM Firmware improvement: does averaging device features over several minutes help?
REM Uses all CPU cores (a few minutes). Log: results\device_avg_log.txt
setlocal
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
echo Running... (log: results\device_avg_log.txt)
python 14_device_avg.py > "%~dp0..\results\device_avg_log.txt" 2>&1 || (echo FAILED - see the log in results & pause & exit /b 1)
type "%~dp0..\results\device_avg_log.txt"
pause
