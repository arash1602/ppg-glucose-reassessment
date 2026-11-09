@echo off
REM Multi-signal improvement: download ECG, arterial pressure, monitor numerics and drug rates
REM for the same VitalDB glucose readings (resumable, a few hours), then evaluate.
REM Log: results\multisignal_log.txt
setlocal
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
echo [1/2] Downloading multi-signal features (can be stopped and restarted)...
python 09_multisignal.py --workers 2 > "%~dp0..\results\multisignal_log.txt" 2>&1 || (echo FAILED - see the log in results & pause & exit /b 1)
echo [2/2] Evaluating...
python 10_multisignal_eval.py >> "%~dp0..\results\multisignal_log.txt" 2>&1 || (echo FAILED - see the log in results & pause & exit /b 1)
echo DONE. Results: results\multisignal_tiers.csv and results\multisignal_calibrated.csv
pause
