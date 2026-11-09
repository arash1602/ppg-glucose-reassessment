@echo off
REM MUST external test for the deep models (needs the GPU environment). Log: results\must_cnn_log.txt
setlocal
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
(python 06_must_external.py --run tiny_beat_subject & python 06_must_external.py --run resnet34_beat_subject) > "%~dp0..\results\must_cnn_log.txt" 2>&1
type "%~dp0..\results\must_cnn_log.txt"
pause
