@echo off
REM Reviewer analyses: causal (pre-draw) PPG windows + leave-one-group-out ablation. CPU only, ~15 min.
REM Can run while run_robustness.bat uses the GPU. Log: results\reviewer_log.txt
setlocal
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
python 16_causal_ablation.py > ..\results\reviewer_log.txt 2>&1
type ..\results\reviewer_log.txt | more
echo.
echo FINISHED.
pause
