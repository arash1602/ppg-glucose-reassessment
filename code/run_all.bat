@echo off
REM Full experiment. Every step is resumable / safe to re-run.
REM Output log: results\run_all_log.txt
setlocal
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
set LOG=%~dp0..\results\run_all_log.txt
if not exist "%~dp0..\results" mkdir "%~dp0..\results"
echo ==== start %date% %time% >> "%LOG%"

echo [1/5] Downloading VitalDB PPG windows (longest step; can be stopped and restarted)
python 01_download_vitaldb.py --workers 6 >> "%LOG%" 2>&1 || goto :err
echo [2/5] Preprocessing
python 02_preprocess.py --workers 6 >> "%LOG%" 2>&1 || goto :err
echo [3/5] Baselines
python 04_baselines.py >> "%LOG%" 2>&1 || goto :err

echo [4/5] Deep models
REM -- A. segment-level protocol of the original study (option name: leaky)
python 03_train_eval.py --model resnet34 --input beat  --split leaky >> "%LOG%" 2>&1 || goto :err
python 03_train_eval.py --model resnet34 --input seg10 --split leaky --bs 256 >> "%LOG%" 2>&1 || goto :err
REM -- B. the same models, split by person
python 03_train_eval.py --model resnet34 --input beat  --split subject >> "%LOG%" 2>&1 || goto :err
python 03_train_eval.py --model resnet34 --input seg10 --split subject --bs 256 >> "%LOG%" 2>&1 || goto :err
python 03_train_eval.py --model resnet34 --input beat  --split subject --demo >> "%LOG%" 2>&1 || goto :err
REM -- C. ESP32-sized model
python 03_train_eval.py --model tiny --input beat --split subject --augment >> "%LOG%" 2>&1 || goto :err
python 03_train_eval.py --model tiny --input beat --split leaky >> "%LOG%" 2>&1 || goto :err

echo [5/5] Personalisation + report
python 05_personalize.py >> "%LOG%" 2>&1 || goto :err
python 07_report.py >> "%LOG%" 2>&1 || goto :err
if exist C:\glucose_data\must python 06_must_external.py --run tiny_beat_subject >> "%LOG%" 2>&1
echo ==== done %date% %time% >> "%LOG%"
echo ALL DONE. Summary: results\summary.md
pause
exit /b 0
:err
echo FAILED - see results\run_all_log.txt (last lines).
pause
exit /b 1
