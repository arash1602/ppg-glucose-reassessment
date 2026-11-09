@echo off
REM ~10-20 minutes: downloads 40 cases and runs every step on them, to check
REM that downloading, preprocessing, training and reporting all work on this PC.
REM Uses separate folders so it does not mix with the real run.
setlocal
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
set GLU_DATA_DIR=C:\glucose_data_quicktest
set GLU_RESULTS_DIR=%~dp0..\results_quicktest
python 01_download_vitaldb.py --limit 40 --workers 4 || goto :err
python 02_preprocess.py --workers 4 || goto :err
python 04_baselines.py || goto :err
python 03_train_eval.py --model tiny --input beat --split subject --epochs 3 || goto :err
python 03_train_eval.py --model resnet34 --input beat --split leaky --epochs 2 || goto :err
python 05_personalize.py || goto :err
python 07_report.py || goto :err
echo.
echo QUICK TEST PASSED. Results in results_quicktest. Next: run_all.bat
pause
exit /b 0
:err
echo.
echo QUICK TEST FAILED at the step above. See the error text above.
pause
exit /b 1
