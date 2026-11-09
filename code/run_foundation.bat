@echo off
REM Pretrained PPG foundation models (PaPaGei-S, Pulse-PPG) under the same person-level evaluation.
REM About 2-3 hours on the RTX 4050. Needs internet once (downloads ~340 MB of model weights from Zenodo).
REM Output: results\foundation_models.csv, results\foundation_models_must.csv, results\preds\fm_*.csv
REM Log:    results\foundation_log.txt
setlocal
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
set LOG=%~dp0..\results\foundation_log.txt
echo ==== start %date% %time% >> "%LOG%"
echo [1/5] Download model weights (once, a few minutes)
python 21_foundation_models.py download >> "%LOG%" 2>&1 || goto :err
echo [2/5] Embeddings of all 10-s segments, both models (20-40 min)
python 21_foundation_models.py embed >> "%LOG%" 2>&1 || goto :err
echo [3/5] Fine-tune PaPaGei-S end to end, PPG only (about 1 h)
python 21_foundation_models.py finetune >> "%LOG%" 2>&1 || goto :err
echo [4/5] Fine-tune PaPaGei-S end to end, PPG + age, sex, BMI (about 1 h)
python 21_foundation_models.py finetune --demo >> "%LOG%" 2>&1 || goto :err
echo [5/5] Evaluation and comparison with the earlier models (5 min)
python 21_foundation_models.py probe >> "%LOG%" 2>&1 || goto :err
echo ==== done %date% %time% >> "%LOG%"
echo ALL DONE.
pause
exit /b 0
:err
echo FAILED - see results\foundation_log.txt
pause
exit /b 1
