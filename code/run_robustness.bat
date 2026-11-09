@echo off
REM Robustness of the deep-model results (reviewer point): repeated seeds + two training variants.
REM Person-level folds stay fixed; only the training randomness changes. About 3 hours on the RTX 4050.
REM Output: results\runs.jsonl and results\preds\*_seed*.csv ; log: results\robustness_log.txt
setlocal
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
set LOG=%~dp0..\results\robustness_log.txt
echo ==== start %date% %time% >> "%LOG%"
echo [1/6] TinyCNN, person split, 4 more seeds (10 min)
for %%S in (1 2 3 4) do python 03_train_eval.py --model tiny --input beat --split subject --augment --seed %%S --tag seed%%S >> "%LOG%" 2>&1 || goto :err
echo [2/6] ResNet34, person split, seed 1 (25 min)
python 03_train_eval.py --model resnet34 --input beat --split subject --seed 1 --tag seed1 >> "%LOG%" 2>&1 || goto :err
echo [3/6] ResNet34, person split, seed 2 (25 min)
python 03_train_eval.py --model resnet34 --input beat --split subject --seed 2 --tag seed2 >> "%LOG%" 2>&1 || goto :err
echo [4/6] ResNet34, person split, lower learning rate 3e-4 (25 min)
python 03_train_eval.py --model resnet34 --input beat --split subject --lr 0.0003 --tag lr3e-4 >> "%LOG%" 2>&1 || goto :err
echo [5/6] ResNet34, person split, 3x more beats per epoch (75 min)
python 03_train_eval.py --model resnet34 --input beat --split subject --per_reading 96 --tag pr96 >> "%LOG%" 2>&1 || goto :err
echo [6/6] ResNet34, segment-level split, seed 1 (30 min)
python 03_train_eval.py --model resnet34 --input beat --split leaky --seed 1 --tag seed1 >> "%LOG%" 2>&1 || goto :err
echo ==== done %date% %time% >> "%LOG%"
echo ALL DONE.
pause
exit /b 0
:err
echo FAILED - see results\robustness_log.txt
pause
exit /b 1
