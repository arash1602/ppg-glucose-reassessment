@echo off
REM Tissue-model sensitivity analysis (alternative continuous 600-1700 nm skin model).
REM Runs the Monte Carlo sweep on the GPU (about 7-9 h), then the analysis and the comparison table.
REM Leave the laptop plugged in and set to never sleep. Progress: results_tissue2_log.txt
setlocal
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
python -m pip install --quiet pmcx pyyaml
set LOG=%~dp0results_tissue2_log.txt
echo ==== start %date% %time% > "%LOG%"
echo [1/5] Monte Carlo sweep, 86 wavelengths x 5 batches (several hours; the window shows progress)
python src\main_tissue2.py 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath '%LOG%' -Append"
if not exist results_tissue2\inputs_per_wavelength.csv goto :err
echo [2/5] Glucose-equivalent errors and net analyte signal
python src\analysis\physio_confounders.py results_tissue2 >> "%LOG%" 2>&1 || goto :err
echo [3/5] Rank, conditioning and Monte Carlo uncertainty
python src\analysis\nas_uncertainty.py results_tissue2 >> "%LOG%" 2>&1 || goto :err
echo [4/5] Comparison with the paper's tissue model
python src\analysis\compare_tissue_models.py results_extended results_tissue2 >> "%LOG%" 2>&1 || goto :err
echo [5/5] Robust multi-wavelength search (all sets of four wavelengths)
python src\analysis\robust_nas.py results_extended results_tissue2 >> "%LOG%" 2>&1 || goto :err
echo ==== done %date% %time% >> "%LOG%"
echo ALL DONE. Results: results_tissue2\physio_analysis\tissue_model_comparison.csv
pause
exit /b 0
:err
echo FAILED - see results_tissue2_log.txt
pause
exit /b 1
