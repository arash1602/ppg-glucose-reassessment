@echo off
REM Extended Monte Carlo sweep (660-1700 nm) + pulsatile/confounder analysis.
REM Uses the same Python environment as the glucose pipeline (C:\glucose_venv).
REM Needs the NVIDIA GPU. If run_all.bat is still training, both will run slower.
setlocal
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
python -m pip install --quiet pmcx pyyaml
python src\analysis\physio_confounders.py results || goto :err
python src\main_extended.py || goto :err
python src\analysis\physio_confounders.py results_extended || goto :err
python src\analysis\nas_uncertainty.py results_extended || goto :err
echo DONE. See results\physio_analysis and results_extended\physio_analysis
pause
exit /b 0
:err
echo FAILED - see the error text above.
pause
exit /b 1
