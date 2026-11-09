@echo off
REM Improvement round (person-level folds). Start it after run_all.bat has finished.
REM Log: results\improve_log.txt
setlocal
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
echo Running improvement round... (log: results\improve_log.txt)
python 08_improve.py > "%~dp0..\results\improve_log.txt" 2>&1 || (echo FAILED - see the log in results & pause & exit /b 1)
python 07_report.py >> "%~dp0..\results\improve_log.txt" 2>&1
echo DONE. Summary: results\improvement_summary.csv
pause
