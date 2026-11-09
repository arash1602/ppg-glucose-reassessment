@echo off
REM Records the exact package versions of your Windows environment next to this file.
call C:\glucose_venv\Scripts\activate.bat
python --version > "%~dp0requirements-lock-windows.txt" 2>&1
python -m pip freeze >> "%~dp0requirements-lock-windows.txt"
echo Written: requirements-lock-windows.txt
pause
