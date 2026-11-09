@echo off
REM One-time setup: creates a Python environment outside the project folder and installs
REM PyTorch with CUDA (GPU) support plus the other packages.
setlocal
set VENV=C:\glucose_venv
where python >nul 2>nul || (echo Python not found. Install Python 3.11 from python.org and tick "Add to PATH". & pause & exit /b 1)
if not exist %VENV% python -m venv %VENV%
call %VENV%\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "%~dp0requirements.txt"
echo.
python -c "import torch; print('PyTorch', torch.__version__, '| GPU available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU - training will be slow')"
echo.
echo Setup finished. Next: run_quicktest.bat
pause
