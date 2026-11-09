@echo off
REM Writes Python / PyTorch / GPU information to results\env_check.txt (read-only check, installs nothing)
set OUT=%~dp0..\results\env_check.txt
if not exist "%~dp0..\results" mkdir "%~dp0..\results"
echo ==== %date% %time% > "%OUT%"
echo ---- where python >> "%OUT%"
where python >> "%OUT%" 2>&1
where py >> "%OUT%" 2>&1
echo ---- python version >> "%OUT%"
python --version >> "%OUT%" 2>&1
py -0p >> "%OUT%" 2>&1
echo ---- packages >> "%OUT%"
python -m pip list 2>nul | findstr /i "torch numpy scipy pandas scikit matplotlib vitaldb tabulate tensorflow" >> "%OUT%"
echo ---- torch + cuda >> "%OUT%"
python -c "import torch,sys; print('python',sys.version); print('torch',torch.__version__,'cuda build',torch.version.cuda); print('cuda available',torch.cuda.is_available()); print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); print('vram GB',round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if torch.cuda.is_available() else '-')" >> "%OUT%" 2>&1
echo ---- nvidia-smi >> "%OUT%"
nvidia-smi >> "%OUT%" 2>&1
echo ---- disk C: >> "%OUT%"
wmic logicaldisk where "DeviceID='C:'" get FreeSpace,Size >> "%OUT%" 2>&1
echo ---- RAM / CPU >> "%OUT%"
wmic computersystem get TotalPhysicalMemory >> "%OUT%" 2>&1
wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors >> "%OUT%" 2>&1
echo Done.
pause
