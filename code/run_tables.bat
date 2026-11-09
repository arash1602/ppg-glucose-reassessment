@echo off
REM Tables 5, 6, 8, 10, 11, 12, 14 and 16 from the outputs of the earlier steps (CPU only, about 15 min).
setlocal
call C:\glucose_venv\Scripts\activate.bat
cd /d "%~dp0"
python 16_causal_ablation.py || goto :err
python 18_robustness_factorial.py || goto :err
python 19_must_matched.py || goto :err
python 20_subgroups_detection.py || goto :err
python 17_device_window_check.py || goto :err
echo Done. Compare results\ with expected_results\.
pause
exit /b 0
:err
echo FAILED
pause
exit /b 1
