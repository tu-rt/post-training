@echo off
cd /d "%~dp0"
python scripts\run_ablation.py %*
pause
