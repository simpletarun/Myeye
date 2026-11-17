@echo off
REM Myeye - one-click setup + run (Windows)
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.9+ is required. Install from https://python.org & pause & exit /b 1)
REM a moved/copied project leaves a .venv with no pip or activate script
if not exist ".venv\Scripts\activate.bat" (
  if exist .venv rmdir /s /q .venv
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (echo Dependency install failed. & pause & exit /b 1)
python run.py
pause
