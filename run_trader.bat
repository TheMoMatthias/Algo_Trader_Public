@echo off
git pull &^

REM Change to the directory containing your Python script
cd /d "%~dp0\Bot models\Advanced models"

REM Activate the virtual environment (if it exists in the Crypto Bot directory)
call "%~dp0\venv\Scripts\activate.bat"

REM Run the Python script
python "AlgoTrader Kucoin.py"

IF "%pau_var%"=="" PAUSE