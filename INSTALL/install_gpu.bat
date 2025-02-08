@echo off
REM Navigate to the parent directory
cd ..

REM Check if the venv directory exists
IF NOT EXIST "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Add the virtual environment's Scripts directory to PATH
SET "PATH=%CD%\venv\Scripts;%PATH%"

echo Activating virtual environment...
CALL .\venv\Scripts\activate.bat

REM Ensure the virtual environment is activated
IF "%VIRTUAL_ENV%"=="" (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Clear pip cache to avoid any old cached conflicts
echo Clearing pip cache...
pip cache purge

REM Check if TA-Lib wheel exists in INSTALL folder and install it
IF EXIST "INSTALL\TA_Lib-0.4.32-cp310-cp310-win_amd64.whl" (
    echo Installing TA-Lib from local wheel...
    pip install INSTALL\TA_Lib-0.4.32-cp310-cp310-win_amd64.whl
) ELSE (
    echo TA-Lib wheel file not found in INSTALL folder.
)

REM Check if requirements.txt exists and then install dependencies with deprecated resolver
IF EXIST "INSTALL\requirements_gpu.txt" (
    echo Installing dependencies from requirements_gpu.txt using legacy resolver...
    pip install --use-deprecated=legacy-resolver -r INSTALL\requirements_gpu.txt
) ELSE (
    echo No dependencies to install.
)

REM Run pip check to identify any dependency issues
echo Checking for dependency conflicts...
pip check

echo Setup complete! Your virtual environment is ready.
pause