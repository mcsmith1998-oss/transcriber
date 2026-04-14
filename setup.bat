@echo off
echo ================================================
echo   Interview Transcriber — Setup
echo ================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not on PATH.
    echo Download it from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo Installing dependencies (this may take a few minutes)...
python -m pip install -r requirements.txt
echo.

echo ================================================
echo   Setup complete!
echo ================================================
echo.
echo NEXT STEPS (one-time only):
echo.
echo   1. Accept the speaker diarization model terms at:
echo      https://huggingface.co/pyannote/speaker-diarization-3.1
echo      https://huggingface.co/pyannote/segmentation-3.0
echo.
echo   2. Log in to HuggingFace - run this in a terminal:
echo      huggingface-cli login
echo.
echo   3. Launch the app by double-clicking run.bat
echo.
pause
