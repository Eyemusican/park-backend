@echo off
echo ============================================================
echo STARTING SMART PARKING VIDEO PROCESSING
echo WITH VEHICLE DETAILS DETECTION
echo ============================================================
echo.
echo Backend must be running on http://localhost:5000
echo.
echo Press Ctrl+C to stop the video processing
echo.
echo Starting in 3 seconds...
timeout /t 3 /nobreak > nul

cd /d "%~dp0"
python smart_parking_mvp.py --run parking_evening_vedio.mp4

pause
