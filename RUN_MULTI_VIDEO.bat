@echo off
echo ========================================
echo MULTI-VIDEO SMART PARKING SYSTEM
echo Running TWO Videos Simultaneously
echo ========================================
echo.
echo Videos:
echo 1. parking_evening_vedio.mp4 (Camera 1)
echo 2. parking_video.mp4 (Camera 2)
echo.
echo Press Ctrl+C or 'q' in any window to stop
echo ========================================
echo.

cd /d "%~dp0"

python smart_parking_balanced_multi_video.py --config camera_config.json

echo.
echo ========================================
echo Multi-video processing stopped
echo ========================================
pause
