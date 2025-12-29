@echo off
echo ========================================
echo PARKING SLOT MAPPER - parking_video.mp4
echo ========================================
echo.
echo This will create a NEW slot configuration for parking_video.mp4
echo Output: configs/parking_slots_video2.json
echo.
echo Instructions:
echo - Click 4 corners to define each parking slot
echo - Press 's' to SAVE and quit
echo - Press 'q' to quit without saving
echo - Press 'r' to remove last slot
echo - Press 'u' to undo last point
echo ========================================
echo.

cd /d "%~dp0"

REM Start fresh - no old slots
python slot_mapper.py parking_video.mp4 configs/parking_slots_video2.json

echo.
echo ========================================
echo Slot mapping complete!
echo Configuration saved to: configs/parking_slots_video2.json
echo ========================================
pause
