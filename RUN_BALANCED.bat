@echo off
REM BALANCED MODE - High FPS + Complete Detections
REM Perfect for production use

echo ========================================
echo SMART PARKING - BALANCED MODE
echo High FPS + All Vehicle Details
echo ========================================
echo.
echo This mode provides:
echo - 10-15 FPS (smooth performance)
echo - Complete vehicle analysis (plates, colors, types)
echo - Smart caching (analyzes each vehicle once)
echo - Real-time parking detection
echo.

python smart_parking_balanced.py parking_evening_vedio.mp4

echo.
echo ========================================
echo Processing complete!
echo ========================================
pause
