@echo off
REM High Performance Smart Parking System Launcher
REM Optimized for maximum FPS

echo ========================================
echo SMART PARKING - HIGH FPS MODE
echo ========================================
echo.

echo Select mode:
echo [1] MAXIMUM SPEED - No vehicle analysis (15-25 FPS)
echo [2] BALANCED - Analyze every 60 frames (10-15 FPS)
echo [3] DETAILED - Analyze every 30 frames (8-12 FPS)
echo [4] CUSTOM - Enter your own settings
echo.

set /p mode="Enter choice (1-4): "

if "%mode%"=="1" (
    echo.
    echo Starting MAXIMUM SPEED mode...
    python smart_parking_mvp_optimized.py parking_evening_vedio.mp4 --model yolov8n.pt
) else if "%mode%"=="2" (
    echo.
    echo Starting BALANCED mode...
    python smart_parking_mvp_optimized.py parking_evening_vedio.mp4 --model yolov8n.pt --analyze --analysis-interval 60
) else if "%mode%"=="3" (
    echo.
    echo Starting DETAILED mode...
    python smart_parking_mvp_optimized.py parking_evening_vedio.mp4 --model yolov8n.pt --analyze --analysis-interval 30
) else if "%mode%"=="4" (
    echo.
    echo Custom mode - Available options:
    echo --model [yolov8n.pt / yolov8s.pt / yolov8m.pt]
    echo --analyze (enable vehicle analysis)
    echo --analysis-interval [number] (frames between analysis)
    echo --skip-frames [number] (skip frames for extra speed)
    echo --output [filename.mp4] (save output video)
    echo --no-display (disable display window)
    echo.
    set /p custom="Enter custom arguments: "
    python smart_parking_mvp_optimized.py parking_evening_vedio.mp4 %custom%
) else (
    echo Invalid choice!
    pause
    exit
)

echo.
echo ========================================
echo Processing complete!
echo ========================================
pause
