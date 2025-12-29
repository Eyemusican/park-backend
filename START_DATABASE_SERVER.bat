@echo off
echo ================================================================
echo SMART PARKING SYSTEM - DATABASE SERVER
echo ================================================================
echo.
echo Starting database-backed server on http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ================================================================
echo.

cd /d "%~dp0"
python run_with_database.py

pause
