@echo off
REM Mammography Agent Patient Simulator Launcher
REM Run this file to start the desktop application

echo.
echo ========================================
echo   Mammography Agent Patient Simulator
echo ========================================
echo.
echo Starting the application...
echo.

REM Change to the project root directory
cd /d "%~dp0.."

REM Run the simulator
python ui/patient_simulator.py

REM Pause to see any error messages
pause 