#!/bin/bash

# Mammography Agent Patient Simulator Launcher
# Run this file to start the desktop application

echo ""
echo "========================================"
echo "  Mammography Agent Patient Simulator"
echo "========================================"
echo ""
echo "Starting the application..."
echo ""

# Change to the project root directory
cd "$(dirname "$0")/.."

# Run the simulator
python ui/patient_simulator.py

# Check if there was an error
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Application exited with an error"
    echo "Check the error messages above for details"
    read -p "Press Enter to continue..."
fi 