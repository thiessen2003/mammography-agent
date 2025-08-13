#!/usr/bin/env python3
"""
Simple launcher script for the Patient Simulator.
Run this script to start the desktop application.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change to project root directory
os.chdir(project_root)

try:
    from ui.patient_simulator import main
    print("🚀 Starting Mammography Agent Patient Simulator...")
    print(f"📁 Working directory: {os.getcwd()}")
    print("🔑 Make sure you have your OpenAI API key ready!")
    print("-" * 50)
    
    # Start the application
    main()
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Make sure you're running this from the project root directory")
    print("2. Install dependencies: pip install -e .")
    print("3. Check that all agent files are present")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    print("\n🔧 Please check:")
    print("1. Python version (requires 3.8+)")
    print("2. All dependencies are installed")
    print("3. OpenAI API key is valid")
    sys.exit(1) 