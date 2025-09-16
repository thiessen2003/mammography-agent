#!/bin/bash

# Installation script for the Multi-Agent Breast Imaging Analysis System

echo "🏥 Installing Multi-Agent Breast Imaging Analysis System"
echo "========================================================"

# Check if Python 3.13+ is available
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.13"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.13+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -e .

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Error installing dependencies"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp env_example.txt .env
    echo "✅ .env file created. Please edit it with your API keys."
else
    echo "✅ .env file already exists"
fi

# Run tests
echo "🧪 Running tests..."
python -m pytest tests/ -v

if [ $? -eq 0 ]; then
    echo "✅ All tests passed"
else
    echo "⚠️  Some tests failed (this is expected without API keys)"
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your OpenAI API key"
echo "2. (Optional) Add LangSmith API key for tracing"
echo "3. Run the demo: python run_demo.py"
echo ""
echo "For more information, see README.md"
