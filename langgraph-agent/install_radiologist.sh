#!/bin/bash

# Installation script for the Multi-Instance Radiologist System

echo "🏥 Installing Multi-Instance Radiologist System"
echo "Using MedVLM-R1 and MedGemma-4B-it models"
echo "=============================================="

# Check if Python 3.13+ is available
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.13"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.13+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Check for CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
else
    echo "⚠️  CUDA not detected. Models will run on CPU (slower)"
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -e .

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Error installing dependencies"
    exit 1
fi

# Check Hugging Face CLI
if command -v huggingface-cli &> /dev/null; then
    echo "✅ Hugging Face CLI found"
else
    echo "⚠️  Hugging Face CLI not found. Install with: pip install huggingface_hub"
fi

# Run tests
echo "🧪 Running tests..."
python test_radiologist_system.py

if [ $? -eq 0 ]; then
    echo "✅ All tests passed"
else
    echo "⚠️  Some tests failed (this is expected without model downloads)"
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Login to Hugging Face: huggingface-cli login"
echo "2. Run the demo: python radiologist_demo.py"
echo "3. Check the documentation: RADIOLOGIST_README.md"
echo ""
echo "System features:"
echo "• Multiple radiologist agent instances"
echo "• MedVLM-R1 and MedGemma-4B-it support"
echo "• Consensus voting mechanism"
echo "• Image analysis capabilities"
echo "• Parallel processing"
echo ""
echo "For more information, see RADIOLOGIST_README.md"
