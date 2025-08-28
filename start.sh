#!/bin/bash

# Deep Agent System Startup Script
echo "🤖 Starting Deep Agent System..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data cache/queries cache/memory

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Please copy env.example to .env and configure your settings."
    echo "   cp env.example .env"
    echo "   # Then edit .env with your actual values"
    exit 1
fi

# Run system test
echo "🧪 Running system test..."
python test_system.py

if [ $? -eq 0 ]; then
    echo "✅ System test passed!"
    echo ""
    echo "🚀 System is ready! You can now run:"
    echo "   python main.py          # Interactive mode"
    echo "   python main.py demo     # Demo mode"
    echo "   python main.py interactive # Explicit interactive mode"
else
    echo "❌ System test failed. Please check the errors above."
    exit 1
fi 