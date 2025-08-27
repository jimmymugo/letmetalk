#!/bin/bash

# FPL Optimizer - Quick Start Script
# This script sets up and runs the FPL Optimizer application

echo "⚽ FPL Optimizer - Starting up..."
echo "=================================="

# Check if virtual environment exists
if [ ! -d "fpl_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv fpl_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source fpl_env/bin/activate

# Install dependencies if needed
if [ ! -f "fpl_env/installed" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    touch fpl_env/installed
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Run tests
echo "🧪 Running system tests..."
python3 test_system.py

if [ $? -eq 0 ]; then
    echo "✅ All tests passed!"
    echo ""
    echo "🚀 Starting Streamlit application..."
    echo "📱 The app will be available at: http://localhost:8501"
    echo "🔄 Press Ctrl+C to stop the application"
    echo ""
    
    # Start the Streamlit app
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
else
    echo "❌ Tests failed. Please check the errors above."
    exit 1
fi