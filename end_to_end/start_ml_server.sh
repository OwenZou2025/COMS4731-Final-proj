#!/bin/bash

# FitSense ML Backend Server Startup Script

cd "$(dirname "$0")"

echo "================================================"
echo "  FitSense ML Backend Server"
echo "================================================"
echo ""

# Check if model file exists
if [ ! -f "./custom_ml_model/model/best_multitask_vit.pth" ]; then
    echo "❌ ERROR: Model file not found!"
    echo "   Expected: ./custom_ml_model/model/best_multitask_vit.pth"
    echo ""
    echo "   Download models using:"
    echo "   cd custom_ml_model/model && bash download_models.sh"
    exit 1
fi

echo "✓ Model file found"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import flask, flask_cors, torch, transformers, google.genai" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Missing Python dependencies"
    echo ""
    echo "   Install using:"
    echo "   pip install -r requirements_ml_backend.txt"
    exit 1
fi

echo "✓ Dependencies installed"
echo ""

# Check API key
if [ -z "$GEMINI_API_KEY" ]; then
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    fi
fi

if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your_actual_api_key_here" ]; then
    echo "⚠️  WARNING: No Gemini API key configured"
    echo "   Outfit recommendations will use fallback mode"
    echo "   Set GEMINI_API_KEY in .env to enable full functionality"
    echo ""
fi

# Start the ML server
echo "Starting ML Backend Server on port 5001..."
echo "(Press Ctrl+C to stop)"
echo ""
echo "================================================"
echo ""

python3 ml_server.py
