#!/bin/bash
# VideoLingo Installation Script
# Handles build dependencies that need to be installed before other packages

set -e  # Exit on error

echo "=== VideoLingo Installation ==="

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected. Consider activating one first."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Install build dependencies first (numpy needed for pkuseg build)
echo ""
echo "[1/2] Installing build dependencies (numpy, cython)..."
pip install numpy==1.25.2 cython

# Step 2: Install all other dependencies
echo ""
echo "[2/2] Installing all dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "=== Installation complete! ==="
echo "Run 'streamlit run st.py' to start VideoLingo"
