#!/bin/bash
# VideoLingo Installation Script
# Handles dependency conflicts between whisperx (tokenizers<0.16) and chatterbox-tts (tokenizers>=0.20)

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
echo "[1/6] Installing build dependencies (numpy, cython)..."
pip install numpy==1.25.2 cython

# Step 2: Install PyTorch
echo ""
echo "[2/6] Installing PyTorch..."
pip install torch>=2.0.0 torchaudio==2.6.0

# Step 3: Install whisperx first (brings faster-whisper with older tokenizers)
echo ""
echo "[3/6] Installing whisperx (audio transcription)..."
pip install "whisperx @ git+https://github.com/m-bain/whisperx.git@7307306a9d8dd0d261e588cc933322454f853853"

# Step 4: Install chatterbox-tts (will upgrade tokenizers - works at runtime despite pip warnings)
echo ""
echo "[4/6] Installing chatterbox-tts..."
pip install chatterbox-tts

# Step 5: Install remaining dependencies
echo ""
echo "[5/6] Installing remaining dependencies..."
pip install -r requirements.txt

# Step 6: Install Ollama and pull model
echo ""
echo "[6/6] Setting up Ollama..."
if command -v ollama &> /dev/null; then
    echo "Ollama is already installed"
else
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "Pulling qwen2.5:14b model (this may take a while)..."
ollama pull qwen2.5:14b

echo ""
echo "=== Installation complete! ==="
echo "Run 'streamlit run st.py' to start VideoLingo"
