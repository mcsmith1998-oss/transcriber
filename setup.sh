#!/bin/bash
# One-time setup for Interview Transcriber
set -e

echo "================================================"
echo "  Interview Transcriber — Setup"
echo "================================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Download it from https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python $PYTHON_VERSION detected."
echo ""

echo "Installing dependencies (this may take a few minutes)..."
python3 -m pip install -r requirements.txt
echo ""

echo "================================================"
echo "  Setup complete!"
echo "================================================"
echo ""
echo "NEXT STEPS (one-time only):"
echo ""
echo "  1. Accept the speaker diarization model terms at:"
echo "     https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "     https://huggingface.co/pyannote/segmentation-3.0"
echo ""
echo "  2. Log in to HuggingFace:"
echo "     huggingface-cli login"
echo ""
echo "  3. Launch the app:"
echo "     ./run.sh"
echo "     (or double-click 'Launch App.command')"
echo ""
