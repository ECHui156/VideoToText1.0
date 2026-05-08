#!/usr/bin/env bash
# requirements_for_gpu.sh - Install CUDA-enabled PyTorch (example uses cu128).
# Run this inside the activated virtualenv or replace "python" with the full path to target python.

set -e
echo "Installing CUDA-enabled PyTorch (cu128) into current Python environment..."
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
echo "Installation finished (or will exit with error)."
