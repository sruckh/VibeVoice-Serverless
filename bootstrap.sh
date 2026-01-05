#!/bin/bash
set -e

echo "=== VibeVoice Runpod Serverless Bootstrap ==="

# Create vibevoice directory structure on network volume
echo "Creating directory structure on network volume..."
mkdir -p /runpod-volume/vibevoice/{models,output,demo/voices,torch_cache}

# Set environment variables for Torch cache
export TORCH_HOME="/runpod-volume/vibevoice/torch_cache"

# Export HF_TOKEN if available
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
    echo "HuggingFace token configured"
else
    echo "WARNING: HF_TOKEN not set. Model download may fail if model requires authentication."
fi

# Virtual Environment Path on Network Volume
VENV_PATH="/runpod-volume/vibevoice/venv"

# Check if this is first run
FIRST_RUN_FLAG="/runpod-volume/vibevoice/.first_run_complete"

if [ ! -f "$FIRST_RUN_FLAG" ]; then
    echo "=== First Run Detected - Setting up Environment ==="

    # Create Virtual Environment
    echo "Creating virtual environment at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
    
    # Activate Virtual Environment
    source "$VENV_PATH/bin/activate"

    # Install PyTorch with CUDA 12.8 support
    echo "Installing PyTorch..."
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128

    # Install flash-attention for optimized inference
    echo "Installing flash-attention..."
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

    # Install huggingface_hub and other dependencies
    echo "Installing additional dependencies..."
    pip install huggingface_hub runpod>=1.6.0 boto3>=1.26.0 toml soundfile>=0.12.1 pyloudnorm>=0.1.0
    echo "Installing LinaCodec..."
    pip install git+https://github.com/ysharma3501/LinaCodec.git

    # Clone VibeVoice repository
    echo "Cloning VibeVoice repository..."
    cd /runpod-volume/vibevoice
    rm -rf vibevoice
    git clone https://github.com/vibevoice-community/VibeVoice.git vibevoice

    # Install VibeVoice
    echo "Installing VibeVoice..."
    cd /runpod-volume/vibevoice/vibevoice
    pip install -e .

    # Pre-download model during first run
    echo "Pre-downloading VibeVoice-7B model..."
    python3 -c "
from huggingface_hub import snapshot_download
import os

cache_dir = os.environ.get('HF_HUB_CACHE')
print(f'Downloading model to {cache_dir or "default HF cache"}...')
kwargs = {"cache_dir": cache_dir} if cache_dir else {}
snapshot_download('vibevoice/VibeVoice-7B', **kwargs)
print('Model download complete')
"

    # Create first run flag
    touch "$FIRST_RUN_FLAG"
    echo "=== First Run Setup Complete ==="

else
    echo "=== Existing Installation Found - Skipping Setup ==="
    # Activate Virtual Environment
    source "$VENV_PATH/bin/activate"
fi

# Start handler
echo "Starting VibeVoice handler..."
exec python /workspace/vibevoice/handler.py
