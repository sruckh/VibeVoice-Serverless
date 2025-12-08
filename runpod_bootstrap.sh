#!/usr/bin/env bash
set -euo pipefail

# --- Diagnostics ---
echo "=== Mount Diagnostics ==="
df -Th
echo "--- Root Listing ---"
ls -l /
echo "--- Volume Check ---"
if [ -d "/runpod-volume" ]; then
    echo "/runpod-volume exists:"
    ls -ld /runpod-volume
    ls -l /runpod-volume
else
    echo "/runpod-volume does NOT exist."
fi
echo "======================="

# Detect Persistent Volume
if [ -d "/runpod-volume" ]; then
    # Use subdirectory as requested
    WORKSPACE="/runpod-volume/VibeVoice"
    echo "Detected /runpod-volume. WORKSPACE set to $WORKSPACE"
else
    # Fallback
    WORKSPACE="/workspace/VibeVoice"
    echo "No volume detected. WORKSPACE set to $WORKSPACE"
fi

VENV="$WORKSPACE/venv"
CACHE="$WORKSPACE/cache"
REPO_DIR="$WORKSPACE/repo"
REPO_URL="https://github.com/sruckh/VibeVoice.git"

# Ensure directories exist
mkdir -p "$CACHE" "$REPO_DIR"

# 1. Clone/Update Repo (contains handler.py)
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Cloning $REPO_URL into $REPO_DIR..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "Repository exists at $REPO_DIR. Pulling latest..."
    cd "$REPO_DIR" && git pull && cd - > /dev/null
fi

# 2. Virtual Environment Setup
if [ ! -f "$VENV/bin/activate" ]; then
    echo "Creating virtual environment at $VENV..."
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --upgrade pip

    # Install PyTorch (Specific Version)
    echo "Installing PyTorch..."
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

    # Install Flash Attention
    echo "Installing Flash Attention..."
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

    # Install Repository Dependencies
    echo "Installing Project Dependencies..."
    cd "$REPO_DIR"
    
    # Modify pyproject.toml to avoid torch conflicts
    if [ -f "pyproject.toml" ]; then
        sed -i '/"torch"/d' pyproject.toml
        sed -i '/"torchvision"/d' pyproject.toml
        sed -i '/"torchaudio"/d' pyproject.toml
        echo "Pruned torch from pyproject.toml"
    fi

    pip install -e .
    pip install boto3
else
    echo "Virtual environment exists. Activating..."
    source "$VENV/bin/activate"
fi

# 3. Environment Config
export HF_HOME="$CACHE"
export HUGGINGFACE_HUB_CACHE="$CACHE"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# 4. S3 Storage Cleanup (if boto3 available)
if python -c "import boto3" &> /dev/null; then
    echo "Running S3 Cleanup..."
    python -c "
import boto3, os
from datetime import datetime, timedelta, timezone
try:
    bucket_name = os.environ.get('BUCKET_NAME')
    endpoint = os.environ.get('BUCKET_ENDPOINT_URL')
    if bucket_name:
        s3 = boto3.resource('s3', endpoint_url=endpoint)
        bucket = s3.Bucket(bucket_name)
        limit = datetime.now(timezone.utc) - timedelta(days=7)
        for obj in bucket.objects.all():
            if obj.last_modified < limit:
                obj.delete()
        print('Cleanup complete.')
except Exception as e:
    print(f'Cleanup skipped/failed: {e}')
"
fi

# 5. Launch Handler
echo "Starting Handler..."
cd "$REPO_DIR"
exec python handler.py
