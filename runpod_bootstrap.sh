#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

WORKSPACE_DIR="/workspace/VibeVoice"
VENV_DIR="/workspace/venv"
SETUP_COMPLETE_MARKER="$WORKSPACE_DIR/.setup_complete"
VIBEVOICE_REPO="https://github.com/sruckh/VibeVoice.git"

echo "Starting RunPod Bootstrap Script..."

# --- 0. Virtual Environment Setup (Persistent) ---
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate Venv
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated."

# --- 1. S3 Cleanup (Storage Management) ---
# Only runs if boto3 is installed (which happens in step 2 if missing)
# On first run, boto3 won't be there yet, so this skips. 
# On subsequent runs (warm volume), boto3 is there, so it runs.
if python -c "import boto3" &> /dev/null; then
    echo "Performing S3 storage cleanup (removing files > 7 days)..."
    python -c "
import boto3, os
from datetime import datetime, timedelta, timezone

bucket_name = os.environ.get('BUCKET_NAME')
endpoint = os.environ.get('BUCKET_ENDPOINT_URL')
# Credentials are auto-picked up from env vars AWS_ACCESS_KEY_ID / SECRET

if bucket_name:
    try:
        s3 = boto3.resource('s3', endpoint_url=endpoint)
        bucket = s3.Bucket(bucket_name)
        limit = datetime.now(timezone.utc) - timedelta(days=7)
        count = 0
        for obj in bucket.objects.all():
            if obj.last_modified < limit:
                obj.delete()
                count += 1
        print(f'Cleanup complete. Deleted {count} old files.')
    except Exception as e:
        print(f'Cleanup failed: {e}')
else:
    print('BUCKET_NAME not set, skipping cleanup.')
"
else
    echo "boto3 not found, skipping cleanup (will install later)."
fi


# --- 2. Installation / Setup ---
if [ -f "$SETUP_COMPLETE_MARKER" ]; then
    echo "Setup already complete. Skipping installation."
else
    echo "First-run setup detected. Proceeding with installation..."

    # Create workspace directory if it doesn't exist
    mkdir -p "$WORKSPACE_DIR"
    cd "$WORKSPACE_DIR"

    # Install PyTorch
    echo "Installing PyTorch..."
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    echo "PyTorch installed."

    # Install sage_attn
    echo "Installing sage_attn..."
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
    echo "sage_attn installed."

    # Clone Repository (Handle non-empty directory)
    echo "Cloning VibeVoice repository..."
    if [ ! -d ".git" ]; then
        git init
        git remote add origin "$VIBEVOICE_REPO"
        git fetch origin
        git reset --hard origin/main
        echo "VibeVoice repository cloned (via init/fetch/reset)."
    else
        echo "VibeVoice repository already initialized."
        git fetch origin
        git reset --hard origin/main
    fi

    # Modify pyproject.toml (remove torch, torchvision, torchaudio)
    echo "Modifying pyproject.toml..."
    if [ -f "pyproject.toml" ]; then
        sed -i '/"torch"/d' pyproject.toml
        sed -i '/"torchvision"/d' pyproject.toml
        sed -i '/"torchaudio"/d' pyproject.toml
        echo "pyproject.toml modified."
    fi

    # Install Dependencies (including boto3 for cleanup/handler)
    echo "Installing VibeVoice dependencies & boto3..."
    pip install -e .
    pip install boto3
    echo "Dependencies installed."

    # Create Marker
    touch "$SETUP_COMPLETE_MARKER"
    echo "Setup complete marker created."
fi

# --- 3. Launch Handler ---
cd "$WORKSPACE_DIR" # Ensure we are in the project root
echo "Launching RunPod handler..."
exec python handler.py