# Suggested Commands

## Development Commands

### Docker Build
```bash
# Build the container image
docker build -t vibevoice-runpod .

# Note: Local testing is limited without RunPod environment and GPU
```

### File Operations
```bash
# List files (standard Linux)
ls -la

# Find files by pattern
find . -name "*.py"

# Search in files
grep -r "pattern" .

# Change directory
cd /path/to/directory

# View file contents
cat filename.py

# Edit files (use your preferred editor)
vim filename.py
nano filename.py
```

### Git Commands
```bash
# Check status
git status

# Stage changes
git add .

# Commit changes
git commit -m "Description of changes"

# Push to remote
git push origin main

# View commit history
git log --oneline

# View diff
git diff
```

## Deployment Commands

### RunPod Deployment
**Note:** Deployment happens via RunPod web interface, not CLI

1. Push code to GitHub repository
2. Create RunPod Serverless Endpoint via web UI:
   - Container Image: Select "From GitHub"
   - Repository: Your GitHub repo URL
   - Branch: `main`
   - Attach Network Volume
   - Set Environment Variables (see below)

### Environment Variables (Set in RunPod UI)

**Required:**
```bash
HF_TOKEN=your_huggingface_token_here
```

**Optional (S3):**
```bash
S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_ACCESS_KEY_ID=your_access_key
S3_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=your_bucket_name
S3_REGION=us-east-1
```

**Optional (Tuning):**
```bash
MAX_TEXT_LENGTH=2000
DEFAULT_SAMPLE_RATE=24000
MAX_CHUNK_CHARS=300
DEFAULT_SPEAKER=Alice
DEFAULT_CFG_SCALE=1.3
```

## Testing Commands

### Manual API Testing
```bash
# Test the deployed RunPod endpoint
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync \
  -H "Authorization: Bearer {RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Hello! This is a test of VibeVoice TTS system.",
      "speaker_name": "Alice",
      "cfg_scale": 1.3,
      "disable_prefill": false
    }
  }'
```

### Voice File Upload (via RunPod Volume SSH/SFTP)
```bash
# Connect to network volume and upload voice files
# (Exact method depends on RunPod volume access configuration)

# Voice files must be .wav format in:
/runpod-volume/vibevoice/demo/voices/Alice.wav
/runpod-volume/vibevoice/demo/voices/Carter.wav
# etc.
```

## Cloudflare Worker Bridge Deployment (Optional)

### Setup
```bash
# Install Wrangler CLI
npm install -g wrangler

# Deploy worker
cd bridge
wrangler deploy

# Set secrets
wrangler secret put RUNPOD_URL
wrangler secret put RUNPOD_API_KEY
wrangler secret put AUTH_TOKEN

# Upload voice mapping to R2
wrangler r2 object put VIBEVOICE_BUCKET/voices.json --file=voices.json
```

### Test Bridge
```bash
# Test OpenAI-compatible endpoint
curl -X POST https://your-worker.workers.dev/v1/audio/speech \
  -H "Authorization: Bearer {AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello! This is a test.",
    "voice": "alloy"
  }' \
  --output test_audio.mp3
```

## Debugging Commands

### View Container Logs (RunPod UI)
- Navigate to your endpoint in RunPod dashboard
- Click on "Logs" tab to view bootstrap and handler output

### Check Network Volume
```bash
# If you have SSH access to RunPod volume
ls -la /runpod-volume/vibevoice/

# Check first run flag
ls -la /runpod-volume/vibevoice/.first_run_complete

# Check voice files
ls -la /runpod-volume/vibevoice/demo/voices/

# Check cached model
ls -la /runpod-volume/vibevoice/hf_cache/

# Check generated outputs
ls -la /runpod-volume/vibevoice/output/
```

### Python Debugging
```bash
# Check Python version in container
python --version

# Check installed packages
pip list

# Check PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check HuggingFace cache
python -c "import os; print(os.environ.get('HF_HUB_CACHE'))"
```

## Maintenance Commands

### Clean Up Old Output Files
**Note:** Automatic cleanup happens in handler.py (files older than 2 days)

Manual cleanup via SSH/SFTP:
```bash
# Remove old output files
find /runpod-volume/vibevoice/output -type f -mtime +2 -delete
```

### Reset First Run Flag (Force Re-setup)
```bash
# Remove first run flag to trigger full bootstrap on next start
rm /runpod-volume/vibevoice/.first_run_complete
```

### Clear Model Cache (Force Re-download)
```bash
# Clear HuggingFace cache
rm -rf /runpod-volume/vibevoice/hf_cache/*
```

## Code Quality Commands

**Note:** This project doesn't currently have linting/formatting configured.
Future additions could include:

```bash
# Format with black (if added)
black handler.py inference.py config.py

# Lint with flake8 (if added)
flake8 handler.py inference.py config.py

# Type check with mypy (if added)
mypy handler.py inference.py config.py

# Run tests (if added)
pytest tests/
```

## Important Notes

### What NOT to Do
❌ Don't run `pip install` locally (dependencies are for container environment)
❌ Don't test GPU code without GPU (will fail or fall back to CPU)
❌ Don't modify files in `chatterbox/` or `VibeVoice/` (reference only)
❌ Don't commit large model files or voice files to git
❌ Don't hardcode API keys or tokens in code

### System-Specific Notes
- **Linux System**: Standard Unix commands work (ls, grep, find, cd)
- **Python 3.12**: Use f-strings, type hints, modern Python features
- **No Direct Container Execution**: Container runs on RunPod, not locally
- **GPU Required**: Model needs CUDA-capable GPU to run efficiently

## Quick Reference

### When Task is Completed
1. Review changes with `git diff`
2. Commit changes with `git commit -m "Description"`
3. Push to GitHub with `git push origin main`
4. RunPod auto-rebuilds from GitHub
5. Test deployed endpoint with curl command

### First Deployment Checklist
- [ ] Set `HF_TOKEN` environment variable in RunPod
- [ ] Attach network volume to endpoint
- [ ] Wait 5-10 minutes for first run (model download)
- [ ] Upload voice files to `/runpod-volume/vibevoice/demo/voices/`
- [ ] Test with sample text via API
- [ ] Verify audio output quality
- [ ] Check logs for any errors
