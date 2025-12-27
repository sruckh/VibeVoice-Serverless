# Tech Stack

## Core Technologies

### Base Infrastructure
- **Base Image**: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`
- **Python**: 3.12
- **PyTorch**: 2.8.0 with CUDA 12.8 support
- **Platform**: RunPod Serverless

### Machine Learning
- **Model**: VibeVoice 7B from HuggingFace (`vibevoice/VibeVoice-7B`)
- **Framework**: PyTorch with transformers
- **Attention**: flash_attention_2 (with sdpa fallback)
- **Inference**: VibeVoiceForConditionalGenerationInference
- **Processing**: VibeVoiceProcessor

### Python Dependencies
```
runpod>=1.6.0              # RunPod serverless SDK
boto3>=1.26.0              # AWS S3 integration
toml                       # Configuration parsing
soundfile>=0.12.1          # Audio I/O
huggingface_hub            # Model downloading
pyloudnorm>=0.1.0          # Loudness normalization
```

### Additional Dependencies (from VibeVoice)
- `transformers` - HuggingFace transformers library
- `accelerate` - Model loading optimization
- `safetensors` - Model weight storage

### System Dependencies
- `ffmpeg` - Audio encoding/decoding
- `libsndfile1` - Sound file library
- `git` - For cloning VibeVoice repository

## Optional Components

### S3 Storage (Optional)
- **boto3** - S3 client
- Presigned URL generation (1 hour expiry)
- Fallback to base64 encoding if S3 not configured

### Cloudflare Worker Bridge (Optional)
- **Runtime**: Cloudflare Workers
- **Storage**: R2 bucket for voice mappings
- **Purpose**: OpenAI TTS API compatibility layer

## File Structure

### Deployment Files
- `Dockerfile` - Container definition
- `requirements.txt` - Python dependencies
- `bootstrap.sh` - Runtime setup script
- `.dockerignore` - Docker build exclusions

### Application Code
- `config.py` - Configuration management
- `handler.py` - RunPod serverless entry point
- `inference.py` - VibeVoice inference engine

### Documentation
- `IMPLEMENTATION.md` - Complete implementation plan
- `CLAUDE.md` - Claude Code guidance
- `FIX_PLAN.md` - Architectural fixes
- `VALIDATION_REPORT.md` - Fix validation

### Optional Components
- `bridge/worker.js` - Cloudflare Worker for OpenAI API compatibility

## Environment Configuration

### Required
- `HF_TOKEN` - HuggingFace authentication token

### Optional (S3)
- `S3_ENDPOINT_URL`
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`
- `S3_BUCKET_NAME`
- `S3_REGION` (default: us-east-1)

### Optional (Tuning)
- `MAX_TEXT_LENGTH` (default: 2000)
- `DEFAULT_SAMPLE_RATE` (default: 24000)
- `MAX_CHUNK_CHARS` (default: 300)
- `DEFAULT_SPEAKER` (default: Alice)
- `DEFAULT_CFG_SCALE` (default: 1.3)

## Network Volume Structure
```
/runpod-volume/vibevoice/
├── hf_home/           # HuggingFace home
├── hf_cache/          # Model cache
├── models/            # Downloaded models
├── output/            # Generated audio
├── demo/voices/       # Voice reference files (.wav)
├── venv/              # Virtual environment
├── vibevoice/         # Cloned VibeVoice repo
└── .first_run_complete # Bootstrap flag
```
