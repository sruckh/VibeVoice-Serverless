# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 🛑 STOP — Run codemap before ANY task

```bash
codemap .                     # Project structure
codemap --deps                # How files connect
codemap --diff                # What changed vs main
codemap --diff --ref <branch> # Changes vs specific branch
```

## Required Usage

**BEFORE starting any task**, run `codemap .` first.

**ALWAYS run `codemap --deps` when:**
- User asks how something works
- Refactoring or moving code
- Tracing imports or dependencies

**ALWAYS run `codemap --diff` when:**
- Reviewing or summarizing changes
- Before committing code
- User asks what changed
- Use `--ref <branch>` when comparing against something other than main

## Project Overview

This is a RunPod Serverless deployment of VibeVoice 7B, a text-to-speech (TTS) model with voice cloning capabilities. The project wraps the VibeVoice model in a serverless API that can be deployed on RunPod infrastructure with optional S3 storage and OpenAI TTS API compatibility via a Cloudflare Worker bridge.

**Key Reference:** See `IMPLEMENTATION.md` for the complete implementation plan and architecture details.

### Reference Directories (Not Part of Deployment)

- `chatterbox/` - Reference implementation of a working RunPod serverless TTS project. Used only as a pattern guide for scaffolding (bootstrap, RunPod integration, S3 storage). **Not deployed.**
- `VibeVoice/` - Upstream source code from the VibeVoice repository. **Not deployed.** The actual VibeVoice code is cloned at runtime by `bootstrap.sh` into `/runpod-volume/vibevoice/vibevoice/`.

## Architecture

### Three-Layer System

1. **Container Layer** (`Dockerfile`, `bootstrap.sh`)
   - CUDA 12.8.1 base with Python 3.12, PyTorch 2.8.0
   - Bootstrap script handles first-run setup (venv creation, VibeVoice installation, model caching)
   - All installations go to `/runpod-volume/vibevoice/` for persistence

2. **Serverless Handler Layer** (`handler.py`, `config.py`)
   - RunPod serverless entry point
   - Input validation, S3 upload (optional), base64 fallback
   - Automatic cleanup of old output files (2+ days)

3. **Inference Layer** (`inference.py`)
   - Loads VibeVoice model from HuggingFace (`vibevoice/VibeVoice-7B`)
   - Smart text chunking for long inputs (max 300 chars/chunk by default)
   - Voice cloning using reference audio from `/runpod-volume/vibevoice/demo/voices/`
   - Generates 24kHz MP3 audio

### Critical Path Dependency

**VibeVoice is cloned at runtime, not build time.** The `bootstrap.sh` script clones the VibeVoice repository from GitHub (`https://github.com/vibevoice-community/VibeVoice.git`) into `/runpod-volume/vibevoice/vibevoice/` during first run. This means:

- `inference.py` imports VibeVoice classes via `sys.path.insert(0, '/runpod-volume/vibevoice/vibevoice')`
- You cannot import VibeVoice modules during local development without cloning the repo
- The actual VibeVoice implementation is external to this repository

### Optional Bridge Layer (`bridge/worker.js`)

Cloudflare Worker that translates OpenAI TTS API requests to the custom VibeVoice API format:
- Exposes POST `/v1/audio/speech` endpoint
- Maps OpenAI voice names (alloy, echo, etc.) to VibeVoice speakers via R2-stored `voices.json`
- Returns raw audio bytes (OpenAI-compatible)

## Building and Deployment

### Local Docker Build

```bash
# Build the container
docker build -t vibevoice-runpod .

# Note: Container requires GPU runtime and network volume mount to actually run
# Local testing is limited without RunPod environment
```

### RunPod Deployment

1. **Push to GitHub** - RunPod builds from GitHub repository
2. **Create Serverless Endpoint** in RunPod dashboard:
   - Container source: GitHub
   - Attach network volume (required for model caching)
   - Set environment variables (see below)
3. **Upload voice files** - `.wav` files go in `/runpod-volume/vibevoice/demo/voices/` on the network volume
4. **First run** - Takes 2-3 minutes (PyTorch + VibeVoice installation + model download)
5. **Subsequent runs** - ~30-60 seconds (model loading only)

### Environment Variables

**Required:**
- `HF_TOKEN` - HuggingFace token for model access

**Optional (S3):**
- `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_BUCKET_NAME`, `S3_REGION`

**Optional (Tuning):**
- `MAX_TEXT_LENGTH` (default: 2000)
- `DEFAULT_SAMPLE_RATE` (default: 24000)
- `MAX_CHUNK_CHARS` (default: 1000)
- `MIN_LAST_CHUNK_CHARS` (default: 150) - minimum last-chunk size before merge
- `CHUNK_SILENCE_MS` (default: 40) - silence padding between chunks
- `DEFAULT_SPEAKER` (default: "Alice")
- `DEFAULT_CFG_SCALE` (default: 1.3)

## API Usage

### Custom API (Direct RunPod)

```bash
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync \
  -H "Authorization: Bearer {RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Hello! This is a test.",
      "speaker_name": "Alice",
      "cfg_scale": 1.3,
      "disable_prefill": false
    }
  }'
```

**Parameters:**
- `text` (required) - Text to synthesize
- `speaker_name` (optional) - Speaker name matching voice file in `demo/voices/`
- `cfg_scale` (optional) - Classifier-free guidance scale
- `disable_prefill` (optional) - Disable voice cloning

**Response:**
```json
{
  "status": "success",
  "sample_rate": 24000,
  "duration_sec": 3.45,
  "audio_url": "https://presigned-s3-url.com/audio.mp3"
}
```

### OpenAI API (via Cloudflare Worker)

```bash
curl -X POST https://your-worker.workers.dev/v1/audio/speech \
  -H "Authorization: Bearer {AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello! This is a test.",
    "voice": "alloy"
  }'
```

## Key Implementation Details

### Voice Cloning

- Voice reference audio must be `.wav` format in `/runpod-volume/vibevoice/demo/voices/`
- `VoiceMapper` class scans this directory and creates name → path mappings
- Speaker names support partial, case-insensitive matching
- If no match found, defaults to first available voice

### Text Chunking

Smart chunking at natural boundaries (sentences → clauses → words):
1. Tries to split at sentence endings (`. `, `! `, `? `)
2. Falls back to clause boundaries (`, `, `; `, `: `)
3. Falls back to word boundaries
4. Each chunk respects `MAX_CHUNK_CHARS` limit
5. Audio chunks are concatenated using `torch.cat()`

### Model Loading

- Model: `vibevoice/VibeVoice-7B` from HuggingFace
- Uses `bfloat16` + `flash_attention_2` on CUDA
- Falls back to `float32` + `sdpa` on CPU
- Processor and model loaded lazily on first inference request
- DDPM inference uses 10 steps (hardcoded in `inference.py:471`)

### Storage Strategy

- **Local volume**: All generated files saved to `/runpod-volume/vibevoice/output/`
- **S3 (optional)**: Files uploaded with 1-hour presigned URLs
- **Base64 fallback**: If S3 not configured, audio returned as base64 string
- **Cleanup**: Files older than 2 days automatically deleted from local volume

## Development Workflow

### Making Changes

1. **Modify Python files** (`config.py`, `handler.py`, `inference.py`)
2. **Test locally if possible** (limited without GPU + VibeVoice clone)
3. **Push to GitHub** - RunPod rebuilds automatically
4. **Test on RunPod** - Use the `/runsync` endpoint

### Debugging

- Check RunPod logs for bootstrap output (first run vs. subsequent)
- Verify `HF_TOKEN` has access to VibeVoice model
- Check network volume mounts and directory structure
- Confirm voice files are present in `/runpod-volume/vibevoice/demo/voices/`
- Verify VRAM availability (VibeVoice 7B requires significant GPU memory)

### Cloudflare Worker Bridge

```bash
# Deploy worker
cd bridge
wrangler deploy

# Set secrets
wrangler secret put RUNPOD_URL
wrangler secret put RUNPOD_API_KEY
wrangler secret put AUTH_TOKEN

# Upload voices.json to R2 bucket
wrangler r2 object put VIBEVOICE_BUCKET/voices.json --file=voices.json
```

## Common Pitfalls

1. **VibeVoice import errors** - Remember it's cloned at runtime, not available during build
2. **Flash Attention failures** - GPU must support flash_attention_2, otherwise falls back to sdpa
3. **Voice file not found** - Check exact filename matches speaker_name (case-insensitive partial matching supported)
4. **Long generation times** - Text chunking activates above `MAX_CHUNK_CHARS` (1000 default), each chunk processed sequentially
5. **S3 upload failures** - Check all S3 environment variables are set; system falls back to base64
6. **First run timeout** - First cold start takes 2-3 minutes; increase RunPod timeout if needed

## File Organization

```
/workspace/vibevoice/          # Container working directory
├── config.py                  # Environment variables and configuration
├── handler.py                 # RunPod serverless entry point
├── inference.py               # VibeVoice inference engine
└── bootstrap.sh               # Runtime setup script

/runpod-volume/vibevoice/      # Network volume (persistent)
├── hf_home/                   # HuggingFace home
├── hf_cache/                  # Model cache
├── models/                    # Downloaded models
├── output/                    # Generated audio (auto-cleanup)
├── demo/voices/               # Voice reference audio (.wav)
├── venv/                      # Virtual environment
├── vibevoice/                 # Cloned VibeVoice repository
└── .first_run_complete        # Flag file for bootstrap logic
```
