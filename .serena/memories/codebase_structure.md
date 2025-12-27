# Codebase Structure

## High-Level Architecture

### Three-Layer System

```
┌─────────────────────────────────────────┐
│         Container Layer                  │
│  (Dockerfile, bootstrap.sh)              │
│  - CUDA base image                       │
│  - First-run setup                       │
│  - Dependency installation               │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      Serverless Handler Layer           │
│  (handler.py, config.py)                │
│  - RunPod entry point                   │
│  - Input validation                     │
│  - S3 upload / base64 fallback          │
│  - Loudness normalization               │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│        Inference Layer                  │
│  (inference.py)                         │
│  - VibeVoice model loading              │
│  - Voice cloning (VoiceMapper)          │
│  - Smart text chunking                  │
│  - Audio generation                     │
└─────────────────────────────────────────┘
```

### Optional Bridge Layer
```
┌─────────────────────────────────────────┐
│   Cloudflare Worker Bridge              │
│   (bridge/worker.js)                    │
│   - OpenAI TTS API compatibility        │
│   - Voice name mapping                  │
│   - Request translation                 │
└─────────────────────────────────────────┘
```

## File Organization

### Core Application Files

#### `config.py`
- **Purpose**: Centralized configuration management
- **Key Components**:
  - Environment variable loading
  - Path definitions for network volume
  - Audio configuration constants
  - Model configuration
- **Important Exports**:
  - `HF_TOKEN`, `HF_HOME`, `HF_HUB_CACHE`
  - `S3_*` variables for S3 integration
  - `AUDIO_PROMPTS_DIR` - Voice reference directory
  - `MAX_TEXT_LENGTH`, `DEFAULT_SAMPLE_RATE`, `MAX_CHUNK_CHARS`
  - `MODEL_PATH` = "vibevoice/VibeVoice-7B"

#### `handler.py`
- **Purpose**: RunPod serverless entry point
- **Key Components**:
  - `cleanup_old_files()` - Remove files older than 2 days
  - `upload_to_s3()` - Upload to S3 or return None
  - `handler()` - Main RunPod handler function
- **Flow**:
  1. Clean up old output files
  2. Extract and validate input parameters
  3. Call inference engine
  4. Apply loudness normalization
  5. Convert to MP3
  6. Upload to S3 or encode as base64
  7. Return response

#### `inference.py`
- **Purpose**: VibeVoice inference engine
- **Key Components**:
  
  **VoiceMapper Class**:
  - `setup_voice_presets()` - Scan voices directory for .wav files
  - `get_voice_path()` - Map speaker name to voice file path
  - Supports partial case-insensitive matching
  
  **VibeVoiceInference Class**:
  - `__init__()` - Initialize with device detection, create temp dir
  - `__del__()` - Cleanup temp directory on deletion
  - `load_model()` - Lazy load VibeVoice model with flash_attention fallback
  - `_smart_chunk_text()` - Intelligent text chunking at sentence/clause boundaries
  - `generate()` - Main generation method with chunking support

#### `bootstrap.sh`
- **Purpose**: Runtime setup script
- **Flow**:
  1. Create directory structure on network volume
  2. Export environment variables (HF_HOME, HF_HUB_CACHE, HF_TOKEN)
  3. Check for first run flag
  
  **First Run**:
  1. Create virtual environment
  2. Install PyTorch 2.8.0 with CUDA 12.8
  3. Install dependencies (huggingface_hub, runpod, boto3, etc.)
  4. Clone VibeVoice repository to `vibevoice` directory
  5. Install VibeVoice with `pip install -e .`
  6. Pre-download VibeVoice-7B model
  7. Create first run flag
  
  **Subsequent Runs**:
  1. Activate existing virtual environment
  2. Start handler immediately

#### `Dockerfile`
- **Base**: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`
- **System Deps**: python3.12, git, ffmpeg, libsndfile1
- **Python Deps**: Install from requirements.txt
- **Copies**: bootstrap.sh, handler.py, inference.py, config.py
- **CMD**: Execute bootstrap.sh

### Documentation Files

- **IMPLEMENTATION.md** - Complete implementation plan with architecture, API specs, deployment guide
- **CLAUDE.md** - Guidance for Claude Code (architecture, commands, pitfalls)
- **FIX_PLAN.md** - Architectural fixes with before/after code
- **VALIDATION_REPORT.md** - Validation of all fixes

### Reference Directories (Not Deployed)

- **chatterbox/** - Reference implementation for patterns (bootstrap, RunPod, S3)
- **VibeVoice/** - Upstream VibeVoice source code (cloned at runtime instead)

### Bridge Components (Optional)

- **bridge/worker.js** - Cloudflare Worker for OpenAI API compatibility
- **bridge/wrangler.toml.example** - Configuration template
- **bridge/voices.json** - Voice name mappings (OpenAI → VibeVoice)
- **bridge/README.md** - Bridge documentation

## Data Flow

### Request Processing Flow

```
1. RunPod receives request
        ↓
2. bootstrap.sh activates venv and starts handler.py
        ↓
3. handler.py receives job input
        ↓
4. Validate parameters (text, cfg_scale, speaker_name)
        ↓
5. inference.py: VoiceMapper.get_voice_path()
        ↓
6. inference.py: Smart text chunking
        ↓
7. inference.py: Generate audio chunks
        ↓
8. handler.py: Concatenate chunks (if multiple)
        ↓
9. handler.py: Apply loudness normalization
        ↓
10. handler.py: Convert to MP3
        ↓
11. handler.py: Upload to S3 OR encode base64
        ↓
12. Return response with audio_url or audio_base64
```

### Bootstrap Flow

```
Container Start
    ↓
Create /runpod-volume/vibevoice/ structure
    ↓
Export environment variables
    ↓
Check .first_run_complete flag
    ↓
┌───────────────────────────────────┐
│ First Run (5-10 min)              │
│ - Create venv                     │
│ - Install PyTorch 2.8.0           │
│ - Install dependencies            │
│ - Clone VibeVoice repo            │
│ - Install VibeVoice               │
│ - Download 7B model (~14GB)       │
│ - Create flag                     │
└───────────────────────────────────┘
    OR
┌───────────────────────────────────┐
│ Subsequent Runs (30-60 sec)       │
│ - Activate existing venv          │
│ - Skip all setup                  │
└───────────────────────────────────┘
    ↓
Start handler.py
```

## Critical Dependencies

### Runtime Dependencies
- VibeVoice repository MUST be cloned at `/runpod-volume/vibevoice/vibevoice/`
- Voice files MUST be in `/runpod-volume/vibevoice/demo/voices/` as .wav files
- HF_TOKEN environment variable MUST be set for model access
- Network volume MUST be attached for persistence

### Import Path Critical
- `inference.py` adds `/runpod-volume/vibevoice/vibevoice` to sys.path
- This allows importing from the cloned VibeVoice repository
- MUST match the git clone directory name in bootstrap.sh

## Design Patterns

### 1. Lazy Loading Pattern
- Model loaded on first `generate()` call, not at handler startup
- Reduces cold start time

### 2. Graceful Degradation Pattern
- flash_attention_2 → sdpa fallback
- S3 upload → base64 encoding fallback
- Loudness normalization → skip on failure

### 3. Smart Chunking Pattern
- Split at sentence boundaries first
- Fall back to clause boundaries
- Fall back to word boundaries
- Ensures natural-sounding concatenated audio

### 4. First-Run Bootstrap Pattern
- Flag file (`.first_run_complete`) prevents re-setup
- All setup happens on network volume for persistence
- Subsequent starts are fast (reuse venv and cached model)

### 5. Cleanup Pattern
- `__del__` methods for instance cleanup
- `cleanup_old_files()` for aged outputs
- Temp file removal in try/except/finally blocks
