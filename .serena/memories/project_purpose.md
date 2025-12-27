# Project Purpose

## VibeVoice RunPod Serverless

This project is a **RunPod Serverless deployment** of the **VibeVoice 7B** text-to-speech (TTS) model with voice cloning capabilities.

### Key Features

1. **Text-to-Speech Generation** - Convert text to natural-sounding speech using VibeVoice 7B model
2. **Voice Cloning** - Clone voices using reference audio files (.wav format)
3. **Smart Text Chunking** - Automatically chunk long texts at natural boundaries (sentences, clauses)
4. **Loudness Normalization** - Normalize audio output to -20 LUFS for consistent volume
5. **S3 Storage Integration** - Optional S3 storage with presigned URLs, fallback to base64
6. **OpenAI TTS API Compatibility** - Optional Cloudflare Worker bridge for OpenAI API compatibility

### Architecture

- **Serverless Deployment** - Runs on RunPod infrastructure with GPU support
- **Network Volume Persistence** - Models, cache, and voice files stored on persistent volume
- **First-Run Bootstrap** - Automatic setup on first deployment (venv, dependencies, model download)
- **Production-Ready** - Comprehensive error handling, fallbacks, and logging

### Primary Use Cases

1. Deploy as RunPod serverless endpoint for TTS generation
2. Generate audio from text with custom voice cloning
3. Integrate with applications via REST API or OpenAI-compatible bridge

### Reference Implementation

Uses patterns from the `chatterbox` reference implementation (in `chatterbox/` directory) for:
- Bootstrap script structure
- RunPod integration patterns
- S3 storage handling
- Chunking strategies

The actual VibeVoice upstream code is in `VibeVoice/` directory but is cloned at runtime, not deployed.
