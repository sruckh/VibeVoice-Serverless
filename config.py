import os

# Environment Variables
HF_TOKEN = os.environ.get("HF_TOKEN")  # Required for model access

# HuggingFace cache configuration (use default unless explicitly set)
HF_HOME = os.environ.get("HF_HOME")
HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
if HF_HOME:
    os.environ["HF_HOME"] = HF_HOME
if HF_HUB_CACHE:
    os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE

# Torch cache configuration
TORCH_HOME = os.environ.get("TORCH_HOME", "/runpod-volume/vibevoice/torch_cache")
os.environ["TORCH_HOME"] = TORCH_HOME

# S3 Configuration
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")

# Runpod volume structure
RUNPOD_VOLUME = "/runpod-volume"
VIBEVOICE_DIR = f"{RUNPOD_VOLUME}/vibevoice"
MODEL_CACHE_DIR = f"{VIBEVOICE_DIR}/models"
OUTPUT_DIR = f"{VIBEVOICE_DIR}/output"
AUDIO_PROMPTS_DIR = f"{VIBEVOICE_DIR}/demo/voices"  # Voice reference audio

# Application Configuration
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", "2000"))
DEFAULT_SAMPLE_RATE = int(os.environ.get("DEFAULT_SAMPLE_RATE", "24000"))
MAX_CHUNK_CHARS = int(os.environ.get("MAX_CHUNK_CHARS", "1000"))
MIN_LAST_CHUNK_CHARS = int(os.environ.get("MIN_LAST_CHUNK_CHARS", "150"))
CHUNK_SILENCE_MS = int(os.environ.get("CHUNK_SILENCE_MS", "40"))

# Audio configuration
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}
MIN_AUDIO_DURATION = 3.0  # seconds
MAX_AUDIO_DURATION = 30.0  # seconds

# VibeVoice model configuration
MODEL_PATH = "vibevoice/VibeVoice-7B"
DEFAULT_SPEAKER = os.environ.get("DEFAULT_SPEAKER", "Alice")
DEFAULT_CFG_SCALE = float(os.environ.get("DEFAULT_CFG_SCALE", "1.3"))
