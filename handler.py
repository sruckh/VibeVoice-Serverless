import runpod
import os
import logging
import base64
import io
import uuid
import soundfile as sf
import boto3
import numpy as np
from botocore.exceptions import NoCredentialsError
import time
from pathlib import Path
import subprocess

from inference import VibeVoiceInference
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Initialize model loader
inference_engine = VibeVoiceInference()

try:
    from linacodec.codec import LinaCodec
    LINACODEC_AVAILABLE = True
except Exception:
    LinaCodec = None
    LINACODEC_AVAILABLE = False

LINA_CODEC = None

def load_linacodec():
    """Load and cache LinaCodec encoder/decoder."""
    global LINA_CODEC
    if not LINACODEC_AVAILABLE:
        raise RuntimeError("LinaCodec is not installed")
    if LINA_CODEC is not None:
        return LINA_CODEC
    LINA_CODEC = LinaCodec()
    return LINA_CODEC

def to_numpy_audio(wav):
    """Convert torch/numpy audio to 1D float32 numpy array."""
    if hasattr(wav, "cpu"):
        wav = wav.float().cpu().numpy()
    elif hasattr(wav, "numpy"):
        wav = wav.numpy()
    if len(wav.shape) > 1:
        wav = wav.squeeze()
    return wav.astype(np.float32)

def encode_to_linacodec(audio):
    """Encode audio to LinaCodec tokens and embedding."""
    lina = load_linacodec()
    tokens, embedding = lina.encode(audio)
    return tokens, embedding

def decode_with_linacodec(audio):
    """Encode and decode audio via LinaCodec to get 48kHz output."""
    lina = load_linacodec()
    tokens, embedding = lina.encode(audio)
    decoded = lina.decode(tokens, embedding)
    if hasattr(decoded, "cpu"):
        decoded = decoded.cpu().numpy()
    return np.asarray(decoded, dtype=np.float32)

def pcm16_base64(audio):
    """Convert float32 audio to base64-encoded PCM16."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode("utf-8")

def encode_mp3_bytes(audio, sample_rate):
    """Encode float32 or int16 mono audio to MP3 bytes using ffmpeg."""
    if audio.dtype != np.int16:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767.0).astype(np.int16)

    raw_bytes = audio.tobytes()

    try:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f", "s16le",
                "-ar", str(sample_rate),
                "-ac", "1",
                "-i", "pipe:0",
                "-f", "mp3",
                "-b:a", "192k",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        mp3_bytes, _ = process.communicate(input=raw_bytes)
        return mp3_bytes
    except Exception as e:
        log.error(f"FFmpeg encoding failed: {e}")
        return b""

def cleanup_old_files(directory, days=2):
    """Delete files older than specified days from directory"""
    try:
        output_dir = Path(directory)
        if not output_dir.exists():
            log.debug(f"Output directory {directory} does not exist, skipping cleanup")
            return

        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)

        deleted_count = 0
        for file_path in output_dir.glob('*'):
            if file_path.is_file():
                file_age = file_path.stat().st_mtime
                if file_age < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        log.debug(f"Deleted old file: {file_path.name}")
                    except Exception as e:
                        log.warning(f"Failed to delete {file_path.name}: {e}")

        if deleted_count > 0:
            log.info(f"Cleaned up {deleted_count} files older than {days} days from {directory}")
    except Exception as e:
        log.error(f"Cleanup failed: {e}")

def upload_to_s3(audio_buffer, filename):
    """Upload generated audio to S3 and return URL"""
    if not config.S3_BUCKET_NAME:
        log.warning("S3_BUCKET_NAME not set, returning base64 audio")
        return None

    # Validate required credentials
    if not all([config.S3_ACCESS_KEY_ID, config.S3_SECRET_ACCESS_KEY]):
        log.warning("S3 credentials incomplete (missing access key or secret), falling back to base64")
        return None

    try:
        s3 = boto3.client(
            's3',
            endpoint_url=config.S3_ENDPOINT_URL,
            aws_access_key_id=config.S3_ACCESS_KEY_ID,
            aws_secret_access_key=config.S3_SECRET_ACCESS_KEY,
            region_name=config.S3_REGION
        )

        s3.upload_fileobj(
            audio_buffer,
            config.S3_BUCKET_NAME,
            filename,
            ExtraArgs={'ContentType': 'audio/mpeg'}
        )

        # Generate presigned URL
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': config.S3_BUCKET_NAME, 'Key': filename},
            ExpiresIn=3600  # 1 hour
        )
        log.info(f"Successfully uploaded to S3: {filename}")
        return url
    except Exception as e:
        log.error(f"S3 upload failed: {e}")
        return None

def stream_audio_chunks(text, speaker_name, cfg_scale, disable_prefill, output_format):
    """Generator for streaming audio or LinaCodec tokens."""
    sample_rate = config.DEFAULT_SAMPLE_RATE
    use_linacodec_decode = output_format in {"pcm_16", "mp3"} and LINACODEC_AVAILABLE

    try:
        chunk_num = 0
        for wav_chunk in inference_engine.generate_stream(
            text=text,
            speaker_name=speaker_name,
            cfg_scale=cfg_scale,
            disable_prefill=disable_prefill,
        ):
            chunk_num += 1
            audio = to_numpy_audio(wav_chunk)

            if output_format == "linacodec_tokens":
                tokens, embedding = encode_to_linacodec(audio)
                tokens_list = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
                embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

                yield {
                    "status": "streaming",
                    "chunk": chunk_num,
                    "format": "linacodec_tokens",
                    "tokens": tokens_list,
                    "embedding": embedding_list,
                    "sample_rate": 48000,
                    "original_sample_rate": sample_rate,
                    "num_tokens": len(tokens_list),
                }
                continue

            if use_linacodec_decode:
                decoded = decode_with_linacodec(audio)
                out_sample_rate = 48000
            else:
                out_sample_rate = sample_rate
                decoded = audio

            if output_format == "mp3":
                mp3_bytes = encode_mp3_bytes(decoded, out_sample_rate)
                audio_b64 = base64.b64encode(mp3_bytes).decode("utf-8")
                fmt = "mp3"
            else:
                audio_b64 = pcm16_base64(decoded)
                fmt = "pcm_16"

            yield {
                "status": "streaming",
                "chunk": chunk_num,
                "format": fmt,
                "audio_chunk": audio_b64,
                "sample_rate": out_sample_rate,
            }

        yield {
            "status": "complete",
            "format": "linacodec_tokens" if output_format == "linacodec_tokens" else output_format,
            "message": "All chunks streamed",
        }
    except Exception as e:
        log.error(f"Streaming failed: {e}")
        yield {"error": str(e)}

def handler(job):
    """Runpod serverless handler

    Expected input format:
    {
        "text": str (required) - Text to synthesize
        "speaker_name": str (optional) - Speaker name for voice cloning (default: "Alice")
        "cfg_scale": float (optional) - CFG scale (default: 1.3)
        "disable_prefill": bool (optional) - Disable voice cloning (default: False)
    }

    Returns:
    {
        "status": "success",
        "sample_rate": int,
        "duration_sec": float,
        "audio_url": str (if S3 configured) OR "audio_base64": str (fallback)
    }
    """
    # Clean up old output files (older than 2 days)
    cleanup_old_files(config.OUTPUT_DIR, days=2)

    job_input = job.get("input", {})

    # Extract parameters
    text = job_input.get("text")
    if not text or not text.strip():
        return {"error": "Missing or empty 'text' parameter"}

    text = text.strip()
    speaker_name = job_input.get("speaker_name", config.DEFAULT_SPEAKER)
    session_id = job_input.get("session_id", str(uuid.uuid4()))

    # VibeVoice generation parameters
    try:
        cfg_scale = float(job_input.get("cfg_scale", config.DEFAULT_CFG_SCALE))
        if cfg_scale <= 0:
            return {"error": "cfg_scale must be a positive number"}
    except (ValueError, TypeError):
        return {"error": f"cfg_scale must be a valid number, got: {job_input.get('cfg_scale')}"}

    disable_prefill = bool(job_input.get("disable_prefill", False))
    stream = bool(job_input.get("stream", False))
    output_format = job_input.get("output_format", "pcm_16")

    # Validate input
    if len(text) > config.MAX_TEXT_LENGTH:
        return {"error": f"Text length exceeds maximum of {config.MAX_TEXT_LENGTH}"}

    try:
        if stream:
            return runpod.stream(
                stream_audio_chunks(
                    text=text,
                    speaker_name=speaker_name,
                    cfg_scale=cfg_scale,
                    disable_prefill=disable_prefill,
                    output_format=output_format,
                )
            )

        # Generate audio (batch)
        wav = inference_engine.generate(
            text=text,
            speaker_name=speaker_name,
            cfg_scale=cfg_scale,
            disable_prefill=disable_prefill,
        )

        if wav is None:
            return {"error": "Failed to generate audio"}

        # Use configured sample rate
        sample_rate = config.DEFAULT_SAMPLE_RATE
        log.info(f"Using sample rate: {sample_rate}")

        # Convert to numpy
        log.info("Converting audio tensor to numpy...")
        wav = to_numpy_audio(wav)

        if output_format != "linacodec_tokens":
            # Apply loudness normalization
            try:
                import pyloudnorm as pyln
                meter = pyln.Meter(sample_rate)  # create BS.1770 meter
                loudness = meter.integrated_loudness(wav)

                # Normalize to -20 LUFS (standard for TTS)
                target_loudness = -20.0
                if loudness > -100:  # Only normalize if loudness is measurable
                    normalized_wav = pyln.normalize.loudness(wav, loudness, target_loudness)
                    log.info(f"Normalized audio from {loudness:.1f} LUFS to {target_loudness} LUFS")
                    wav = normalized_wav
                else:
                    log.warning(f"Audio too quiet to measure loudness ({loudness:.1f} LUFS), skipping normalization")
            except Exception as e:
                log.warning(f"Loudness normalization failed: {e}, using unnormalized audio")

        if output_format == "linacodec_tokens":
            if not LINACODEC_AVAILABLE:
                return {"error": "LinaCodec not available"}
            tokens, embedding = encode_to_linacodec(wav)
            tokens_list = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
            embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            return {
                "status": "success",
                "format": "linacodec_tokens",
                "tokens": tokens_list,
                "embedding": embedding_list,
                "sample_rate": 48000,
                "duration_sec": len(wav) / sample_rate,
            }

        if output_format == "pcm_16":
            return {
                "status": "success",
                "format": "pcm_16",
                "sample_rate": sample_rate,
                "duration_sec": len(wav) / sample_rate,
                "audio_base64": pcm16_base64(wav),
            }

        # Default: MP3
        audio_buffer = io.BytesIO()
        log.info(f"Writing audio to buffer (shape: {wav.shape})...")
        sf.write(audio_buffer, wav, sample_rate, format="MP3")
        audio_buffer.seek(0)

        # Upload to S3 or return base64
        filename = f"{session_id}_{uuid.uuid4()}.mp3"

        # Local output path for persistence in volume
        output_path = os.path.join(config.OUTPUT_DIR, filename)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        log.info(f"Saving audio locally to {output_path}...")
        with open(output_path, "wb") as f:
            f.write(audio_buffer.getbuffer())

        # Reset buffer for S3 upload
        audio_buffer.seek(0)

        log.info("Uploading to S3 (if configured)...")
        s3_url = upload_to_s3(audio_buffer, filename)

        response = {
            "status": "success",
            "sample_rate": sample_rate,
            "duration_sec": len(wav) / sample_rate,
        }

        if s3_url:
            response["audio_url"] = s3_url
        else:
            # Fallback to base64
            audio_buffer.seek(0)
            b64_audio = base64.b64encode(audio_buffer.read()).decode("utf-8")
            response["audio_base64"] = b64_audio

        log.info("Handler completed successfully.")
        return response

    except Exception as e:
        log.error(f"Inference failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
