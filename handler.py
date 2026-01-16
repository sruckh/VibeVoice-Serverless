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
import tempfile

from inference import VibeVoiceInference
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Initialize model loader
inference_engine = VibeVoiceInference()

def to_numpy_audio(wav):
    """Convert torch/numpy audio to 1D float32 numpy array."""
    if hasattr(wav, "cpu"):
        wav = wav.float().cpu().numpy()
    elif hasattr(wav, "numpy"):
        wav = wav.numpy()
    if len(wav.shape) > 1:
        wav = wav.squeeze()
    return wav.astype(np.float32)

def pcm16_base64(audio):
    """Convert float32 audio to base64-encoded PCM16."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode("utf-8")

def encode_mp3_bytes(audio, src_rate, dst_rate=48000):
    """Encode float32 or int16 mono audio to MP3 bytes using ffmpeg, with resampling."""
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
                "-ar", str(src_rate),  # Input sample rate
                "-ac", "1",
                "-i", "pipe:0",
                "-f", "mp3",
                "-ar", str(dst_rate),  # Output sample rate
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

def resample_pcm_bytes(audio_bytes, src_rate, dst_rate=48000):
    """Resample PCM bytes using scipy.signal.resample. High quality, no artifacts."""
    if src_rate == dst_rate:
        return audio_bytes

    try:
        import scipy.signal

        # Convert bytes to numpy array
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        
        if len(audio) == 0:
            return audio_bytes

        # Calculate number of samples
        num_samples = int(len(audio) * dst_rate / src_rate)
        
        # Resample using Fourier method (best for maintaining signal integrity)
        resampled = scipy.signal.resample(audio, num_samples)
        
        # Convert back to int16 bytes
        resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
        return resampled.tobytes()
        
    except ImportError:
        log.error("Scipy not found, falling back to numpy linear interp (lower quality)")
        # Fallback to linear
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        duration_sec = len(audio) / src_rate
        new_length = int(duration_sec * dst_rate)
        x_old = np.linspace(0, duration_sec, len(audio))
        x_new = np.linspace(0, duration_sec, new_length)
        resampled = np.interp(x_new, x_old, audio)
        resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
        return resampled.tobytes()
        
    except Exception as e:
        log.error(f"Resampling failed: {e}")
        return audio_bytes

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

def upload_to_s3(audio_buffer, filename, content_type='application/octet-stream'):
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
            ExtraArgs={'ContentType': content_type}
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

def stream_audio_chunks(text, speaker_name, cfg_scale, disable_prefill, output_format, max_chunk_chars=None):
    """Generator for streaming audio chunks (pcm_16 or mp3).
    
    Sends base64 chunks directly for true streaming (no S3 upload).
    """
    session_id = str(uuid.uuid4())
    
    log.info(f"[Streaming][{session_id}] Starting stream: output_format={output_format}, max_chunk_chars={max_chunk_chars}")

    try:
        chunk_count = 0
        
        # Use the new generate_audio_stream_decoded method with LinaCodec
        for response_item in inference_engine.generate_audio_stream_decoded(
            text=text,
            speaker_name=speaker_name,
            cfg_scale=cfg_scale,
            disable_prefill=disable_prefill,
            max_chunk_chars=max_chunk_chars,
            output_format=output_format,
        ):
            # For streaming mode: Send base64 directly (no S3 upload)
            # This enables true streaming with minimal latency
            if response_item.get('status') == 'streaming':
                chunk_count += 1
                log.info(f"[Streaming][{session_id}] Yielding chunk {chunk_count} (format={response_item.get('format')}, size={len(response_item.get('audio_chunk', ''))} bytes base64)")
            elif response_item.get('status') == 'complete':
                log.info(f"[Streaming][{session_id}] Stream complete: {chunk_count} total chunks")
            
            # Yield chunk immediately without S3 upload
            yield response_item

    except Exception as e:
        log.error(f"[Streaming][{session_id}] Streaming failed: {e}")
        yield {
            "status": "error",
            "error": str(e)
        }

def _extract_and_validate_params(job_input):
    """Extract and validate parameters from job input."""
    text = job_input.get("text") or job_input.get("input")
    if not text or not text.strip():
        return None, {"error": "Missing or empty 'text' parameter"}

    text = text.strip()
    speaker_name = job_input.get("speaker_name", config.DEFAULT_SPEAKER)
    session_id = job_input.get("session_id", str(uuid.uuid4()))

    try:
        cfg_scale = float(job_input.get("cfg_scale", config.DEFAULT_CFG_SCALE))
        if cfg_scale <= 0:
            return None, {"error": "cfg_scale must be a positive number"}
    except (ValueError, TypeError):
        return None, {"error": f"cfg_scale must be a valid number, got: {job_input.get('cfg_scale')}"}

    disable_prefill = bool(job_input.get("disable_prefill", False))

    if len(text) > config.MAX_TEXT_LENGTH:
        return None, {"error": f"Text length exceeds maximum of {config.MAX_TEXT_LENGTH}"}

    return {
        "text": text,
        "speaker_name": speaker_name,
        "session_id": session_id,
        "cfg_scale": cfg_scale,
        "disable_prefill": disable_prefill,
    }, None

def handler_stream(job_input, output_format, request_id):
    """Streaming mode handler - yields audio chunks as they're generated."""
    log.info(f"[Handler][{request_id}] handler_stream called")
    
    params, error = _extract_and_validate_params(job_input)
    if error:
        log.error(f"[Handler][{request_id}] Validation error: {error}")
        yield error
        return

    if output_format not in {"pcm_16", "mp3"}:
        yield {"error": f"Unknown output_format: {output_format}"}
        return

    # Use 250 chars for streaming - balances prosody quality with progressive playback
    # 250 chars → ~17s audio → ~250KB MP3 → reasonable time-to-first-byte
    # Smaller chunks = faster TTFB but worse prosody; 250 is minimum for good quality
    max_chunk_chars = 250
    log.info(f"[Handler][{request_id}] max_chunk_chars={max_chunk_chars} for streaming (format={output_format})")

    yield from stream_audio_chunks(
        text=params["text"],
        speaker_name=params["speaker_name"],
        cfg_scale=params["cfg_scale"],
        disable_prefill=params["disable_prefill"],
        output_format=output_format,
        max_chunk_chars=max_chunk_chars,
    )

def handler_batch(job, output_format):
    """Batch mode handler - generates complete audio and returns S3 URL (like chatterbox)"""
    # Clean up old output files (older than 2 days)
    cleanup_old_files(config.OUTPUT_DIR, days=2)

    job_input = job.get("input", {})
    params, error = _extract_and_validate_params(job_input)
    if error:
        return error

    # Use larger chunks for batch mode to preserve quality
    # This is internal chunking only - final output is single complete file
    max_chunk_chars = 300

    try:
        wav = inference_engine.generate(
            text=params["text"],
            speaker_name=params["speaker_name"],
            cfg_scale=params["cfg_scale"],
            disable_prefill=params["disable_prefill"],
            max_chunk_chars=max_chunk_chars,
        )

        if wav is None:
            return {"error": "Failed to generate audio"}

        src_rate = config.DEFAULT_SAMPLE_RATE # 24000
        dst_rate = 48000 # Upsample to 48kHz
        log.info(f"Upsampling batch audio from {src_rate} to {dst_rate}")

        wav = to_numpy_audio(wav)

        # Loudness normalization
        if output_format != "linacodec_tokens":
            try:
                import pyloudnorm as pyln
                meter = pyln.Meter(src_rate)
                loudness = meter.integrated_loudness(wav)

                target_loudness = -20.0
                if loudness > -100:
                    normalized_wav = pyln.normalize.loudness(wav, loudness, target_loudness)
                    log.info(f"Normalized audio from {loudness:.1f} LUFS to {target_loudness} LUFS")
                    wav = normalized_wav
                else:
                    log.warning(f"Audio too quiet to measure loudness ({loudness:.1f} LUFS), skipping normalization")
            except Exception as e:
                log.warning(f"Loudness normalization failed: {e}, using unnormalized audio")

        # Prepare audio buffer and filename based on format
        audio_buffer = io.BytesIO()
        
        if output_format == "pcm_16":
            # Save as WAV file for PCM (with headers so it's playable)
            import soundfile as sf
            wav_clipped = np.clip(wav, -1.0, 1.0)
            # Resample to 48kHz
            import librosa
            wav_resampled = librosa.resample(wav_clipped, orig_sr=src_rate, target_sr=dst_rate)
            
            # Write WAV to buffer
            sf.write(audio_buffer, wav_resampled, dst_rate, format='WAV', subtype='PCM_16')
            audio_buffer.seek(0)
            
            filename = f"{params['session_id']}_{uuid.uuid4()}.wav"
            content_type = 'audio/wav'
            
        else:  # MP3
            mp3_bytes = encode_mp3_bytes(wav, src_rate, dst_rate)
            audio_buffer.write(mp3_bytes)
            audio_buffer.seek(0)
            
            filename = f"{params['session_id']}_{uuid.uuid4()}.mp3"
            content_type = 'audio/mpeg'

        # Save locally
        output_path = os.path.join(config.OUTPUT_DIR, filename)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        log.info(f"Saving audio locally to {output_path}...")
        with open(output_path, "wb") as f:
            f.write(audio_buffer.getbuffer())

        audio_buffer.seek(0)

        # Upload to S3 (like chatterbox does)
        log.info("Uploading to S3 (if configured)...")
        s3_url = upload_to_s3(audio_buffer, filename, content_type=content_type)

        duration_sec = len(wav) / src_rate
        
        response = {
            "status": "success",
            "format": output_format,
            "sample_rate": dst_rate,
            "duration_sec": duration_sec,
        }

        if s3_url:
            response["audio_url"] = s3_url
            log.info(f"Batch audio uploaded to S3: {s3_url}")
        else:
            # Fallback to base64 if S3 not configured
            audio_buffer.seek(0)
            b64_audio = base64.b64encode(audio_buffer.read()).decode("utf-8")
            response["audio_base64"] = b64_audio
            log.warning("S3 not configured, returning base64 (may be large)")

        log.info("Handler completed successfully.")
        return response

    except Exception as e:
        log.error(f"Inference failed: {e}")
        import traceback
        log.error(traceback.format_exc())
        return {"error": str(e)}

def handler(job):
    """Runpod serverless handler (streaming + batch)."""
    # Generate unique request ID to track double calls
    request_id = str(uuid.uuid4())[:8]

    job_input = job.get("input", {})

    # Robust stream parameter parsing - handle bool, string, int
    stream_value = job_input.get("stream", False)
    stream = stream_value is True or str(stream_value).lower() == "true"

    log.info(f"[Handler][{request_id}] stream_value={stream_value!r} (type={type(stream_value).__name__}) -> stream={stream}")

    output_format = job_input.get("output_format")

    if stream and not output_format:
        output_format = "pcm_16"
    if not stream and not output_format:
        output_format = "mp3"

    log.info(f"[Handler][{request_id}] {'STREAM' if stream else 'BATCH'} mode requested: format={output_format}")

    if stream:
        yield from handler_stream(job_input, output_format, request_id)
        log.info(f"[Handler][{request_id}] Stream handler complete")
        return

    result = handler_batch(job, output_format)
    log.info(f"[Handler][{request_id}] Batch handler complete")
    yield result

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True  # CRITICAL: Enables /runsync to capture generator yields
    })