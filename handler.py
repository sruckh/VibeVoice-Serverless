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
    """Resample PCM bytes using torchaudio (24k -> 48k)."""
    if src_rate == dst_rate:
        return audio_bytes

    try:
        # Convert bytes (int16) to float32 tensor
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0) # (1, time)

        # Resample
        resampler = torchaudio.transforms.Resample(src_rate, dst_rate)
        resampled_tensor = resampler(audio_tensor)

        # Convert back to int16 bytes
        resampled_np = resampled_tensor.squeeze(0).numpy()
        resampled_np = np.clip(resampled_np, -1.0, 1.0)
        pcm16 = (resampled_np * 32767.0).astype(np.int16)
        return pcm16.tobytes()
        
    except Exception as e:
        log.error(f"Torchaudio resampling failed: {e}")
        # Fallback to ffmpeg if torch fails? Or just return raw?
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

def stream_audio_chunks(text, speaker_name, cfg_scale, disable_prefill, output_format, max_chunk_chars=None):
    """Generator for streaming audio chunks (pcm_16 or mp3)."""
    src_rate = config.DEFAULT_SAMPLE_RATE # 24000
    dst_rate = 48000 # Upsample to 48kHz to match client expectations

    try:
        chunk_num = 0
        log.info(
            f"[Streaming] output_format={output_format}, src_rate={src_rate}, dst_rate={dst_rate}, max_chunk_chars={max_chunk_chars}"
        )

        for wav_chunk in inference_engine.generate_stream(
            text=text,
            speaker_name=speaker_name,
            cfg_scale=cfg_scale,
            disable_prefill=disable_prefill,
            max_chunk_chars=max_chunk_chars,
        ):
            chunk_num += 1
            audio = to_numpy_audio(wav_chunk)
            
            if output_format == "mp3":
                # FFmpeg handles resampling internally
                mp3_bytes = encode_mp3_bytes(audio, src_rate, dst_rate)
                audio_b64 = base64.b64encode(mp3_bytes).decode("utf-8")
                fmt = "mp3"
            else:
                # PCM: Must resample explicitly to 48kHz
                audio = np.clip(audio, -1.0, 1.0)
                pcm16 = (audio * 32767.0).astype(np.int16)
                raw_bytes = pcm16.tobytes()
                
                resampled_bytes = resample_pcm_bytes(raw_bytes, src_rate, dst_rate)
                audio_b64 = base64.b64encode(resampled_bytes).decode("utf-8")
                fmt = "pcm_16"

            yield {
                "status": "streaming",
                "chunk": chunk_num,
                "format": fmt,
                "audio_chunk": audio_b64,
                "sample_rate": dst_rate,
            }

        yield {
            "status": "complete",
            "format": output_format,
            "message": "All chunks streamed",
        }
    except Exception as e:
        log.error(f"Streaming failed: {e}")

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

def handler_stream(job_input, output_format):
    """Streaming mode handler - yields audio chunks as they're generated."""
    params, error = _extract_and_validate_params(job_input)
    if error:
        yield error
        return

    if output_format not in {"pcm_16", "mp3"}:
        yield {"error": f"Unknown output_format: {output_format}"}
        return

    # Force chunk size to ~500 chars (~20s audio).
    # This provides good prosody while keeping payload size very safe.
    max_chunk_chars = 500
    log.info(f"Using forced max_chunk_chars={max_chunk_chars} for better prosody")

    yield from stream_audio_chunks(
        text=params["text"],
        speaker_name=params["speaker_name"],
        cfg_scale=params["cfg_scale"],
        disable_prefill=params["disable_prefill"],
        output_format=output_format,
        max_chunk_chars=max_chunk_chars,
    )

def handler_batch(job, output_format):
    """Batch mode handler - generates complete audio and returns URL/base64"""
    # Clean up old output files (older than 2 days)
    cleanup_old_files(config.OUTPUT_DIR, days=2)

    job_input = job.get("input", {})
    params, error = _extract_and_validate_params(job_input)
    if error:
        return error

    # Use larger chunks for batch mode to preserve quality
    max_chunk_chars = 1000

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

        if output_format == "pcm_16":
            # PCM: Must resample explicitly to 48kHz
            wav = np.clip(wav, -1.0, 1.0)
            pcm16 = (wav * 32767.0).astype(np.int16)
            resampled_bytes = resample_pcm_bytes(pcm16.tobytes(), src_rate, dst_rate)
            
            return {
                "status": "success",
                "format": "pcm_16",
                "sample_rate": dst_rate,
                "duration_sec": len(wav) / src_rate,
                "audio_base64": base64.b64encode(resampled_bytes).decode("utf-8"),
            }

        # MP3 handling
        audio_buffer = io.BytesIO()
        mp3_bytes = encode_mp3_bytes(wav, src_rate, dst_rate)
        audio_buffer.write(mp3_bytes)
        audio_buffer.seek(0)

        filename = f"{params['session_id']}_{uuid.uuid4()}.mp3"

        output_path = os.path.join(config.OUTPUT_DIR, filename)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        log.info(f"Saving audio locally to {output_path}...")
        with open(output_path, "wb") as f:
            f.write(audio_buffer.getbuffer())

        audio_buffer.seek(0)

        log.info("Uploading to S3 (if configured)...")
        s3_url = upload_to_s3(audio_buffer, filename)

        response = {
            "status": "success",
            "sample_rate": dst_rate,
            "duration_sec": len(wav) / src_rate,
        }

        if s3_url:
            response["audio_url"] = s3_url
        else:
            audio_buffer.seek(0)
            b64_audio = base64.b64encode(audio_buffer.read()).decode("utf-8")
            response["audio_base64"] = b64_audio

        log.info("Handler completed successfully.")
        return response

    except Exception as e:
        log.error(f"Inference failed: {e}")

import torchaudio

def handler(job):
    """Runpod serverless handler (streaming + batch)."""
    job_input = job.get("input", {})
    stream = bool(job_input.get("stream", False))
    output_format = job_input.get("output_format")

    if stream and not output_format:
        output_format = "pcm_16"
    if not stream and not output_format:
        output_format = "mp3"

    if stream:
        log.info(f"[Handler] Streaming mode requested: format={output_format}")
        yield from handler_stream(job_input, output_format)
        return

    result = handler_batch(job, output_format)
    yield result

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})