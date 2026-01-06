import sys
import os
import gc
import torch
import numpy as np
import logging
import soundfile as sf
import tempfile
import random
from pathlib import Path

# Add VibeVoice to path (it's cloned at runtime in bootstrap.sh)
sys.path.insert(0, '/runpod-volume/vibevoice/vibevoice')

import config

# Import VibeVoice classes
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

log = logging.getLogger(__name__)

class VoiceMapper:
    """Maps speaker names to voice file paths"""
    
    def __init__(self):
        self.setup_voice_presets()

    def setup_voice_presets(self):
        """Setup voice presets by scanning voices directory."""
        voices_dir = Path(config.AUDIO_PROMPTS_DIR)
        
        if not voices_dir.exists():
            log.warning(f"Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            return
        
        # Scan for all WAV files in voices directory
        self.voice_presets = {}
        wav_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith('.wav') and os.path.isfile(os.path.join(voices_dir, f))]
        
        for wav_file in wav_files:
            name = os.path.splitext(wav_file)[0]
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path
        
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        log.info(f"Found {len(self.voice_presets)} voice files in {voices_dir}")
        if self.voice_presets:
            log.info(f"Available voices: {', '.join(self.voice_presets.keys())}")
        else:
            log.warning(f"No .wav files found in {voices_dir}")

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name"""
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]
        
        # Try partial matching (case insensitive)
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path

        # Default to first voice if no match found
        if not self.voice_presets:
            raise ValueError(
                f"No voice files found in {config.AUDIO_PROMPTS_DIR}. "
                f"Please upload at least one .wav file to the voices directory."
            )

        default_voice = list(self.voice_presets.values())[0]
        log.warning(f"No voice preset found for '{speaker_name}', using default voice: {default_voice}")
        return default_voice

class VibeVoiceInference:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_chunk_chars = config.MAX_CHUNK_CHARS
        self.voice_mapper = VoiceMapper()
        self.temp_dir = tempfile.mkdtemp()
        log.info(f"Created temporary directory: {self.temp_dir}")

    def __del__(self):
        """Cleanup temporary directory on instance deletion"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                log.debug(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                log.warning(f"Failed to cleanup temp directory {self.temp_dir}: {e}")

    def load_model(self):
        """Load VibeVoice 7B model"""
        if self.model is not None:
            return self.model

        log.info(f"Loading VibeVoice 7B model on {self.device}...")
        
        # Clear any existing GPU memory before loading
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            # Load processor
            self.processor = VibeVoiceProcessor.from_pretrained(config.MODEL_PATH)

            # Determine dtype and attention implementation
            if self.device == "cuda":
                load_dtype = torch.bfloat16
                # Use sdpa directly - flash_attention_2 uses too much memory during loading
                attn_impl = "sdpa"
                
                log.info(f"Loading model with sdpa (memory-efficient for 24GB GPUs)...")
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    config.MODEL_PATH,
                    torch_dtype=load_dtype,
                    device_map="cuda",
                    attn_implementation=attn_impl,
                    low_cpu_mem_usage=True,  # Reduce CPU→GPU transfer memory spikes
                )
                log.info(f"Model loaded successfully with sdpa")
            else:
                load_dtype = torch.float32
                attn_impl = "sdpa"
                log.info(f"Using device: {self.device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl}")

                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    config.MODEL_PATH,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl,
                    low_cpu_mem_usage=True,
                )

            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=10)

            log.info(f"Model loaded successfully (sample rate: {config.DEFAULT_SAMPLE_RATE})")
            return self.model
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            # Clean up on failure
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            raise

    def _smart_chunk_text(self, text: str, max_chars: int = None) -> list[str]:
        """Split text into chunks only at sentence boundaries to preserve prosody."""
        if max_chars is None:
            max_chars = self.max_chunk_chars

        if len(text) <= max_chars:
            return [text]

        chunks = []
        # ONLY split at sentence endings. Splitting at commas (clauses) destroys VibeVoice prosody.
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']

        def split_keep_boundaries(text_segment: str, boundaries: list[str]) -> list[str]:
            parts = []
            current = ""
            i = 0
            while i < len(text_segment):
                match_found = False
                for boundary in boundaries:
                    if text_segment[i:i+len(boundary)] == boundary:
                        current += boundary
                        parts.append(current)
                        current = ""
                        i += len(boundary)
                        match_found = True
                        break
                
                if not match_found:
                    current += text_segment[i]
                    i += 1
            
            if current:
                parts.append(current)
            return parts

        # Split into sentences
        segments = split_keep_boundaries(text, sentence_endings)
        current_chunk = ""

        for segment in segments:
            if len(segment) > max_chars:
                # Flush existing buffer
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Sentence itself is too long. This is rare (>800 chars).
                # Fall back to word-level split as a last resort.
                words = segment.split()
                for word in words:
                    prefix = " " if current_chunk else ""
                    if len(current_chunk) + len(word) + 1 <= max_chars:
                        current_chunk += prefix + word
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = word
            else:
                # Segment fits. Try to group sentences up to max_chars.
                prefix = "" if not current_chunk or current_chunk.endswith(" ") else " "
                if len(current_chunk) + len(segment) + len(prefix) <= max_chars:
                    current_chunk += prefix + segment
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = segment

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Cleanup
        chunks = [c.strip() for c in chunks if c.strip()]

        if not chunks:
            return [text]

        log.info(f"Split text into {len(chunks)} chunks (max {max_chars} chars each, sentence-level only)")
        return chunks

    def _set_seed(self, seed: int) -> None:
        """Helper to set seed across all libraries for consistency"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate(
        self,
        text,
        speaker_name=None,
        cfg_scale=1.3,
        disable_prefill=False,
        max_chunk_chars=None,
    ):
        """Generate audio from text using VibeVoice 7B with smart chunking"""
        if self.model is None:
            self.load_model()

        log.info(f"Generating audio for text ({len(text)} chars): {text[:50]}...")

        session_seed = int.from_bytes(os.urandom(4), "little")
        log.info(f"Using session seed: {session_seed}")

        # Get voice path
        speaker_name = speaker_name or config.DEFAULT_SPEAKER
        voice_path = self.voice_mapper.get_voice_path(speaker_name)

        # Smart chunk text if it's too long
        chunk_limit = max_chunk_chars if max_chunk_chars is not None else self.max_chunk_chars
        chunks = self._smart_chunk_text(text, chunk_limit)

        if len(chunks) == 1:
            # Single chunk, generate directly
            formatted_text = f"Speaker 1: {chunks[0]}"
            temp_txt_path = None
            try:
                temp_txt_path = os.path.join(self.temp_dir, f"temp_{os.getpid()}.txt")
                with open(temp_txt_path, 'w') as f:
                    f.write(formatted_text)

                with torch.no_grad():
                    self._set_seed(session_seed)
                    # voice_samples must be a list of lists: [[path1, path2, ...]]
                    inputs = self.processor(
                        text=[formatted_text],
                        voice_samples=[[voice_path]],  # List of lists!
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )

                    # Move tensors to target device
                    for k, v in inputs.items():
                        if torch.is_tensor(v):
                            inputs[k] = v.to(self.device)

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={'do_sample': False},
                        verbose=False,
                        is_prefill=not disable_prefill,
                    )
                
                if temp_txt_path and os.path.exists(temp_txt_path):
                    os.remove(temp_txt_path)

                if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                    return outputs.speech_outputs[0]
                return None

            except Exception:
                if temp_txt_path and os.path.exists(temp_txt_path):
                    os.remove(temp_txt_path)
                raise

        # Multiple chunks, generate and concatenate
        log.info(f"Generating {len(chunks)} chunks (max {chunk_limit} chars each)...")
        combined_wav = []
        
        for i, chunk_text in enumerate(chunks, 1):
            log.info(f"Generating chunk {i}/{len(chunks)} ({len(chunk_text)} chars)...")
            
            formatted_chunk = f"Speaker 1: {chunk_text}"
            temp_txt_path = None
            
            try:
                temp_txt_path = os.path.join(self.temp_dir, f"temp_{os.getpid()}_{i}.txt")
                with open(temp_txt_path, 'w') as f:
                    f.write(formatted_chunk)
                    
                with torch.no_grad():
                    self._set_seed(session_seed)
                    inputs = self.processor(
                        text=[formatted_chunk],
                        voice_samples=[[voice_path]],
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )
                    
                    for k, v in inputs.items():
                        if torch.is_tensor(v):
                            inputs[k] = v.to(self.device)
                            
                    chunk_wav = self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={'do_sample': False},
                        verbose=False,
                        is_prefill=not disable_prefill,
                    )
                
                if temp_txt_path and os.path.exists(temp_txt_path):
                    os.remove(temp_txt_path)
                
                if chunk_wav.speech_outputs and chunk_wav.speech_outputs[0] is not None:
                    wav_data = chunk_wav.speech_outputs[0]
                    if torch.is_tensor(wav_data):
                        wav_data = wav_data.cpu().numpy().squeeze()
                    combined_wav.append(wav_data)
                    
                    # Add silence between chunks
                    if i < len(chunks):
                        silence = np.zeros(int(config.DEFAULT_SAMPLE_RATE * (config.CHUNK_SILENCE_MS / 1000)))
                        combined_wav.append(silence)
                        
            except Exception as e:
                log.error(f"Error generating chunk {i}: {e}")
                if temp_txt_path and os.path.exists(temp_txt_path):
                    os.remove(temp_txt_path)
                # Continue with other chunks or fail? For now fail
                raise e

        if not combined_wav:
            return None
            
        return np.concatenate(combined_wav)

    def generate_stream(
        self,
        text,
        speaker_name=None,
        cfg_scale=1.3,
        disable_prefill=False,
        max_chunk_chars=None,
    ):
        """Generate audio in chunks and yield each chunk as it completes."""
        if self.model is None:
            self.load_model()

        session_seed = int.from_bytes(os.urandom(4), "little")
        log.info(f"Using session seed: {session_seed}")

        # Get voice path
        speaker_name = speaker_name or config.DEFAULT_SPEAKER
        voice_path = self.voice_mapper.get_voice_path(speaker_name)

        # Smart chunk text if it's too long
        chunk_limit = max_chunk_chars if max_chunk_chars is not None else self.max_chunk_chars
        chunks = self._smart_chunk_text(text, chunk_limit)

        log.info(f"Streaming {len(chunks)} chunks (max_chunk_chars={chunk_limit})...")

        for i, chunk_text in enumerate(chunks, 1):
            log.info(f"Generating chunk {i}/{len(chunks)} ({len(chunk_text)} chars)...")

            formatted_chunk = f"Speaker 1: {chunk_text}"
            temp_txt_path = None

            try:
                # VibeVoice might rely on file existence in some internal path checks?
                # Keeping file writing just in case, matching original behavior.
                temp_txt_path = os.path.join(self.temp_dir, f"temp_{os.getpid()}_{i}.txt")
                with open(temp_txt_path, 'w') as f:
                    f.write(formatted_chunk)

                with torch.no_grad():
                    self._set_seed(session_seed)
                    inputs = self.processor(
                        text=[formatted_chunk],
                        voice_samples=[[voice_path]],
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )

                    # Move tensors to target device
                    for k, v in inputs.items():
                        if torch.is_tensor(v):
                            inputs[k] = v.to(self.device)

                    chunk_wav = self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={'do_sample': False},
                        verbose=False,
                        is_prefill=not disable_prefill,
                    )

                if temp_txt_path and os.path.exists(temp_txt_path):
                    os.remove(temp_txt_path)

                if chunk_wav.speech_outputs and chunk_wav.speech_outputs[0] is not None:
                    yield chunk_wav.speech_outputs[0]

            except Exception:
                if temp_txt_path and os.path.exists(temp_txt_path):
                    os.remove(temp_txt_path)
                raise
