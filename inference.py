import sys
import os
import torch
import numpy as np
import logging
import soundfile as sf
import tempfile
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
        log.info(f"Available voices: {', '.join(self.voice_presets.keys())}")

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
        try:
            # Load processor
            self.processor = VibeVoiceProcessor.from_pretrained(config.MODEL_PATH)

            # Determine dtype and attention implementation
            if self.device == "cuda":
                load_dtype = torch.bfloat16
                attn_impl = "flash_attention_2"

                # Try flash_attention_2 first, fallback to sdpa if it fails
                try:
                    log.info(f"Attempting to load model with flash_attention_2...")
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        config.MODEL_PATH,
                        torch_dtype=load_dtype,
                        device_map="cuda",
                        attn_implementation=attn_impl,
                    )
                    log.info(f"Model loaded successfully with flash_attention_2")
                except Exception as e:
                    log.warning(f"flash_attention_2 failed ({e}), falling back to sdpa")
                    attn_impl = "sdpa"
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        config.MODEL_PATH,
                        torch_dtype=load_dtype,
                        device_map="cuda",
                        attn_implementation=attn_impl,
                    )
                    log.info(f"Model loaded successfully with sdpa fallback")
            else:
                load_dtype = torch.float32
                attn_impl = "sdpa"
                log.info(f"Using device: {self.device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl}")

                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    config.MODEL_PATH,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl,
                )

            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=10)

            log.info(f"Model loaded successfully (sample rate: {config.DEFAULT_SAMPLE_RATE})")
            return self.model
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise

    def _smart_chunk_text(self, text: str, max_chars: int = None) -> list[str]:
        """Split text into chunks at natural boundaries (sentences, clauses)"""
        if max_chars is None:
            max_chars = self.max_chunk_chars

        if len(text) <= max_chars:
            return [text]

        chunks = []
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        clause_boundaries = [', ', '; ', ': ', ' - ', ' — ']

        def split_at_boundaries(text_segment: str, boundaries: list[str]) -> list[str]:
            parts = []
            current = ""
            i = 0
            while i < len(text_segment):
                current += text_segment[i]
                for boundary in boundaries:
                    if text_segment[i:i+len(boundary)] == boundary:
                        if len(current) > 0:
                            parts.append(current)
                            current = ""
                        i += len(boundary) - 1
                        break
                i += 1
            if current:
                parts.append(current)
            return parts

        segments = split_at_boundaries(text, sentence_endings)
        current_chunk = ""

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            if len(segment) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                clauses = split_at_boundaries(segment, clause_boundaries)
                for clause in clauses:
                    clause = clause.strip()
                    if not clause:
                        continue

                    if len(clause) > max_chars:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""

                        words = clause.split()
                        for word in words:
                            if len(current_chunk) + len(word) + 1 <= max_chars:
                                current_chunk += " " + word if current_chunk else word
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = word
                    else:
                        if len(current_chunk) + len(clause) + 1 <= max_chars:
                            current_chunk += " " + clause if current_chunk else clause
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = clause
            else:
                if len(current_chunk) + len(segment) + 1 <= max_chars:
                    current_chunk += " " + segment if current_chunk else segment
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = segment

        if current_chunk:
            chunks.append(current_chunk.strip())

        chunks = [c for c in chunks if c.strip()]

        if not chunks:
            return [text]

        log.info(f"Split text into {len(chunks)} chunks (max {max_chars} chars each)")
        return chunks

    def generate(
        self,
        text,
        speaker_name=None,
        cfg_scale=1.3,
        disable_prefill=False,
    ):
        """Generate audio from text using VibeVoice 7B with smart chunking"""
        if self.model is None:
            self.load_model()

        log.info(f"Generating audio for text ({len(text)} chars): {text[:50]}...")

        # Get voice path
        speaker_name = speaker_name or config.DEFAULT_SPEAKER
        voice_path = self.voice_mapper.get_voice_path(speaker_name)

        # Smart chunk text if it's too long
        chunks = self._smart_chunk_text(text, self.max_chunk_chars)

        if len(chunks) == 1:
            # Single chunk, generate directly
            temp_txt_path = None
            try:
                temp_txt_path = os.path.join(self.temp_dir, f"temp_{os.getpid()}.txt")
                with open(temp_txt_path, 'w') as f:
                    f.write(text)

                with torch.no_grad():
                    inputs = self.processor(
                        text=[text],
                        voice_samples=[voice_path],
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

                # Cleanup temp file
                if temp_txt_path and os.path.exists(temp_txt_path):
                    os.remove(temp_txt_path)
                
                return outputs.speech_outputs[0] if outputs.speech_outputs else None
            except Exception as e:
                if temp_txt_path and os.path.exists(temp_txt_path):
                    os.remove(temp_txt_path)
                raise
        else:
            # Multiple chunks, generate and concatenate
            log.info(f"Processing {len(chunks)} chunks...")
            audio_chunks = []

            for i, chunk_text in enumerate(chunks, 1):
                log.info(f"Generating chunk {i}/{len(chunks)} ({len(chunk_text)} chars)...")

                temp_txt_path = None
                try:
                    temp_txt_path = os.path.join(self.temp_dir, f"temp_{os.getpid()}_{i}.txt")
                    with open(temp_txt_path, 'w') as f:
                        f.write(chunk_text)

                    with torch.no_grad():
                        inputs = self.processor(
                            text=[chunk_text],
                            voice_samples=[voice_path],
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

                        # Cleanup temp file
                        if temp_txt_path and os.path.exists(temp_txt_path):
                            os.remove(temp_txt_path)

                        if chunk_wav.speech_outputs and chunk_wav.speech_outputs[0] is not None:
                            chunk_audio = chunk_wav.speech_outputs[0]
                            audio_chunks.append(chunk_audio)

                except Exception as e:
                    if temp_txt_path and os.path.exists(temp_txt_path):
                        os.remove(temp_txt_path)
                    raise

            # Concatenate all audio chunks
            if audio_chunks:
                concatenated_audio = torch.cat(audio_chunks, dim=-1)
                log.info(f"Concatenated {len(audio_chunks)} chunks → {concatenated_audio.shape[-1]} samples ({concatenated_audio.shape[-1]/config.DEFAULT_SAMPLE_RATE:.1f}s)")
                return concatenated_audio
            else:
                return None
