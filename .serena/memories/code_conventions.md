# Code Conventions and Style

## Python Style

### General Guidelines
- **Python Version**: 3.12
- **Type Hints**: Used for function parameters and return types (`str`, `float`, `bool`, `list[str]`)
- **Docstrings**: Google-style docstrings for classes and functions
- **Line Length**: No strict limit observed, but generally readable
- **Indentation**: 4 spaces (standard Python)

### Naming Conventions
- **Classes**: PascalCase (e.g., `VoiceMapper`, `VibeVoiceInference`)
- **Functions/Methods**: snake_case (e.g., `load_model`, `get_voice_path`, `cleanup_old_files`)
- **Variables**: snake_case (e.g., `sample_rate`, `temp_dir`, `voice_path`)
- **Constants**: UPPER_SNAKE_CASE in config.py (e.g., `MAX_TEXT_LENGTH`, `DEFAULT_SPEAKER`)
- **Private Methods**: Prefix with underscore (e.g., `_smart_chunk_text`)

### Import Organization
```python
# Standard library imports
import sys
import os
import logging

# Third-party imports
import torch
import soundfile as sf

# Local imports
import config
from vibevoice.modular.modeling_vibevoice_inference import ...
```

## Error Handling

### Pattern
- Use try/except blocks for operations that may fail
- Log errors with `log.error()`, warnings with `log.warning()`
- Provide informative error messages
- Graceful fallbacks where possible

### Example
```python
try:
    # Attempt primary method
    result = primary_operation()
except Exception as e:
    log.warning(f"Primary failed ({e}), falling back to secondary")
    result = fallback_operation()
```

## Logging

### Levels
- `log.info()` - Important operations and milestones
- `log.warning()` - Recoverable issues, fallbacks
- `log.error()` - Errors that prevent operations
- `log.debug()` - Detailed debugging information

### Pattern
```python
log = logging.getLogger(__name__)
log.info(f"Loading model on {device}...")
log.warning(f"No voice preset found for '{name}', using default")
log.error(f"Failed to load model: {e}")
```

## Function Design

### Docstrings
- Include purpose, parameters, and return value
- Use triple quotes
- Example:
```python
def handler(job):
    """Runpod serverless handler

    Expected input format:
    {
        "text": str (required) - Text to synthesize
        "speaker_name": str (optional) - Speaker name
    }

    Returns:
    {
        "status": "success",
        "sample_rate": int,
        "audio_url": str
    }
    """
```

### Parameter Validation
- Validate inputs early in functions
- Return informative error dictionaries
- Example:
```python
if not text or not text.strip():
    return {"error": "Missing or empty 'text' parameter"}
```

## Configuration Management

### Pattern
- All configuration in `config.py`
- Use environment variables with defaults
- Document each config variable

### Example
```python
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", "2000"))
DEFAULT_SPEAKER = os.environ.get("DEFAULT_SPEAKER", "Alice")
```

## Resource Management

### Temporary Files
- Create temp files in dedicated temp directory
- Clean up in try/finally or except blocks
- Use `__del__` for instance cleanup

### Example
```python
temp_txt_path = None
try:
    temp_txt_path = os.path.join(self.temp_dir, f"temp_{uuid.uuid4()}.txt")
    # Use file
finally:
    if temp_txt_path and os.path.exists(temp_txt_path):
        os.remove(temp_txt_path)
```

## Comments

### When to Comment
- Complex logic that isn't self-evident
- Fallback strategies
- Important warnings or gotchas
- NOT for obvious operations

### Style
```python
# Move tensors to target device
for k, v in inputs.items():
    if torch.is_tensor(v):
        inputs[k] = v.to(self.device)
```

## Shell Scripts (bootstrap.sh)

### Style
- Use `set -e` to exit on errors
- Echo progress messages
- Export environment variables explicitly
- Use meaningful variable names in UPPER_CASE

### Example
```bash
#!/bin/bash
set -e

echo "=== Starting Setup ==="
VENV_PATH="/runpod-volume/vibevoice/venv"

if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
    echo "Token configured"
fi
```

## Important Patterns

### 1. Lazy Loading
```python
def load_model(self):
    if self.model is not None:
        return self.model
    # Load model...
```

### 2. Graceful Degradation
```python
try:
    # Try optimal approach
except Exception as e:
    log.warning(f"Optimal failed, using fallback: {e}")
    # Use fallback
```

### 3. Device-Aware Code
```python
if self.device == "cuda":
    # CUDA-specific settings
else:
    # CPU fallback
```

### 4. Progress Logging
```python
log.info(f"Processing chunk {i}/{total} ({len(text)} chars)...")
```

## Anti-Patterns to Avoid

❌ Hardcoded paths (use config variables)
❌ Silent failures (always log errors)
❌ Broad exception catching without logging
❌ Missing cleanup for temporary resources
❌ Assumptions about file existence (check first)
