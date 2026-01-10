#!/usr/bin/env python3
"""
Test script for VibeVoice batch mode (no streaming)

This script tests the handler directly to verify batch mode output format.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from handler import handler
import json

def test_batch_mode():
    """Test batch mode with a simple text input"""

    print("=" * 60)
    print("Testing VibeVoice Batch Mode")
    print("=" * 60)

    # Create test job
    test_job = {
        "input": {
            "text": "Hello, this is a test of the VibeVoice batch mode.",
            "speaker_name": "Alice",
            "cfg_scale": 1.3,
            "disable_prefill": False,
            "output_format": "mp3"
        }
    }

    print(f"\nTest Input:")
    print(json.dumps(test_job, indent=2))

    print(f"\nCalling handler...")

    # Call handler (it's a generator, so we need to consume it)
    results = list(handler(test_job))

    print(f"\nReceived {len(results)} result(s)")

    # Display results
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")

        # Create a display version without huge base64 strings
        display_result = result.copy()
        if 'audio_base64' in display_result:
            b64_len = len(display_result['audio_base64'])
            display_result['audio_base64'] = f"<base64 data: {b64_len} chars>"

        print(json.dumps(display_result, indent=2))

        # Validate result
        if 'error' in result:
            print(f"❌ ERROR: {result['error']}")
            return False

        if result.get('status') == 'success':
            print("✓ Status: success")

            if 'sample_rate' in result:
                print(f"✓ Sample rate: {result['sample_rate']} Hz")

            if 'duration_sec' in result:
                print(f"✓ Duration: {result['duration_sec']:.2f} seconds")

            if 'audio_url' in result:
                print(f"✓ Audio URL: {result['audio_url']}")
            elif 'audio_base64' in result:
                print(f"✓ Audio base64: {len(result['audio_base64'])} chars")
            else:
                print("❌ No audio output found!")
                return False
        else:
            print(f"❌ Unexpected status: {result.get('status')}")
            return False

    print("\n" + "=" * 60)
    print("✓ Batch mode test PASSED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = test_batch_mode()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
