#!/usr/bin/env python3
"""
Test script for VibeVoice streaming mode

This script tests the handler directly to verify streaming output format.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from handler import handler
import json

def test_streaming_mode():
    """Test streaming mode with a simple text input"""

    print("=" * 60)
    print("Testing VibeVoice Streaming Mode")
    print("=" * 60)

    # Create test job with stream=true
    test_job = {
        "input": {
            "text": "Hello, this is a test of the VibeVoice streaming mode. "
                   "It should produce multiple chunks of audio data. "
                   "Each chunk will be encoded with LinaCodec for quality.",
            "speaker_name": "Alice",
            "cfg_scale": 1.3,
            "disable_prefill": False,
            "stream": True,
            "output_format": "pcm_16"
        }
    }

    print(f"\nTest Input:")
    print(json.dumps(test_job, indent=2))

    print(f"\nCalling handler (streaming)...")
    print("-" * 60)

    # Call handler (it's a generator)
    chunk_count = 0
    completion_received = False

    for result in handler(test_job):
        # Create a display version without huge base64 strings
        display_result = result.copy()
        if 'audio_chunk' in display_result:
            b64_len = len(display_result['audio_chunk'])
            display_result['audio_chunk'] = f"<base64 data: {b64_len} chars>"
        if 'audio_url' in display_result:
            display_result['audio_url'] = f"<S3 URL: {display_result['audio_url'][:50]}...>"

        print(f"\n--- Chunk {chunk_count + 1} ---")
        print(json.dumps(display_result, indent=2))

        # Validate result
        if result.get('status') == 'streaming':
            chunk_count += 1

            # Check required fields
            if 'chunk' not in result:
                print("❌ Missing 'chunk' field in streaming response")
                return False

            if result['chunk'] != chunk_count:
                print(f"❌ Chunk number mismatch: expected {chunk_count}, got {result['chunk']}")
                return False

            if 'format' not in result:
                print("❌ Missing 'format' field in streaming response")
                return False

            if 'sample_rate' not in result:
                print("❌ Missing 'sample_rate' field in streaming response")
                return False

            # Check for audio data
            has_audio = 'audio_chunk' in result or 'audio_url' in result
            if not has_audio:
                print("❌ No audio data in streaming chunk (missing audio_chunk or audio_url)")
                return False

            print(f"✓ Chunk {chunk_count}: format={result['format']}, "
                  f"sample_rate={result['sample_rate']} Hz, "
                  f"has_audio={'audio_url' if 'audio_url' in result else 'audio_chunk'}")

        elif result.get('status') == 'complete':
            completion_received = True
            print("✓ Stream completion signal received")

            if 'total_chunks' in result:
                if result['total_chunks'] != chunk_count:
                    print(f"⚠ Total chunks mismatch: received {chunk_count}, "
                          f"completion says {result['total_chunks']}")

        elif result.get('status') == 'error':
            print(f"❌ ERROR: {result.get('error', 'Unknown error')}")
            return False

        else:
            print(f"⚠ Unexpected status: {result.get('status')}")

    print("\n" + "-" * 60)

    # Validate streaming results
    if chunk_count == 0:
        print("❌ No streaming chunks received!")
        return False

    if not completion_received:
        print("⚠ Warning: No completion signal received")

    print(f"\n✓ Received {chunk_count} streaming chunk(s)")
    print("=" * 60)
    print("✓ Streaming mode test PASSED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = test_streaming_mode()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
