"""
Test script for Chatterbox TTS HTTP API integration with VideoLingo
Run this to verify the integration is working correctly

Requires:
- Chatterbox TTS API running (docker compose -f docker/docker-compose.gpu.yml up -d)
- API endpoint configured in config.yaml (default: http://localhost:4123)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_api_health():
    """Test if Chatterbox API is reachable"""
    print("\n" + "="*60)
    print("Test 0: API Health Check")
    print("="*60)

    from core.tts_backend.chatterbox_tts import check_api_health, get_api_url

    api_url = get_api_url()
    print(f"\nAPI URL: {api_url}")

    if check_api_health():
        print("✓ API is healthy and reachable")
        return True
    else:
        print("✗ API is not reachable")
        print(f"  Make sure Chatterbox API is running at {api_url}")
        print("  Start with: docker compose -f docker/docker-compose.gpu.yml up -d")
        return False


def test_basic_tts():
    """Test basic TTS without voice cloning"""
    print("\n" + "="*60)
    print("Test 1: Basic TTS (No Voice Cloning)")
    print("="*60)

    from core.tts_backend.chatterbox_tts import chatterbox_tts

    # Test in multiple languages
    test_cases = [
        ("Hello, this is a test of Chatterbox TTS.", "en", "test_en.wav"),
        ("Bonjour, ceci est un test.", "fr", "test_fr.wav"),
        ("Hola, esta es una prueba.", "es", "test_es.wav"),
        ("これはテストです。", "ja", "test_ja.wav"),
        ("这是一个测试。", "zh", "test_zh.wav"),
    ]

    output_dir = Path("output/test_chatterbox")
    output_dir.mkdir(parents=True, exist_ok=True)

    for text, lang, filename in test_cases:
        try:
            save_path = output_dir / filename
            print(f"\nGenerating: {text} [{lang}]")
            chatterbox_tts(
                text=text,
                save_path=str(save_path),
                language_id=lang
            )
            print(f"✓ Success: {save_path}")
        except Exception as e:
            print(f"✗ Error: {str(e)}")

def test_voice_cloning():
    """Test voice cloning with reference audio"""
    print("\n" + "="*60)
    print("Test 2: Voice Cloning (with Reference Audio)")
    print("="*60)

    from core.tts_backend.chatterbox_tts import chatterbox_tts

    # Check if reference audio exists
    ref_audio = Path("output/audio/refers/1.wav")
    if not ref_audio.exists():
        print(f"⚠ Warning: Reference audio not found at {ref_audio}")
        print("  Voice cloning test skipped. Run video processing first to generate reference audio.")
        return

    output_dir = Path("output/test_chatterbox")
    output_dir.mkdir(parents=True, exist_ok=True)

    text = "This is a test of voice cloning with Chatterbox TTS."
    save_path = output_dir / "test_voice_clone.wav"

    try:
        print(f"\nGenerating with voice cloning: {text}")
        print(f"Using reference: {ref_audio}")
        chatterbox_tts(
            text=text,
            save_path=str(save_path),
            language_id="en",
            audio_prompt=str(ref_audio),
            exaggeration=0.5,
            cfg_weight=0.4
        )
        print(f"✓ Success: {save_path}")
    except Exception as e:
        print(f"✗ Error: {str(e)}")

def test_videolingo_integration():
    """Test VideoLingo integration"""
    print("\n" + "="*60)
    print("Test 3: VideoLingo Integration Test")
    print("="*60)

    try:
        from core.utils import load_key
        from core.tts_backend.chatterbox_tts import get_language_code

        # Test language mapping
        print("\nTesting language code mapping:")
        test_languages = [
            "简体中文", "English", "日语", "Русский", "Français"
        ]

        for lang in test_languages:
            code = get_language_code(lang)
            print(f"  {lang} → {code}")

        # Test configuration loading
        print("\nTesting configuration loading:")
        try:
            config = load_key("chatterbox_tts")
            print(f"  API URL: {config.get('api_url', 'http://localhost:4123')}")
            print(f"  Voice clone mode: {config.get('voice_clone_mode', 1)}")
            print(f"  Exaggeration: {config.get('exaggeration', 0.5)}")
            print(f"  CFG weight: {config.get('cfg_weight', 0.4)}")
            print("  ✓ Configuration loaded successfully")
        except Exception as e:
            print(f"  ✗ Configuration error: {str(e)}")

        print("\n✓ VideoLingo integration test completed")

    except Exception as e:
        print(f"✗ Integration test failed: {str(e)}")

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("Chatterbox TTS HTTP API - Usage Instructions")
    print("="*60)
    print("""
1. START CHATTERBOX API:
   # Clone the API repo
   git clone https://github.com/travisvn/chatterbox-tts-api
   cd chatterbox-tts-api

   # Start with GPU support
   docker compose -f docker/docker-compose.gpu.yml up -d

   # Or CPU only
   docker compose -f docker/docker-compose.yml up -d

2. CONFIGURATION (in config.yaml):

   tts_method: 'chatterbox_tts'

   chatterbox_tts:
     api_url: 'http://localhost:4123'  # Chatterbox API endpoint
     voice_clone_mode: 1  # 1=No cloning, 2=Single ref, 3=Per-segment ref
     exaggeration: 0.5    # Emotionality (0.25-2.0)
     cfg_weight: 0.4      # Voice cloning strength (0.0-1.0)

3. MODES:
   - Mode 1: Basic TTS without voice cloning (fastest)
   - Mode 2: Clone voice using optimal reference audio (5-10s)
   - Mode 3: Clone voice per segment (output/audio/refers/{N}.wav)

4. SUPPORTED LANGUAGES (22 total):
   ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr
   Plus zh (Chinese) - may need to be enabled in API

5. VOICE CLONING:
   - Language is assigned when uploading voice sample
   - Reference audio: 5-10 seconds optimal, 3s minimum
   - Reference quality affects output quality
""")

if __name__ == "__main__":
    print("\n🎙️  Chatterbox TTS HTTP API Test Suite")
    print_usage_instructions()

    # Run tests
    try:
        # First check API health
        if not test_api_health():
            print("\n⚠️  Skipping TTS tests - API not available")
            print("  Start the API first, then run this test again.")
            sys.exit(1)

        test_videolingo_integration()
        test_basic_tts()
        test_voice_cloning()

        print("\n" + "="*60)
        print("All tests completed! Check output/test_chatterbox/ for audio files.")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n✗ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
