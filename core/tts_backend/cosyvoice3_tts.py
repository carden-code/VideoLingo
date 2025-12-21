"""
CosyVoice 3.0 TTS HTTP Client for VideoLingo
Connects to CosyVoice FastAPI server (https://github.com/FunAudioLLM/CosyVoice)

Supports multilingual TTS with zero-shot voice cloning via REST API.
Supported languages: Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian
"""

from pathlib import Path
import requests
import hashlib
import wave
import struct
from core.utils import *


def get_api_url():
    """Get CosyVoice API URL from config"""
    config = load_key("cosyvoice3")
    return config.get("api_url", "http://localhost:50000")


def check_api_health():
    """Check if CosyVoice API is available"""
    api_url = get_api_url()
    try:
        # CosyVoice doesn't have a dedicated health endpoint, try a simple request
        response = requests.get(f"{api_url}/", timeout=5)
        return response.status_code in (200, 404, 405)  # Server is running
    except requests.RequestException:
        return False


def get_language_code(language_name):
    """
    Map VideoLingo language names to CosyVoice language codes

    Supported languages (9 total):
    Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian
    """
    language_map = {
        # English
        'english': 'en',
        'en': 'en',

        # Chinese
        'chinese': 'zh',
        'zh': 'zh',
        'zh-cn': 'zh',

        # Japanese
        'japanese': 'ja',
        'ja': 'ja',

        # Korean
        'korean': 'ko',
        'ko': 'ko',

        # German
        'german': 'de',
        'de': 'de',

        # Spanish
        'spanish': 'es',
        'es': 'es',

        # French
        'french': 'fr',
        'fr': 'fr',

        # Italian
        'italian': 'it',
        'it': 'it',

        # Russian
        'russian': 'ru',
        'ru': 'ru',
    }

    normalized = language_name.lower().strip()

    if normalized in language_map:
        return language_map[normalized]

    # Default to English if not found
    rprint(f"[yellow]Language '{language_name}' not supported by CosyVoice, defaulting to 'en'[/yellow]")
    return 'en'


def find_optimal_reference(min_duration: float = 10.0, max_duration: float = 30.0, fallback_min: float = 5.0, enhance: bool = True) -> str:
    """
    Find optimal reference audio for CosyVoice voice cloning.

    Uses SNR analysis to select the cleanest audio segment.
    CosyVoice works best with reference audio between 10-30 seconds.

    Args:
        min_duration: Minimum duration (default 10.0s for better voice cloning)
        max_duration: Maximum duration (default 30.0s)
        fallback_min: Minimum acceptable duration if no ideal found (default 5.0s)
        enhance: Apply noise reduction and normalization

    Returns:
        Path to optimal reference audio, or None if none suitable found
    """
    try:
        from core.utils.reference_audio_utils import get_best_enhanced_reference
        return get_best_enhanced_reference(
            refers_dir=str(Path.cwd() / "output/audio/refers"),
            min_duration=min_duration,
            max_duration=max_duration,
            fallback_min=fallback_min,
            enhance=enhance
        )
    except ImportError:
        # Fallback to simple selection if new module not available
        from core.asr_backend.audio_preprocess import get_audio_duration

        refers_dir = Path.cwd() / "output/audio/refers"
        if not refers_dir.exists():
            return None

        ref_files = list(refers_dir.glob("*.wav"))
        if not ref_files:
            return None

        ref_files.sort(key=lambda x: int(x.stem) if x.stem.isdigit() else 999)

        # First pass: find optimal duration (10-30s)
        candidates = []
        for ref_file in ref_files:
            try:
                duration = get_audio_duration(str(ref_file))
                if duration >= fallback_min:
                    candidates.append((ref_file, duration))
                    if min_duration <= duration <= max_duration:
                        rprint(f"[green]✓ Found optimal reference: {ref_file.name} ({duration:.1f}s)[/green]")
                        return str(ref_file)
            except Exception:
                continue

        # Fallback: use longest available clip >= fallback_min
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0]
            rprint(f"[yellow]Using best available: {best[0].name} ({best[1]:.1f}s)[/yellow]")
            return str(best[0])

        return None


def save_pcm_to_wav(pcm_data: bytes, save_path: str, sample_rate: int = 22050):
    """
    Save raw PCM data to WAV file

    Args:
        pcm_data: Raw 16-bit PCM audio data
        save_path: Output WAV file path
        sample_rate: Audio sample rate (default 22050 for CosyVoice)
    """
    with wave.open(save_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)


def cosyvoice3_tts(text: str, save_path: str, reference_audio: str = None,
                   reference_text: str = None, mode: str = "zero_shot"):
    """
    Generate speech using CosyVoice 3.0 API

    Args:
        text: Text to synthesize
        save_path: Path to save the generated audio
        reference_audio: Path to reference audio for voice cloning
        reference_text: Transcript of reference audio (required for zero_shot mode)
        mode: "zero_shot" or "cross_lingual"
    """
    api_url = get_api_url()

    # Check API health
    if not check_api_health():
        raise ConnectionError(f"CosyVoice API not available at {api_url}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare request based on mode
    if mode == "zero_shot" and reference_audio and reference_text:
        # Zero-shot voice cloning with reference text
        endpoint = f"{api_url}/inference_zero_shot"

        with open(reference_audio, 'rb') as f:
            files = {'prompt_wav': (Path(reference_audio).name, f, 'audio/wav')}
            data = {
                'tts_text': text,
                'prompt_text': reference_text
            }

            rprint(f"[cyan]CosyVoice zero-shot: {text[:50]}...[/cyan]")
            response = requests.post(endpoint, files=files, data=data, timeout=120)

    elif reference_audio:
        # Cross-lingual mode (no reference text needed)
        endpoint = f"{api_url}/inference_cross_lingual"

        with open(reference_audio, 'rb') as f:
            files = {'prompt_wav': (Path(reference_audio).name, f, 'audio/wav')}
            data = {'tts_text': text}

            rprint(f"[cyan]CosyVoice cross-lingual: {text[:50]}...[/cyan]")
            response = requests.post(endpoint, files=files, data=data, timeout=120)

    else:
        raise ValueError("CosyVoice requires reference audio for voice cloning")

    if response.status_code != 200:
        raise Exception(f"CosyVoice API error ({response.status_code}): {response.text}")

    # Save audio - CosyVoice returns raw PCM data
    pcm_data = response.content

    # Get sample rate from config (default 22050 for CosyVoice)
    config = load_key("cosyvoice3")
    sample_rate = config.get("sample_rate", 22050)

    save_pcm_to_wav(pcm_data, str(save_path), sample_rate)

    rprint(f"[bold green]✓ Audio saved: {save_path}[/bold green]")
    return True


def cosyvoice3_tts_for_videolingo(text, save_as, number, task_df):
    """
    CosyVoice 3.0 TTS integration for VideoLingo pipeline

    Supports two modes:
    - zero_shot: Voice cloning with reference audio and text (best quality)
    - cross_lingual: Voice cloning without reference text (for different languages)

    Falls back to silent audio if CosyVoice fails.

    Args:
        text: Text to synthesize
        save_as: Output file path
        number: Current subtitle number
        task_df: DataFrame containing subtitle tasks
    """
    config = load_key("cosyvoice3")

    # Get configuration
    MODE = config.get("mode", "cross_lingual")  # zero_shot or cross_lingual

    # Find reference audio FIRST
    current_dir = Path.cwd()
    refers_dir = current_dir / "output/audio/refers"

    if not refers_dir.exists() or not list(refers_dir.glob("*.wav")):
        rprint("[yellow]Reference audio not found, extracting...[/yellow]")
        try:
            from core._9_refer_audio import extract_refer_audio_main
            extract_refer_audio_main()
        except Exception as e:
            rprint(f"[bold red]Failed to extract reference audio: {str(e)}[/bold red]")
            raise

    # Find optimal reference audio
    audio_prompt = find_optimal_reference()

    if not audio_prompt:
        raise Exception("No suitable reference audio found for CosyVoice")

    # Get reference text for the SELECTED audio segment (must match audio!)
    reference_text = None
    if MODE == "zero_shot":
        # Extract segment number from the selected audio file
        # Strip _enhanced suffix if present (e.g., "2_enhanced" -> "2")
        ref_stem = Path(audio_prompt).stem.replace("_enhanced", "")
        ref_number = int(ref_stem) if ref_stem.isdigit() else 1
        try:
            reference_text = task_df.loc[task_df['number'] == ref_number, 'origin'].values[0]
            rprint(f"[cyan]Zero-shot mode: using reference segment {ref_number}[/cyan]")
        except (KeyError, IndexError):
            rprint(f"[yellow]Could not get reference text for segment {ref_number}, switching to cross_lingual mode[/yellow]")
            MODE = "cross_lingual"

    # Generate TTS with fallback to silent audio
    try:
        success = cosyvoice3_tts(
            text=text,
            save_path=save_as,
            reference_audio=audio_prompt,
            reference_text=reference_text,
            mode=MODE
        )
        return success
    except Exception as e:
        # Log the error and create silent audio as fallback
        rprint(f"[bold yellow]⚠️ FALLBACK: CosyVoice failed for segment {number}: {str(e)}[/bold yellow]")
        rprint(f"[bold yellow]⚠️ FALLBACK: Text was: '{text}'[/bold yellow]")
        rprint(f"[bold cyan]🔄 FALLBACK: Creating silent audio for this segment[/bold cyan]")

        try:
            from pydub import AudioSegment
            # Create 1 second of silence as placeholder
            silence = AudioSegment.silent(duration=1000)
            silence.export(save_as, format="wav")
            rprint(f"[bold green]✓ FALLBACK: Silent audio created for segment {number}[/bold green]")
            return True
        except Exception as fallback_error:
            rprint(f"[bold red]❌ FALLBACK: Silent audio creation also failed: {str(fallback_error)}[/bold red]")
            raise Exception(f"CosyVoice failed for segment {number}: {str(e)}")
