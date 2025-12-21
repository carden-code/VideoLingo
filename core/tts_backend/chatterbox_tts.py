"""
Chatterbox TTS HTTP Client for VideoLingo
Connects to Chatterbox TTS API (https://github.com/travisvn/chatterbox-tts-api)

Supports multilingual TTS with zero-shot voice cloning via REST API.
"""

from pathlib import Path
import requests
import hashlib
from core.utils import *

# Cache for uploaded voice IDs (voice_path -> voice_name)
_uploaded_voices = {}


def get_api_url():
    """Get Chatterbox API URL from config"""
    config = load_key("chatterbox_tts")
    return config.get("api_url", "http://localhost:4123")


def check_api_health():
    """Check if Chatterbox API is available"""
    api_url = get_api_url()
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def get_language_code(language_name):
    """
    Map VideoLingo language names to Chatterbox language IDs

    Supported languages (22 total):
    ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr
    Plus zh (Chinese) - may need to be enabled in API
    """
    language_map = {
        # English
        'english': 'en',
        '英文': 'en',
        'en': 'en',

        # Chinese
        'chinese': 'zh',
        '中文': 'zh',
        '简体中文': 'zh',
        'zh': 'zh',
        'zh-cn': 'zh',

        # Spanish
        'spanish': 'es',
        '西班牙语': 'es',
        'es': 'es',

        # French
        'french': 'fr',
        '法语': 'fr',
        'fr': 'fr',

        # German
        'german': 'de',
        '德语': 'de',
        'de': 'de',

        # Italian
        'italian': 'it',
        '意大利语': 'it',
        'it': 'it',

        # Japanese
        'japanese': 'ja',
        '日语': 'ja',
        'ja': 'ja',

        # Korean
        'korean': 'ko',
        '韩语': 'ko',
        'ko': 'ko',

        # Russian
        'russian': 'ru',
        'русский': 'ru',
        'Русский': 'ru',
        '俄语': 'ru',
        'ru': 'ru',

        # Portuguese
        'portuguese': 'pt',
        '葡萄牙语': 'pt',
        'pt': 'pt',

        # Polish
        'polish': 'pl',
        '波兰语': 'pl',
        'pl': 'pl',

        # Turkish
        'turkish': 'tr',
        '土耳其语': 'tr',
        'tr': 'tr',

        # Dutch
        'dutch': 'nl',
        '荷兰语': 'nl',
        'nl': 'nl',

        # Arabic
        'arabic': 'ar',
        '阿拉伯语': 'ar',
        'ar': 'ar',

        # Hindi
        'hindi': 'hi',
        '印地语': 'hi',
        'hi': 'hi',

        # Hebrew
        'hebrew': 'he',
        'עברית': 'he',
        '希伯来语': 'he',
        'he': 'he',

        # Ukrainian - not in API list, fallback to Russian
        'ukrainian': 'ru',
        'українська': 'ru',
        '乌克兰语': 'ru',
        'uk': 'ru',

        # Greek
        'greek': 'el',
        'ελληνικά': 'el',
        '希腊语': 'el',
        'el': 'el',

        # Finnish
        'finnish': 'fi',
        '芬兰语': 'fi',
        'fi': 'fi',

        # Swedish
        'swedish': 'sv',
        '瑞典语': 'sv',
        'sv': 'sv',

        # Norwegian
        'norwegian': 'no',
        '挪威语': 'no',
        'no': 'no',

        # Danish
        'danish': 'da',
        '丹麦语': 'da',
        'da': 'da',

        # Malay
        'malay': 'ms',
        '马来语': 'ms',
        'ms': 'ms',

        # Swahili
        'swahili': 'sw',
        '斯瓦希里语': 'sw',
        'sw': 'sw',
    }

    # Normalize input
    normalized = language_name.lower().strip()

    if normalized in language_map:
        return language_map[normalized]

    # Default to English if not found
    rprint(f"[yellow]Language '{language_name}' not found in map, defaulting to 'en'[/yellow]")
    return 'en'


def upload_voice(voice_path: str, language_id: str) -> str:
    """
    Upload voice sample to Chatterbox API for voice cloning.

    Args:
        voice_path: Path to voice sample audio file
        language_id: ISO 639-1 language code (e.g., 'ru', 'en')

    Returns:
        voice_name for use in TTS requests
    """
    global _uploaded_voices

    api_url = get_api_url()

    # Generate unique voice name based on full file hash
    with open(voice_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()[:12]
    voice_name = f"vl_{language_id}_{file_hash}"

    # Check cache (includes hash so changed files get re-uploaded)
    cache_key = f"{voice_path}:{language_id}:{file_hash}"
    if cache_key in _uploaded_voices:
        return _uploaded_voices[cache_key]

    # Check if voice already exists
    try:
        response = requests.get(f"{api_url}/voices", timeout=10)
        if response.status_code == 200:
            voices = response.json()
            if isinstance(voices, list):
                for v in voices:
                    if v.get('name') == voice_name:
                        rprint(f"[cyan]Voice '{voice_name}' already uploaded[/cyan]")
                        _uploaded_voices[cache_key] = voice_name
                        return voice_name
    except requests.RequestException:
        pass

    # Upload new voice
    rprint(f"[cyan]Uploading voice '{voice_name}' with language '{language_id}'...[/cyan]")

    with open(voice_path, 'rb') as f:
        files = {'voice_file': (Path(voice_path).name, f, 'audio/wav')}
        data = {'voice_name': voice_name, 'language': language_id}

        try:
            response = requests.post(
                f"{api_url}/voices",
                files=files,
                data=data,
                timeout=30
            )

            if response.status_code in (200, 201):
                rprint(f"[bold green]✓ Voice uploaded: {voice_name}[/bold green]")
                _uploaded_voices[cache_key] = voice_name
                return voice_name
            else:
                rprint(f"[yellow]Voice upload failed ({response.status_code}): {response.text}[/yellow]")
                return None

        except requests.RequestException as e:
            rprint(f"[yellow]Voice upload error: {e}[/yellow]")
            return None


def find_optimal_reference(min_duration: float = 10.0, max_duration: float = 30.0, fallback_min: float = 5.0, enhance: bool = True) -> str:
    """
    Find optimal reference audio for Chatterbox voice cloning.

    Uses SNR analysis to select the cleanest audio segment.
    Chatterbox works best with reference audio between 10-30 seconds.

    Args:
        min_duration: Ideal minimum duration (default 10.0s for better cloning)
        max_duration: Ideal maximum duration (default 30.0s)
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


def chatterbox_tts(text, save_path, language_id='en', audio_prompt=None, exaggeration=0.5, cfg_weight=0.4):
    """
    Generate speech using Chatterbox TTS API

    Args:
        text: Text to synthesize
        save_path: Path to save the generated audio
        language_id: Language code (e.g., 'en', 'zh', 'ja')
        audio_prompt: Optional path to reference audio for voice cloning
        exaggeration: Control emotionality (0.25-2.0, default 0.5)
        cfg_weight: Influence of audio prompt (0.0-1.0, default 0.4)
    """
    api_url = get_api_url()

    # Check API health
    if not check_api_health():
        raise ConnectionError(f"Chatterbox API not available at {api_url}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine voice to use
    voice_name = None

    if audio_prompt and Path(audio_prompt).exists():
        # Upload voice with language and get voice_name
        voice_name = upload_voice(audio_prompt, language_id)
        if voice_name:
            rprint(f"[cyan]Using voice cloning: {voice_name}[/cyan]")

    if voice_name:
        # Use uploaded voice (language inherited from voice metadata)
        payload = {
            'input': text,
            'voice': voice_name,
            'exaggeration': exaggeration,
            'cfg_weight': cfg_weight
        }

        response = requests.post(
            f"{api_url}/v1/audio/speech",
            json=payload,
            timeout=120
        )
    else:
        # Use with voice file directly (single request with file upload)
        if audio_prompt and Path(audio_prompt).exists():
            rprint(f"[cyan]Using voice cloning with direct upload (language: {language_id})[/cyan]")
            with open(audio_prompt, 'rb') as f:
                files = {'voice_file': (Path(audio_prompt).name, f, 'audio/wav')}
                data = {
                    'input': text,
                    'language': language_id,
                    'exaggeration': str(exaggeration),
                    'cfg_weight': str(cfg_weight)
                }
                response = requests.post(
                    f"{api_url}/v1/audio/speech/upload",
                    files=files,
                    data=data,
                    timeout=120
                )
        else:
            # Basic TTS without voice cloning
            rprint(f"[cyan]Generating audio (language: {language_id}, no voice cloning)[/cyan]")
            payload = {
                'input': text,
                'language': language_id,
                'exaggeration': exaggeration
            }
            response = requests.post(
                f"{api_url}/v1/audio/speech",
                json=payload,
                timeout=120
            )

    if response.status_code != 200:
        raise Exception(f"TTS API error ({response.status_code}): {response.text}")

    # Save audio file
    with open(save_path, 'wb') as f:
        f.write(response.content)

    rprint(f"[bold green]✓ Audio saved: {save_path}[/bold green]")
    return True


def chatterbox_tts_for_videolingo(text, save_as, number, task_df):
    """
    Chatterbox TTS integration for VideoLingo pipeline

    Supports three modes:
    - Mode 1: Basic TTS without voice cloning
    - Mode 2: Voice cloning with single reference audio
    - Mode 3: Voice cloning with per-segment reference audio

    Falls back to silent audio if Chatterbox fails (e.g., on very short text).

    Args:
        text: Text to synthesize
        save_as: Output file path
        number: Current subtitle number
        task_df: DataFrame containing subtitle tasks
    """
    chatterbox_config = load_key("chatterbox_tts")

    # Get configuration
    VOICE_CLONE_MODE = chatterbox_config.get("voice_clone_mode", 1)
    EXAGGERATION = chatterbox_config.get("exaggeration", 0.5)
    CFG_WEIGHT = chatterbox_config.get("cfg_weight", 0.4)

    # Get target language
    TARGET_LANGUAGE = load_key("target_language")
    language_id = get_language_code(TARGET_LANGUAGE)

    # Determine reference audio based on mode
    audio_prompt = None
    current_dir = Path.cwd()

    if VOICE_CLONE_MODE == 2:
        # Use optimal reference audio for all segments (5-10s ideal, 3s minimum)
        refers_dir = current_dir / "output/audio/refers"
        if not refers_dir.exists() or not list(refers_dir.glob("*.wav")):
            rprint("[yellow]Reference audio not found, extracting...[/yellow]")
            try:
                from core._9_refer_audio import extract_refer_audio_main
                extract_refer_audio_main()
            except Exception as e:
                rprint(f"[bold red]Failed to extract reference audio: {str(e)}[/bold red]")
                rprint("[yellow]Continuing without voice cloning...[/yellow]")

        # Find optimal reference (5-10s ideal, 3s minimum fallback)
        audio_prompt = find_optimal_reference()

    elif VOICE_CLONE_MODE == 3:
        # Use per-segment reference audio
        ref_path = current_dir / f"output/audio/refers/{number}.wav"
        if ref_path.exists():
            audio_prompt = str(ref_path)
        else:
            rprint(f"[yellow]Segment reference audio not found at {ref_path}, extracting...[/yellow]")
            try:
                from core._9_refer_audio import extract_refer_audio_main
                extract_refer_audio_main()
                if ref_path.exists():
                    audio_prompt = str(ref_path)
                else:
                    # Fallback to optimal reference selection
                    rprint("[yellow]Falling back to optimal reference selection...[/yellow]")
                    audio_prompt = find_optimal_reference()
            except Exception as e:
                rprint(f"[bold red]Failed to extract reference audio: {str(e)}[/bold red]")
                rprint("[yellow]Continuing without voice cloning...[/yellow]")

    # Generate TTS with fallback to silent audio
    try:
        success = chatterbox_tts(
            text=text,
            save_path=save_as,
            language_id=language_id,
            audio_prompt=audio_prompt,
            exaggeration=EXAGGERATION,
            cfg_weight=CFG_WEIGHT
        )
        return success
    except Exception as e:
        # Log the error and create silent audio as fallback
        rprint(f"[bold yellow]⚠️ FALLBACK: Chatterbox failed for segment {number}: {str(e)}[/bold yellow]")
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
            raise Exception(f"Chatterbox failed for segment {number}: {str(e)}")
