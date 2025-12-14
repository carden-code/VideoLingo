"""
Chatterbox TTS Integration for VideoLingo
Supports multilingual TTS with zero-shot voice cloning
GitHub: https://github.com/resemble-ai/chatterbox
"""

from pathlib import Path
import torch
from core.utils import *

def check_chatterbox_installed():
    """Check if chatterbox-tts is installed"""
    try:
        import chatterbox
        return True
    except ImportError:
        raise ImportError(
            "Chatterbox TTS is not installed. Please install it using:\n"
            "pip install chatterbox-tts\n"
            "or from source:\n"
            "git clone https://github.com/resemble-ai/chatterbox.git && cd chatterbox && pip install -e ."
        )

# Global model cache to avoid reloading
_chatterbox_model = None
_chatterbox_multilingual_model = None

def get_chatterbox_model(multilingual=False, device="cuda"):
    """
    Get or initialize the Chatterbox model

    Args:
        multilingual: Whether to use multilingual model (supports 23 languages)
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Chatterbox model instance
    """
    global _chatterbox_model, _chatterbox_multilingual_model

    check_chatterbox_installed()

    # Auto-detect device if cuda not available
    if device == "cuda" and not torch.cuda.is_available():
        rprint("[yellow]CUDA not available, falling back to CPU[/yellow]")
        device = "cpu"

    if multilingual:
        if _chatterbox_multilingual_model is None:
            rprint(f"[bold cyan]Loading Chatterbox Multilingual model on {device}...[/bold cyan]")
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            _chatterbox_multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            rprint("[bold green]✓ Chatterbox Multilingual model loaded successfully[/bold green]")
        return _chatterbox_multilingual_model
    else:
        if _chatterbox_model is None:
            rprint(f"[bold cyan]Loading Chatterbox model on {device}...[/bold cyan]")
            from chatterbox.tts import ChatterboxTTS
            _chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
            rprint("[bold green]✓ Chatterbox model loaded successfully[/bold green]")
        return _chatterbox_model

def get_language_code(language_name):
    """
    Map VideoLingo language names to Chatterbox language IDs

    Supported languages (23 total):
    - en (English), es (Spanish), fr (French), de (German), it (Italian)
    - pt (Portuguese), pl (Polish), tr (Turkish), ru (Russian), nl (Dutch)
    - cs (Czech), ar (Arabic), zh-cn (Chinese), hu (Hungarian), ko (Korean)
    - ja (Japanese), hi (Hindi), th (Thai), vi (Vietnamese), id (Indonesian)
    - he (Hebrew), uk (Ukrainian), el (Greek)
    """
    language_map = {
        # English
        'english': 'en',
        '英文': 'en',
        'en': 'en',

        # Chinese
        'chinese': 'zh-cn',
        '中文': 'zh-cn',
        '简体中文': 'zh-cn',
        'zh': 'zh-cn',
        'zh-cn': 'zh-cn',

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

        # Thai
        'thai': 'th',
        '泰语': 'th',
        'th': 'th',

        # Vietnamese
        'vietnamese': 'vi',
        '越南语': 'vi',
        'vi': 'vi',

        # Indonesian
        'indonesian': 'id',
        '印尼语': 'id',
        'id': 'id',

        # Czech
        'czech': 'cs',
        'čeština': 'cs',
        '捷克语': 'cs',
        'cs': 'cs',

        # Hungarian
        'hungarian': 'hu',
        'magyar': 'hu',
        '匈牙利语': 'hu',
        'hu': 'hu',

        # Hebrew
        'hebrew': 'he',
        'עברית': 'he',
        '希伯来语': 'he',
        'he': 'he',

        # Ukrainian
        'ukrainian': 'uk',
        'українська': 'uk',
        '乌克兰语': 'uk',
        'uk': 'uk',

        # Greek
        'greek': 'el',
        'ελληνικά': 'el',
        '希腊语': 'el',
        'el': 'el',
    }

    # Normalize input
    normalized = language_name.lower().strip()

    if normalized in language_map:
        return language_map[normalized]

    # Default to English if not found
    rprint(f"[yellow]Language '{language_name}' not found in map, defaulting to 'en'[/yellow]")
    return 'en'

def chatterbox_tts(text, save_path, language_id='en', audio_prompt=None, exaggeration=0.5, cfg_weight=0.4, device="cuda"):
    """
    Generate speech using Chatterbox TTS

    Args:
        text: Text to synthesize
        save_path: Path to save the generated audio
        language_id: Language code (e.g., 'en', 'zh-cn', 'ja')
        audio_prompt: Optional path to reference audio for voice cloning
        exaggeration: Control emotionality (0.0-1.0, default 0.5)
        cfg_weight: Influence of audio prompt (0.0-1.0, default 0.4)
        device: Device to use ('cuda' or 'cpu')
    """
    import soundfile as sf
    import numpy as np

    # Get model
    model = get_chatterbox_model(multilingual=True, device=device)

    # Generate audio
    if audio_prompt and Path(audio_prompt).exists():
        rprint(f"[cyan]Using voice cloning with reference: {audio_prompt}[/cyan]")
        wav = model.generate(
            text,
            language_id=language_id,
            audio_prompt_path=audio_prompt,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )
    else:
        rprint(f"[cyan]Generating audio for language: {language_id}[/cyan]")
        wav = model.generate(
            text,
            language_id=language_id,
            exaggeration=exaggeration
        )

    # Save audio file
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tensor to numpy if needed
    if torch.is_tensor(wav):
        wav = wav.cpu().numpy()

    # Ensure correct shape (channels, samples) -> (samples, channels) or (samples,)
    if wav.ndim > 1 and wav.shape[0] < wav.shape[1]:
        wav = wav.T

    # Get sample rate from model (Chatterbox uses .sr)
    sample_rate = model.sr if hasattr(model, 'sr') else 24000

    # Save as WAV
    sf.write(str(save_path), wav, sample_rate)
    rprint(f"[bold green]✓ Audio saved: {save_path}[/bold green]")

    return True

def chatterbox_tts_for_videolingo(text, save_as, number, task_df):
    """
    Chatterbox TTS integration for VideoLingo pipeline

    Supports three modes:
    - Mode 1: Basic TTS without voice cloning
    - Mode 2: Voice cloning with single reference audio
    - Mode 3: Voice cloning with per-segment reference audio

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
    DEVICE = chatterbox_config.get("device", "cuda")

    # Get target language
    TARGET_LANGUAGE = load_key("target_language")
    language_id = get_language_code(TARGET_LANGUAGE)

    # Determine reference audio based on mode
    audio_prompt = None
    current_dir = Path.cwd()

    if VOICE_CLONE_MODE == 2:
        # Use single reference audio for all segments
        ref_path = current_dir / "output/audio/refers/1.wav"
        if ref_path.exists():
            audio_prompt = str(ref_path)
        else:
            rprint(f"[yellow]Reference audio not found at {ref_path}, extracting...[/yellow]")
            try:
                from core._9_refer_audio import extract_refer_audio_main
                extract_refer_audio_main()
                if ref_path.exists():
                    audio_prompt = str(ref_path)
            except Exception as e:
                rprint(f"[bold red]Failed to extract reference audio: {str(e)}[/bold red]")
                rprint("[yellow]Continuing without voice cloning...[/yellow]")

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
                    # Fallback to mode 2
                    rprint("[yellow]Falling back to mode 2 (single reference)...[/yellow]")
                    ref_path = current_dir / "output/audio/refers/1.wav"
                    if ref_path.exists():
                        audio_prompt = str(ref_path)
            except Exception as e:
                rprint(f"[bold red]Failed to extract reference audio: {str(e)}[/bold red]")
                rprint("[yellow]Continuing without voice cloning...[/yellow]")

    # Generate TTS
    try:
        success = chatterbox_tts(
            text=text,
            save_path=save_as,
            language_id=language_id,
            audio_prompt=audio_prompt,
            exaggeration=EXAGGERATION,
            cfg_weight=CFG_WEIGHT,
            device=DEVICE
        )
        return success
    except Exception as e:
        raise Exception(f"Chatterbox TTS failed: {str(e)}")
