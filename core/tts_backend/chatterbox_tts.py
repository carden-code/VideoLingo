"""
Chatterbox TTS Integration for VideoLingo
Supports multilingual TTS with zero-shot voice cloning
GitHub: https://github.com/resemble-ai/chatterbox
"""

from pathlib import Path
import torch
import threading
import atexit
from queue import Queue, Empty
from contextlib import contextmanager
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


class ChatterboxModelPool:
    """
    Thread-safe pool of Chatterbox models for parallel TTS generation.

    Each worker thread gets its own model instance, preventing the
    alignment_stream_analyzer state conflicts that cause RuntimeError.
    """

    VRAM_PER_MODEL_GB = 3.25  # Empirically measured

    def __init__(self, pool_size: int = 4, device: str = "cuda"):
        self.requested_pool_size = pool_size  # Config value (for comparison)
        self.pool_size = pool_size  # Actual loaded count (may be lower due to OOM)
        self.device = device
        self._pool: Queue = None
        self._init_lock = threading.Lock()
        self._initialized = False
        self._models = []  # Keep references for cleanup

    def initialize(self):
        """Lazy initialization of model pool on first access"""
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return

            check_chatterbox_installed()

            # Auto-detect device
            if self.device == "cuda" and not torch.cuda.is_available():
                rprint("[yellow]CUDA not available, falling back to CPU[/yellow]")
                self.device = "cpu"

            self._pool = Queue()
            estimated_vram = self.pool_size * self.VRAM_PER_MODEL_GB

            rprint(f"[bold cyan]Initializing Chatterbox model pool ({self.pool_size} models) on {self.device}...[/bold cyan]")

            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            loaded_count = 0
            for i in range(self.pool_size):
                try:
                    rprint(f"[cyan]Loading model {i + 1}/{self.pool_size}...[/cyan]")
                    model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
                    self._pool.put(model)
                    self._models.append(model)
                    loaded_count += 1
                except torch.cuda.OutOfMemoryError:
                    rprint(f"[yellow]CUDA OOM at model {i + 1}, stopping at {loaded_count} models[/yellow]")
                    break
                except Exception as e:
                    rprint(f"[red]Failed to load model {i + 1}: {e}[/red]")
                    break

            if loaded_count == 0:
                raise RuntimeError("Failed to initialize any Chatterbox models")

            self.pool_size = loaded_count
            actual_vram = loaded_count * self.VRAM_PER_MODEL_GB
            rprint(f"[bold green]✓ Model pool initialized ({loaded_count} models, ~{actual_vram:.1f} GB VRAM)[/bold green]")

            self._initialized = True

    @contextmanager
    def acquire(self):
        """
        Context manager for safely acquiring and releasing a model.

        Usage:
            with pool.acquire() as model:
                wav = model.generate(...)
        """
        self.initialize()

        model = self._pool.get()  # Blocks if all models are in use
        try:
            yield model
        finally:
            self._pool.put(model)  # Always return to pool

    def shutdown(self):
        """Release all models and free VRAM"""
        if not self._initialized:
            return

        rprint("[cyan]Shutting down Chatterbox model pool...[/cyan]")

        # Clear the queue
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except Empty:
                break

        # Delete model references
        for model in self._models:
            del model
        self._models.clear()

        # Force CUDA memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._initialized = False
        rprint("[green]✓ Model pool shutdown complete[/green]")


# Global model pool instance
_model_pool: ChatterboxModelPool = None
_pool_lock = threading.Lock()
_atexit_registered = False


def get_model_pool(pool_size: int = None, device: str = None) -> ChatterboxModelPool:
    """
    Get or create the global model pool.

    Automatically rebuilds pool if config values (pool_size, device) have changed.

    Args:
        pool_size: Number of models in pool (from config if None)
        device: Device to use (from config if None)

    Returns:
        ChatterboxModelPool instance
    """
    global _model_pool, _atexit_registered

    # Load config values if not provided
    try:
        chatterbox_config = load_key("chatterbox_tts")
        if pool_size is None:
            pool_size = chatterbox_config.get("pool_size", 4)
        if device is None:
            device = chatterbox_config.get("device", "cuda")
    except:
        pool_size = pool_size or 4
        device = device or "cuda"

    with _pool_lock:
        # Check if we need to rebuild the pool (config changed)
        if _model_pool is not None:
            # Compare against requested_pool_size, not actual pool_size (which may be lower due to OOM)
            config_changed = (
                _model_pool.requested_pool_size != pool_size or
                _model_pool.device != device
            )
            if config_changed and not _model_pool._initialized:
                # Pool not yet initialized, just update settings
                _model_pool.requested_pool_size = pool_size
                _model_pool.pool_size = pool_size
                _model_pool.device = device
            elif config_changed and _model_pool._initialized:
                # Pool already initialized with different settings
                rprint(f"[yellow]Config changed (pool_size: {_model_pool.requested_pool_size}→{pool_size}, device: {_model_pool.device}→{device}), rebuilding pool...[/yellow]")
                _model_pool.shutdown()
                _model_pool = None

        # Create new pool if needed
        if _model_pool is None:
            _model_pool = ChatterboxModelPool(pool_size=pool_size, device=device)
            # Register cleanup on exit (only once)
            if not _atexit_registered:
                atexit.register(lambda: _model_pool.shutdown() if _model_pool else None)
                _atexit_registered = True

    return _model_pool


# Legacy support - single model cache (for pool_size=1 or backward compatibility)
_chatterbox_model = None
_chatterbox_multilingual_model = None

def get_chatterbox_model(multilingual=False, device="cuda"):
    """
    Get or initialize a single Chatterbox model (legacy, for backward compatibility).

    For parallel generation, use get_model_pool() instead.
    """
    global _chatterbox_model, _chatterbox_multilingual_model

    check_chatterbox_installed()

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
    - cs (Czech), ar (Arabic), zh (Chinese), hu (Hungarian), ko (Korean)
    - ja (Japanese), hi (Hindi), th (Thai), vi (Vietnamese), id (Indonesian)
    - he (Hebrew), uk (Ukrainian), el (Greek)
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

def chatterbox_tts(text, save_path, language_id='en', audio_prompt=None, exaggeration=0.5, cfg_weight=0.4, device="cuda", use_pool=True):
    """
    Generate speech using Chatterbox TTS

    Args:
        text: Text to synthesize
        save_path: Path to save the generated audio
        language_id: Language code (e.g., 'en', 'zh', 'ja')
        audio_prompt: Optional path to reference audio for voice cloning
        exaggeration: Control emotionality (0.0-1.0, default 0.5)
        cfg_weight: Influence of audio prompt (0.0-1.0, default 0.4)
        device: Device to use ('cuda' or 'cpu')
        use_pool: Whether to use model pool for thread-safe generation (default True)
    """
    import soundfile as sf
    import numpy as np

    def generate_with_model(model):
        """Generate audio with a given model instance"""
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
        return wav, model.sr if hasattr(model, 'sr') else 24000

    # Generate using pool or single model
    if use_pool:
        # Pool reads device from config, override only if explicitly specified
        pool = get_model_pool(device=device if device != "cuda" else None)
        with pool.acquire() as model:
            wav, sample_rate = generate_with_model(model)
    else:
        model = get_chatterbox_model(multilingual=True, device=device)
        wav, sample_rate = generate_with_model(model)

    # Save audio file
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tensor to numpy if needed
    if torch.is_tensor(wav):
        wav = wav.cpu().numpy()

    # Ensure correct shape (channels, samples) -> (samples, channels) or (samples,)
    if wav.ndim > 1 and wav.shape[0] < wav.shape[1]:
        wav = wav.T

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
