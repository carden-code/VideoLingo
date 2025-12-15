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


_alignment_patch_applied = False

def apply_alignment_bugfix():
    """
    Monkey-patch fix for Chatterbox alignment_stream_analyzer bug.

    Bug: IndexError in step() when alignment matrix is empty after forced EOS.
    Location: chatterbox/models/t3/inference/alignment_stream_analyzer.py:139

    Original code crashes on: A[self.completed_at:, :-5].max(dim=1)
    when the sliced matrix has zero size.
    """
    global _alignment_patch_applied
    if _alignment_patch_applied:
        return

    try:
        from chatterbox.models.t3.inference import alignment_stream_analyzer

        # Store original step method
        original_step = alignment_stream_analyzer.AlignmentStreamAnalyzer.step

        def patched_step(self, logits, next_token):
            """Patched step method with empty matrix check"""
            import torch

            # Call most of original logic but handle the problematic line
            self.step_count += 1
            last_token = next_token

            if self.complete:
                return logits

            A = self.A
            attn = self.get_attention()

            if attn is not None:
                if A is None:
                    A = attn
                else:
                    A = torch.cat([A, attn], dim=0)
                self.A = A

            if A is None:
                return logits

            # Check completion conditions
            alignment_complete = (A[:, -5:].sum() > 5)
            if alignment_complete and not self.complete:
                self.complete = True
                self.completed_at = A.shape[0]

            # Detect repetition patterns
            long_tail = self.complete and (self.step_count - self.completed_at > 50)

            # BUGFIX: Check matrix size before calling max()
            sliced_A = A[self.completed_at:, :-5] if self.complete else torch.tensor([])
            if sliced_A.numel() == 0 or sliced_A.shape[1] == 0:
                alignment_repetition = False
            else:
                alignment_repetition = self.complete and (sliced_A.max(dim=1).values.sum() > 5)

            # Check token repetition
            self.recent_tokens.append(last_token)
            if len(self.recent_tokens) > 10:
                self.recent_tokens.pop(0)

            token_repetition = False
            if len(self.recent_tokens) >= 4:
                last_4 = self.recent_tokens[-4:]
                if last_4[0] == last_4[2] and last_4[1] == last_4[3]:
                    import logging
                    logging.warning(f"🚨 Detected 2x repetition of token {last_4[0]}")
                    token_repetition = True

            # Force EOS if needed
            if long_tail or alignment_repetition or token_repetition:
                import logging
                logging.warning(f"forcing EOS token, long_tail={long_tail}, alignment_repetition={alignment_repetition}, token_repetition={token_repetition}")
                logits[:, self.eos_token] = 1e9

            return logits

        # Apply patch
        alignment_stream_analyzer.AlignmentStreamAnalyzer.step = patched_step
        _alignment_patch_applied = True
        rprint("[green]✓ Applied Chatterbox alignment bugfix[/green]")

    except Exception as e:
        rprint(f"[yellow]⚠ Could not apply Chatterbox bugfix: {e}[/yellow]")


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
            apply_alignment_bugfix()  # Fix IndexError in alignment_stream_analyzer

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
    apply_alignment_bugfix()  # Fix IndexError in alignment_stream_analyzer

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


def find_optimal_reference(min_duration: float = 5.0, max_duration: float = 10.0, fallback_min: float = 3.0) -> str:
    """
    Find optimal reference audio for Chatterbox voice cloning.

    Chatterbox works best with reference audio between 5-10 seconds.
    Short references (<3s) can cause alignment errors.

    Args:
        min_duration: Ideal minimum duration (default 5.0s)
        max_duration: Ideal maximum duration (default 10.0s)
        fallback_min: Minimum acceptable duration if no ideal found (default 3.0s)

    Returns:
        Path to optimal reference audio, or None if none suitable found
    """
    from core.asr_backend.audio_preprocess import get_audio_duration

    refers_dir = Path.cwd() / "output/audio/refers"
    if not refers_dir.exists():
        return None

    ref_files = list(refers_dir.glob("*.wav"))
    if not ref_files:
        return None

    # Sort by segment number to process in order
    ref_files.sort(key=lambda x: int(x.stem) if x.stem.isdigit() else 999)

    best_ref = None
    best_duration = 0

    for ref_file in ref_files:
        try:
            duration = get_audio_duration(str(ref_file))
        except Exception:
            continue

        # Ideal: 5-10 seconds - return immediately
        if min_duration <= duration <= max_duration:
            rprint(f"[green]✓ Found optimal reference: {ref_file.name} ({duration:.1f}s)[/green]")
            return str(ref_file)

        # Track longest acceptable for fallback (>= 3s)
        if duration >= fallback_min and duration > best_duration:
            best_ref = str(ref_file)
            best_duration = duration

    if best_ref:
        rprint(f"[yellow]Using best available reference: {Path(best_ref).name} ({best_duration:.1f}s)[/yellow]")
    else:
        rprint(f"[red]No reference audio >= {fallback_min}s found[/red]")

    return best_ref


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

    Falls back to edge_tts if Chatterbox fails (e.g., on very short text).

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

    # Generate TTS with fallback to edge_tts
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
        # Log the error and fall back to edge_tts
        rprint(f"[bold yellow]⚠️ FALLBACK: Chatterbox failed for segment {number}: {str(e)}[/bold yellow]")
        rprint(f"[bold yellow]⚠️ FALLBACK: Text was: '{text}'[/bold yellow]")
        rprint(f"[bold cyan]🔄 FALLBACK: Switching to edge_tts for this segment (no voice cloning)[/bold cyan]")

        try:
            from core.tts_backend.edge_tts import edge_tts
            edge_tts(text, save_as)
            rprint(f"[bold green]✓ FALLBACK: edge_tts succeeded for segment {number}[/bold green]")
            return True
        except Exception as edge_error:
            rprint(f"[bold red]❌ FALLBACK: edge_tts also failed: {str(edge_error)}[/bold red]")
            raise Exception(f"Both Chatterbox and edge_tts failed for segment {number}: Chatterbox: {str(e)}, edge_tts: {str(edge_error)}")
