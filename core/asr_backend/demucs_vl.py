"""
Audio separation module for VideoLingo

Supports multiple backends:
- audio-separator: BS-Roformer, MelBand-Roformer (best quality)
- demucs: htdemucs (fast, baseline)

Configure in config.yaml:
  audio_separator:
    model: "bs_roformer"  # or "mel_band_roformer" or "htdemucs"
"""

import os
import torch
from rich.console import Console
from rich import print as rprint
from torch.cuda import is_available as is_cuda_available
from typing import Optional
import gc
from core.utils.models import *
from core.utils import load_key


def get_separator_model():
    """Get configured separator model from config"""
    try:
        config = load_key("audio_separator")
        return config.get("model", "bs_roformer")
    except (KeyError, TypeError):
        return "bs_roformer"


def demucs_audio_legacy():
    """Original htdemucs implementation using demucs library directly"""
    from demucs.pretrained import get_model
    from demucs.audio import save_audio
    from demucs.api import Separator
    from demucs.apply import BagOfModels

    class PreloadedSeparator(Separator):
        def __init__(self, model: BagOfModels, shifts: int = 1, overlap: float = 0.25,
                     split: bool = True, segment: Optional[int] = None, jobs: int = 0):
            self._model, self._audio_channels, self._samplerate = model, model.audio_channels, model.samplerate
            device = "cuda" if is_cuda_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            self.update_parameter(device=device, shifts=shifts, overlap=overlap, split=split,
                                segment=segment, jobs=jobs, progress=True, callback=None, callback_arg=None)

    console = Console()
    os.makedirs(_AUDIO_DIR, exist_ok=True)

    console.print("🤖 Loading <htdemucs> model...")
    model = get_model('htdemucs')
    separator = PreloadedSeparator(model=model, shifts=1, overlap=0.25)

    console.print("🎵 Separating audio...")
    _, outputs = separator.separate_audio_file(_RAW_AUDIO_FILE)

    kwargs = {"samplerate": model.samplerate, "bitrate": 128, "preset": 2,
             "clip": "rescale", "as_float": False, "bits_per_sample": 16}

    console.print("🎤 Saving vocals track...")
    save_audio(outputs['vocals'].cpu(), _VOCAL_AUDIO_FILE, **kwargs)

    console.print("🎹 Saving background music...")
    background = sum(audio for source, audio in outputs.items() if source != 'vocals')
    save_audio(background.cpu(), _BACKGROUND_AUDIO_FILE, **kwargs)

    # Clean up memory
    del outputs, background, model, separator
    gc.collect()

    console.print("[green]✨ Audio separation completed![/green]")


def demucs_audio():
    """
    Main audio separation function - dispatches to appropriate backend
    """
    if os.path.exists(_VOCAL_AUDIO_FILE) and os.path.exists(_BACKGROUND_AUDIO_FILE):
        rprint(f"[yellow]⚠️ {_VOCAL_AUDIO_FILE} and {_BACKGROUND_AUDIO_FILE} already exist, skip audio separation.[/yellow]")
        return

    model = get_separator_model()

    # Use audio-separator for Roformer models (better quality)
    if model in ("bs_roformer", "mel_band_roformer"):
        try:
            from core.asr_backend.audio_separator_vl import separate_audio
            rprint(f"[cyan]Using audio-separator with {model} model (best quality)[/cyan]")
            separate_audio()
            return
        except ImportError as e:
            rprint(f"[yellow]audio-separator not installed, falling back to htdemucs: {e}[/yellow]")
            rprint("[yellow]Install with: pip install audio-separator[gpu][/yellow]")

    # Fallback to htdemucs (legacy demucs library)
    rprint("[cyan]Using htdemucs model (fast baseline)[/cyan]")
    demucs_audio_legacy()


if __name__ == "__main__":
    demucs_audio()
