"""
Audio Separator module for VideoLingo
Uses audio-separator library with support for multiple models:
- BS-Roformer (best quality, SDR ~12.9)
- MelBand-Roformer (good for complex mixes, SDR ~11.4)
- htdemucs (fast, baseline quality, SDR ~10-11)

Install: pip install audio-separator[gpu]
"""

import os
import shutil
import gc
from pathlib import Path
from rich.console import Console
from rich import print as rprint
from core.utils.models import _AUDIO_DIR, _RAW_AUDIO_FILE, _VOCAL_AUDIO_FILE, _BACKGROUND_AUDIO_FILE
from core.utils import load_key

# Available models with their filenames
SEPARATOR_MODELS = {
    "bs_roformer": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "mel_band_roformer": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
    "htdemucs": "htdemucs_ft.yaml",  # Uses demucs backend
}

DEFAULT_MODEL = "bs_roformer"


def get_separator_model():
    """Get configured separator model from config"""
    try:
        config = load_key("audio_separator")
        return config.get("model", DEFAULT_MODEL)
    except (KeyError, TypeError):
        return DEFAULT_MODEL


def separate_audio_with_model(input_file: str, output_dir: str, model_name: str = None):
    """
    Separate audio using audio-separator library

    Args:
        input_file: Path to input audio file
        output_dir: Directory to save output files
        model_name: Model to use (bs_roformer, mel_band_roformer, htdemucs)

    Returns:
        Tuple of (vocals_path, background_path)
    """
    from audio_separator.separator import Separator

    console = Console()

    if model_name is None:
        model_name = get_separator_model()

    model_file = SEPARATOR_MODELS.get(model_name, SEPARATOR_MODELS[DEFAULT_MODEL])

    console.print(f"🤖 Loading <{model_name}> model...")

    # Initialize separator
    separator = Separator(
        output_dir=output_dir,
        output_format="mp3",
    )

    # Load the model
    separator.load_model(model_filename=model_file)

    console.print("🎵 Separating audio...")

    # Separate audio - returns list of output file paths
    output_files = separator.separate(input_file)

    # audio-separator outputs files like:
    # input_(Vocals)_model.mp3 and input_(Instrumental)_model.mp3
    # We need to find and rename them

    vocals_file = None
    instrumental_file = None

    for f in output_files:
        f_lower = f.lower()
        if 'vocal' in f_lower:
            vocals_file = f
        elif 'instrumental' in f_lower or 'no_vocal' in f_lower or 'accompaniment' in f_lower:
            instrumental_file = f

    # Clean up separator
    del separator
    gc.collect()

    return vocals_file, instrumental_file


def separate_audio():
    """
    Main function to separate audio - compatible with existing VideoLingo interface
    """
    if os.path.exists(_VOCAL_AUDIO_FILE) and os.path.exists(_BACKGROUND_AUDIO_FILE):
        rprint(f"[yellow]⚠️ {_VOCAL_AUDIO_FILE} and {_BACKGROUND_AUDIO_FILE} already exist, skip audio separation.[/yellow]")
        return

    console = Console()
    os.makedirs(_AUDIO_DIR, exist_ok=True)

    model_name = get_separator_model()

    # Create temp directory for separator output
    temp_dir = os.path.join(_AUDIO_DIR, "temp_separator")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        vocals_file, instrumental_file = separate_audio_with_model(
            input_file=_RAW_AUDIO_FILE,
            output_dir=temp_dir,
            model_name=model_name
        )

        # Move files to expected locations
        if vocals_file and os.path.exists(vocals_file):
            console.print("🎤 Saving vocals track...")
            shutil.move(vocals_file, _VOCAL_AUDIO_FILE)
        else:
            raise FileNotFoundError("Vocals file not found in separator output")

        if instrumental_file and os.path.exists(instrumental_file):
            console.print("🎹 Saving background music...")
            shutil.move(instrumental_file, _BACKGROUND_AUDIO_FILE)
        else:
            raise FileNotFoundError("Instrumental file not found in separator output")

        console.print("[green]✨ Audio separation completed![/green]")

    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    separate_audio()
