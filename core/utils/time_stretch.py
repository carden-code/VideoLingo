"""
High-quality time-stretching for VideoLingo

Supports multiple backends:
- pyrubberband: Best quality, preserves pitch and natural sound
- ffmpeg atempo: Fast fallback, acceptable quality

Rubberband is significantly better for speech at speed factors > 1.3x
"""

import os
import shutil
import subprocess
from pathlib import Path
from rich import print as rprint
import numpy as np


def time_stretch_rubberband(input_file: str, output_file: str, speed_factor: float) -> bool:
    """
    Time-stretch audio using pyrubberband (high quality).

    Args:
        input_file: Input audio file path
        output_file: Output audio file path
        speed_factor: Speed multiplier (>1 = faster, <1 = slower)

    Returns:
        True if successful, False otherwise
    """
    try:
        import pyrubberband as pyrb
        import soundfile as sf

        # Read audio
        audio, sr = sf.read(input_file)

        # Time stretch (pyrubberband expects time_ratio = 1/speed_factor)
        # speed_factor > 1 means faster, so time_ratio < 1
        time_ratio = 1.0 / speed_factor

        # Use speech-optimized options
        stretched = pyrb.time_stretch(
            audio,
            sr,
            time_ratio,
            rbargs={
                '--fine': '',           # Fine pitch detection for speech
                '--formant': '',        # Preserve formants (important for speech)
            }
        )

        # Write output
        sf.write(output_file, stretched, sr)
        return True

    except ImportError:
        return False
    except Exception as e:
        rprint(f"[yellow]Rubberband failed: {e}[/yellow]")
        return False


def time_stretch_ffmpeg(input_file: str, output_file: str, speed_factor: float) -> bool:
    """
    Time-stretch audio using ffmpeg atempo filter (fallback).

    Note: ffmpeg atempo is limited to 0.5-2.0 range per filter,
    so we chain multiple filters for extreme values.

    Args:
        input_file: Input audio file path
        output_file: Output audio file path
        speed_factor: Speed multiplier (>1 = faster, <1 = slower)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Build atempo filter chain for factors outside 0.5-2.0 range
        atempo_filters = []
        remaining = speed_factor

        while remaining > 2.0:
            atempo_filters.append('atempo=2.0')
            remaining /= 2.0
        while remaining < 0.5:
            atempo_filters.append('atempo=0.5')
            remaining *= 2.0

        if abs(remaining - 1.0) > 0.001:
            atempo_filters.append(f'atempo={remaining}')

        if not atempo_filters:
            # No change needed, just copy
            shutil.copy2(input_file, output_file)
            return True

        filter_str = ','.join(atempo_filters)
        cmd = ['ffmpeg', '-y', '-i', input_file, '-filter:a', filter_str, output_file]

        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0

    except Exception as e:
        rprint(f"[red]FFmpeg time stretch failed: {e}[/red]")
        return False


def adjust_audio_speed(input_file: str, output_file: str, speed_factor: float) -> None:
    """
    Adjust audio speed with automatic backend selection.

    Tries rubberband first (best quality), falls back to ffmpeg.

    Args:
        input_file: Input audio file path
        output_file: Output audio file path
        speed_factor: Speed multiplier (>1 = faster, <1 = slower)
    """
    from core.asr_backend.audio_preprocess import get_audio_duration
    from pydub import AudioSegment

    # If speed factor is close to 1, just copy
    if abs(speed_factor - 1.0) < 0.001:
        shutil.copy2(input_file, output_file)
        return

    input_duration = get_audio_duration(input_file)
    expected_duration = input_duration / speed_factor

    # Try rubberband first (best quality for speech)
    success = time_stretch_rubberband(input_file, output_file, speed_factor)

    if not success:
        # Fall back to ffmpeg
        rprint("[dim]Using ffmpeg atempo (rubberband not available)[/dim]")
        success = time_stretch_ffmpeg(input_file, output_file, speed_factor)

        if not success:
            raise Exception(f"Both rubberband and ffmpeg failed for {input_file}")

    # Verify output duration
    output_duration = get_audio_duration(output_file)

    # Handle edge case: output slightly longer than expected
    if output_duration >= expected_duration * 1.02:
        diff = output_duration - expected_duration

        if input_duration < 3 and diff <= 0.1:
            # Short audio with small overshoot - trim it
            audio = AudioSegment.from_wav(output_file)
            trimmed_audio = audio[:int(expected_duration * 1000)]
            trimmed_audio.export(output_file, format="wav")
            rprint(f"[dim]✂️ Trimmed to {expected_duration:.2f}s[/dim]")
        else:
            raise Exception(
                f"Duration mismatch: expected {expected_duration:.2f}s, "
                f"got {output_duration:.2f}s (input: {input_duration:.2f}s, "
                f"speed: {speed_factor})"
            )


def get_stretch_backend() -> str:
    """Check which time-stretch backend is available."""
    try:
        import pyrubberband
        return "rubberband"
    except ImportError:
        return "ffmpeg"
