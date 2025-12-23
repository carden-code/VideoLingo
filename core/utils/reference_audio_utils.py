"""
Reference Audio Enhancement Utilities for VideoLingo

Provides functions to select and enhance reference audio for TTS voice cloning:
- SNR (Signal-to-Noise Ratio) analysis for quality assessment
- Smart segment selection based on duration and quality
- Optional stitching of multiple short references to reach minimum length
- Noise reduction for cleaner reference audio
- Audio normalization

Optimal reference audio for voice cloning: 10-30 seconds, clean voice
"""

import os
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from rich import print as rprint
import soundfile as sf


# Default parameters for reference audio selection
DEFAULT_MIN_DURATION = 10.0  # seconds - minimum for good voice cloning
DEFAULT_MAX_DURATION = 30.0  # seconds - practical upper limit
DEFAULT_FALLBACK_MIN = 5.0   # seconds - fallback if no long segments
DEFAULT_STITCH_MAX_CLIPS = 3
DEFAULT_STITCH_CROSSFADE_MS = 120


def calculate_snr(audio: np.ndarray, sr: int, frame_length: int = 2048) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR) for audio.

    Higher SNR = cleaner audio with less noise.
    Uses RMS energy to estimate signal vs noise.

    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        frame_length: Frame length for analysis

    Returns:
        SNR in dB (higher is better, typically 10-40 dB for speech)
    """
    if len(audio) == 0:
        return -np.inf

    # Handle stereo by converting to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Calculate RMS energy in frames
    num_frames = len(audio) // frame_length
    if num_frames == 0:
        return -np.inf

    frame_energies = []
    for i in range(num_frames):
        frame = audio[i * frame_length:(i + 1) * frame_length]
        energy = np.sqrt(np.mean(frame ** 2))
        if energy > 0:
            frame_energies.append(energy)

    if len(frame_energies) < 2:
        return -np.inf

    frame_energies = np.array(frame_energies)

    # Estimate noise as the lowest 10% of frame energies
    noise_threshold = np.percentile(frame_energies, 10)
    noise_frames = frame_energies[frame_energies <= noise_threshold]
    noise_rms = np.mean(noise_frames) if len(noise_frames) > 0 else 1e-10

    # Estimate signal as the highest 50% of frame energies
    signal_threshold = np.percentile(frame_energies, 50)
    signal_frames = frame_energies[frame_energies >= signal_threshold]
    signal_rms = np.mean(signal_frames) if len(signal_frames) > 0 else noise_rms

    # Calculate SNR in dB
    if noise_rms > 0:
        snr_db = 20 * np.log10(signal_rms / noise_rms)
    else:
        snr_db = 60.0  # Very clean audio

    return snr_db


def get_audio_info(file_path: str) -> Tuple[float, float, int]:
    """
    Get audio file info: duration, SNR, and sample rate.

    Args:
        file_path: Path to audio file

    Returns:
        Tuple of (duration_seconds, snr_db, sample_rate)
    """
    try:
        audio, sr = sf.read(file_path)
        duration = len(audio) / sr
        snr = calculate_snr(audio, sr)
        return duration, snr, sr
    except Exception as e:
        rprint(f"[yellow]Error reading {file_path}: {e}[/yellow]")
        return 0.0, -np.inf, 0


def _load_reference_candidates(
    refers_dir: str,
    fallback_min: float,
) -> List[Tuple[str, float, float]]:
    refers_path = Path(refers_dir)
    if not refers_path.exists():
        rprint(f"[red]Reference directory not found: {refers_dir}[/red]")
        return []

    # Exclude already enhanced or stitched files to prevent double processing
    ref_files = [
        f for f in refers_path.glob("*.wav")
        if "_enhanced" not in f.stem and not f.stem.startswith("stitched_")
    ]
    if not ref_files:
        rprint(f"[red]No WAV files found in {refers_dir}[/red]")
        return []

    candidates: List[Tuple[str, float, float]] = []  # (path, duration, snr)

    rprint(f"[cyan]Analyzing {len(ref_files)} reference segments...[/cyan]")

    for ref_file in ref_files:
        duration, snr, _ = get_audio_info(str(ref_file))
        if duration >= fallback_min:
            candidates.append((str(ref_file), duration, snr))

    if not candidates:
        rprint(f"[red]No reference audio >= {fallback_min}s found[/red]")

    return candidates


def _select_best_reference(
    candidates: List[Tuple[str, float, float]],
    min_duration: float,
    max_duration: float,
    prefer_snr: bool
) -> Tuple[Optional[Tuple[str, float, float]], bool]:
    if not candidates:
        return None, False

    optimal = [(p, d, s) for p, d, s in candidates if min_duration <= d <= max_duration]

    if optimal:
        if prefer_snr:
            optimal.sort(key=lambda x: (x[2], x[1]), reverse=True)
        else:
            optimal.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return optimal[0], True

    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return candidates[0], False


def _build_stitched_reference(
    candidates: List[Tuple[str, float, float]],
    refers_dir: str,
    min_duration: float,
    max_duration: float,
    prefer_snr: bool = True,
    max_clips: int = DEFAULT_STITCH_MAX_CLIPS,
    crossfade_ms: int = DEFAULT_STITCH_CROSSFADE_MS
) -> Optional[str]:
    if len(candidates) < 2:
        return None

    if prefer_snr:
        candidates = sorted(candidates, key=lambda x: (x[2], x[1]), reverse=True)
    else:
        candidates = sorted(candidates, key=lambda x: (x[1], x[2]), reverse=True)

    selected = []
    total_duration = 0.0
    for candidate in candidates:
        if len(selected) >= max_clips:
            break
        selected.append(candidate)
        total_duration += candidate[1]
        if total_duration >= min_duration:
            break

    if total_duration < min_duration:
        return None

    signature = "|".join(Path(item[0]).name for item in selected)
    signature += f":{min_duration}:{max_duration}:{crossfade_ms}"
    digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:10]
    stitched_path = Path(refers_dir) / f"stitched_{digest}.wav"

    if stitched_path.exists():
        rprint(f"[cyan]Using cached stitched reference: {stitched_path.name}[/cyan]")
        return str(stitched_path)

    try:
        from pydub import AudioSegment
    except ImportError:
        rprint("[yellow]pydub not installed, cannot stitch references[/yellow]")
        return None

    merged = None
    for path, _, _ in selected:
        clip = AudioSegment.from_file(path)
        if merged is None:
            merged = clip
        else:
            overlap = min(crossfade_ms, len(merged), len(clip))
            merged = merged.append(clip, crossfade=overlap)

    if merged is None:
        return None

    if max_duration > 0 and len(merged) > int(max_duration * 1000):
        merged = merged[:int(max_duration * 1000)]

    stitched_path.parent.mkdir(parents=True, exist_ok=True)
    merged.export(stitched_path, format="wav")
    rprint(f"[green]✓ Stitched reference created: {stitched_path.name}[/green]")

    return str(stitched_path)


def find_best_reference(
    refers_dir: str = "output/audio/refers",
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    fallback_min: float = DEFAULT_FALLBACK_MIN,
    prefer_snr: bool = True
) -> Optional[str]:
    """
    Find the best reference audio for voice cloning.

    Selection criteria (in order of priority):
    1. Duration in optimal range (10-30 seconds)
    2. Highest SNR (cleanest audio)

    Args:
        refers_dir: Directory containing reference audio segments
        min_duration: Minimum ideal duration (default 10s)
        max_duration: Maximum ideal duration (default 30s)
        fallback_min: Minimum acceptable duration if no ideal found (default 5s)
        prefer_snr: If True, prefer higher SNR over longer duration

    Returns:
        Path to best reference audio, or None if none suitable found
    """
    candidates = _load_reference_candidates(refers_dir, fallback_min)
    if not candidates:
        return None

    best, is_optimal = _select_best_reference(candidates, min_duration, max_duration, prefer_snr)
    if not best:
        return None

    if is_optimal:
        rprint(f"[green]✓ Found optimal reference: {Path(best[0]).name} "
               f"({best[1]:.1f}s, SNR: {best[2]:.1f}dB)[/green]")
    else:
        rprint(f"[yellow]Using best available: {Path(best[0]).name} "
               f"({best[1]:.1f}s, SNR: {best[2]:.1f}dB)[/yellow]")

    return best[0]


def apply_noise_reduction(
    audio: np.ndarray,
    sr: int,
    noise_reduce_strength: float = 0.5
) -> np.ndarray:
    """
    Apply noise reduction to audio.

    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        noise_reduce_strength: Strength of noise reduction (0.0-1.0)

    Returns:
        Noise-reduced audio
    """
    try:
        import noisereduce as nr

        # Handle stereo
        if len(audio.shape) > 1:
            # Process each channel
            reduced = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                reduced[:, ch] = nr.reduce_noise(
                    y=audio[:, ch],
                    sr=sr,
                    prop_decrease=noise_reduce_strength,
                    stationary=False
                )
            return reduced
        else:
            return nr.reduce_noise(
                y=audio,
                sr=sr,
                prop_decrease=noise_reduce_strength,
                stationary=False
            )
    except ImportError:
        rprint("[yellow]noisereduce not installed, skipping noise reduction[/yellow]")
        return audio
    except Exception as e:
        rprint(f"[yellow]Noise reduction failed: {e}[/yellow]")
        return audio


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target dB level.

    Args:
        audio: Audio data as numpy array
        target_db: Target RMS level in dB (default -20 dB)

    Returns:
        Normalized audio
    """
    if len(audio) == 0:
        return audio

    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio

    current_db = 20 * np.log10(rms)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)

    normalized = audio * gain

    # Prevent clipping
    max_val = np.max(np.abs(normalized))
    if max_val > 0.99:
        normalized = normalized * (0.99 / max_val)

    return normalized


def enhance_reference_audio(
    input_path: str,
    output_path: Optional[str] = None,
    apply_nr: bool = True,
    apply_norm: bool = True,
    noise_reduce_strength: float = 0.3,
    target_db: float = -20.0
) -> str:
    """
    Enhance reference audio with noise reduction and normalization.

    Args:
        input_path: Path to input audio file
        output_path: Path for output (if None, creates _enhanced suffix)
        apply_nr: Apply noise reduction
        apply_norm: Apply normalization
        noise_reduce_strength: Noise reduction strength (0.0-1.0)
        target_db: Target normalization level in dB

    Returns:
        Path to enhanced audio file
    """
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_enhanced{p.suffix}")

    # Read audio
    audio, sr = sf.read(input_path)
    original_snr = calculate_snr(audio, sr)

    # Apply enhancements
    if apply_nr:
        audio = apply_noise_reduction(audio, sr, noise_reduce_strength)

    if apply_norm:
        audio = normalize_audio(audio, target_db)

    # Save enhanced audio
    sf.write(output_path, audio, sr)

    enhanced_snr = calculate_snr(audio, sr)
    rprint(f"[green]✓ Enhanced reference audio saved: {output_path}[/green]")
    rprint(f"[cyan]  SNR: {original_snr:.1f}dB → {enhanced_snr:.1f}dB[/cyan]")

    return output_path


def get_best_enhanced_reference(
    refers_dir: str = "output/audio/refers",
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    fallback_min: float = DEFAULT_FALLBACK_MIN,
    enhance: bool = True,
    noise_reduce_strength: float = 0.3,
    prefer_snr: bool = True,
    stitch_short_refs: bool = True,
    stitch_max_clips: int = DEFAULT_STITCH_MAX_CLIPS,
    stitch_crossfade_ms: int = DEFAULT_STITCH_CROSSFADE_MS
) -> Optional[str]:
    """
    Get the best reference audio, optionally enhanced.

    This is the main function to use for TTS backends.

    Args:
        refers_dir: Directory containing reference audio segments
        min_duration: Minimum ideal duration (default 10s)
        max_duration: Maximum ideal duration (default 30s)
        fallback_min: Minimum acceptable duration if no ideal found (default 5s)
        enhance: Whether to apply noise reduction/normalization
        noise_reduce_strength: Strength of noise reduction
        prefer_snr: Prefer higher SNR when selecting candidates
        stitch_short_refs: Stitch multiple clips if best is shorter than min_duration
        stitch_max_clips: Max clips to stitch
        stitch_crossfade_ms: Crossfade between stitched clips (ms)

    Returns:
        Path to best (optionally enhanced) reference audio
    """
    # Find best raw reference + candidates for stitching
    candidates = _load_reference_candidates(refers_dir, fallback_min)
    if not candidates:
        return None

    best, is_optimal = _select_best_reference(candidates, min_duration, max_duration, prefer_snr)
    if not best:
        return None
    best_ref, best_duration, best_snr = best

    if is_optimal:
        rprint(f"[green]✓ Found optimal reference: {Path(best_ref).name} "
               f"({best_duration:.1f}s, SNR: {best_snr:.1f}dB)[/green]")
    else:
        rprint(f"[yellow]Using best available: {Path(best_ref).name} "
               f"({best_duration:.1f}s, SNR: {best_snr:.1f}dB)[/yellow]")

    if stitch_short_refs and best_duration < min_duration:
        stitched = _build_stitched_reference(
            candidates=candidates,
            refers_dir=refers_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            prefer_snr=prefer_snr,
            max_clips=stitch_max_clips,
            crossfade_ms=stitch_crossfade_ms
        )
        if stitched:
            best_ref = stitched

    if not enhance:
        return best_ref

    # Check if enhanced version already exists
    p = Path(best_ref)
    enhanced_path = str(p.parent / f"{p.stem}_enhanced{p.suffix}")

    if os.path.exists(enhanced_path):
        rprint(f"[cyan]Using cached enhanced reference: {Path(enhanced_path).name}[/cyan]")
        return enhanced_path

    # Create enhanced version
    return enhance_reference_audio(
        input_path=best_ref,
        output_path=enhanced_path,
        noise_reduce_strength=noise_reduce_strength
    )
