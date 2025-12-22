from typing import List, Tuple

from pydub import AudioSegment
from pydub.silence import detect_silence


def detect_speech_segments(
    audio_file: str,
    min_silence_len_ms: int = 300,
    silence_offset_db: float = -16.0,
    sample_ms: int = 30000,
    max_segments: int = 20000,
    min_speech_ratio: float = 0.15,
    max_speech_ratio: float = 0.98
) -> List[Tuple[float, float]]:
    audio = AudioSegment.from_file(audio_file)
    if len(audio) == 0:
        return []

    sample = audio[: min(sample_ms, len(audio))]
    noise_floor = sample.dBFS
    silence_thresh = noise_floor + silence_offset_db

    silence_regions = detect_silence(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh
    )

    speech_segments = []
    prev_end = 0
    for start_ms, end_ms in silence_regions:
        if start_ms > prev_end:
            speech_segments.append((prev_end / 1000.0, start_ms / 1000.0))
        prev_end = end_ms
    if prev_end < len(audio):
        speech_segments.append((prev_end / 1000.0, len(audio) / 1000.0))

    if len(speech_segments) > max_segments:
        return []

    total_speech = sum(end - start for start, end in speech_segments)
    ratio = total_speech / (len(audio) / 1000.0)
    if ratio < min_speech_ratio or ratio > max_speech_ratio:
        return []

    return speech_segments


def snap_to_vad_onset(
    time_stamps: List[Tuple[float, float]],
    speech_segments: List[Tuple[float, float]],
    max_shift_sec: float = 0.25
) -> List[Tuple[float, float]]:
    if not speech_segments:
        return time_stamps

    onsets = [start for start, _ in speech_segments]
    adjusted = []
    prev_start = 0.0
    for start, end in time_stamps:
        candidate = None
        best_delta = None
        for onset in onsets:
            delta = abs(onset - start)
            if delta <= max_shift_sec and (best_delta is None or delta < best_delta):
                candidate = onset
                best_delta = delta

        if candidate is not None:
            start = candidate

        if start < prev_start:
            start = prev_start
        if start > end - 0.05:
            start = max(prev_start, end - 0.05)

        adjusted.append((start, end))
        prev_start = start

    return adjusted
