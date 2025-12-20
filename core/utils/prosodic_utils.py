"""
Prosodic analysis utilities for VideoLingo segmentation.

Uses word-level timestamps from WhisperX to detect natural speech boundaries:
- Pause detection: Long pauses indicate phrase/sentence boundaries
- Pre-boundary lengthening: Words before boundaries are often elongated
- Speaking rate changes: Rate variations mark prosodic boundaries

Based on research:
- "Prosody-Based Automatic Segmentation of Speech" (Shriberg et al.)
- "Automatic Detection of Prosodic Boundaries" (PLOS ONE)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from rich import print as rprint


@dataclass
class ProsodyInfo:
    """Prosodic information for a text segment."""
    start_time: float
    end_time: float
    text: str
    pause_before: float  # Pause duration before this segment
    pause_after: float   # Pause duration after this segment
    words_per_second: float  # Speaking rate
    is_boundary_candidate: bool  # Whether this is a good split point


# Thresholds based on prosodic research
PAUSE_THRESHOLD_PHRASE = 0.3    # 300ms - phrase boundary
PAUSE_THRESHOLD_SENTENCE = 0.5  # 500ms - sentence boundary
PAUSE_THRESHOLD_PARAGRAPH = 1.0 # 1000ms - topic/paragraph boundary
MIN_SEGMENT_DURATION = 1.0      # Minimum segment duration in seconds
MIN_WORDS_PER_SEGMENT = 3       # Minimum words per segment


def load_word_timestamps(chunks_file: str = "output/log/cleaned_chunks.xlsx") -> pd.DataFrame:
    """
    Load word-level timestamps from cleaned_chunks.xlsx.

    Returns DataFrame with columns: text, start, end, speaker_id
    """
    df = pd.read_excel(chunks_file)
    # Remove quotes from text if present
    df['text'] = df['text'].apply(lambda x: str(x).strip('"').strip())
    return df


def calculate_pauses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate pause duration after each word.

    Args:
        df: DataFrame with 'start' and 'end' columns

    Returns:
        DataFrame with added 'pause_after' column
    """
    df = df.copy()
    df['pause_after'] = 0.0

    for i in range(len(df) - 1):
        current_end = df.iloc[i]['end']
        next_start = df.iloc[i + 1]['start']
        pause = max(0, next_start - current_end)
        df.iloc[i, df.columns.get_loc('pause_after')] = pause

    return df


def detect_prosodic_boundaries(df: pd.DataFrame,
                                min_pause: float = PAUSE_THRESHOLD_PHRASE) -> List[int]:
    """
    Detect prosodic boundary indices based on pause duration.

    Args:
        df: DataFrame with word timestamps and pauses
        min_pause: Minimum pause duration to consider as boundary

    Returns:
        List of word indices that mark prosodic boundaries
    """
    if 'pause_after' not in df.columns:
        df = calculate_pauses(df)

    boundaries = []
    for i, row in df.iterrows():
        if row['pause_after'] >= min_pause:
            boundaries.append(i)

    return boundaries


def get_pause_at_position(df: pd.DataFrame, word_index: int) -> float:
    """
    Get pause duration after a specific word.

    Args:
        df: DataFrame with word timestamps
        word_index: Index of the word

    Returns:
        Pause duration in seconds (0 if last word or invalid index)
    """
    if word_index < 0 or word_index >= len(df) - 1:
        return 0.0

    current_end = df.iloc[word_index]['end']
    next_start = df.iloc[word_index + 1]['start']
    return max(0, next_start - current_end)


def find_best_split_point(df: pd.DataFrame,
                          start_idx: int,
                          end_idx: int,
                          min_words: int = MIN_WORDS_PER_SEGMENT) -> Optional[int]:
    """
    Find the best prosodic split point within a range of words.

    Prioritizes:
    1. Longest pause (strongest prosodic boundary)
    2. Must leave at least min_words on each side

    Args:
        df: DataFrame with word timestamps and pauses
        start_idx: Start word index
        end_idx: End word index (exclusive)
        min_words: Minimum words required on each side of split

    Returns:
        Best split index, or None if no valid split found
    """
    if 'pause_after' not in df.columns:
        df = calculate_pauses(df)

    # Need enough words for valid split
    if end_idx - start_idx < min_words * 2:
        return None

    # Find valid range for splits
    valid_start = start_idx + min_words - 1  # -1 because we split AFTER this index
    valid_end = end_idx - min_words

    if valid_start >= valid_end:
        return None

    # Find longest pause in valid range
    best_idx = None
    best_pause = 0.0

    for i in range(valid_start, valid_end):
        pause = df.iloc[i]['pause_after']
        if pause > best_pause:
            best_pause = pause
            best_idx = i

    # Only return if pause is significant
    if best_pause >= PAUSE_THRESHOLD_PHRASE:
        return best_idx

    return None


def calculate_speaking_rate(df: pd.DataFrame,
                            start_idx: int,
                            end_idx: int) -> float:
    """
    Calculate speaking rate (words per second) for a segment.

    Args:
        df: DataFrame with word timestamps
        start_idx: Start word index
        end_idx: End word index (exclusive)

    Returns:
        Words per second
    """
    if start_idx >= end_idx or start_idx < 0 or end_idx > len(df):
        return 0.0

    segment_df = df.iloc[start_idx:end_idx]
    duration = segment_df.iloc[-1]['end'] - segment_df.iloc[0]['start']

    if duration <= 0:
        return 0.0

    word_count = len(segment_df)
    return word_count / duration


def analyze_segment_prosody(df: pd.DataFrame,
                            start_idx: int,
                            end_idx: int) -> ProsodyInfo:
    """
    Analyze prosodic features of a segment.

    Args:
        df: DataFrame with word timestamps
        start_idx: Start word index
        end_idx: End word index (exclusive)

    Returns:
        ProsodyInfo with prosodic features
    """
    if 'pause_after' not in df.columns:
        df = calculate_pauses(df)

    segment_df = df.iloc[start_idx:end_idx]

    # Get text
    text = ' '.join(segment_df['text'].tolist())

    # Get timing
    start_time = segment_df.iloc[0]['start']
    end_time = segment_df.iloc[-1]['end']

    # Get pauses
    pause_before = get_pause_at_position(df, start_idx - 1) if start_idx > 0 else 0.0
    pause_after = df.iloc[end_idx - 1]['pause_after'] if end_idx <= len(df) else 0.0

    # Speaking rate
    wps = calculate_speaking_rate(df, start_idx, end_idx)

    # Boundary candidate if significant pause after
    is_boundary = pause_after >= PAUSE_THRESHOLD_PHRASE

    return ProsodyInfo(
        start_time=start_time,
        end_time=end_time,
        text=text,
        pause_before=pause_before,
        pause_after=pause_after,
        words_per_second=wps,
        is_boundary_candidate=is_boundary
    )


def should_merge_segments(seg1_end_idx: int,
                          seg2_start_idx: int,
                          df: pd.DataFrame,
                          max_gap: float = 0.5) -> bool:
    """
    Determine if two segments should be merged based on prosody.

    Short pause between segments suggests they belong together.

    Args:
        seg1_end_idx: End index of first segment
        seg2_start_idx: Start index of second segment
        df: DataFrame with word timestamps
        max_gap: Maximum pause to allow merging

    Returns:
        True if segments should be merged
    """
    if seg1_end_idx < 0 or seg2_start_idx >= len(df):
        return False

    # Get pause between segments
    pause = get_pause_at_position(df, seg1_end_idx - 1)

    # Merge if pause is short (continuous speech)
    return pause < max_gap


def get_prosodic_split_points(text_segments: List[str],
                               df: pd.DataFrame,
                               min_pause: float = PAUSE_THRESHOLD_PHRASE) -> List[Tuple[int, float]]:
    """
    Find prosodic split points that align with text segments.

    Args:
        text_segments: List of text segments from NLP splitting
        df: DataFrame with word timestamps
        min_pause: Minimum pause for split point

    Returns:
        List of (word_index, pause_duration) for each split point
    """
    if 'pause_after' not in df.columns:
        df = calculate_pauses(df)

    split_points = []
    current_word_idx = 0

    for segment in text_segments:
        words = segment.split()
        segment_end_idx = current_word_idx + len(words) - 1

        if segment_end_idx < len(df):
            pause = df.iloc[segment_end_idx]['pause_after']
            if pause >= min_pause:
                split_points.append((segment_end_idx, pause))

        current_word_idx += len(words)

    return split_points


def refine_nlp_boundaries_with_prosody(nlp_segments: List[str],
                                        df: pd.DataFrame,
                                        context_words: int = 3) -> List[str]:
    """
    Refine NLP-based segmentation using prosodic information.

    If an NLP boundary doesn't align with a prosodic boundary,
    try to find a better split point nearby.

    Args:
        nlp_segments: Text segments from NLP splitting
        df: DataFrame with word timestamps
        context_words: How many words to search around boundary

    Returns:
        Refined list of text segments
    """
    if 'pause_after' not in df.columns:
        df = calculate_pauses(df)

    refined_segments = []
    all_words = df['text'].tolist()
    current_idx = 0

    for i, segment in enumerate(nlp_segments):
        segment_words = segment.split()
        segment_end_idx = current_idx + len(segment_words) - 1

        # Check if current boundary has a good pause
        current_pause = get_pause_at_position(df, segment_end_idx)

        if current_pause >= PAUSE_THRESHOLD_PHRASE:
            # Good prosodic boundary, keep as is
            refined_segments.append(segment)
        else:
            # Look for better boundary nearby
            search_start = max(current_idx + MIN_WORDS_PER_SEGMENT - 1,
                             segment_end_idx - context_words)
            search_end = min(len(df) - MIN_WORDS_PER_SEGMENT,
                           segment_end_idx + context_words)

            best_idx = segment_end_idx
            best_pause = current_pause

            for j in range(search_start, search_end + 1):
                pause = get_pause_at_position(df, j)
                if pause > best_pause:
                    best_pause = pause
                    best_idx = j

            if best_idx != segment_end_idx and best_pause >= PAUSE_THRESHOLD_PHRASE:
                # Found better boundary
                new_text = ' '.join(all_words[current_idx:best_idx + 1])
                refined_segments.append(new_text)
                rprint(f"[dim]Prosody: adjusted boundary at '{segment_words[-1]}' "
                       f"(pause: {current_pause:.2f}s → {best_pause:.2f}s)[/dim]")

                # Handle remaining words from this segment
                if best_idx < segment_end_idx:
                    remaining = ' '.join(all_words[best_idx + 1:segment_end_idx + 1])
                    if i + 1 < len(nlp_segments):
                        # Prepend to next segment
                        nlp_segments[i + 1] = remaining + ' ' + nlp_segments[i + 1]
                # If we extended into next segment, that will be handled next iteration
            else:
                refined_segments.append(segment)

        current_idx = segment_end_idx + 1

    return refined_segments


def get_pause_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate pause statistics for the audio.

    Args:
        df: DataFrame with word timestamps

    Returns:
        Dictionary with pause statistics
    """
    if 'pause_after' not in df.columns:
        df = calculate_pauses(df)

    pauses = df['pause_after'].values
    pauses = pauses[pauses > 0]  # Only non-zero pauses

    if len(pauses) == 0:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'max': 0.0,
            'phrase_boundaries': 0,
            'sentence_boundaries': 0
        }

    return {
        'mean': float(np.mean(pauses)),
        'median': float(np.median(pauses)),
        'std': float(np.std(pauses)),
        'max': float(np.max(pauses)),
        'phrase_boundaries': int(np.sum(pauses >= PAUSE_THRESHOLD_PHRASE)),
        'sentence_boundaries': int(np.sum(pauses >= PAUSE_THRESHOLD_SENTENCE))
    }


def print_prosody_analysis(df: pd.DataFrame):
    """Print prosodic analysis summary."""
    stats = get_pause_statistics(df)

    rprint("\n[bold cyan]Prosodic Analysis Summary[/bold cyan]")
    rprint(f"  Mean pause: {stats['mean']*1000:.0f}ms")
    rprint(f"  Median pause: {stats['median']*1000:.0f}ms")
    rprint(f"  Max pause: {stats['max']*1000:.0f}ms")
    rprint(f"  Phrase boundaries (>{PAUSE_THRESHOLD_PHRASE*1000:.0f}ms): {stats['phrase_boundaries']}")
    rprint(f"  Sentence boundaries (>{PAUSE_THRESHOLD_SENTENCE*1000:.0f}ms): {stats['sentence_boundaries']}")


if __name__ == "__main__":
    # Test with sample data
    import os
    if os.path.exists("output/log/cleaned_chunks.xlsx"):
        df = load_word_timestamps()
        df = calculate_pauses(df)
        print_prosody_analysis(df)

        # Show some boundaries
        boundaries = detect_prosodic_boundaries(df)
        rprint(f"\n[bold]Found {len(boundaries)} prosodic boundaries[/bold]")
        for idx in boundaries[:10]:
            word = df.iloc[idx]['text']
            pause = df.iloc[idx]['pause_after']
            rprint(f"  '{word}' → pause {pause*1000:.0f}ms")
