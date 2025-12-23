import datetime
import re
import os
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from core.prompts import get_subtitle_trim_prompt
from core.tts_backend.estimate_duration import init_estimator, estimate_duration
from core.utils import *
from core.utils.segment_utils import encode_parent_list, parse_parent_list
from core.utils.models import *
from core.utils.prosodic_utils import (
    load_word_timestamps,
    calculate_pauses,
    get_pause_statistics,
    health_check_timestamps,
    PAUSE_THRESHOLD_PHRASE,
    PAUSE_THRESHOLD_SENTENCE
)

console = Console()
speed_factor = load_key("speed_factor")

TRANS_SUBS_FOR_AUDIO_FILE = 'output/audio/trans_subs_for_audio.srt'
SRC_SUBS_FOR_AUDIO_FILE = 'output/audio/src_subs_for_audio.srt'
ESTIMATOR = None
PROSODY_DF = None  # Cached prosodic data

# Minimum words for TTS - short texts cause Chatterbox issues (token repetition, empty alignment)
MIN_WORDS_FOR_TTS = 3
# Maximum gap (seconds) to consider merging with neighbor
MAX_MERGE_GAP = 3.0

def normalize_segment_id(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text if text else None

def merge_segment_lineage(left_id, left_parent, right_id, right_parent):
    parent_ids = []
    for pid in parse_parent_list(left_parent) + parse_parent_list(right_parent):
        if pid not in parent_ids:
            parent_ids.append(pid)
    for seg_id in (left_id, right_id):
        if seg_id and seg_id not in parent_ids:
            parent_ids.append(seg_id)
    if not parent_ids:
        return None, encode_parent_list([])
    if len(parent_ids) == 1:
        merged_segment_id = f"{parent_ids[0]}_m1"
    else:
        merged_segment_id = f"merge_{parent_ids[0]}_{parent_ids[-1]}"
    return merged_segment_id, encode_parent_list(parent_ids)


def load_prosody_data():
    """
    Load and cache prosodic data from word timestamps.

    Returns:
        DataFrame with word timestamps and pauses, or None if not available
    """
    global PROSODY_DF

    if PROSODY_DF is not None:
        return PROSODY_DF

    chunks_file = "output/log/cleaned_chunks.xlsx"
    if not os.path.exists(chunks_file):
        rprint("[dim]Prosody: word timestamps not available[/dim]")
        return None

    try:
        prosody_df = load_word_timestamps(chunks_file)
        prosody_df = calculate_pauses(prosody_df)

        health = health_check_timestamps(prosody_df)
        if not health.get('valid', True):
            rprint(f"[yellow]⚠️ Prosody: {health.get('reason', 'invalid timestamps')}[/yellow]")
            return None

        if not health.get('usable_for_prosody', False):
            rprint(
                f"[yellow]⚠️ Prosody: timestamps quality too low "
                f"({health.get('zero_duration_pct', 0)}% zero-duration, "
                f"{health.get('non_monotonic_pct', 0)}% non-monotonic, "
                f"{health.get('invalid_pct', 0)}% invalid), skipping[/yellow]"
            )
            return None

        PROSODY_DF = prosody_df

        # Print prosody statistics
        stats = get_pause_statistics(PROSODY_DF)
        rprint(f"[cyan]🎵 Prosody: {stats['phrase_boundaries']} phrase / {stats['sentence_boundaries']} sentence boundaries[/cyan]")

        # Show health info if degraded
        if health['quality'] == 'degraded':
            rprint(
                f"[yellow]⚠️ Timestamps: {health['zero_duration_pct']}% zero-duration, "
                f"{health['non_monotonic_pct']}% non-monotonic, "
                f"{health['invalid_pct']}% invalid[/yellow]"
            )

        return PROSODY_DF
    except Exception as e:
        rprint(f"[yellow]⚠️ Could not load prosody data: {e}[/yellow]")
        return None


def find_prosodic_pause_at_time(target_time: float, tolerance: float = 0.5) -> float:
    """
    Find the prosodic pause near a specific timestamp.

    Args:
        target_time: Time in seconds to search near
        tolerance: Search window in seconds

    Returns:
        Pause duration if found, 0.0 otherwise
    """
    prosody_df = load_prosody_data()
    if prosody_df is None:
        return 0.0

    # Find words near the target time
    mask = (prosody_df['end'] >= target_time - tolerance) & \
           (prosody_df['end'] <= target_time + tolerance)

    nearby_words = prosody_df[mask]

    if nearby_words.empty:
        return 0.0

    # Return the maximum pause in the window
    return nearby_words['pause_after'].max()


def is_prosodic_boundary(start_time: float, end_time: float) -> bool:
    """
    Check if there's a prosodic boundary (long pause) between two timestamps.

    Args:
        start_time: End time of first segment
        end_time: Start time of second segment

    Returns:
        True if there's a significant prosodic pause
    """
    prosody_df = load_prosody_data()
    if prosody_df is None:
        return False

    # Find words that end around start_time
    mask = (prosody_df['end'] >= start_time - 0.2) & \
           (prosody_df['end'] <= start_time + 0.2)

    nearby_words = prosody_df[mask]

    if nearby_words.empty:
        return False

    # Check if any word has a significant pause after it
    max_pause = nearby_words['pause_after'].max()
    return max_pause >= PAUSE_THRESHOLD_SENTENCE


def check_len_then_trim(text, duration):
    global ESTIMATOR
    if ESTIMATOR is None:
        ESTIMATOR = init_estimator()
    estimated_duration = estimate_duration(text, ESTIMATOR) / speed_factor['max']
    
    console.print(f"Subtitle text: {text}, "
                  f"[bold green]Estimated reading duration: {estimated_duration:.2f} seconds[/bold green]")

    if estimated_duration > duration:
        rprint(Panel(f"Estimated reading duration {estimated_duration:.2f} seconds exceeds given duration {duration:.2f} seconds, shortening...", title="Processing", border_style="yellow"))
        original_text = text
        prompt = get_subtitle_trim_prompt(text, duration)
        # JSON schema for structured output
        trim_schema = {
            "name": "subtitle_trim",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "string"},
                    "result": {"type": "string"}
                },
                "required": ["analysis", "result"],
                "additionalProperties": False
            }
        }
        def valid_trim(response):
            if 'result' not in response:
                return {'status': 'error', 'message': 'No result in response'}
            return {'status': 'success', 'message': ''}
        try:
            response = ask_gpt(prompt, resp_type='json', log_title='sub_trim', valid_def=valid_trim, json_schema=trim_schema)
            shortened_text = response['result']
        except Exception:
            rprint("[bold red]🚫 AI refused to answer due to sensitivity, so manually remove punctuation[/bold red]")
            shortened_text = re.sub(r'[,.!?;:，。！？；：]', ' ', text).strip()
        rprint(Panel(f"Subtitle before shortening: {original_text}\nSubtitle after shortening: {shortened_text}", title="Subtitle Shortening Result", border_style="green"))
        return shortened_text
    else:
        return text

def time_diff_seconds(t1, t2, base_date):
    """Calculate the difference in seconds between two time objects"""
    dt1 = datetime.datetime.combine(base_date, t1)
    dt2 = datetime.datetime.combine(base_date, t2)
    return (dt2 - dt1).total_seconds()


def count_words(text):
    """
    Count words in text, handling both space-separated and CJK languages.
    For CJK (Chinese, Japanese, Korean), count characters as "words".
    """
    if not text or not isinstance(text, str):
        return 0

    text = text.strip()

    # Check if text contains CJK characters
    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or  # Chinese
                   '\u3040' <= c <= '\u309f' or  # Hiragana
                   '\u30a0' <= c <= '\u30ff' or  # Katakana
                   '\uac00' <= c <= '\ud7af')    # Korean

    if cjk_chars > len(text) * 0.3:  # Mostly CJK
        # For CJK, count characters (excluding spaces and punctuation)
        return sum(1 for c in text if c.isalnum())
    else:
        # For space-separated languages, count words
        return len(text.split())


def should_merge_with_neighbor(df, i, today, min_words=MIN_WORDS_FOR_TTS, max_gap=MAX_MERGE_GAP):
    """
    Decide which neighbor to merge with for short text segments.

    Uses prosodic analysis to make smarter decisions:
    - If there's a long pause (prosodic boundary), prefer NOT to merge
    - Priority: NEXT segment (short replies usually relate to following phrase in speech)

    Args:
        df: DataFrame with subtitles
        i: Current row index
        today: Date for time calculations
        min_words: Minimum word count threshold
        max_gap: Maximum gap in seconds to consider merging

    Returns:
        'next', 'prev', or None (if no merge needed or possible)
    """
    text = df.loc[i, 'text'].strip()
    word_count = count_words(text)

    if word_count >= min_words:
        return None  # Text is long enough

    # Calculate gaps to neighbors
    gap_to_prev = float('inf')
    gap_to_next = float('inf')
    prosody_pause_prev = 0.0
    prosody_pause_next = 0.0

    if i > 0:
        prev_end = df.loc[i - 1, 'end_time']
        curr_start = df.loc[i, 'start_time']
        gap_to_prev = time_diff_seconds(prev_end, curr_start, today)
        # Get prosodic pause at the boundary
        prev_end_seconds = time_diff_seconds(datetime.time(0, 0, 0), prev_end, today)
        prosody_pause_prev = find_prosodic_pause_at_time(prev_end_seconds)

    if i < len(df) - 1:
        curr_end = df.loc[i, 'end_time']
        next_start = df.loc[i + 1, 'start_time']
        gap_to_next = time_diff_seconds(curr_end, next_start, today)
        # Get prosodic pause at the boundary
        curr_end_seconds = time_diff_seconds(datetime.time(0, 0, 0), curr_end, today)
        prosody_pause_next = find_prosodic_pause_at_time(curr_end_seconds)

    # Prosody-aware decision:
    # If one direction has a strong prosodic boundary (long pause), prefer the other
    next_has_boundary = prosody_pause_next >= PAUSE_THRESHOLD_SENTENCE
    prev_has_boundary = prosody_pause_prev >= PAUSE_THRESHOLD_SENTENCE

    # If merging with NEXT would cross a sentence boundary, try PREV
    if gap_to_next <= max_gap and not next_has_boundary:
        return 'next'
    elif gap_to_prev <= max_gap and not prev_has_boundary:
        return 'prev'
    # Fallback: even if there's a boundary, merge if we must (short text is worse)
    elif gap_to_next <= max_gap:
        if next_has_boundary:
            rprint(f"[dim]Prosody: crossing sentence boundary to merge '{text}'[/dim]")
        return 'next'
    elif gap_to_prev <= max_gap:
        if prev_has_boundary:
            rprint(f"[dim]Prosody: crossing sentence boundary to merge '{text}'[/dim]")
        return 'prev'

    return None  # Both neighbors too far - will rely on monkey-patch fallback


def merge_short_text_segments(df):
    """
    Merge segments with too few words to prevent TTS issues.

    Short texts (< MIN_WORDS_FOR_TTS words) can cause:
    - Token repetition in Chatterbox
    - Empty alignment matrices
    - IndexError crashes

    Returns:
        Modified DataFrame with short segments merged
    """
    today = datetime.date.today()
    merged_count = 0

    i = 0
    while i < len(df):
        merge_direction = should_merge_with_neighbor(df, i, today)

        if merge_direction == 'next' and i < len(df) - 1:
            # Merge current with next
            word_count = count_words(df.loc[i, 'text'])
            rprint(f"[bold magenta]📝 Short text merge: '{df.loc[i, 'text']}' ({word_count} words) → merging with NEXT segment[/bold magenta]")

            if 'segment_id' in df.columns:
                merged_id, merged_parent = merge_segment_lineage(
                    normalize_segment_id(df.loc[i, 'segment_id']),
                    df.loc[i, 'parent_segment_id'] if 'parent_segment_id' in df.columns else None,
                    normalize_segment_id(df.loc[i + 1, 'segment_id']),
                    df.loc[i + 1, 'parent_segment_id'] if 'parent_segment_id' in df.columns else None
                )
                df.loc[i + 1, 'segment_id'] = merged_id
                if 'parent_segment_id' in df.columns:
                    df.loc[i + 1, 'parent_segment_id'] = merged_parent

            df.loc[i + 1, 'text'] = df.loc[i, 'text'] + ' ' + df.loc[i + 1, 'text']
            df.loc[i + 1, 'origin'] = df.loc[i, 'origin'] + ' ' + df.loc[i + 1, 'origin']
            df.loc[i + 1, 'start_time'] = df.loc[i, 'start_time']
            df.loc[i + 1, 'duration'] = time_diff_seconds(
                df.loc[i + 1, 'start_time'],
                df.loc[i + 1, 'end_time'],
                today
            )
            df = df.drop(i).reset_index(drop=True)
            merged_count += 1
            # Don't increment i - check the merged result

        elif merge_direction == 'prev' and i > 0:
            # Merge current with previous
            word_count = count_words(df.loc[i, 'text'])
            rprint(f"[bold magenta]📝 Short text merge: '{df.loc[i, 'text']}' ({word_count} words) → merging with PREV segment[/bold magenta]")

            if 'segment_id' in df.columns:
                merged_id, merged_parent = merge_segment_lineage(
                    normalize_segment_id(df.loc[i - 1, 'segment_id']),
                    df.loc[i - 1, 'parent_segment_id'] if 'parent_segment_id' in df.columns else None,
                    normalize_segment_id(df.loc[i, 'segment_id']),
                    df.loc[i, 'parent_segment_id'] if 'parent_segment_id' in df.columns else None
                )
                df.loc[i - 1, 'segment_id'] = merged_id
                if 'parent_segment_id' in df.columns:
                    df.loc[i - 1, 'parent_segment_id'] = merged_parent

            df.loc[i - 1, 'text'] = df.loc[i - 1, 'text'] + ' ' + df.loc[i, 'text']
            df.loc[i - 1, 'origin'] = df.loc[i - 1, 'origin'] + ' ' + df.loc[i, 'origin']
            df.loc[i - 1, 'end_time'] = df.loc[i, 'end_time']
            df.loc[i - 1, 'duration'] = time_diff_seconds(
                df.loc[i - 1, 'start_time'],
                df.loc[i - 1, 'end_time'],
                today
            )
            df = df.drop(i).reset_index(drop=True)
            merged_count += 1
            # Don't increment i - recheck from same position

        else:
            if merge_direction is None and count_words(df.loc[i, 'text']) < MIN_WORDS_FOR_TTS:
                word_count = count_words(df.loc[i, 'text'])
                rprint(f"[bold yellow]⚠️ Short text '{df.loc[i, 'text']}' ({word_count} words) - no close neighbor, keeping as-is (fallback will handle)[/bold yellow]")
            i += 1

    if merged_count > 0:
        rprint(f"[bold green]✓ Merged {merged_count} short text segments[/bold green]")

    return df


def process_srt():
    """Process srt file, generate audio tasks"""

    # Load prosodic data early for analysis
    rprint("[bold cyan]🎵 Loading prosodic data from word timestamps...[/bold cyan]")
    load_prosody_data()

    with open(TRANS_SUBS_FOR_AUDIO_FILE, 'r', encoding='utf-8') as file:
        content = file.read()

    with open(SRC_SUBS_FOR_AUDIO_FILE, 'r', encoding='utf-8') as src_file:
        src_content = src_file.read()

    segment_id_map = {}
    parent_id_map = {}
    if os.path.exists(_5_REMERGED):
        df_remerged = pd.read_excel(_5_REMERGED)
        if 'segment_id' in df_remerged.columns:
            segment_id_map = {
                i + 1: normalize_segment_id(v)
                for i, v in enumerate(df_remerged['segment_id'].tolist())
            }
        if 'parent_segment_id' in df_remerged.columns:
            parent_id_map = {}
            for i, v in enumerate(df_remerged['parent_segment_id'].tolist()):
                if v is None or (isinstance(v, float) and pd.isna(v)) or not str(v).strip():
                    parent_id_map[i + 1] = encode_parent_list([])
                elif isinstance(v, (list, tuple)):
                    parent_id_map[i + 1] = encode_parent_list(v)
                else:
                    parent_id_map[i + 1] = str(v)
    
    subtitles = []
    src_subtitles = {}
    
    for block in src_content.strip().split('\n\n'):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 3:
            continue
        
        number = int(lines[0])
        src_text = ' '.join(lines[2:])
        src_subtitles[number] = src_text
    
    for block in content.strip().split('\n\n'):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 3:
            continue
        
        try:
            number = int(lines[0])
            start_time, end_time = lines[1].split(' --> ')
            start_time = datetime.datetime.strptime(start_time, '%H:%M:%S,%f').time()
            end_time = datetime.datetime.strptime(end_time, '%H:%M:%S,%f').time()
            duration = time_diff_seconds(start_time, end_time, datetime.date.today())
            text = ' '.join(lines[2:])
            # Remove content within parentheses (including English and Chinese parentheses)
            text = re.sub(r'\([^)]*\)', '', text).strip()
            text = re.sub(r'（[^）]*）', '', text).strip()
            # Remove '-' character, can continue to add illegal characters that cause errors
            text = text.replace('-', '')

            # Add the original text from src_subs_for_audio.srt
            origin = src_subtitles.get(number, '')

        except ValueError as e:
            rprint(Panel(f"Unable to parse subtitle block '{block}', error: {str(e)}, skipping this subtitle block.", title="Error", border_style="red"))
            continue
        
        subtitles.append({
            'number': number,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'text': text,
            'origin': origin,
            'segment_id': segment_id_map.get(number),
            'parent_segment_id': parent_id_map.get(number)
        })
    
    df = pd.DataFrame(subtitles)

    # 🔄 First pass: Merge short TEXT segments (prevents TTS issues with short words)
    rprint("[bold cyan]📝 Checking for short text segments to merge...[/bold cyan]")
    df = merge_short_text_segments(df)

    # 🔄 Second pass: Merge short DURATION segments (with prosody-aware logic)
    i = 0
    MIN_SUB_DUR = load_key("min_subtitle_duration")
    while i < len(df):
        today = datetime.date.today()
        if df.loc[i, 'duration'] < MIN_SUB_DUR:
            # Check if we can merge with next segment
            can_merge_next = i < len(df) - 1 and \
                             time_diff_seconds(df.loc[i, 'start_time'], df.loc[i+1, 'start_time'], today) < MIN_SUB_DUR

            if can_merge_next:
                # VAD-guided: Check for prosodic boundary before merging
                curr_end = df.loc[i, 'end_time']
                curr_end_seconds = time_diff_seconds(datetime.time(0, 0, 0), curr_end, today)
                prosody_pause = find_prosodic_pause_at_time(curr_end_seconds)

                if prosody_pause >= PAUSE_THRESHOLD_SENTENCE:
                    # Strong prosodic boundary - prefer extending over merging
                    rprint(f"[dim]VAD: pause {prosody_pause*1000:.0f}ms detected, extending instead of merging[/dim]")
                    df.loc[i, 'end_time'] = (datetime.datetime.combine(today, df.loc[i, 'start_time']) +
                                            datetime.timedelta(seconds=MIN_SUB_DUR)).time()
                    df.loc[i, 'duration'] = MIN_SUB_DUR
                    i += 1
                else:
                    # No strong boundary - safe to merge
                    rprint(f"[bold yellow]Merging subtitles {i+1} and {i+2}[/bold yellow]")
                    if 'segment_id' in df.columns:
                        merged_id, merged_parent = merge_segment_lineage(
                            normalize_segment_id(df.loc[i, 'segment_id']),
                            df.loc[i, 'parent_segment_id'] if 'parent_segment_id' in df.columns else None,
                            normalize_segment_id(df.loc[i + 1, 'segment_id']),
                            df.loc[i + 1, 'parent_segment_id'] if 'parent_segment_id' in df.columns else None
                        )
                        df.loc[i, 'segment_id'] = merged_id
                        if 'parent_segment_id' in df.columns:
                            df.loc[i, 'parent_segment_id'] = merged_parent
                    df.loc[i, 'text'] += ' ' + df.loc[i+1, 'text']
                    df.loc[i, 'origin'] += ' ' + df.loc[i+1, 'origin']
                    df.loc[i, 'end_time'] = df.loc[i+1, 'end_time']
                    df.loc[i, 'duration'] = time_diff_seconds(df.loc[i, 'start_time'], df.loc[i, 'end_time'], today)
                    df = df.drop(i+1).reset_index(drop=True)
            else:
                if i < len(df) - 1:  # Not the last audio
                    rprint(f"[bold blue]Extending subtitle {i+1} duration to {MIN_SUB_DUR} seconds[/bold blue]")
                    df.loc[i, 'end_time'] = (datetime.datetime.combine(today, df.loc[i, 'start_time']) +
                                            datetime.timedelta(seconds=MIN_SUB_DUR)).time()
                    df.loc[i, 'duration'] = MIN_SUB_DUR
                else:
                    rprint(f"[bold red]The last subtitle {i+1} duration is less than {MIN_SUB_DUR} seconds, but not extending[/bold red]")
                i += 1
        else:
            i += 1
    
    df['start_time'] = df['start_time'].apply(lambda x: x.strftime('%H:%M:%S.%f')[:-3])
    df['end_time'] = df['end_time'].apply(lambda x: x.strftime('%H:%M:%S.%f')[:-3])

    ##! No longer perform secondary trim
    # check and trim subtitle length, for twice to ensure the subtitle length is within the limit, 允许tolerance
    # df['text'] = df.apply(lambda x: check_len_then_trim(x['text'], x['duration']+x['tolerance']), axis=1)

    return df

@check_file_exists(_8_1_AUDIO_TASK)
def gen_audio_task_main():
    df = process_srt()
    console.print(df)
    df.to_excel(_8_1_AUDIO_TASK, index=False)
    rprint(Panel(f"Successfully generated {_8_1_AUDIO_TASK}", title="Success", border_style="green"))

if __name__ == '__main__':
    gen_audio_task_main()
