import datetime
import re
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from core.prompts import get_subtitle_trim_prompt
from core.tts_backend.estimate_duration import init_estimator, estimate_duration
from core.utils import *
from core.utils.models import *

console = Console()
speed_factor = load_key("speed_factor")

TRANS_SUBS_FOR_AUDIO_FILE = 'output/audio/trans_subs_for_audio.srt'
SRC_SUBS_FOR_AUDIO_FILE = 'output/audio/src_subs_for_audio.srt'
ESTIMATOR = None

# Minimum words for TTS - short texts cause Chatterbox issues (token repetition, empty alignment)
MIN_WORDS_FOR_TTS = 3
# Maximum gap (seconds) to consider merging with neighbor
MAX_MERGE_GAP = 3.0

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
        def valid_trim(response):
            if 'result' not in response:
                return {'status': 'error', 'message': 'No result in response'}
            return {'status': 'success', 'message': ''}
        try:    
            response = ask_gpt(prompt, resp_type='json', log_title='sub_trim', valid_def=valid_trim)
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

    Priority: NEXT segment (short replies usually relate to following phrase in speech)

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

    if i > 0:
        prev_end = df.loc[i - 1, 'end_time']
        curr_start = df.loc[i, 'start_time']
        gap_to_prev = time_diff_seconds(prev_end, curr_start, today)

    if i < len(df) - 1:
        curr_end = df.loc[i, 'end_time']
        next_start = df.loc[i + 1, 'start_time']
        gap_to_next = time_diff_seconds(curr_end, next_start, today)

    # Priority: next segment (speech context usually flows forward)
    if gap_to_next <= max_gap:
        return 'next'
    elif gap_to_prev <= max_gap:
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
    
    with open(TRANS_SUBS_FOR_AUDIO_FILE, 'r', encoding='utf-8') as file:
        content = file.read()
    
    with open(SRC_SUBS_FOR_AUDIO_FILE, 'r', encoding='utf-8') as src_file:
        src_content = src_file.read()
    
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
        
        subtitles.append({'number': number, 'start_time': start_time, 'end_time': end_time, 'duration': duration, 'text': text, 'origin': origin})
    
    df = pd.DataFrame(subtitles)

    # 🔄 First pass: Merge short TEXT segments (prevents TTS issues with short words)
    rprint("[bold cyan]📝 Checking for short text segments to merge...[/bold cyan]")
    df = merge_short_text_segments(df)

    # 🔄 Second pass: Merge short DURATION segments (existing logic)
    i = 0
    MIN_SUB_DUR = load_key("min_subtitle_duration")
    while i < len(df):
        today = datetime.date.today()
        if df.loc[i, 'duration'] < MIN_SUB_DUR:
            if i < len(df) - 1 and time_diff_seconds(df.loc[i, 'start_time'],df.loc[i+1, 'start_time'],today) < MIN_SUB_DUR:
                rprint(f"[bold yellow]Merging subtitles {i+1} and {i+2}[/bold yellow]")
                df.loc[i, 'text'] += ' ' + df.loc[i+1, 'text']
                df.loc[i, 'origin'] += ' ' + df.loc[i+1, 'origin']
                df.loc[i, 'end_time'] = df.loc[i+1, 'end_time']
                df.loc[i, 'duration'] = time_diff_seconds(df.loc[i, 'start_time'],df.loc[i, 'end_time'],today)
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