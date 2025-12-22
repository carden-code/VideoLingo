import pandas as pd
import json
import os
import concurrent.futures
from core.translate_lines import translate_lines
from core._4_1_summarize import search_things_to_note_in_prompt
from core._8_1_audio_task import check_len_then_trim
from core._6_gen_sub import align_timestamp
from core.utils import *
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from difflib import SequenceMatcher
from core.utils.models import *
console = Console()


def verify_translation_quality(src_text: list, trans_text: list, sample_size: int = 5) -> bool:
    """
    Optional LLM-based verification of translation quality.
    Samples a few translations and asks LLM to verify they make sense.

    Args:
        src_text: List of source text lines
        trans_text: List of translated text lines
        sample_size: Number of samples to verify (default 5)

    Returns:
        True if verification passes, raises ValueError if critical issues found
    """
    if not load_key("verify_translation", False):
        return True  # Skip if not enabled in config

    console.print("[cyan]🔍 Running LLM translation verification...[/cyan]")

    # Sample evenly distributed lines
    total = len(src_text)
    if total <= sample_size:
        indices = list(range(total))
    else:
        step = total // sample_size
        indices = [i * step for i in range(sample_size)]

    # Build verification prompt
    samples = []
    for idx in indices:
        if idx < len(src_text) and idx < len(trans_text):
            src = src_text[idx].strip()
            trans = trans_text[idx].strip()
            if src and trans:
                samples.append(f"{idx + 1}. Source: \"{src}\"\n   Translation: \"{trans}\"")

    if not samples:
        console.print("[yellow]⚠️ No samples to verify[/yellow]")
        return True

    target_lang = load_key("target_language", "English")
    prompt = f"""You are a translation quality checker. Verify these {len(samples)} translation samples.

Target language: {target_lang}

Samples:
{chr(10).join(samples)}

Check each sample for:
1. Is the translation semantically correct (meaning preserved)?
2. Is it in the correct target language?
3. Is it complete (not truncated or garbled)?

Respond in JSON format:
{{
  "passed": true/false,
  "issues": ["list of issues if any, empty if passed"],
  "critical": true/false (true = translation is unusable)
}}"""

    try:
        result = ask_gpt(prompt, resp_type='json', log_title='verify_translation')

        if result.get('critical', False):
            issues = result.get('issues', ['Unknown critical issue'])
            console.print(f"[bold red]❌ CRITICAL translation issues found:[/bold red]")
            for issue in issues:
                console.print(f"[red]   • {issue}[/red]")
            raise ValueError(f"Translation verification failed: {'; '.join(issues)}")

        if not result.get('passed', True):
            issues = result.get('issues', [])
            console.print(f"[yellow]⚠️ Translation quality warnings:[/yellow]")
            for issue in issues:
                console.print(f"[yellow]   • {issue}[/yellow]")
        else:
            console.print(f"[green]✓ Translation verification passed ({len(samples)} samples checked)[/green]")

        return True

    except Exception as e:
        if "Translation verification failed" in str(e):
            raise
        console.print(f"[yellow]⚠️ Verification skipped due to error: {e}[/yellow]")
        return True

# Global duration map for duration-aware translation
_DURATION_MAP = None


def load_duration_map():
    """
    Load duration data from cleaned_chunks.xlsx for duration-aware translation.
    Returns a dict mapping normalized text to duration info.
    """
    global _DURATION_MAP
    if _DURATION_MAP is not None:
        return _DURATION_MAP

    _DURATION_MAP = {}

    if not os.path.exists(_2_CLEANED_CHUNKS):
        console.print("[dim]Duration-aware translation: cleaned_chunks.xlsx not found[/dim]")
        return _DURATION_MAP

    try:
        df = pd.read_excel(_2_CLEANED_CHUNKS)
        if 'text' in df.columns and 'start' in df.columns and 'end' in df.columns:
            for _, row in df.iterrows():
                text = str(row['text']).strip().strip('"')
                duration = float(row['end']) - float(row['start'])
                # Store with normalized text as key
                normalized = ''.join(text.lower().split())
                _DURATION_MAP[normalized] = {
                    'duration': duration,
                    'chars': len(text)
                }
            console.print(f"[cyan]⏱️ Duration-aware translation: loaded {len(_DURATION_MAP)} segments[/cyan]")
        else:
            console.print("[dim]Duration-aware translation: required columns not found[/dim]")
    except Exception as e:
        console.print(f"[dim]Duration-aware translation: failed to load ({e})[/dim]")

    return _DURATION_MAP


def estimate_chunk_duration(chunk_text):
    """
    Estimate total duration for a chunk of text by matching sentences.
    Falls back to character-based estimation if no matches found.

    Args:
        chunk_text: Multi-line text chunk

    Returns:
        dict with 'total_duration' and 'src_chars', or None if not available
    """
    duration_map = load_duration_map()
    if not duration_map:
        return None

    sentences = chunk_text.strip().split('\n')
    total_duration = 0
    total_chars = 0
    matched = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        normalized = ''.join(sentence.lower().split())
        total_chars += len(sentence)

        # Try exact match first
        if normalized in duration_map:
            total_duration += duration_map[normalized]['duration']
            matched += 1
        else:
            # Try partial match (sentence might be split differently)
            for key, value in duration_map.items():
                if normalized in key or key in normalized:
                    # Proportional duration based on character ratio
                    ratio = len(normalized) / len(key) if len(key) > 0 else 1
                    total_duration += value['duration'] * min(ratio, 1.5)
                    matched += 1
                    break

    # If less than half matched, use character-based estimation
    # Average speaking rate: ~15 chars/sec for most languages
    if matched < len(sentences) / 2:
        estimated_duration = total_chars / 15.0
        return {
            'total_duration': estimated_duration,
            'src_chars': total_chars,
            'estimated': True
        }

    if total_duration > 0:
        return {
            'total_duration': total_duration,
            'src_chars': total_chars,
            'estimated': False
        }

    return None


# Function to split text into chunks
def split_chunks_by_chars(chunk_size, max_i, sentences=None):
    """Split text into chunks based on character count, return a list of multi-line text chunks"""
    if sentences is None:
        with open(_3_2_SPLIT_BY_MEANING, "r", encoding="utf-8") as file:
            sentences = file.read().strip().split('\n')

    chunks = []
    chunk = ''
    sentence_count = 0
    for sentence in sentences:
        if len(chunk) + len(sentence + '\n') > chunk_size or sentence_count == max_i:
            chunks.append(chunk.strip())
            chunk = sentence + '\n'
            sentence_count = 1
        else:
            chunk += sentence + '\n'
            sentence_count += 1
    chunks.append(chunk.strip())
    return chunks

def load_segments_df():
    if os.path.exists(_3_2_SEGMENTS):
        return pd.read_excel(_3_2_SEGMENTS)
    return None

# Get context from surrounding chunks
def get_previous_content(chunks, chunk_index):
    return None if chunk_index == 0 else chunks[chunk_index - 1].split('\n')[-3:] # Get last 3 lines
def get_after_content(chunks, chunk_index):
    return None if chunk_index == len(chunks) - 1 else chunks[chunk_index + 1].split('\n')[:2] # Get first 2 lines

# 🔍 Translate a single chunk
def translate_chunk(chunk, chunks, theme_prompt, i):
    things_to_note_prompt = search_things_to_note_in_prompt(chunk)
    previous_content_prompt = get_previous_content(chunks, i)
    after_content_prompt = get_after_content(chunks, i)

    # Calculate duration info for duration-aware translation
    duration_info = estimate_chunk_duration(chunk)

    translation, english_result = translate_lines(
        chunk, previous_content_prompt, after_content_prompt,
        things_to_note_prompt, theme_prompt, i,
        duration_info=duration_info
    )
    return i, english_result, translation

# Add similarity calculation function
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# 🚀 Main function to translate all chunks
@check_file_exists(_4_2_TRANSLATION)
def translate_all():
    console.print("[bold green]Start Translating All...[/bold green]")

    # Pre-load duration map for duration-aware translation
    load_duration_map()

    segments_df = load_segments_df()
    if segments_df is not None and 'text' in segments_df.columns:
        sentences = segments_df['text'].fillna("").astype(str).tolist()
        segments_df = segments_df.reset_index(drop=True)
    else:
        sentences = None
    chunks = split_chunks_by_chars(chunk_size=600, max_i=10, sentences=sentences)
    with open(_4_1_TERMINOLOGY, 'r', encoding='utf-8') as file:
        theme_prompt = json.load(file).get('theme')

    # 🔄 Use concurrent execution for translation
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("[cyan]Translating chunks...", total=len(chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                future = executor.submit(translate_chunk, chunk, chunks, theme_prompt, i)
                futures.append(future)
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                progress.update(task, advance=1)

    results.sort(key=lambda x: x[0])  # Sort results based on original order
    
    # 💾 Save results to lists and Excel file
    src_text, trans_text = [], []
    segment_ids = []
    parent_segment_ids = []
    word_start_idxs = []
    word_end_idxs = []
    segment_cursor = 0
    for i, chunk in enumerate(chunks):
        chunk_lines = chunk.split('\n')
        src_text.extend(chunk_lines)

        if segments_df is not None:
            chunk_seg = segments_df.iloc[segment_cursor:segment_cursor + len(chunk_lines)]
            segment_ids.extend(chunk_seg['segment_id'].tolist())
            if 'parent_segment_id' in chunk_seg.columns:
                parent_segment_ids.extend(chunk_seg['parent_segment_id'].tolist())
            if 'word_start_idx' in chunk_seg.columns:
                word_start_idxs.extend(chunk_seg['word_start_idx'].tolist())
            if 'word_end_idx' in chunk_seg.columns:
                word_end_idxs.extend(chunk_seg['word_end_idx'].tolist())
            segment_cursor += len(chunk_lines)
        
        # Calculate similarity between current chunk and translation results
        chunk_text = ''.join(chunk_lines).lower()
        matching_results = [(r, similar(''.join(r[1].split('\n')).lower(), chunk_text)) 
                          for r in results]
        best_match = max(matching_results, key=lambda x: x[1])
        
        # Check similarity and handle exceptions
        if best_match[1] < 0.9:
            console.print(f"[yellow]Warning: No matching translation found for chunk {i}[/yellow]")
            raise ValueError(f"Translation matching failed (chunk {i})")
        elif best_match[1] < 1.0:
            console.print(f"[yellow]Warning: Similar match found (chunk {i}, similarity: {best_match[1]:.3f})[/yellow]")
            
        # Validate and extend translations
        translations = best_match[0][2].split('\n')

        # Check for empty translations (critical validation)
        for j, trans in enumerate(translations):
            if not trans or not trans.strip():
                console.print(f"[bold red]❌ CRITICAL: Empty translation detected![/bold red]")
                console.print(f"[red]   Chunk {i}, line {j}: source='{chunk_lines[j] if j < len(chunk_lines) else 'N/A'}'[/red]")
                raise ValueError(f"Empty translation in chunk {i}, line {j}. LLM returned empty result. Check output/gpt_log/error.json")

        trans_text.extend(translations)

    # Final validation: source and translation must have same line count
    if len(src_text) != len(trans_text):
        console.print(f"[bold red]❌ CRITICAL: Line count mismatch![/bold red]")
        console.print(f"[red]   Source lines: {len(src_text)}, Translation lines: {len(trans_text)}[/red]")
        raise ValueError(f"Translation alignment failed: {len(src_text)} source lines vs {len(trans_text)} translation lines")

    # Optional LLM-based quality verification (enable with verify_translation: true in config)
    verify_translation_quality(src_text, trans_text)

    # Trim long translation text
    df_text = pd.read_excel(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    df_translate = pd.DataFrame({'Source': src_text, 'Translation': trans_text})
    if segment_ids:
        df_translate.insert(0, 'segment_id', segment_ids)
    if parent_segment_ids:
        df_translate.insert(1, 'parent_segment_id', parent_segment_ids)
    if word_start_idxs:
        df_translate['word_start_idx'] = word_start_idxs
    if word_end_idxs:
        df_translate['word_end_idx'] = word_end_idxs
    subtitle_output_configs = [('trans_subs_for_audio.srt', ['Translation'])]
    df_time = align_timestamp(df_text, df_translate, subtitle_output_configs, output_dir=None, for_display=False)
    console.print(df_time)
    # apply check_len_then_trim to df_time['Translation'], only when duration > MIN_TRIM_DURATION.
    df_time['Translation'] = df_time.apply(lambda x: check_len_then_trim(x['Translation'], x['duration']) if x['duration'] > load_key("min_trim_duration") else x['Translation'], axis=1)
    console.print(df_time)
    
    df_time.to_excel(_4_2_TRANSLATION, index=False)
    console.print("[bold green]✅ Translation completed and results saved.[/bold green]")

if __name__ == '__main__':
    translate_all()
