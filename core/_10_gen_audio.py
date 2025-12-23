import json
import os
import time
import shutil
import subprocess
from typing import Tuple
from threading import Lock

import pandas as pd
import requests
from pydub import AudioSegment
from rich.console import Console
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils import *
from core.utils.models import *
from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.tts_main import tts_main
from core.tts_backend.estimate_duration import init_estimator, estimate_duration
from core.prompts import get_subtitle_trim_prompt
from core.utils.anchor_utils import (
    load_terms,
    build_anchor_requirements,
    build_anchor_constraints,
    validate_anchor_requirements
)
from core.utils.time_stretch import adjust_audio_speed, get_stretch_backend

console = Console()


def free_vram_for_tts():
    """
    Free VRAM before TTS generation by unloading Ollama model and clearing CUDA cache.

    This is necessary because Ollama keeps the LLM model (~10GB) in VRAM after translation,
    which conflicts with Chatterbox Model Pool (~13-15GB) on limited VRAM GPUs.
    """
    # 1. Unload Ollama model if using local Ollama
    try:
        base_url = load_key("api.base_url")
        model_name = load_key("api.model")

        # Check if using local Ollama (localhost:11434)
        if "localhost:11434" in base_url or "127.0.0.1:11434" in base_url:
            rprint(f"[cyan]🔄 Unloading Ollama model '{model_name}' to free VRAM...[/cyan]")
            response = requests.post(
                f"{base_url}/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=30
            )
            if response.status_code == 200:
                rprint("[green]✓ Ollama model unloaded successfully[/green]")
            else:
                rprint(f"[yellow]⚠ Ollama unload returned status {response.status_code}[/yellow]")
    except Exception as e:
        # Non-critical error - continue anyway
        rprint(f"[dim]Ollama unload skipped: {e}[/dim]")

    # 2. Clear CUDA cache
    try:
        import torch  # Lazy import to avoid dependency on CPU-only setups
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            rprint("[green]✓ CUDA cache cleared[/green]")
    except Exception as e:
        rprint(f"[dim]CUDA cache clear skipped: {e}[/dim]")

TEMP_FILE_TEMPLATE = f"{_AUDIO_TMP_DIR}/{{}}_temp.wav"
OUTPUT_FILE_TEMPLATE = f"{_AUDIO_SEGS_DIR}/{{}}.wav"
WARMUP_SIZE = 5
ESTIMATOR = None
LOG_LOCK = Lock()
DURATION_FIX_LOG = "output/log/duration_fixes.jsonl"


def get_duration_estimator():
    global ESTIMATOR
    if ESTIMATOR is None:
        ESTIMATOR = init_estimator()
    return ESTIMATOR


def log_duration_fix(payload: dict):
    os.makedirs(os.path.dirname(DURATION_FIX_LOG), exist_ok=True)
    with LOG_LOCK:
        with open(DURATION_FIX_LOG, "a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=True) + "\n")


def allocate_line_targets(lines, total_target):
    if not lines:
        return []
    estimator = get_duration_estimator()
    estimates = [estimate_duration(line, estimator) for line in lines]
    total_est = sum(estimates)
    if total_est <= 0:
        return [total_target / len(lines)] * len(lines)
    return [total_target * (est / total_est) for est in estimates]


def shorten_line_with_anchors(line, src_line, target_duration, terms, strict=False):
    anchors = build_anchor_requirements(src_line or "", terms)
    constraints = build_anchor_constraints([src_line or ""], [anchors]) if anchors else ""
    prompt = get_subtitle_trim_prompt(
        line,
        target_duration,
        anchor_constraints=constraints,
        strict=strict
    )
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
    response = ask_gpt(prompt, resp_type='json', log_title='duration_fix', json_schema=trim_schema)
    shortened = str(response.get('result', '')).strip()
    if not shortened:
        return None
    if anchors:
        missing = validate_anchor_requirements(shortened, anchors)
        if missing:
            return None
    return shortened


def split_text_simple(text, parts=2):
    text = str(text).strip()
    if parts <= 1 or not text:
        return [text]
    if parts != 2:
        size = max(1, len(text) // parts)
        return [text[i:i + size].strip() for i in range(0, len(text), size)]
    punctuation = [',', '.', ';', ':', '!', '?', '，', '。', '；', '：', '！', '？']
    mid = len(text) // 2
    split_at = None
    for i in range(mid, len(text)):
        if text[i] in punctuation:
            split_at = i + 1
            break
    if split_at is None:
        for i in range(mid, -1, -1):
            if text[i] in punctuation:
                split_at = i + 1
                break
    if split_at is None and " " in text:
        left = text[:mid].rfind(" ")
        right = text[mid:].find(" ")
        candidates = []
        if left != -1:
            candidates.append(left)
        if right != -1:
            candidates.append(mid + right)
        if candidates:
            split_at = min(candidates, key=lambda x: abs(x - mid))
    if split_at is None:
        split_at = mid
    left_part = text[:split_at].strip()
    right_part = text[split_at:].strip()
    if not left_part or not right_part:
        return [text]
    return [left_part, right_part]

def parse_df_srt_time(time_str: str) -> float:
    """Convert SRT time format to seconds"""
    hours, minutes, seconds = time_str.strip().split(':')
    seconds, milliseconds = seconds.split('.')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def process_row(row: pd.Series, tasks_df: pd.DataFrame, terms: list, max_speed: float) -> Tuple[int, float, list, list]:
    """Helper function for processing single row data"""
    number = row['number']
    lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
    src_lines = eval(row['src_lines']) if isinstance(row.get('src_lines', []), str) else row.get('src_lines', [])
    if not isinstance(src_lines, list):
        src_lines = []
    total_target = row.get('tol_dur', row.get('duration', 0))
    line_targets = allocate_line_targets(lines, total_target) if total_target else []
    out_lines = []
    out_src_lines = []
    real_dur = 0
    out_index = 0
    for line_index, line in enumerate(lines):
        src_line = src_lines[line_index] if line_index < len(src_lines) else ""
        target_duration = line_targets[line_index] if line_index < len(line_targets) else total_target
        temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{out_index}")
        tts_main(line, temp_file, number, tasks_df)
        duration = get_audio_duration(temp_file)

        if target_duration and duration > target_duration * max_speed:
            log_duration_fix({
                "event": "overrun",
                "number": number,
                "line_index": line_index,
                "target_duration": target_duration,
                "duration": duration,
                "text": line,
                "source": src_line
            })
            shortened = shorten_line_with_anchors(line, src_line, target_duration, terms, strict=False)
            if shortened is None:
                shortened = shorten_line_with_anchors(line, src_line, target_duration, terms, strict=True)
            if shortened and shortened != line:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                line = shortened
                tts_main(line, temp_file, number, tasks_df)
                duration = get_audio_duration(temp_file)
                log_duration_fix({
                    "event": "shorten_success",
                    "number": number,
                    "line_index": line_index,
                    "target_duration": target_duration,
                    "duration": duration,
                    "text": line,
                    "source": src_line
                })
            elif shortened is None:
                log_duration_fix({
                    "event": "shorten_failed",
                    "number": number,
                    "line_index": line_index,
                    "target_duration": target_duration,
                    "duration": duration,
                    "text": line,
                    "source": src_line
                })

        if target_duration and duration > target_duration * max_speed:
            split_parts = split_text_simple(line, parts=2)
            if len(split_parts) > 1:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                src_parts = split_text_simple(src_line, parts=len(split_parts))
                if len(src_parts) != len(split_parts):
                    src_parts = [src_line] * len(split_parts)
                part_durations = []
                for offset, part in enumerate(split_parts):
                    part_file = TEMP_FILE_TEMPLATE.format(f"{number}_{out_index + offset}")
                    tts_main(part, part_file, number, tasks_df)
                    part_durations.append(get_audio_duration(part_file))
                log_duration_fix({
                    "event": "split_applied",
                    "number": number,
                    "line_index": line_index,
                    "target_duration": target_duration,
                    "durations": part_durations,
                    "parts": split_parts,
                    "source_parts": src_parts
                })
                out_lines.extend(split_parts)
                out_src_lines.extend(src_parts)
                real_dur += sum(part_durations)
                out_index += len(split_parts)
                continue

        out_lines.append(line)
        out_src_lines.append(src_line)
        real_dur += duration
        out_index += 1

    return number, real_dur, out_lines, out_src_lines

def generate_tts_audio(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Generate TTS audio sequentially and calculate actual duration"""
    tasks_df['real_dur'] = 0
    rprint("[bold green]🎯 Starting TTS audio generation...[/bold green]")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]🔄 Generating TTS audio...", total=len(tasks_df))
        
        # warm up for first 5 rows
        warmup_size = min(WARMUP_SIZE, len(tasks_df))
        terms = load_terms()
        max_speed = load_key("speed_factor.max")

        for idx, row in tasks_df.head(warmup_size).iterrows():
            try:
                number, real_dur, out_lines, out_src_lines = process_row(row, tasks_df, terms, max_speed)
                tasks_df.at[idx, 'real_dur'] = real_dur
                tasks_df.at[idx, 'lines'] = out_lines
                if 'src_lines' in tasks_df.columns:
                    tasks_df.at[idx, 'src_lines'] = out_src_lines
                progress.advance(task)
            except Exception as e:
                rprint(f"[red]❌ Error in warmup: {str(e)}[/red]")
                raise e
        
        # for gpt_sovits, chatterbox_tts and cosyvoice3, do not use parallel to avoid GPU conflicts
        tts_method = load_key("tts_method")
        max_workers = 1 if tts_method in ("gpt_sovits", "chatterbox_tts", "cosyvoice3") else load_key("max_workers")
        # parallel processing for remaining tasks
        if len(tasks_df) > warmup_size:
            remaining_tasks = tasks_df.iloc[warmup_size:].copy()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_row, row, tasks_df.copy(), terms, max_speed)
                    for _, row in remaining_tasks.iterrows()
                ]
                
                for future in as_completed(futures):
                    try:
                        number, real_dur, out_lines, out_src_lines = future.result()
                        idx = tasks_df.index[tasks_df['number'] == number][0]
                        tasks_df.at[idx, 'real_dur'] = real_dur
                        tasks_df.at[idx, 'lines'] = out_lines
                        if 'src_lines' in tasks_df.columns:
                            tasks_df.at[idx, 'src_lines'] = out_src_lines
                        progress.advance(task)
                    except Exception as e:
                        rprint(f"[red]❌ Error: {str(e)}[/red]")
                        raise e

    rprint("[bold green]✨ TTS audio generation completed![/bold green]")
    return tasks_df

def process_chunk(chunk_df: pd.DataFrame, accept: float, min_speed: float) -> tuple[float, bool]:
    """Process audio chunk and calculate speed factor"""
    chunk_durs = chunk_df['real_dur'].sum()
    tol_durs = chunk_df['tol_dur'].sum()
    durations = tol_durs - chunk_df.iloc[-1]['tolerance']
    all_gaps = chunk_df['gap'].sum() - chunk_df.iloc[-1]['gap']
    
    keep_gaps = True
    speed_var_error = 0.1

    if (chunk_durs + all_gaps) / accept < durations:
        speed_factor = max(min_speed, (chunk_durs + all_gaps) / (durations-speed_var_error))
    elif chunk_durs / accept < durations:
        speed_factor = max(min_speed, chunk_durs / (durations-speed_var_error))
        keep_gaps = False
    elif (chunk_durs + all_gaps) / accept < tol_durs:
        speed_factor = max(min_speed, (chunk_durs + all_gaps) / (tol_durs-speed_var_error))
    else:
        speed_factor = chunk_durs / (tol_durs-speed_var_error)
        keep_gaps = False
        
    return round(speed_factor, 3), keep_gaps

def merge_chunks(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Merge audio chunks and adjust timeline"""
    rprint("[bold blue]🔄 Starting audio chunks processing...[/bold blue]")
    accept = load_key("speed_factor.accept")
    min_speed = load_key("speed_factor.min")
    chunk_start = 0
    
    tasks_df['new_sub_times'] = None
    
    for index, row in tasks_df.iterrows():
        if row['cut_off'] == 1:
            chunk_df = tasks_df.iloc[chunk_start:index+1].reset_index(drop=True)
            speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)
            
            # 🎯 Step1: Start processing new timeline
            chunk_start_time = parse_df_srt_time(chunk_df.iloc[0]['start_time'])
            chunk_end_time = parse_df_srt_time(chunk_df.iloc[-1]['end_time']) + chunk_df.iloc[-1]['tolerance'] # 加上tolerance才是这一块的结束
            cur_time = chunk_start_time
            for i, row in chunk_df.iterrows():
                # If i is not 0, which is not the first row of the chunk, cur_time needs to be added with the gap of the previous row, remember to divide by speed_factor
                if i != 0 and keep_gaps:
                    cur_time += chunk_df.iloc[i-1]['gap']/speed_factor
                new_sub_times = []
                number = row['number']
                lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
                for line_index, line in enumerate(lines):
                    # 🔄 Step2: Start speed change and save as OUTPUT_FILE_TEMPLATE
                    temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
                    output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
                    adjust_audio_speed(temp_file, output_file, speed_factor)
                    ad_dur = get_audio_duration(output_file)
                    new_sub_times.append([cur_time, cur_time+ad_dur])
                    cur_time += ad_dur
                # 🔄 Step3: Find corresponding main DataFrame index and update new_sub_times
                main_df_idx = tasks_df[tasks_df['number'] == row['number']].index[0]
                tasks_df.at[main_df_idx, 'new_sub_times'] = new_sub_times
                # 🎯 Step4: Choose emoji based on speed_factor and accept comparison
                emoji = "⚡" if speed_factor <= accept else "⚠️"
                rprint(f"[cyan]{emoji} Processed chunk {chunk_start} to {index} with speed factor {speed_factor}[/cyan]")
            # 🔄 Step5: Check if the last row exceeds the range
            if cur_time > chunk_end_time:
                time_diff = cur_time - chunk_end_time
                if time_diff <= 0.6:  # If exceeding time is within 0.6 seconds, truncate the last audio
                    rprint(f"[yellow]⚠️ Chunk {chunk_start} to {index} exceeds by {time_diff:.3f}s, truncating last audio[/yellow]")
                    # Get the last audio file
                    last_number = tasks_df.iloc[index]['number']
                    last_lines = eval(tasks_df.iloc[index]['lines']) if isinstance(tasks_df.iloc[index]['lines'], str) else tasks_df.iloc[index]['lines']
                    last_line_index = len(last_lines) - 1
                    last_file = OUTPUT_FILE_TEMPLATE.format(f"{last_number}_{last_line_index}")
                    
                    # Calculate the duration to keep
                    audio = AudioSegment.from_wav(last_file)
                    original_duration = len(audio) / 1000  # Convert to seconds
                    new_duration = original_duration - time_diff
                    trimmed_audio = audio[:(new_duration * 1000)]  # pydub uses milliseconds
                    trimmed_audio.export(last_file, format="wav")
                    
                    # Update the last timestamp
                    last_times = tasks_df.at[index, 'new_sub_times']
                    last_times[-1][1] = chunk_end_time
                    tasks_df.at[index, 'new_sub_times'] = last_times
                else:
                    raise Exception(f"Chunk {chunk_start} to {index} exceeds the chunk end time {chunk_end_time:.2f} seconds with current time {cur_time:.2f} seconds")
            chunk_start = index+1
    
    rprint("[bold green]✅ Audio chunks processing completed![/bold green]")
    return tasks_df

def gen_audio() -> None:
    """Main function: Generate audio and process timeline"""
    rprint("[bold magenta]🚀 Starting audio generation process...[/bold magenta]")

    # Show time-stretch backend
    backend = get_stretch_backend()
    if backend == "rubberband":
        rprint("[green]🎚️ Time-stretch: rubberband (high quality)[/green]")
    else:
        rprint("[yellow]🎚️ Time-stretch: ffmpeg atempo (install pyrubberband for better quality)[/yellow]")

    # 🧹 Step0: Free VRAM from Ollama before loading TTS models
    free_vram_for_tts()

    # 🎯 Step1: Create necessary directories
    os.makedirs(_AUDIO_TMP_DIR, exist_ok=True)
    os.makedirs(_AUDIO_SEGS_DIR, exist_ok=True)
    
    # 📝 Step2: Load task file
    tasks_df = pd.read_excel(_8_1_AUDIO_TASK)
    rprint("[green]📊 Loaded task file successfully[/green]")
    
    # 🔊 Step3: Generate TTS audio
    tasks_df = generate_tts_audio(tasks_df)
    
    # 🔄 Step4: Merge audio chunks
    tasks_df = merge_chunks(tasks_df)
    
    # 💾 Step5: Save results
    tasks_df.to_excel(_8_1_AUDIO_TASK, index=False)
    rprint("[bold green]🎉 Audio generation completed successfully![/bold green]")

if __name__ == "__main__":
    gen_audio()
