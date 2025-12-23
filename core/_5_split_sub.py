import pandas as pd
from typing import List, Tuple
import concurrent.futures

from core._3_2_split_meaning import split_sentence
from core.prompts import get_align_prompt
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from core.utils import *
from core.utils.segment_utils import encode_parent_list, parse_parent_list
from core.utils.span_utils import map_sentences_to_spans
from core.utils.models import *
console = Console()

# ! You can modify your own weights here
# Chinese and Japanese 2.5 characters, Korean 2 characters, Thai 1.5 characters, full-width symbols 2 characters, other English-based and half-width symbols 1 character
def calc_len(text: str) -> float:
    text = str(text) # force convert
    def char_weight(char):
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # Chinese and Japanese
            return 1.75
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # Korean
            return 1.5
        elif 0x0E00 <= code <= 0x0E7F:  # Thai
            return 1
        elif 0xFF01 <= code <= 0xFF5E:  # full-width symbols
            return 1.75
        else:  # other characters (e.g. English and half-width symbols)
            return 1

    return sum(char_weight(char) for char in text)

def align_subs(src_sub: str, tr_sub: str, src_part: str) -> Tuple[List[str], List[str], str]:
    align_prompt = get_align_prompt(src_sub, tr_sub, src_part)
    src_parts = src_part.split('\n')
    num_parts = len(src_parts)
    def valid_align(response_data):
        if 'align' not in response_data:
            return {"status": "error", "message": "Missing required key: `align`"}
        if 'analysis' not in response_data:
            return {"status": "error", "message": "Missing required key: `analysis`"}
        if len(response_data['align']) != num_parts:
            return {"status": "error", "message": f"Align parts mismatch: expected {num_parts}, got {len(response_data['align'])}"}
        for idx, item in enumerate(response_data['align']):
            src_key = f"src_part_{idx + 1}"
            tgt_key = f"target_part_{idx + 1}"
            missing = [key for key in (src_key, tgt_key) if key not in item]
            if missing:
                return {"status": "error", "message": f"Missing required key(s) in align item {idx + 1}: {missing}"}
        return {"status": "success", "message": "Align completed"}
    align_item_props = {
        "^src_part_\\d+$": {"type": "string"},
        "^target_part_\\d+$": {"type": "string"}
    }
    schema = {
        "name": "align_subs",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string"},
                "align": {
                    "type": "array",
                    "minItems": num_parts,
                    "maxItems": num_parts,
                    "items": {
                        "type": "object",
                        "patternProperties": align_item_props,
                        "minProperties": 2,
                        "maxProperties": 2,
                        "additionalProperties": False
                    }
                }
            },
            "required": ["analysis", "align"],
            "additionalProperties": False
        }
    }
    parsed = ask_gpt(
        align_prompt,
        resp_type='json',
        valid_def=valid_align,
        log_title='align_subs',
        json_schema=schema
    )
    align_data = parsed['align']
    tr_parts = [item[f'target_part_{i+1}'].strip() for i, item in enumerate(align_data)]
    
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)
    tr_remerged = joiner.join(tr_parts)
    
    table = Table(title="🔗 Aligned parts")
    table.add_column("Language", style="cyan")
    table.add_column("Parts", style="magenta")
    table.add_row("SRC_LANG", "\n".join(src_parts))
    table.add_row("TARGET_LANG", "\n".join(tr_parts))
    console.print(table)
    
    return src_parts, tr_parts, tr_remerged

def merge_parent_list(parent_ids, segment_id):
    items = list(parent_ids) if parent_ids else []
    if segment_id and segment_id not in items:
        items.append(segment_id)
    return tuple(items)

def split_align_subs(src_lines: List[str], tr_lines: List[str], segment_ids: List[str], parent_segment_ids: List[tuple]):
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    remerged_tr_lines = tr_lines.copy()
    remerged_segment_ids = segment_ids.copy()
    remerged_parent_ids = parent_segment_ids.copy()
    segment_id_lines = segment_ids.copy()
    split_parent_lines = parent_segment_ids.copy()
    
    to_split = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src, tr = str(src), str(tr)
        if len(src) > MAX_SUB_LENGTH or calc_len(tr) * TARGET_SUB_MULTIPLIER > MAX_SUB_LENGTH:
            to_split.append(i)
            table = Table(title=f"📏 Line {i} needs to be split")
            table.add_column("Type", style="cyan")
            table.add_column("Content", style="magenta")
            table.add_row("Source Line", src)
            table.add_row("Target Line", tr)
            console.print(table)
    
    @except_handler("Error in split_align_subs", retry=2)
    def process(i):
        console.print(f"[dim]🔧 Processing line {i}...[/dim]")
        split_src = split_sentence(src_lines[i], num_parts=2).strip()
        console.print(f"[dim]🔧 Line {i}: split done, aligning...[/dim]")
        src_parts, tr_parts, tr_remerged = align_subs(src_lines[i], tr_lines[i], split_src)
        console.print(f"[dim]🔧 Line {i}: align done[/dim]")
        src_lines[i] = src_parts
        tr_lines[i] = tr_parts
        remerged_tr_lines[i] = tr_remerged
        parent_id = segment_id_lines[i] or f"seg_{i + 1:04d}"
        part_ids = [f"{parent_id}_s{j + 1}" for j in range(len(src_parts))]
        segment_id_lines[i] = part_ids
        parent_for_parts = merge_parent_list(split_parent_lines[i], parent_id)
        split_parent_lines[i] = [parent_for_parts] * len(src_parts)
    
    console.print(f"[cyan]🔄 Starting {len(to_split)} split tasks (max_workers={load_key('max_workers')})...[/cyan]")
    with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
        list(executor.map(process, to_split))  # list() forces completion
    console.print(f"[green]✅ All {len(to_split)} split tasks completed[/green]")
    
    def flatten(values):
        return [item for sublist in values for item in (sublist if isinstance(sublist, list) else [sublist])]

    # Flatten `src_lines` and `tr_lines`
    src_lines = flatten(src_lines)
    tr_lines = flatten(tr_lines)
    split_segment_ids = flatten(segment_id_lines)
    split_parent_ids = flatten(split_parent_lines)
    
    return src_lines, tr_lines, remerged_tr_lines, split_segment_ids, split_parent_ids, remerged_segment_ids, remerged_parent_ids

def split_for_sub_main():
    console.print("[bold green]🚀 Start splitting subtitles...[/bold green]")
    
    df = pd.read_excel(_4_2_TRANSLATION)
    src = df['Source'].tolist()
    trans = df['Translation'].tolist()
    if 'segment_id' in df.columns:
        segment_ids = [
            sid if pd.notna(sid) and str(sid).strip() else None
            for sid in df['segment_id'].tolist()
        ]
    else:
        segment_ids = [None] * len(df)
    segment_ids = [
        sid if sid is not None else f"seg_{i + 1:04d}"
        for i, sid in enumerate(segment_ids)
    ]
    if 'parent_segment_id' in df.columns:
        parent_segment_ids = [
            tuple(parse_parent_list(pid)) for pid in df['parent_segment_id'].tolist()
        ]
    else:
        parent_segment_ids = [()] * len(df)
    
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    
    for attempt in range(3):  # 多次切割
        console.print(Panel(f"🔄 Split attempt {attempt + 1}", expand=False))
        split_src, split_trans, remerged, split_segment_ids, split_parent_ids, remerged_segment_ids, remerged_parent_ids = split_align_subs(
            src.copy(), trans, segment_ids, parent_segment_ids
        )
        
        # 检查是否所有字幕都符合长度要求
        if all(len(src) <= MAX_SUB_LENGTH for src in split_src) and \
           all(calc_len(tr) * TARGET_SUB_MULTIPLIER <= MAX_SUB_LENGTH for tr in split_trans):
            break
        
        # 更新源数据继续下一轮分割
        src, trans = split_src, split_trans
        segment_ids = split_segment_ids
        parent_segment_ids = split_parent_ids

    # 确保二者有相同的长度，防止报错
    if len(src) > len(remerged):
        remerged += [None] * (len(src) - len(remerged))
        remerged_segment_ids += [None] * (len(src) - len(remerged_segment_ids))
        remerged_parent_ids += [None] * (len(src) - len(remerged_parent_ids))
    elif len(remerged) > len(src):
        src += [None] * (len(remerged) - len(src))
        segment_ids += [None] * (len(remerged) - len(segment_ids))
        parent_segment_ids += [None] * (len(remerged) - len(parent_segment_ids))

    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)

    df_words = pd.read_excel(_2_CLEANED_CHUNKS)
    df_words['text'] = df_words['text'].str.strip('"').str.strip()
    word_list = df_words['text'].tolist()

    def build_spans(sentences):
        def is_empty(value):
            return value is None or pd.isna(value) or not str(value).strip()

        non_empty = [s for s in sentences if not is_empty(s)]
        spans = map_sentences_to_spans(non_empty, word_list, joiner)
        span_iter = iter(spans)
        results = []
        for sentence in sentences:
            if not is_empty(sentence):
                results.append(next(span_iter))
            else:
                results.append((None, None))
        return results

    split_spans = build_spans(split_src)
    remerged_spans = build_spans(src)

    df_split = pd.DataFrame({
        'segment_id': split_segment_ids,
        'parent_segment_id': [encode_parent_list(p) for p in split_parent_ids],
        'Source': split_src,
        'Translation': split_trans
    })
    df_split['word_start_idx'] = [s[0] for s in split_spans]
    df_split['word_end_idx'] = [s[1] for s in split_spans]
    df_split.to_excel(_5_SPLIT_SUB, index=False)

    df_remerged = pd.DataFrame({
        'segment_id': remerged_segment_ids,
        'parent_segment_id': [encode_parent_list(p) for p in remerged_parent_ids],
        'Source': src,
        'Translation': remerged
    })
    df_remerged['word_start_idx'] = [s[0] for s in remerged_spans]
    df_remerged['word_end_idx'] = [s[1] for s in remerged_spans]
    df_remerged.to_excel(_5_REMERGED, index=False)

if __name__ == '__main__':
    split_for_sub_main()
