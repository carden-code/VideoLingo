from core.prompts import generate_shared_prompt, get_prompt_faithfulness, get_prompt_expressiveness
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich import box
from core.utils import *
from core.utils.anchor_utils import build_anchor_requirements, build_anchor_constraints, validate_anchor_requirements
console = Console()

def valid_translate_result(result: dict, required_keys: list, required_sub_keys: list):
    # Check for the required key
    if not all(key in result for key in required_keys):
        return {"status": "error", "message": f"Missing required key(s): {', '.join(set(required_keys) - set(result.keys()))}"}

    # Check for required sub-keys in all items
    for key in result:
        if not all(sub_key in result[key] for sub_key in required_sub_keys):
            return {"status": "error", "message": f"Missing required sub-key(s) in item {key}: {', '.join(set(required_sub_keys) - set(result[key].keys()))}"}

        # Check for empty translations (critical validation)
        for sub_key in required_sub_keys:
            value = result[key].get(sub_key, '')
            if not value or not str(value).strip():
                return {"status": "error", "message": f"Empty translation in item {key}, field '{sub_key}'"}

    return {"status": "success", "message": "Translation completed"}

def translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt, index=0, duration_info=None, terms=None):
    """
    Translate lines with optional duration awareness for video dubbing.

    Args:
        lines: Text lines to translate (newline-separated)
        previous_content_prompt: Context from previous chunk
        after_cotent_prompt: Context from next chunk
        things_to_note_prompt: Terminology notes
        summary_prompt: Video summary
        index: Chunk index for logging
        duration_info: Optional dict with duration info for dubbing:
            - total_duration: Total seconds for this chunk
            - src_chars: Character count of source text
            - line_durations: Optional per-line durations
            - line_chars: Optional per-line character counts
        terms: Optional terminology list for anchor validation
    """
    shared_prompt = generate_shared_prompt(previous_content_prompt, after_cotent_prompt, summary_prompt, things_to_note_prompt)

    lines_list = lines.split('\n')
    anchors_by_line = [
        build_anchor_requirements(line, terms or []) for line in lines_list
    ]

    def slice_duration_info(info, start, end, sub_lines):
        if not info:
            return None
        line_durations = info.get("line_durations")
        line_chars = info.get("line_chars")
        if not line_durations:
            return info
        sub_durations = line_durations[start:end]
        sub_chars = line_chars[start:end] if line_chars else [len(line) for line in sub_lines]
        return {
            "total_duration": sum(sub_durations),
            "src_chars": sum(sub_chars),
            "line_durations": sub_durations,
            "line_chars": sub_chars
        }

    def anchor_constraints_for(sub_lines, sub_anchors):
        return build_anchor_constraints(sub_lines, sub_anchors)

    def validate_anchor_result(result, sub_anchors, field_name):
        issues = []
        for idx, anchors in enumerate(sub_anchors, start=1):
            if not anchors:
                continue
            translation = result[str(idx)].get(field_name, "")
            missing = validate_anchor_requirements(translation, anchors)
            if missing:
                issues.append(f"{idx}: {', '.join(missing)}")
        return issues

    def merge_results(left, right):
        merged = {}
        idx = 1
        for part in (left, right):
            for key in sorted(part.keys(), key=lambda x: int(x)):
                merged[str(idx)] = part[key]
                idx += 1
        return merged

    def translate_step_with_escalation(start, end, build_prompt, field_name, step_name):
        sub_lines = lines_list[start:end]
        sub_anchors = anchors_by_line[start:end]

        def valid_result(response_data):
            return valid_translate_result(response_data, [str(i) for i in range(1, len(sub_lines) + 1)], [field_name])

        last_error = None
        for attempt in range(2):
            strict = attempt == 1
            prompt = build_prompt(start, end, strict)
            try:
                result = ask_gpt(prompt + attempt * " ", resp_type='json', valid_def=valid_result, log_title=f'translate_{step_name}')
                missing = validate_anchor_result(result, sub_anchors, field_name)
                if missing:
                    raise ValueError(f"Anchor validation failed: {missing}")
                return result
            except Exception as exc:
                last_error = exc
                console.print(f'[yellow]⚠️ {step_name.capitalize()} retry {attempt + 1} failed: {exc}[/yellow]')

        if len(sub_lines) <= 1:
            raise ValueError(f'[red]❌ {step_name.capitalize()} translation of block {index} failed: {last_error}[/red]')

        mid = start + len(sub_lines) // 2
        left_result = translate_step_with_escalation(start, mid, build_prompt, field_name, step_name)
        right_result = translate_step_with_escalation(mid, end, build_prompt, field_name, step_name)
        return merge_results(left_result, right_result)

    ## Step 1: Faithful to the Original Text
    def build_faith_prompt(start, end, strict):
        sub_lines = lines_list[start:end]
        sub_anchors = anchors_by_line[start:end]
        sub_text = "\n".join(sub_lines)
        sub_duration = slice_duration_info(duration_info, start, end, sub_lines)
        anchor_constraints = anchor_constraints_for(sub_lines, sub_anchors)
        return get_prompt_faithfulness(sub_text, shared_prompt, sub_duration, anchor_constraints, strict)

    faith_result = translate_step_with_escalation(
        0,
        len(lines_list),
        build_faith_prompt,
        "direct",
        "faithfulness"
    )

    for i in faith_result:
        faith_result[i]["direct"] = faith_result[i]["direct"].replace('\n', ' ')

    # If reflect_translate is False or not set, use faithful translation directly
    reflect_translate = load_key('reflect_translate')
    if not reflect_translate:
        # If reflect_translate is False or not set, use faithful translation directly
        translate_result = "\n".join([faith_result[i]["direct"].strip() for i in faith_result])
        
        table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
        table.add_column("Translations", style="bold")
        for i, key in enumerate(faith_result):
            table.add_row(f"[cyan]Origin:  {faith_result[key]['origin']}[/cyan]")
            table.add_row(f"[magenta]Direct:  {faith_result[key]['direct']}[/magenta]")
            if i < len(faith_result) - 1:
                table.add_row("[yellow]" + "-" * 50 + "[/yellow]")
        
        console.print(table)
        return translate_result, lines

    ## Step 2: Express Smoothly
    def slice_result(result, start, end):
        subset = {}
        offset = 0
        for idx in range(start, end):
            subset[str(offset + 1)] = result[str(idx + 1)]
            offset += 1
        return subset

    def build_express_prompt(start, end, strict):
        sub_lines = lines_list[start:end]
        sub_anchors = anchors_by_line[start:end]
        sub_text = "\n".join(sub_lines)
        sub_duration = slice_duration_info(duration_info, start, end, sub_lines)
        anchor_constraints = anchor_constraints_for(sub_lines, sub_anchors)
        sub_faith = slice_result(faith_result, start, end)
        return get_prompt_expressiveness(sub_faith, sub_text, shared_prompt, sub_duration, anchor_constraints, strict)

    express_result = translate_step_with_escalation(
        0,
        len(lines_list),
        build_express_prompt,
        "free",
        "expressiveness"
    )

    table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
    table.add_column("Translations", style="bold")
    for i, key in enumerate(express_result):
        table.add_row(f"[cyan]Origin:  {faith_result[key]['origin']}[/cyan]")
        table.add_row(f"[magenta]Direct:  {faith_result[key]['direct']}[/magenta]")
        table.add_row(f"[green]Free:    {express_result[key]['free']}[/green]")
        if i < len(express_result) - 1:
            table.add_row("[yellow]" + "-" * 50 + "[/yellow]")

    console.print(table)

    translate_result = "\n".join([express_result[i]["free"].replace('\n', ' ').strip() for i in express_result])

    if len(lines.split('\n')) != len(translate_result.split('\n')):
        console.print(Panel(f'[red]❌ Translation of block {index} failed, Length Mismatch, Please check `output/gpt_log/translate_expressiveness.json`[/red]'))
        raise ValueError(f'Origin ···{lines}···,\nbut got ···{translate_result}···')

    return translate_result, lines


if __name__ == '__main__':
    # test e.g.
    lines = '''All of you know Andrew Ng as a famous computer science professor at Stanford.
He was really early on in the development of neural networks with GPUs.
Of course, a creator of Coursera and popular courses like deeplearning.ai.
Also the founder and creator and early lead of Google Brain.'''
    previous_content_prompt = None
    after_cotent_prompt = None
    things_to_note_prompt = None
    summary_prompt = None
    translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt)
