# VideoLingo Quality Improvements - Implementation Plan

> План реализации критических улучшений качества пайплайна
> Приоритеты: P0 (must do) → P1 (next) → P2 (nice to have)

---

## P0.1 — Overlap + дедупликация на границах чанков ✅ DONE

> **PR:** https://github.com/carden-code/VideoLingo/pull/22

### Проблема
Сейчас в `audio_preprocess.py:split_audio()` чанки режутся последовательно без overlap.
При склейке в `_2_asr.py` результаты просто `.extend()` без проверки дублей.

### Решение

**Файл:** `core/asr_backend/audio_preprocess.py`

```python
def split_audio(audio_file: str, target_len: float = 30*60, win: float = 60,
                overlap: float = 1.5) -> List[Tuple[float, float]]:
    """
    Разрезает аудио на чанки с overlap для защиты от обрезанных слов на границах.

    Args:
        overlap: Перекрытие между чанками в секундах (default 1.5s)

    Returns:
        List of (start, end) tuples with overlapping regions
    """
    audio = AudioSegment.from_file(audio_file)
    duration = float(mediainfo(audio_file)["duration"])

    if duration <= target_len + win:
        return [(0, duration)]

    segments, pos = [], 0.0
    safe_margin = 0.5

    while pos < duration:
        if duration - pos <= target_len:
            segments.append((pos, duration))
            break

        threshold = pos + target_len
        ws, we = int((threshold - win) * 1000), int((threshold + win) * 1000)

        silence_regions = detect_silence(audio[ws:we], min_silence_len=500, silence_thresh=-30)
        silence_regions = [(s/1000 + (threshold - win), e/1000 + (threshold - win))
                          for s, e in silence_regions]

        valid_regions = [
            (start, end) for start, end in silence_regions
            if (end - start) >= (safe_margin * 2) and threshold <= start + safe_margin <= threshold + win
        ]

        if valid_regions:
            split_at = valid_regions[0][0] + safe_margin
        else:
            split_at = threshold

        # Добавляем overlap к концу текущего чанка (перекрытие с началом следующего)
        chunk_end = min(split_at + overlap, duration)
        segments.append((pos, chunk_end))

        # Следующий чанк начинается БЕЗ overlap (overlap только в конце)
        pos = split_at

    rprint(f"[green]🎙️ Audio split: {len(segments)} segments with {overlap}s overlap[/green]")
    return segments
```

**Файл:** `core/_2_asr.py` — добавить дедупликацию

```python
from core.utils import *
from core.asr_backend.demucs_vl import demucs_audio
from core.asr_backend.audio_preprocess import (
    process_transcription, convert_video_to_audio, split_audio,
    save_results, normalize_audio_volume, deduplicate_segments
)
from core._1_ytdlp import find_video_files
from core.utils.models import *

@check_file_exists(_2_CLEANED_CHUNKS)
def transcribe():
    video_file = find_video_files()
    convert_video_to_audio(video_file)

    if load_key("demucs"):
        demucs_audio()
        vocal_audio = normalize_audio_volume(_VOCAL_AUDIO_FILE, _VOCAL_AUDIO_FILE, format="mp3")
    else:
        vocal_audio = _RAW_AUDIO_FILE

    # Получаем сегменты с overlap
    segments = split_audio(_RAW_AUDIO_FILE, overlap=1.5)

    from core.asr_backend.whisperX_local import transcribe_audio as ts

    all_results = []
    for start, end in segments:
        result = ts(_RAW_AUDIO_FILE, vocal_audio, start, end)
        all_results.append((start, end, result))

    # НОВОЕ: Дедупликация с учётом overlap
    combined_result = deduplicate_segments(all_results, segments, overlap=1.5)

    df = process_transcription(combined_result)
    save_results(df)
```

**Файл:** `core/asr_backend/audio_preprocess.py` — функция дедупликации

```python
def deduplicate_segments(all_results: List[Tuple[float, float, Dict]],
                         segments: List[Tuple[float, float]],
                         overlap: float = 1.5,
                         tolerance: float = 0.05) -> Dict:
    """
    Дедупликация слов на границах overlap-чанков.

    Логика: если слово попадает в overlap-зону И уже есть в предыдущем чанке
    (по времени start/end), то пропускаем его.
    """
    combined = {'segments': []}
    drop_before = None  # До какого времени пропускаем (конец overlap)

    for i, (chunk_start, chunk_end, result) in enumerate(all_results):
        if i > 0:
            prev_end = segments[i - 1][1]
            drop_before = max(chunk_start, prev_end - tolerance)

        for segment in result['segments']:
            new_words = []
            for word in segment.get('words', []):
                word_start = word.get('start')
                word_end = word.get('end', word_start)

                if word_start is None:
                    word_start = segment.get('start')  # if still None -> mark segment suspect

                # Пропускаем слова, которые уже покрыты предыдущим чанком
                if word_start is not None and drop_before is not None:
                    if word_start < (drop_before - tolerance):
                        continue

                new_words.append(word)

            if new_words:
                new_segment = segment.copy()
                new_segment['words'] = new_words
                # Пересчитываем start/end сегмента
                new_segment['start'] = new_words[0].get('start', segment['start'])
                new_segment['end'] = new_words[-1].get('end', segment['end'])
                combined['segments'].append(new_segment)

    rprint(f"[cyan]🔗 Deduplicated: {sum(len(s.get('words', [])) for s in combined['segments'])} words[/cyan]")
    return combined
```

**Acceptance criteria (P0.1):**
- На стыках чанков не повторяются 1–3 слова.
- Нет обрезанных слов на границе (проверка на файле >30 мин).

---

## P0.2 — Заменить substring-matching на word-index spans ✅ DONE

> **PR:** https://github.com/carden-code/VideoLingo/pull/23

### Проблема
В `_6_gen_sub.py:get_sentence_timestamps()` таймкоды предложений получаются через substring matching в "слепленном" потоке слов. Это ломается при любой нормализации/пунктуации.

### Решение: Word-Index Spans (спаны рождаются в split)

Ключевое правило: **никакого восстановления спанов через substring/SequenceMatcher/difflib**.
Стадии split (`_3_1_split_nlp.py` / `_3_2_split_meaning.py`) работают на word timeline
и возвращают `segment_id + word_start_idx/word_end_idx + text`.
Тайминги берутся строго по индексам слов.

**Новый файл:** `core/utils/segment_index.py`

```python
"""
Segment Index - стабильная связь между текстом и word timestamps.

Каждый сегмент хранит диапазон индексов слов [word_start_idx, word_end_idx],
а не текст для substring matching.
"""

import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
import uuid
from rich import print as rprint


@dataclass
class Segment:
    """Атомарный сегмент с привязкой к word timeline."""
    id: str
    text: str
    word_start_idx: int  # Индекс первого слова в cleaned_chunks
    word_end_idx: int    # Индекс последнего слова (inclusive)
    start_time: float = 0.0
    end_time: float = 0.0
    speaker_id: Optional[str] = None
    translation: Optional[str] = None

    # ASR confidence (P1.1)
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None

    @classmethod
    def generate_id(cls) -> str:
        return str(uuid.uuid4())[:8]


class SegmentIndex:
    """
    Индекс сегментов с привязкой к word-level timestamps.

    Решает проблему "substring matching" — теперь связь через индексы слов.
    """

    def __init__(self, words_df: pd.DataFrame):
        """
        Args:
            words_df: DataFrame из cleaned_chunks.xlsx с колонками:
                      text, start, end, speaker_id
        """
        self.words_df = words_df.reset_index(drop=True)
        self.words_df['word_idx'] = self.words_df.index
        self.segments: List[Segment] = []

        # word_idx only; spans are produced during split stage

    def create_segment_from_span(self,
                                 segment_id: str,
                                 text: str,
                                 word_start_idx: int,
                                 word_end_idx: int,
                                 speaker_id: Optional[str] = None) -> Segment:
        """
        Создаёт сегмент напрямую из word-span.
        """
        start_time = float(self.words_df.loc[word_start_idx, 'start'])
        end_time = float(self.words_df.loc[word_end_idx, 'end'])
        if speaker_id is None:
            speaker_id = self.words_df.loc[word_start_idx, 'speaker_id']

        segment = Segment(
            id=segment_id,
            text=text,
            word_start_idx=word_start_idx,
            word_end_idx=word_end_idx,
            start_time=start_time,
            end_time=end_time,
            speaker_id=speaker_id
        )

        self.segments.append(segment)
        return segment

    def build_from_sentences(self, segments: List[dict]) -> List[Segment]:
        """
        Создаёт сегменты из списка структур с готовыми word-span.

        Args:
            segments: [{segment_id, text, word_start_idx, word_end_idx, speaker_id?}]

        Returns:
            Список сегментов с правильными индексами и таймкодами
        """
        result = []
        for item in segments:
            seg = self.create_segment_from_span(
                segment_id=item['segment_id'],
                text=item['text'],
                word_start_idx=item['word_start_idx'],
                word_end_idx=item['word_end_idx'],
                speaker_id=item.get('speaker_id')
            )
            result.append(seg)

        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Export segments to DataFrame."""
        return pd.DataFrame([
            {
                'segment_id': s.id,
                'text': s.text,
                'word_start_idx': s.word_start_idx,
                'word_end_idx': s.word_end_idx,
                'start_time': s.start_time,
                'end_time': s.end_time,
                'duration': s.end_time - s.start_time,
                'speaker_id': s.speaker_id,
                'translation': s.translation,
            }
            for s in self.segments
        ])

    def update_segment_translation(self, segment_id: str, translation: str):
        """Update translation for a segment by ID."""
        for seg in self.segments:
            if seg.id == segment_id:
                seg.translation = translation
                return
        raise ValueError(f"Segment not found: {segment_id}")

    def merge_segments(self, seg1_id: str, seg2_id: str) -> Segment:
        """
        Merge two adjacent segments.

        Returns new merged segment, removes originals from index.
        """
        seg1 = next((s for s in self.segments if s.id == seg1_id), None)
        seg2 = next((s for s in self.segments if s.id == seg2_id), None)

        if not seg1 or not seg2:
            raise ValueError("Segment not found")

        if seg1.word_end_idx + 1 != seg2.word_start_idx:
            rprint(f"[yellow]⚠️ Merging non-adjacent segments[/yellow]")

        merged = Segment(
            id=Segment.generate_id(),
            text=seg1.text + ' ' + seg2.text,
            word_start_idx=seg1.word_start_idx,
            word_end_idx=seg2.word_end_idx,
            start_time=seg1.start_time,
            end_time=seg2.end_time,
            speaker_id=seg1.speaker_id,
            translation=(seg1.translation or '') + ' ' + (seg2.translation or '')
                        if seg1.translation or seg2.translation else None
        )

        # Remove old, add new
        self.segments = [s for s in self.segments if s.id not in (seg1_id, seg2_id)]
        self.segments.append(merged)
        self.segments.sort(key=lambda s: s.word_start_idx)

        return merged


def load_segment_index(words_file: str = "output/log/cleaned_chunks.xlsx") -> SegmentIndex:
    """Load words and create segment index."""
    df = pd.read_excel(words_file)
    df['text'] = df['text'].str.strip('"').str.strip()
    return SegmentIndex(df)
```

Примечание: split-стадия работает по word timeline, а не по голому тексту.
Для CJK и языков без пробелов сегментация строится по индексам слов/символов,
а не через восстановление спанов по строкам.

Пример контракта split-стадии:
```python
words_df = pd.read_excel(_2_CLEANED_CHUNKS)
tokens = words_df['text'].tolist()
# split_tokens(...) возвращает список (word_start_idx, word_end_idx)
spans = split_tokens(tokens)
segments = [
    {
        'segment_id': f'seg_{i:04d}',
        'word_start_idx': start,
        'word_end_idx': end,
        'text': ' '.join(tokens[start:end + 1]),
    }
    for i, (start, end) in enumerate(spans)
]
pd.DataFrame(segments).to_excel(_3_2_SEGMENTS, index=False)
```

### Интеграция в `_6_gen_sub.py`

```python
# Заменить get_sentence_timestamps() на:

from core.utils.segment_index import load_segment_index, SegmentIndex

def align_timestamp_v2(df_translate: pd.DataFrame, output_dir: str):
    """
    Новая версия align_timestamp через SegmentIndex.
    """
    # Load word-level timestamps
    index = load_segment_index()

    # Build segments from spans with stable IDs
    segments_df = pd.read_excel(_3_2_SEGMENTS)
    segments = index.build_from_sentences(segments_df.to_dict('records'))

    # Add translations by segment_id
    translations = dict(zip(df_translate['segment_id'], df_translate['Translation']))
    for seg in segments:
        if seg.id in translations:
            seg.translation = translations[seg.id]

    # Export
    df_result = index.to_dataframe()

    # Generate SRT files
    for filename, columns in SUBTITLE_OUTPUT_CONFIGS:
        generate_srt(df_result, columns, os.path.join(output_dir, filename))

    return df_result
```

**Acceptance criteria (P0.2):**
- 100% сегментов имеют `segment_id`, `word_start_idx`, `word_end_idx`.
- Таймкоды берутся строго из spans, без substring/SequenceMatcher.

---

## P0.3 — Стабильный segment_id через весь пайплайн

### Изменения в форматах файлов

**Новый формат `cleaned_chunks.xlsx` (word-level, без segment_id):**
```
| word_idx | text    | start  | end    | speaker_id |
|----------|---------|--------|--------|------------|
| 0        | Hello   | 0.240  | 0.560  | SPEAKER_00 |
| 1        | world   | 0.580  | 0.920  | SPEAKER_00 |
```

**Новый формат `segments.xlsx` (segment-level):**
```
| segment_id | parent_segment_id | source_chunk_id | text         | word_start_idx | word_end_idx | start  | end    | speaker_id |
|------------|-------------------|-----------------|--------------|----------------|--------------|--------|--------|------------|
| seg_0001   |                   | chunk_0001      | Hello world  | 0              | 1            | 0.240  | 0.920  | SPEAKER_00 |
```

**Новый формат `translation_results.xlsx`:**
```
| segment_id | parent_segment_id | Source              | Translation          | start  | end    |
|------------|-------------------|---------------------|----------------------|--------|--------|
| seg_0001   |                   | Hello world         | Привет мир           | 0.240  | 0.920  |
| seg_0002   |                   | This is a test      | Это тест             | 1.100  | 2.300  |
```

**Новый формат `tts_tasks.xlsx`:**
```
| segment_id | parent_segment_id | number | start_time | end_time | text         | origin      | est_dur |
|------------|-------------------|--------|------------|----------|--------------|-------------|---------|
| seg_0001   |                   | 1      | 00:00:00.2 | 00:00:00.9| Привет мир   | Hello world | 0.8     |
```

**Правила идентификаторов:**
- `segment_id` уникален и неизменяем.
- Любой split/merge создаёт новый `segment_id`, а исходные сохраняются в `parent_segment_id`.
- `source_chunk_id` опционален, но обязателен для трассировки ASR чанка.

**Acceptance criteria (P0.3):**
- От любой строки в `tts_tasks.xlsx` можно пройти до исходного ASR чанка по id.

### Проброс segment_id

```python
# В _3_2_split_meaning.py - после финального split создаём segment_id
def split_by_meaning():
    # ... existing split logic ...
    final_sentences = load_final_sentences()

    segments_with_ids = []
    for i, sentence in enumerate(final_sentences):
        segments_with_ids.append({
            'segment_id': f'seg_{i:04d}',
            'text': sentence
        })

    # Save with IDs (segment-level table)
    pd.DataFrame(segments_with_ids).to_excel(_3_2_SEGMENTS, index=False)

# Важно: при split/merge в _5_split_sub.py создаём новые segment_id,
# а старые сохраняем как parent_segment_id (для трассировки).
# source_chunk_id переносится от исходных сегментов.
```

---

## P1.1 — Сохранение ASR confidence сигналов

**Файл:** `core/asr_backend/whisperX_local.py`

```python
def transcribe_audio(raw_audio_file, vocal_audio_file, start, end):
    # ... existing code ...

    result = model.transcribe(raw_audio_segment, batch_size=batch_size, print_progress=True)
    raw_transcribe = result  # сохранить до align

    # ... alignment ...
    aligned = whisperx.align(...)

    # НОВОЕ: переносим confidence метрики на aligned сегменты
    for seg, raw_seg in zip(aligned['segments'], raw_transcribe['segments']):
        seg['_confidence'] = {
            'avg_logprob': raw_seg.get('avg_logprob', None),
            'no_speech_prob': raw_seg.get('no_speech_prob', None),
            'compression_ratio': raw_seg.get('compression_ratio', None),
            'temperature': raw_seg.get('temperature', None),
        }

    return aligned
```

Примечание: если количество сегментов после align отличается, переносить метрики
по пересечению временных интервалов, а не по zip.

**Файл:** `core/asr_backend/audio_preprocess.py`

```python
def process_transcription(result: Dict) -> pd.DataFrame:
    all_words = []
    for segment in result['segments']:
        speaker_id = segment.get('speaker_id', None)
        confidence = segment.get('_confidence', {})

        for word in segment['words']:
            word_dict = {
                'text': word["word"],
                'start': word.get('start', ...),
                'end': word['end'],
                'speaker_id': speaker_id,
                # НОВОЕ: Confidence на уровне сегмента
                'segment_avg_logprob': confidence.get('avg_logprob'),
                'segment_no_speech_prob': confidence.get('no_speech_prob'),
                # Флаг проблемного слова
                'is_zero_duration': word.get('end', 0) <= word.get('start', 0),
            }
            all_words.append(word_dict)

    return pd.DataFrame(all_words)


def flag_suspicious_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Помечает подозрительные сегменты для ручной проверки или re-ASR.

    Критерии:
    - avg_logprob < -1.0 (низкая уверенность)
    - no_speech_prob > 0.5 (вероятно тишина)
    - >20% zero-duration слов в сегменте
    """
    df['is_suspicious'] = False

    # Low confidence
    if 'segment_avg_logprob' in df.columns:
        df.loc[df['segment_avg_logprob'] < -1.0, 'is_suspicious'] = True

    # Likely silence
    if 'segment_no_speech_prob' in df.columns:
        df.loc[df['segment_no_speech_prob'] > 0.5, 'is_suspicious'] = True

    # Many zero-duration words
    if 'is_zero_duration' in df.columns:
        zero_pct = df['is_zero_duration'].mean()
        if zero_pct > 0.2:
            rprint(f"[yellow]⚠️ High zero-duration rate: {zero_pct:.1%}[/yellow]")

    suspicious_count = df['is_suspicious'].sum()
    if suspicious_count > 0:
        rprint(f"[yellow]⚠️ Flagged {suspicious_count} suspicious words for review[/yellow]")

    return df
```

---

## P1.2 — Hard constraints для терминов

**Файл:** `core/_4_2_translate.py`

```python
def validate_anchors(source: str, translation: str, terms: List[dict]) -> dict:
    """
    Проверяет, что ключевые anchors сохранились в переводе.

    Args:
        source: Исходный текст
        translation: Перевод
        terms: Список терминов из terminology.json

    Returns:
        {'valid': bool, 'missing': [...], 'wrong': [...]}
    """
    issues = {'valid': True, 'missing': [], 'wrong': []}

    for term in terms:
        src_term = term.get('src', '')
        tgt_term = term.get('tgt', '')
        note = term.get('note', '')

        # Check if source term appears in source text
        if src_term.lower() not in source.lower():
            continue

        # Case 1: "keep" - term should appear unchanged
        if 'keep' in note.lower() or 'не переводить' in note.lower():
            if src_term.lower() not in translation.lower():
                issues['missing'].append(src_term)
                issues['valid'] = False

        # Case 2: Specific translation required
        elif tgt_term and tgt_term != src_term:
            if tgt_term.lower() not in translation.lower():
                issues['wrong'].append({'expected': tgt_term, 'source': src_term})
                issues['valid'] = False

    # Check numbers preserved
    src_numbers = re.findall(r'\d+(?:\.\d+)?%?', source)
    for num in src_numbers:
        if num not in translation:
            issues['missing'].append(f"number: {num}")
            issues['valid'] = False

    return issues


def translate_with_anchor_validation(chunk, terms, max_retries=3):
    """
    Перевод с проверкой anchors.
    При провале - retry с более строгим промптом.
    """
    for attempt in range(max_retries):
        translation = translate_chunk_basic(chunk)

        anchor_check = validate_anchors(chunk, translation, terms)

        if anchor_check['valid']:
            return translation

        if attempt < max_retries - 1:
            rprint(f"[yellow]⚠️ Anchor validation failed: {anchor_check}[/yellow]")
            # Более строгий промпт на следующей попытке
            # (добавить в промпт явное указание сохранить конкретные термины)

    raise ValueError(f"Anchor validation failed after {max_retries} attempts")
```

---

## P1.3 — Эскалация retry перевода

**Файл:** `core/translate_lines.py`

```python
def retry_translation_with_escalation(prompt_fn, lines, step_name, max_retries=3):
    """
    Retry с эскалацией:
    1. Обычный промпт
    2. Строгий промпт (literal, no paraphrase)
    3. Разбиение чанка пополам
    """

    for attempt in range(max_retries):
        try:
            if attempt == 0:
                # Стандартный промпт
                prompt = prompt_fn(lines, strict=False)
            elif attempt == 1:
                # Строгий промпт
                prompt = prompt_fn(lines, strict=True)
                console.print(f"[yellow]🔄 Retry #{attempt+1}: strict mode[/yellow]")
            else:
                # Разбиение чанка пополам
                mid = len(lines.split('\n')) // 2
                lines_list = lines.split('\n')
                first_half = '\n'.join(lines_list[:mid])
                second_half = '\n'.join(lines_list[mid:])

                console.print(f"[yellow]🔄 Retry #{attempt+1}: splitting chunk[/yellow]")

                result1 = retry_translation_with_escalation(prompt_fn, first_half, step_name, 2)
                result2 = retry_translation_with_escalation(prompt_fn, second_half, step_name, 2)

                # Merge results
                return merge_translation_results(result1, result2)

            result = ask_gpt(prompt, resp_type='json', ...)

            if validate_result(result, lines):
                return result

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            console.print(f"[yellow]⚠️ Attempt {attempt+1} failed: {e}[/yellow]")

    raise ValueError(f"Translation failed after {max_retries} attempts with escalation")
```

---

## P1.4 — Пер-сегментное ограничение длительности перевода

### Проблема
Сейчас duration-aware подсказки применяются на уровне чанка, но не каждого сегмента.
Из-за этого отдельные фразы могут выходить длиннее/короче и «ползти» по таймлайну.

### Решение
Передавать LLM длительность **каждого сегмента** и требовать перевод, укладывающийся в её окно.
Опираемся на word-span тайминги (P0.2).

**Идея:**
```python
for segment in segments:
    duration_info = {
        "total_duration": segment.end_time - segment.start_time,
        "src_chars": len(segment.text),
    }
    translation = translate_lines(segment.text, ..., duration_info=duration_info)
```

**Acceptance criteria (P1.4):**
- 80–90% сегментов укладываются без time-stretch.
- Нет массовых «перелётов» в соседние сегменты.

---

## P2.1 — Ducking для фона

**Файл:** `core/_12_dub_to_vid.py`

```python
def merge_video_audio_with_ducking():
    """
    Финальный микс с sidechain compression (ducking).
    Фон приглушается когда играет речь.
    """
    VIDEO_FILE = find_video_files()
    background_file = _BACKGROUND_AUDIO_FILE

    # Normalize dub audio
    normalized_dub_audio = 'output/normalized_dub.wav'
    normalize_audio_volume(DUB_AUDIO, normalized_dub_audio)

    # FFmpeg filter с sidechaincompress
    # Когда dub громче threshold — background приглушается
    audio_filter = '''
    [1:a]asplit=2[bg][sc];
    [2:a]asplit=2[dub][ducksig];
    [bg][ducksig]sidechaincompress=
        threshold=0.02:
        ratio=4:
        attack=50:
        release=300:
        makeup=1
    [bgducked];
    [bgducked][dub]amix=inputs=2:duration=first:weights=0.3 1[a]
    '''

    # Alternative: простой lowpass на фоне во время речи
    # Менее агрессивно, но легче настроить
    audio_filter_simple = '''
    [1:a][2:a]sidechaincompress=
        threshold=0.01:
        ratio=3:
        attack=20:
        release=200
    [compressed];
    [compressed][2:a]amix=inputs=2:duration=first[a]
    '''

    cmd = [
        'ffmpeg', '-y',
        '-i', VIDEO_FILE,
        '-i', background_file,
        '-i', normalized_dub_audio,
        '-filter_complex', audio_filter,
        '-map', '0:v',
        '-map', '[a]',
        '-c:a', 'aac', '-b:a', '128k',
        DUB_VIDEO
    ]

    subprocess.run(cmd)
```

---

## P2.2 — Правильная ремедиация duration mismatch

**Файл:** `core/_10_gen_audio.py`

```python
def remediate_duration_mismatch(segment_id: str,
                                 source_text: str,
                                 text: str,
                                 target_duration: float,
                                 actual_duration: float,
                                 attempt: int = 0) -> str:
    """
    Ремедиация когда TTS длиннее таргета.

    Порядок действий:
    1. Перевести короче (length-aware retranslate)
    2. Split сегмента на 2
    3. Speedup (в пределах max)
    4. Trim — только если край
    """
    ratio = actual_duration / target_duration
    max_speed = load_key("speed_factor.max")

    if ratio <= max_speed:
        # Можем просто ускорить
        return apply_speedup(segment_id, ratio)

    if attempt == 0:
        # Попытка 1: Попросить LLM сократить перевод
        console.print(f"[yellow]📝 Duration {ratio:.1f}x too long, requesting shorter translation[/yellow]")

        anchors = extract_anchors(source_text)
        shorter_text = request_shorter_translation(
            source_text,
            text,
            target_duration,
            actual_duration,
            anchors
        )

        if not validate_anchors(source_text, shorter_text, anchors):
            console.print("[yellow]⚠️ Anchor validation failed, splitting segment instead[/yellow]")
            return remediate_duration_mismatch(
                segment_id, source_text, text, target_duration, actual_duration, attempt + 1
            )

        # Regenerate TTS
        new_audio = regenerate_tts(segment_id, shorter_text)
        new_duration = get_audio_duration(new_audio)

        if new_duration / target_duration <= max_speed:
            return apply_speedup(segment_id, new_duration / target_duration)

        # Recurse with next strategy
        return remediate_duration_mismatch(
            segment_id, source_text, shorter_text, target_duration, new_duration, attempt + 1
        )

    elif attempt == 1:
        # Попытка 2: Split сегмента
        console.print(f"[yellow]✂️ Splitting segment {segment_id}[/yellow]")
        # ... split logic ...

    else:
        # Last resort: trim
        if ratio <= max_speed * 1.1:  # 10% tolerance
            console.print(f"[yellow]⚠️ Trimming audio (last resort)[/yellow]")
            return trim_audio(segment_id, target_duration * max_speed)
        else:
            raise ValueError(f"Cannot remediate duration: {ratio:.1f}x too long")


def request_shorter_translation(source_text: str,
                                text: str,
                                target_dur: float,
                                actual_dur: float,
                                anchors: dict) -> str:
    """Запрос к LLM на сокращение перевода с сохранением anchors."""

    chars_to_remove = int(len(text) * (1 - target_dur / actual_dur))

    prompt = f"""
Shorten this translation to fit in {target_dur:.1f} seconds (currently takes {actual_dur:.1f}s).
Remove {chars_to_remove} characters while preserving meaning.
Do NOT change anchors (numbers, terms, acronyms, currencies).

Source: "{source_text}"
Text: "{text}"

Output JSON: {{"shortened": "shorter version here"}}
"""

    result = ask_gpt(prompt, resp_type='json', log_title='duration_fix')
    return result['shortened']
```

---

## P2.3 — VAD для уточнения стартов сегментов

### Проблема
Разные языки имеют разную длину и паузы, из-за чего старт фразы может уезжать от движения губ.

### Решение
Использовать VAD (например, Silero) для детекции реальных участков речи в исходнике
и корректировать стартовые метки сегментов в пределах небольшого окна.

**Авто-калибровка (без ручной настройки под ролик):**
- Оценить шумовой фон на первых 30–60 сек и выставить порог VAD относительно него.
- Если VAD даёт слишком много коротких сегментов или слишком длинную «речь без пауз»,
  автоматически отключить VAD для этого файла (fail-safe).
- Ограничить сдвиг `max_shift_ms` (например, 200–300 мс), чтобы не ломать таймлайн.

**Пресеты (опционально):**
- `lecture` (длинные фразы, меньше пауз)
- `interview` (частые смены спикеров)
- `noisy` (агрессивнее фильтрация)

**Идея:**
```python
vad_segments = detect_speech_segments(audio_path)  # [(start, end), ...]
segment.start_time = snap_to_nearest_vad_onset(segment.start_time, vad_segments, max_shift=0.3)
```

**Acceptance criteria (P2.3):**
- Старты сегментов совпадают с VAD-onset (в пределах 200–300 мс).
- Не ухудшается общая длина таймлайна.
- При плохом качестве VAD автоматически отключается (fail-safe).

---

## Порядок внедрения

### Неделя 1: P0 (критическое)
1. P0.1: Overlap + дедуп в `audio_preprocess.py` и `_2_asr.py`
2. P0.2: `SegmentIndex` class
3. P0.3: segment_id в форматах файлов

### Неделя 2: P1 (укрепление)
4. P1.1: ASR confidence в `cleaned_chunks.xlsx`
5. P1.2: Anchor validation
6. P1.3: Retry escalation
7. P1.4: Пер-сегментное duration-aware ограничение

### Неделя 3: P2 (polish)
8. P2.1: Ducking
9. P2.2: Duration remediation
10. P2.3: VAD-aligned starts

---

## Тестирование

### Тест для P0.1 (overlap + дедуп)
```bash
# Найти видео >35 минут
# Проверить что на границах 30-минутных чанков нет:
# - Обрезанных слов
# - Дублированных слов
# - Gaps в таймлайне
```

### Тест для P0.2 (word-index spans)
```bash
# Подать заранее известный word timeline + split
# Все сегменты должны иметь word_start_idx/word_end_idx
# Таймкоды считаются только по span (никаких substring/SequenceMatcher)
```

### Тест для P0.3 (segment_id)
```bash
# После merge/split операций проверить:
# - Все segment_id уникальны
# - parent_segment_id сохранён для каждого результата split/merge
# - Связь origin↔translation не потеряна
# - tts_tasks ссылаются на корректные сегменты
```

### Тест для P1.4 (per-segment duration-aware)
```bash
# Выбрать 20–30 сегментов с разной длиной
# Проверить: доля сегментов, не требующих time-stretch > 80%
# Нет систематического переразмера (ratio > 1.25)
```

### Тест для P2.2 (duration remediation)
```bash
# Вход: сегмент с переводом, который заметно длиннее таргета
# Ожидание: shorten/split снижает ratio < 1.5 и сохраняет anchors (числа/термины)
```

### Тест для P2.3 (VAD alignment)
```bash
# Вход: видео с заметными паузами в речи
# Ожидание: старт сегментов совпадает с VAD-onset (±300ms)
# Fail-safe: при "шумном" VAD он отключается, таймлайн остаётся стабильным
```

---

*Документ создан: Декабрь 2024*
