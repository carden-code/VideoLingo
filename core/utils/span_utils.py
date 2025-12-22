import re
from typing import List, Tuple


def normalize_token(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\W+", "", text, flags=re.UNICODE).strip().lower()


def build_word_index(words: List[str]) -> List[Tuple[int, str]]:
    word_tokens = []
    for idx, word in enumerate(words):
        token = normalize_token(word)
        if token:
            word_tokens.append((idx, token))
    return word_tokens


def match_span_space(tokens: List[str], word_tokens: List[Tuple[int, str]], start_idx: int):
    if not tokens:
        return None
    for i in range(start_idx, len(word_tokens) - len(tokens) + 1):
        window = [t for _, t in word_tokens[i:i + len(tokens)]]
        if window == tokens:
            start_word_idx = word_tokens[i][0]
            end_word_idx = word_tokens[i + len(tokens) - 1][0]
            return start_word_idx, end_word_idx, i + len(tokens)
    return None


def match_span_no_space(sentence_norm: str, word_tokens: List[Tuple[int, str]], start_idx: int):
    if not sentence_norm:
        return None
    for i in range(start_idx, len(word_tokens)):
        buf = ""
        for j in range(i, len(word_tokens)):
            buf += word_tokens[j][1]
            if len(buf) >= len(sentence_norm):
                if buf == sentence_norm:
                    return word_tokens[i][0], word_tokens[j][0], j + 1
                break
    return None


def map_sentences_to_spans(sentences: List[str], words: List[str], joiner: str):
    word_tokens = build_word_index(words)
    token_cursor = 0
    spans = []

    for sentence in sentences:
        sentence = str(sentence).strip()
        if not sentence:
            continue

        if joiner == " ":
            tokens = [normalize_token(t) for t in sentence.split()]
            tokens = [t for t in tokens if t]
            match = match_span_space(tokens, word_tokens, token_cursor)
        else:
            sentence_norm = normalize_token(sentence)
            match = match_span_no_space(sentence_norm, word_tokens, token_cursor)

        if not match:
            raise ValueError(f"Failed to map sentence to word span: {sentence}")

        word_start_idx, word_end_idx, token_cursor = match
        spans.append((word_start_idx, word_end_idx))

    return spans
