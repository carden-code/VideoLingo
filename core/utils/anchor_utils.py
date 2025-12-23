import json
import os
import re
from typing import List, Dict

from core.utils.models import _4_1_TERMINOLOGY

NUMBER_RE = re.compile(r'(?<!\w)[\$\u20AC\u00A3\u00A5]?\d+(?:[.,]\d+)?%?')
ACRONYM_RE = re.compile(r'\b[A-Z]{2,}(?:[0-9]{0,2})\b')


def load_terms(path: str = _4_1_TERMINOLOGY) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data.get('terms', [])


def _should_keep_source(note: str, target: str, source: str) -> bool:
    if not note:
        return False
    lowered = note.lower()
    if 'keep' in lowered or 'do not translate' in lowered or 'не переводить' in lowered:
        return True
    if target and source and target.strip().lower() == source.strip().lower():
        return True
    return False


def _normalize_number_token(token: str) -> str:
    token = token.replace(' ', '')
    if token.count('.') and token.count(','):
        token = token.replace(',', '')
    else:
        token = token.replace(',', '.')
    return token


def _normalize_number_text(text: str) -> str:
    return text.replace(' ', '').replace(',', '.')


def build_anchor_requirements(source: str, terms: List[Dict]) -> List[Dict]:
    anchors = []
    src_lower = source.lower()
    for term in terms:
        src_term = str(term.get('src', '')).strip()
        tgt_term = str(term.get('tgt', '')).strip()
        note = str(term.get('note', '')).strip()
        aliases = term.get('aliases', [])
        if isinstance(aliases, str):
            aliases = [a.strip() for a in aliases.split(',') if a.strip()]
        if not isinstance(aliases, list):
            aliases = []
        if not src_term:
            continue
        if src_term.lower() not in src_lower:
            continue
        if _should_keep_source(note, tgt_term, src_term) or not tgt_term:
            anchors.append({"type": "term", "value": src_term})
        else:
            anchors.append({"type": "term", "value": tgt_term, "aliases": aliases})

    for match in NUMBER_RE.findall(source):
        normalized = _normalize_number_token(match)
        anchors.append({
            "type": "number",
            "value": match,
            "normalized": normalized,
            "digits": re.sub(r'\D', '', match)
        })

    for match in ACRONYM_RE.findall(source):
        anchors.append({"type": "acronym", "value": match})

    return anchors


def build_anchor_constraints(lines: List[str], anchors_by_line: List[List[Dict]]) -> str:
    rows = []
    for idx, anchors in enumerate(anchors_by_line, start=1):
        if not anchors:
            continue
        terms = [a["value"] for a in anchors if a["type"] in ("term", "acronym")]
        numbers = [a["value"] for a in anchors if a["type"] == "number"]
        parts = []
        if terms:
            parts.append(f"terms: {', '.join(terms)}")
        if numbers:
            parts.append(f"numbers: {', '.join(numbers)}")
        rows.append(f"{idx}. Keep {', '.join(parts)}")
    if not rows:
        return ""
    return "\n".join(rows)


def _extract_content_words(term: str) -> List[str]:
    """Extract significant content words from a term (skip stopwords like 'to', 'of', 'the')."""
    stopwords = {'to', 'of', 'the', 'a', 'an', 'and', 'or', 'for', 'in', 'on', 'at', 'by', 'with'}
    words = re.findall(r'\b\w+\b', term.lower())
    content = [w for w in words if w not in stopwords and len(w) > 2]
    return content if content else words  # fallback to all words if no content words


def _term_matches(term: str, translation_lower: str, aliases: List[str]) -> bool:
    """
    Check if term matches translation using fuzzy logic:
    1. Exact substring match
    2. Aliases match
    3. Any significant content word from term appears in translation
    """
    term_lower = term.lower()

    # 1. Exact match
    if term_lower in translation_lower:
        return True

    # 2. Aliases match
    for alias in aliases:
        if alias.lower() in translation_lower:
            return True

    # 3. Content words match (e.g., "to create" → check "create")
    content_words = _extract_content_words(term)
    for word in content_words:
        # Use word boundary to avoid partial matches like "create" in "recreate"
        if re.search(rf'\b{re.escape(word)}\b', translation_lower):
            return True

    return False


def validate_anchor_requirements(translation: str, anchors: List[Dict]) -> List[str]:
    """
    Validate that required anchors appear in translation.

    Only validates numbers and acronyms - these MUST be preserved exactly.
    Terms are NOT validated because:
    - They're already in the prompt as guidance
    - LLM may use valid synonyms (e.g., "create" vs "to create")
    - Strict validation causes false failures across languages
    """
    missing = []
    translation_norm = _normalize_number_text(translation)
    translation_digits = re.sub(r'\D', '', translation)

    for anchor in anchors:
        anchor_type = anchor.get("type")
        value = anchor.get("value", "")

        # Skip term validation - terms are guidance, not strict requirements
        if anchor_type == "term":
            continue

        elif anchor_type == "acronym":
            if value not in translation:
                missing.append(value)
        elif anchor_type == "number":
            normalized = anchor.get("normalized", "")
            digits = anchor.get("digits", "")
            if normalized and normalized not in translation_norm:
                if digits and digits not in translation_digits:
                    missing.append(value)
    return missing
