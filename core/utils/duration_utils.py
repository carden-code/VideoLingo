from typing import Optional, Dict, List, Tuple

from core.utils import load_key


DEFAULT_CHAR_RATE = 15.0
DEFAULT_CHAR_RATE_MAP = {
    "en": 15.0,
    "ru": 13.5,
    "zh": 7.0,
    "ja": 7.0,
    "ko": 7.5,
    "es": 15.0,
    "fr": 14.5,
    "de": 14.0,
    "it": 15.0,
    "pt": 14.5,
}
DEFAULT_SPEED_RATIO_MIN = 0.75
DEFAULT_SPEED_RATIO_MAX = 1.25


def _safe_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def get_duration_aware_config() -> Dict:
    try:
        cfg = load_key("duration_aware")
    except KeyError:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def normalize_language_code(language_name: Optional[str]) -> str:
    if not language_name:
        return "en"
    raw = str(language_name).strip()
    if not raw:
        return "en"

    lowered = raw.lower()
    if len(lowered) <= 6 and all(c.isalpha() or c in ("-", "_") for c in lowered):
        return lowered.split("-")[0].split("_")[0]

    try:
        from core.tts_backend.chatterbox_tts import get_language_code
        return get_language_code(raw)
    except Exception:
        basic_map = {
            "english": "en",
            "russian": "ru",
            "chinese": "zh",
            "japanese": "ja",
            "korean": "ko",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "italian": "it",
            "portuguese": "pt",
        }
        return basic_map.get(lowered, "en")


def resolve_language_codes(
    src_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
) -> Tuple[str, str]:
    if not src_lang:
        try:
            src_lang = load_key("whisper.detected_language")
        except KeyError:
            src_lang = None
    if not src_lang or src_lang == "auto":
        try:
            src_lang = load_key("whisper.language")
        except KeyError:
            src_lang = "en"

    if not target_lang:
        try:
            target_lang = load_key("target_language")
        except KeyError:
            target_lang = "en"

    return normalize_language_code(src_lang), normalize_language_code(target_lang)


def get_chars_per_sec(lang_code: str, cfg: Optional[Dict] = None) -> float:
    cfg = cfg or get_duration_aware_config()
    rates = cfg.get("chars_per_sec", {}) if isinstance(cfg, dict) else {}
    default_rate = _safe_float(cfg.get("default_chars_per_sec", DEFAULT_CHAR_RATE), DEFAULT_CHAR_RATE)

    if isinstance(rates, dict):
        base = lang_code.split("-")[0].split("_")[0]
        for key in (lang_code, base):
            if key in rates:
                return _safe_float(rates[key], default_rate)
        if "default" in rates:
            return _safe_float(rates["default"], default_rate)

    if lang_code in DEFAULT_CHAR_RATE_MAP:
        return DEFAULT_CHAR_RATE_MAP[lang_code]
    return default_rate


def enrich_duration_info(
    duration_info: Optional[Dict],
    lines_list: Optional[List[str]] = None,
    src_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
) -> Optional[Dict]:
    if not duration_info:
        return None

    cfg = get_duration_aware_config()
    if isinstance(cfg, dict) and cfg.get("enabled") is False:
        return duration_info

    info = dict(duration_info)
    lines_list = lines_list or []

    if not info.get("line_chars") and lines_list:
        info["line_chars"] = [len(line) for line in lines_list]

    if info.get("line_chars") and not info.get("src_chars"):
        info["src_chars"] = sum(info["line_chars"])

    total_duration = _safe_float(info.get("total_duration", 0.0), 0.0)
    if total_duration <= 0:
        return info

    if not info.get("line_durations") and info.get("line_chars"):
        total_chars = _safe_float(info.get("src_chars", 0.0), 0.0)
        if total_chars > 0:
            info["line_durations"] = [
                total_duration * (chars / total_chars) for chars in info["line_chars"]
            ]
        else:
            info["line_durations"] = [0.0 for _ in info["line_chars"]]

    src_lang_code, target_lang_code = resolve_language_codes(src_lang, target_lang)
    src_cps_typical = get_chars_per_sec(src_lang_code, cfg)
    tgt_cps_typical = get_chars_per_sec(target_lang_code, cfg)

    src_chars = _safe_float(info.get("src_chars", 0.0), 0.0)
    src_cps_actual = src_chars / total_duration if total_duration > 0 else 0.0

    speed_cfg = cfg.get("speed_ratio", {}) if isinstance(cfg, dict) else {}
    min_ratio = _safe_float(speed_cfg.get("min", DEFAULT_SPEED_RATIO_MIN), DEFAULT_SPEED_RATIO_MIN)
    max_ratio = _safe_float(speed_cfg.get("max", DEFAULT_SPEED_RATIO_MAX), DEFAULT_SPEED_RATIO_MAX)

    speed_ratio = src_cps_actual / src_cps_typical if src_cps_typical else 1.0
    speed_ratio = min(max(speed_ratio, min_ratio), max_ratio)

    target_cps = tgt_cps_typical * speed_ratio
    target_chars = target_cps * total_duration

    info["src_chars_per_sec"] = src_cps_actual
    info["target_chars_per_sec"] = target_cps
    info["target_chars"] = target_chars
    info["speed_ratio"] = speed_ratio

    if info.get("line_durations"):
        info["line_target_chars"] = [dur * target_cps for dur in info["line_durations"]]

    if src_chars > 0:
        info["target_char_ratio"] = target_chars / src_chars

    return info
