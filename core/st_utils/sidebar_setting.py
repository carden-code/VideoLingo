import streamlit as st
from translations.translations import translate as t
from translations.translations import DISPLAY_LANGUAGES
from core.utils import *

def config_input(label, key, help=None):
    """Generic config input handler"""
    val = st.text_input(label, value=load_key(key), help=help)
    if val != load_key(key):
        update_key(key, val)
    return val

def page_setting():

    display_language = st.selectbox("Display Language 🌐",
                                  options=list(DISPLAY_LANGUAGES.keys()),
                                  index=list(DISPLAY_LANGUAGES.values()).index(load_key("display_language")))
    if DISPLAY_LANGUAGES[display_language] != load_key("display_language"):
        update_key("display_language", DISPLAY_LANGUAGES[display_language])
        st.rerun()

    with st.expander(t("LLM Configuration"), expanded=True):
        config_input(t("API_KEY"), "api.key")
        config_input(t("BASE_URL"), "api.base_url", help=t("Openai format, will add /v1/chat/completions automatically"))

        c1, c2 = st.columns([4, 1])
        with c1:
            config_input(t("MODEL"), "api.model", help=t("click to check API validity")+ " 👉")
        with c2:
            if st.button("📡", key="api"):
                st.toast(t("API Key is valid") if check_api() else t("API Key is invalid"),
                        icon="✅" if check_api() else "❌")
        llm_support_json = st.toggle(t("LLM JSON Format Support"), value=load_key("api.llm_support_json"), help=t("Enable if your LLM supports JSON mode output"))
        if llm_support_json != load_key("api.llm_support_json"):
            update_key("api.llm_support_json", llm_support_json)
            st.rerun()

    with st.expander(t("Subtitles Settings"), expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            langs = {
                "🇺🇸 English": "en",
                "🇨🇳 简体中文": "zh",
                "🇪🇸 Español": "es",
                "🇷🇺 Русский": "ru",
                "🇫🇷 Français": "fr",
                "🇩🇪 Deutsch": "de",
                "🇮🇹 Italiano": "it",
                "🇯🇵 日本語": "ja"
            }
            lang = st.selectbox(
                t("Recog Lang"),
                options=list(langs.keys()),
                index=list(langs.values()).index(load_key("whisper.language"))
            )
            if langs[lang] != load_key("whisper.language"):
                update_key("whisper.language", langs[lang])
                st.rerun()

        with c2:
            target_language = st.text_input(t("Target Lang"), value=load_key("target_language"), help=t("Input any language in natural language, as long as llm can understand"))
            if target_language != load_key("target_language"):
                update_key("target_language", target_language)
                st.rerun()

        demucs = st.toggle(t("Vocal separation enhance"), value=load_key("demucs"), help=t("Recommended for videos with loud background noise, but will increase processing time"))
        if demucs != load_key("demucs"):
            update_key("demucs", demucs)
            st.rerun()

        burn_subtitles = st.toggle(t("Burn-in Subtitles"), value=load_key("burn_subtitles"), help=t("Whether to burn subtitles into the video, will increase processing time"))
        if burn_subtitles != load_key("burn_subtitles"):
            update_key("burn_subtitles", burn_subtitles)
            st.rerun()

    with st.expander(t("Dubbing Settings"), expanded=True):
        tts_methods = ["chatterbox_tts", "cosyvoice3"]
        select_tts = st.selectbox(t("TTS Method"), options=tts_methods, index=tts_methods.index(load_key("tts_method")) if load_key("tts_method") in tts_methods else 0)
        if select_tts != load_key("tts_method"):
            update_key("tts_method", select_tts)
            st.rerun()

        if select_tts == "chatterbox_tts":
            st.info("🐳 Requires Chatterbox TTS API running in Docker")

            # Ensure chatterbox_tts config section exists with defaults
            from core.utils.config_utils import ensure_section
            ensure_section('chatterbox_tts', {
                'api_url': 'http://localhost:4123',
                'voice_clone_mode': 2,
                'exaggeration': 0.5,
                'cfg_weight': 0.4
            })

            # Helper function to safely load config
            def load_chatterbox_config(key, default):
                try:
                    return load_key(f"chatterbox_tts.{key}")
                except KeyError:
                    return default

            # Voice clone mode
            mode_options = {
                1: "Mode 1: Basic TTS (No cloning, fastest)",
                2: "Mode 2: Single reference (Balanced)",
                3: "Mode 3: Per-segment reference (Best quality)"
            }
            current_mode = load_chatterbox_config("voice_clone_mode", 1)
            selected_mode = st.selectbox(
                "Voice Clone Mode",
                options=list(mode_options.keys()),
                format_func=lambda x: mode_options[x],
                index=list(mode_options.keys()).index(current_mode),
                help="Mode 1: Fast, no voice cloning. Mode 2: Clone using single reference. Mode 3: Clone per segment (slowest, best quality)"
            )
            if selected_mode != current_mode:
                update_key("chatterbox_tts.voice_clone_mode", selected_mode)
                st.rerun()

            # Exaggeration control
            current_exaggeration = load_chatterbox_config("exaggeration", 0.5)
            exaggeration = st.slider(
                "Exaggeration (Emotionality)",
                min_value=0.0,
                max_value=1.0,
                value=float(current_exaggeration),
                step=0.1,
                help="Control speech emotionality: 0.0=monotone, 0.5=balanced (recommended), 1.0=very expressive"
            )
            if exaggeration != current_exaggeration:
                update_key("chatterbox_tts.exaggeration", exaggeration)
                st.rerun()

            # CFG weight (only for modes 2 and 3)
            if selected_mode in [2, 3]:
                current_cfg = load_chatterbox_config("cfg_weight", 0.4)
                cfg_weight = st.slider(
                    "CFG Weight (Voice Clone Strength)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(current_cfg),
                    step=0.1,
                    help="Influence of reference audio: 0.3-0.5 recommended for best results"
                )
                if cfg_weight != current_cfg:
                    update_key("chatterbox_tts.cfg_weight", cfg_weight)
                    st.rerun()

            # API URL
            current_api_url = load_chatterbox_config("api_url", "http://localhost:4123")
            api_url = st.text_input(
                "Chatterbox API URL",
                value=current_api_url,
                help="URL of Chatterbox TTS API (Docker: docker compose -f docker/docker-compose.gpu.yml up -d)"
            )
            if api_url != current_api_url:
                update_key("chatterbox_tts.api_url", api_url)
                st.rerun()

        elif select_tts == "cosyvoice3":
            st.info("🎙️ CosyVoice 3.0 - Multilingual TTS with voice cloning (ZH, EN, JA, KO, DE, ES, FR, IT, RU)")

            # Ensure cosyvoice3 config section exists with defaults
            from core.utils.config_utils import ensure_section
            ensure_section('cosyvoice3', {
                'api_url': 'http://localhost:50000',
                'mode': 'cross_lingual',
                'sample_rate': 22050
            })

            # Helper function to safely load config
            def load_cosyvoice3_config(key, default):
                try:
                    return load_key(f"cosyvoice3.{key}")
                except KeyError:
                    return default

            # Mode selection
            mode_options = {
                "cross_lingual": "Cross-lingual (voice cloning without reference text)",
                "zero_shot": "Zero-shot (voice cloning with reference text, better quality)"
            }
            current_mode = load_cosyvoice3_config("mode", "cross_lingual")
            selected_mode = st.selectbox(
                "Voice Clone Mode",
                options=list(mode_options.keys()),
                format_func=lambda x: mode_options[x],
                index=list(mode_options.keys()).index(current_mode) if current_mode in mode_options else 0,
                help="cross_lingual: Works across languages. zero_shot: Better quality but requires reference text"
            )
            if selected_mode != current_mode:
                update_key("cosyvoice3.mode", selected_mode)
                st.rerun()

            # API URL
            current_api_url = load_cosyvoice3_config("api_url", "http://localhost:50000")
            api_url = st.text_input(
                "CosyVoice API URL",
                value=current_api_url,
                help="URL of CosyVoice 3.0 FastAPI server (default port: 50000)"
            )
            if api_url != current_api_url:
                update_key("cosyvoice3.api_url", api_url)
                st.rerun()

            # Sample rate
            current_sample_rate = load_cosyvoice3_config("sample_rate", 22050)
            sample_rate = st.number_input(
                "Sample Rate (Hz)",
                min_value=16000,
                max_value=48000,
                value=int(current_sample_rate),
                step=1000,
                help="Audio sample rate. Default: 22050 Hz for CosyVoice"
            )
            if sample_rate != current_sample_rate:
                update_key("cosyvoice3.sample_rate", sample_rate)
                st.rerun()

def check_api():
    try:
        # JSON schema for structured output
        check_schema = {
            "name": "api_check",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"}
                },
                "required": ["message"],
                "additionalProperties": False
            }
        }
        resp = ask_gpt(
            "This is a test. Respond with JSON: {\"message\": \"success\"}",
            resp_type="json",
            log_title='api_check',
            json_schema=check_schema
        )
        return resp.get('message') == 'success'
    except Exception:
        return False

if __name__ == "__main__":
    check_api()
