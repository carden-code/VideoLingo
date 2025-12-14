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

    # with st.expander(t("Youtube Settings"), expanded=True):
    #     config_input(t("Cookies Path"), "youtube.cookies_path")

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

        runtime = st.selectbox(t("WhisperX Runtime"), options=["local", "cloud", "elevenlabs"], index=["local", "cloud", "elevenlabs"].index(load_key("whisper.runtime")), help=t("Local runtime requires >8GB GPU, cloud runtime requires 302ai API key, elevenlabs runtime requires ElevenLabs API key"))
        if runtime != load_key("whisper.runtime"):
            update_key("whisper.runtime", runtime)
            st.rerun()
        if runtime == "cloud":
            config_input(t("WhisperX 302ai API"), "whisper.whisperX_302_api_key")
        if runtime == "elevenlabs":
            config_input(("ElevenLabs API"), "whisper.elevenlabs_api_key")

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
        tts_methods = ["azure_tts", "openai_tts", "fish_tts", "sf_fish_tts", "edge_tts", "gpt_sovits", "custom_tts", "sf_cosyvoice2", "f5tts", "chatterbox_tts"]
        select_tts = st.selectbox(t("TTS Method"), options=tts_methods, index=tts_methods.index(load_key("tts_method")))
        if select_tts != load_key("tts_method"):
            update_key("tts_method", select_tts)
            st.rerun()

        # sub settings for each tts method
        if select_tts == "sf_fish_tts":
            config_input(t("SiliconFlow API Key"), "sf_fish_tts.api_key")
            
            # Add mode selection dropdown
            mode_options = {
                "preset": t("Preset"),
                "custom": t("Refer_stable"),
                "dynamic": t("Refer_dynamic")
            }
            selected_mode = st.selectbox(
                t("Mode Selection"),
                options=list(mode_options.keys()),
                format_func=lambda x: mode_options[x],
                index=list(mode_options.keys()).index(load_key("sf_fish_tts.mode")) if load_key("sf_fish_tts.mode") in mode_options.keys() else 0
            )
            if selected_mode != load_key("sf_fish_tts.mode"):
                update_key("sf_fish_tts.mode", selected_mode)
                st.rerun()
            if selected_mode == "preset":
                config_input("Voice", "sf_fish_tts.voice")

        elif select_tts == "openai_tts":
            config_input("302ai API", "openai_tts.api_key")
            config_input(t("OpenAI Voice"), "openai_tts.voice")

        elif select_tts == "fish_tts":
            config_input("302ai API", "fish_tts.api_key")
            fish_tts_character = st.selectbox(t("Fish TTS Character"), options=list(load_key("fish_tts.character_id_dict").keys()), index=list(load_key("fish_tts.character_id_dict").keys()).index(load_key("fish_tts.character")))
            if fish_tts_character != load_key("fish_tts.character"):
                update_key("fish_tts.character", fish_tts_character)
                st.rerun()

        elif select_tts == "azure_tts":
            config_input("302ai API", "azure_tts.api_key")
            config_input(t("Azure Voice"), "azure_tts.voice")
        
        elif select_tts == "gpt_sovits":
            st.info(t("Please refer to Github homepage for GPT_SoVITS configuration"))
            config_input(t("SoVITS Character"), "gpt_sovits.character")
            
            refer_mode_options = {1: t("Mode 1: Use provided reference audio only"), 2: t("Mode 2: Use first audio from video as reference"), 3: t("Mode 3: Use each audio from video as reference")}
            selected_refer_mode = st.selectbox(
                t("Refer Mode"),
                options=list(refer_mode_options.keys()),
                format_func=lambda x: refer_mode_options[x],
                index=list(refer_mode_options.keys()).index(load_key("gpt_sovits.refer_mode")),
                help=t("Configure reference audio mode for GPT-SoVITS")
            )
            if selected_refer_mode != load_key("gpt_sovits.refer_mode"):
                update_key("gpt_sovits.refer_mode", selected_refer_mode)
                st.rerun()
                
        elif select_tts == "edge_tts":
            config_input(t("Edge TTS Voice"), "edge_tts.voice")

        elif select_tts == "sf_cosyvoice2":
            config_input(t("SiliconFlow API Key"), "sf_cosyvoice2.api_key")
        
        elif select_tts == "f5tts":
            config_input("302ai API", "f5tts.302_api")

        elif select_tts == "chatterbox_tts":
            st.info("📦 Install: `pip install chatterbox-tts soundfile`")

            # Initialize chatterbox_tts config section if it doesn't exist
            def ensure_chatterbox_config():
                """Ensure chatterbox_tts config section exists with defaults"""
                try:
                    load_key("chatterbox_tts")
                except KeyError:
                    # Section doesn't exist, create it with defaults
                    from ruamel.yaml import YAML
                    import threading

                    yaml = YAML()
                    yaml.preserve_quotes = True
                    lock = threading.Lock()

                    with lock:
                        with open('config.yaml', 'r', encoding='utf-8') as file:
                            data = yaml.load(file)

                        # Add chatterbox_tts section with defaults
                        data['chatterbox_tts'] = {
                            'voice_clone_mode': 2,
                            'exaggeration': 0.5,
                            'cfg_weight': 0.4,
                            'device': 'cuda'
                        }

                        with open('config.yaml', 'w', encoding='utf-8') as file:
                            yaml.dump(data, file)

            # Ensure config exists before loading
            ensure_chatterbox_config()

            # Helper function to safely load config
            def load_chatterbox_config(key, default):
                try:
                    return load_key(f"chatterbox_tts.{key}")
                except KeyError:
                    # Key missing (shouldn't happen after ensure), use default
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

            # Device selection
            device_options = ["cuda", "cpu"]
            current_device = load_chatterbox_config("device", "cuda")
            selected_device = st.selectbox(
                "Device",
                options=device_options,
                index=device_options.index(current_device),
                help="Use CUDA for GPU acceleration (faster) or CPU"
            )
            if selected_device != current_device:
                update_key("chatterbox_tts.device", selected_device)
                st.rerun()

def check_api():
    try:
        resp = ask_gpt("This is a test, response 'message':'success' in json format.", 
                      resp_type="json", log_title='None')
        return resp.get('message') == 'success'
    except Exception:
        return False
    
if __name__ == "__main__":
    check_api()
