import os
import re
from pydub import AudioSegment

from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.chatterbox_tts import chatterbox_tts_for_videolingo
from core.tts_backend.cosyvoice3_tts import cosyvoice3_tts_for_videolingo
from core.prompts import get_correct_text_prompt
from core.utils import *

def clean_text_for_tts(text):
    """Remove problematic characters for TTS"""
    chars_to_remove = ['&', '®', '™', '©']
    for char in chars_to_remove:
        text = text.replace(char, '')
    return text.strip()

def tts_main(text, save_as, number, task_df):
    text = clean_text_for_tts(text)
    # Check if text is empty or single character, single character voiceovers are prone to bugs
    cleaned_text = re.sub(r'[^\w\s]', '', text).strip()
    if not cleaned_text or len(cleaned_text) <= 1:
        silence = AudioSegment.silent(duration=100)  # 100ms = 0.1s
        silence.export(save_as, format="wav")
        rprint(f"Created silent audio for empty/single-char text: {save_as}")
        return

    # Skip if file exists
    if os.path.exists(save_as):
        return

    print(f"Generating <{text}...>")
    TTS_METHOD = load_key("tts_method")

    # Validate TTS method, fallback to chatterbox_tts for unsupported methods
    supported_methods = ['chatterbox_tts', 'cosyvoice3']
    if TTS_METHOD not in supported_methods:
        rprint(f"[yellow]⚠️ tts_method='{TTS_METHOD}' not supported, using chatterbox_tts[/yellow]")
        TTS_METHOD = 'chatterbox_tts'

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt >= max_retries - 1:
                print("Asking GPT to correct text...")
                # JSON schema for structured output
                correct_schema = {
                    "name": "text_correction",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"}
                        },
                        "required": ["text"],
                        "additionalProperties": False
                    }
                }
                correct_text = ask_gpt(get_correct_text_prompt(text), resp_type="json", log_title='tts_correct_text', json_schema=correct_schema)
                text = correct_text['text']

            if TTS_METHOD == 'chatterbox_tts':
                chatterbox_tts_for_videolingo(text, save_as, number, task_df)
            elif TTS_METHOD == 'cosyvoice3':
                cosyvoice3_tts_for_videolingo(text, save_as, number, task_df)

            # Check generated audio duration
            duration = get_audio_duration(save_as)
            if duration > 0:
                break
            else:
                if os.path.exists(save_as):
                    os.remove(save_as)
                if attempt == max_retries - 1:
                    print(f"Warning: Generated audio duration is 0 for text: {text}")
                    # Create silent audio file
                    silence = AudioSegment.silent(duration=100)  # 100ms silence
                    silence.export(save_as, format="wav")
                    return
                print(f"Attempt {attempt + 1} failed, retrying...")
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to generate audio after {max_retries} attempts: {str(e)}")
            print(f"Attempt {attempt + 1} failed, retrying...")
