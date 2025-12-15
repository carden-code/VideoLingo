import platform
import subprocess

import cv2
from rich.console import Console

from core._1_ytdlp import find_video_files
from core.asr_backend.audio_preprocess import normalize_audio_volume
from core.utils import *
from core.utils.models import *

console = Console()

DUB_VIDEO = "output/output_dub.mp4"
DUB_SUB_FILE = 'output/dub.srt'
DUB_AUDIO = 'output/dub.mp3'

TRANS_FONT_SIZE = 17
TRANS_FONT_NAME = 'Arial'
if platform.system() == 'Linux':
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
if platform.system() == 'Darwin':
    TRANS_FONT_NAME = 'Arial Unicode MS'

TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1 
TRANS_BACK_COLOR = '&H33000000'

def merge_video_audio():
    """Merge video and audio, optionally with burned-in subtitles"""
    VIDEO_FILE = find_video_files()
    background_file = _BACKGROUND_AUDIO_FILE
    burn_subs = load_key("burn_subtitles")

    # Normalize dub audio
    normalized_dub_audio = 'output/normalized_dub.wav'
    normalize_audio_volume(DUB_AUDIO, normalized_dub_audio)

    # Get video resolution
    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")

    # Build ffmpeg command based on burn_subtitles setting
    if burn_subs:
        rprint("[bold cyan]Burning subtitles into video...[/bold cyan]")
        subtitle_filter = (
            f"subtitles={DUB_SUB_FILE}:force_style='FontSize={TRANS_FONT_SIZE},"
            f"FontName={TRANS_FONT_NAME},PrimaryColour={TRANS_FONT_COLOR},"
            f"OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
            f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=27,BorderStyle=4'"
        )
        video_filter = (
            f'[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,'
            f'pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,'
            f'{subtitle_filter}[v]'
        )
    else:
        rprint("[bold cyan]Assembling video without subtitles...[/bold cyan]")
        video_filter = (
            f'[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,'
            f'pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2[v]'
        )

    cmd = [
        'ffmpeg', '-y', '-i', VIDEO_FILE, '-i', background_file, '-i', normalized_dub_audio,
        '-filter_complex',
        f'{video_filter};[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=3[a]'
    ]

    if load_key("ffmpeg_gpu"):
        rprint("[bold green]Using GPU acceleration...[/bold green]")
        cmd.extend(['-map', '[v]', '-map', '[a]', '-c:v', 'h264_nvenc'])
    else:
        cmd.extend(['-map', '[v]', '-map', '[a]'])

    cmd.extend(['-c:a', 'aac', '-b:a', '96k', DUB_VIDEO])

    subprocess.run(cmd)
    rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")

if __name__ == '__main__':
    merge_video_audio()
