import json
import shutil
import subprocess
from typing import Any, Dict, List


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"命令执行失败: {' '.join(cmd)}\n{result.stderr.strip()}")
    return result


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("未检测到 ffmpeg/ffprobe，请先安装并配置到 PATH。")


def ffprobe_subtitle_streams(video_path: str) -> List[Dict[str, Any]]:
    ensure_ffmpeg()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "s",
        "-show_entries",
        "stream=index,codec_name:stream_tags=language",
        "-of",
        "json",
        video_path,
    ]
    data = json.loads(run_cmd(cmd).stdout or "{}")
    streams = []
    for stream in data.get("streams", []):
        streams.append(
            {
                "index": stream.get("index"),
                "codec": stream.get("codec_name", ""),
                "language": (stream.get("tags") or {}).get("language", ""),
            }
        )
    return streams


def extract_audio(video_path: str, output_wav: str) -> str:
    ensure_ffmpeg()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        output_wav,
    ]
    run_cmd(cmd)
    return output_wav


def extract_subtitle_to_srt(video_path: str, stream_index: int, output_srt: str) -> str:
    ensure_ffmpeg()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-map",
        f"0:{stream_index}",
        "-c:s",
        "srt",
        output_srt,
    ]
    run_cmd(cmd)
    return output_srt
