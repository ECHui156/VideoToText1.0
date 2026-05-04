import os
import re
from typing import Callable, List, Optional

from .media_utils import extract_subtitle_to_srt, ffprobe_subtitle_streams
from .models import Segment

# 支持使用逗号或小数点作为毫秒分隔符，匹配形式 00:00:00,000 或 00:00:00.000
_TIME_RE = re.compile(
    r"(\d+):(\d+):(\d+)[,\.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,\.](\d+)"
)


def _time_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(srt_path: str, log_cb: Optional[Callable[[str], None]] = None) -> List[Segment]:
    """解析 SRT 文件。

    变更点：
    - 以二进制读取并尝试多种常见编码（utf-8, gbk, gb18030, latin-1 等），提高对非 UTF-8 SRT 的兼容性。
    - 放宽时间戳匹配（接受 "," 或 "." 作为毫秒分隔符），并使用 search() 以容忍行首的 BOM 或其它前缀。
    - 可选地通过 `log_cb` 返回所用的编码和调试信息。
    """
    segments: List[Segment] = []
    if not os.path.exists(srt_path):
        return segments

    # 读取原始字节，尝试多种解码以识别编码
    with open(srt_path, "rb") as f:
        raw = f.read()

    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936", "latin-1")
    text = None
    used_enc = None
    for enc in encodings:
        try:
            text = raw.decode(enc)
            used_enc = enc
            break
        except Exception:
            continue

    if text is None:
        text = raw.decode("latin-1", errors="replace")
        used_enc = "latin-1-replace"

    if log_cb:
        log_cb(f"解析 SRT 使用编码: {used_enc}")

    lines = [line.rstrip("\r\n") for line in text.splitlines()]

    # 移除可能存在的 BOM
    if lines and lines[0].startswith("\ufeff"):
        lines[0] = lines[0].lstrip("\ufeff")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.isdigit():
            i += 1
        if i >= len(lines):
            break
        time_line = lines[i].strip()
        match = _TIME_RE.search(time_line)
        if not match:
            i += 1
            continue
        start = _time_to_seconds(match.group(1), match.group(2), match.group(3), match.group(4))
        end = _time_to_seconds(match.group(5), match.group(6), match.group(7), match.group(8))
        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1
        text = " ".join(text_lines).strip()
        segments.append(Segment(start, end, text))
        i += 1
    return segments


def extract_soft_subtitles(
    video_path: str,
    temp_dir: str,
    log_cb: Optional[Callable[[str], None]] = None,
) -> Optional[List[Segment]]:
    streams = ffprobe_subtitle_streams(video_path)
    if not streams:
        return None
    stream = streams[0]
    if log_cb:
        lang = stream.get("language") or "unknown"
        codec = stream.get("codec") or "unknown"
        log_cb(f"检测到软字幕流 index={stream.get('index')} lang={lang} codec={codec}")
    srt_path = os.path.join(temp_dir, "soft_subs.srt")
    extract_subtitle_to_srt(video_path, stream.get("index"), srt_path)
    segments = parse_srt(srt_path, log_cb=log_cb)
    if not segments:
        if log_cb:
            try:
                with open(srt_path, "rb") as f:
                    sample = f.read(200).decode("latin-1", errors="replace")
                log_cb(f"SRT 内容预览（前200字节）:\n{sample}")
            except Exception:
                pass
        return None
    return segments
