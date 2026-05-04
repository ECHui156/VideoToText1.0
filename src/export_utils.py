import os
from typing import List

from .models import Segment

INVALID_CHARS = r'<>:"/\\|?*'


def sanitize_filename(name: str) -> str:
    # 清理 Windows 不允许的文件名字符
    cleaned = (name or "").strip()
    for ch in INVALID_CHARS:
        cleaned = cleaned.replace(ch, "_")
    cleaned = cleaned.rstrip(". ")
    return cleaned or "video"


def segments_to_plain_text(segments: List[Segment]) -> str:
    # 只输出文本内容，保留时间戳在内存结构里
    lines = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def write_txt(segments: List[Segment], output_path: str) -> str:
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    content = segments_to_plain_text(segments)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path
