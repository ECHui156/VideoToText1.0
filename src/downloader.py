import os
import re
from typing import Callable, Optional

from yt_dlp import YoutubeDL


def is_url(text: str) -> bool:
    return bool(re.match(r"^https?://", text or ""))


def is_bilibili_url(text: str) -> bool:
    return "bilibili.com" in (text or "") or "b23.tv" in (text or "")


def _pick_latest_file(folder: str) -> Optional[str]:
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def download_bilibili(
    url: str, out_dir: str, progress_cb: Optional[Callable[[float, str], None]] = None
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    def _hook(d):
        if progress_cb is None:
            return
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes") or 0
            if total:
                progress_cb(min(downloaded / total, 1.0), "下载中")
            else:
                progress_cb(0.0, "下载中")
        elif status == "finished":
            progress_cb(1.0, "下载完成，正在合并")

    ydl_opts = {
        "outtmpl": os.path.join(out_dir, "%(title).200s.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_hook],
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    file_path = None
    requested = info.get("requested_downloads") or []
    if requested:
        file_path = requested[0].get("filepath")
    if not file_path:
        file_path = info.get("_filename") or info.get("filepath")
    if not file_path or not os.path.exists(file_path):
        file_path = _pick_latest_file(out_dir)

    if not file_path or not os.path.exists(file_path):
        raise RuntimeError("下载完成但未找到视频文件，请检查下载目录。")

    return file_path
