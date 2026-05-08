import os
import tempfile
from typing import Callable, Optional, Tuple

from .audio_transcribe import transcribe_audio
from .downloader import download_bilibili, is_bilibili_url
from .export_utils import sanitize_filename, write_txt
from .media_utils import ensure_ffmpeg, extract_audio
from .models import PipelineOutput
from .ocr_utils import ocr_subtitles_from_video
from .subtitles import extract_soft_subtitles


def run_pipeline(
    input_type: str,
    local_video_path: Optional[str],
    bilibili_url: Optional[str],
    output_dir: Optional[str],
    model_size: str,
    language: str,
    ocr_fps: float,
    ocr_lang: str,
    ocr_similarity: float,
    ocr_region: Optional[Tuple[float, float, float, float]] = None,
    do_transcribe: bool = True,
    do_ocr: bool = True,
    keep_temp: bool = False,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    device: str = "cpu",
    use_fp16: bool = False,
) -> PipelineOutput:
    ensure_ffmpeg()
    logs = []

    def log(msg: str) -> None:
        logs.append(msg)

    def progress(value: float, desc: str) -> None:
        if progress_cb:
            progress_cb(value, desc)

    if input_type == "bilibili":
        if not bilibili_url:
            raise ValueError("请输入 Bilibili 视频网址。")
        if not is_bilibili_url(bilibili_url):
            raise ValueError("当前仅支持 Bilibili 网址。")
    else:
        if not local_video_path or not os.path.exists(local_video_path):
            raise ValueError("本地视频路径无效，请重新选择文件。")

    if keep_temp:
        work_dir = tempfile.mkdtemp(prefix="video_to_text_")
        should_cleanup = False
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="video_to_text_")
        work_dir = temp_dir_obj.name
        should_cleanup = True

    try:
        if input_type == "bilibili":
            progress(0.02, "下载视频")
            download_dir = os.path.join(work_dir, "download")
            video_path = download_bilibili(bilibili_url, download_dir, progress_cb=progress)
            log(f"下载完成: {video_path}")
        else:
            video_path = local_video_path
            log(f"使用本地视频: {video_path}")

        progress(0.15, "提取音频")
        audio_path = os.path.join(work_dir, "audio.wav")
        extract_audio(video_path, audio_path, progress_cb=progress)

        progress(0.25, "语音转写")
        audio_segments = transcribe_audio(
            audio_path, model_size=model_size, language=language, progress_cb=progress
        )
        log(f"语音转写段数: {len(audio_segments)}")

        progress(0.6, "检测软字幕")
        subtitle_segments = extract_soft_subtitles(
            video_path, work_dir, log_cb=log, progress_cb=progress
        )
        if subtitle_segments:
            log(f"软字幕段数: {len(subtitle_segments)}")
        else:
            log("跳过音频转写（用户选择）")
            audio_segments = []

        if do_ocr:
            progress(0.6, "检测软字幕")
            subtitle_segments = extract_soft_subtitles(
                video_path, work_dir, log_cb=log, progress_cb=progress
            )
            if subtitle_segments:
                log(f"软字幕段数: {len(subtitle_segments)}")
            else:
                progress(0.7, "无软字幕，开始 OCR")
                subtitle_segments = ocr_subtitles_from_video(
                    video_path,
                    ocr_fps=ocr_fps,
                    lang=ocr_lang,
                    similarity_threshold=ocr_similarity,
                    ocr_region=ocr_region,
                    progress_cb=progress,
                )
                log(f"OCR 字幕段数: {len(subtitle_segments)}")
        else:
            log("跳过 OCR 字幕识别（用户选择）")
            subtitle_segments = []

        output_dir = output_dir or "outputs"
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        base_name = sanitize_filename(os.path.splitext(os.path.basename(video_path))[0])
        audio_txt_path = os.path.join(output_dir, f"{base_name}_音频稿.txt")
        subtitle_txt_path = os.path.join(output_dir, f"{base_name}_字幕稿.txt")

        audio_out_path = ""
        subtitle_out_path = ""

        if audio_segments:
            write_txt(audio_segments, audio_txt_path)
            audio_out_path = audio_txt_path
            log(f"输出音频稿: {audio_txt_path}")
        else:
            log("未生成音频稿（无转写或已跳过）")

        if subtitle_segments:
            write_txt(subtitle_segments, subtitle_txt_path)
            subtitle_out_path = subtitle_txt_path
            log(f"输出字幕稿: {subtitle_txt_path}")
        else:
            log("未生成字幕稿（无软字幕/OCR 或已跳过）")

        progress(1.0, "完成")
        return PipelineOutput(
            video_path=video_path,
            audio_segments=audio_segments,
            subtitle_segments=subtitle_segments,
            audio_txt_path=audio_out_path,
            subtitle_txt_path=subtitle_out_path,
            logs=logs,
        )
    finally:
        if should_cleanup:
            temp_dir_obj.cleanup()
