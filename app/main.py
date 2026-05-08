import os
import sys

# 避免代理设置影响本地回环地址访问
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

import cv2
import gradio as gr
import threading
import queue

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ocr_utils import (
    DEFAULT_OCR_REGION,
    draw_ocr_region,
    extract_frame_at_timestamp,
    get_video_duration,
    validate_ocr_region,
)
from src.pipeline import run_pipeline


def _build_ocr_region(x: float, y: float, w: float, h: float):
    try:
        return validate_ocr_region((x, y, w, h))
    except ValueError as exc:
        raise gr.Error(str(exc))


def _build_ocr_preview(
    video_path: str, preview_time: float, x: float, y: float, w: float, h: float
):
    if not video_path:
        return None
    region = _build_ocr_region(x, y, w, h)
    try:
        frame = extract_frame_at_timestamp(video_path, preview_time or 0.0)
    except RuntimeError as exc:
        raise gr.Error(str(exc))
    if frame is None:
        return None
    preview = draw_ocr_region(frame, region)
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def _on_video_selected(video_path: str, x: float, y: float, w: float, h: float):
    if not video_path:
        return gr.update(maximum=0, value=0), None
    try:
        duration = get_video_duration(video_path)
    except RuntimeError as exc:
        raise gr.Error(str(exc))
    preview_time = 0.0
    preview_image = _build_ocr_preview(video_path, preview_time, x, y, w, h)
    return gr.update(maximum=max(duration, 0.0), value=preview_time), preview_image


def run_action(
    input_mode: str,
    video_path: str,
    bilibili_url: str,
    output_dir: str,
    model_size: str,
    language: str,
    ocr_fps: float,
    ocr_lang: str,
    ocr_similarity: float,
    ocr_region_x: float,
    ocr_region_y: float,
    ocr_region_w: float,
    ocr_region_h: float,
    do_transcribe: bool,
    do_ocr: bool,
    progress=gr.Progress(),
):
    mode = "local" if input_mode == "本地视频" else "bilibili"
    ocr_region = _build_ocr_region(
        ocr_region_x, ocr_region_y, ocr_region_w, ocr_region_h
    )

    q: "queue.Queue" = queue.Queue()

    def progress_cb(value: float, desc: str) -> None:
        # 将描述推入队列以便生成器输出，同时更新 Gradio 进度条
        if desc:
            q.put(("msg", desc))
        if value is None:
            progress(0, desc=desc)
        else:
            progress(value, desc=desc)

    def worker():
        try:
            result = run_pipeline(
                input_type=mode,
                local_video_path=video_path,
                bilibili_url=bilibili_url,
                output_dir=output_dir,
                model_size=model_size,
                language=language,
                ocr_fps=ocr_fps,
                ocr_lang=ocr_lang,
                ocr_similarity=ocr_similarity,
                ocr_region=ocr_region,
                progress_cb=progress_cb,
            )
            q.put(("done", result))
        except Exception as exc:
            q.put(("error", str(exc)))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    logs: list[str] = []

    while True:
        typ, payload = q.get()
        if typ == "msg":
            desc = payload
            if not logs or logs[-1] != desc:
                logs.append(desc)
            yield None, None, "\n".join(logs)
        elif typ == "done":
            result = payload
            # 合并并去重连续重复条目
            for l in (result.logs or []):
                if not logs or logs[-1] != l:
                    logs.append(l)
            yield result.audio_txt_path, result.subtitle_txt_path, "\n".join(logs)
            break
        elif typ == "error":
            err = payload
            logs.append(f"错误: {err}")
            yield None, None, "\n".join(logs)
            break


with gr.Blocks(title="视频转文字") as demo:
    gr.Markdown("# 视频转文字（本地版）\n支持本地视频或 Bilibili 网址输入，生成音频稿与字幕稿。")

    with gr.Row():
        input_mode = gr.Radio(
            ["本地视频", "Bilibili URL"], value="本地视频", label="输入方式"
        )

    with gr.Row():
        video_path = gr.File(
            label="本地视频文件",
            file_types=[".mp4", ".mkv", ".flv", ".mov", ".avi"],
            type="filepath",
        )
        bilibili_url = gr.Textbox(label="Bilibili 网址", placeholder="https://www.bilibili.com/video/...", lines=1)

    output_dir = gr.Textbox(label="输出目录", value="outputs", lines=1)

    with gr.Row():
        do_transcribe = gr.Checkbox(value=True, label="启用音频转写 (Whisper)")
        do_ocr = gr.Checkbox(value=True, label="启用 OCR 字幕识别 (Tesseract)")

    with gr.Accordion("高级设置", open=False):
        model_size = gr.Dropdown(
            ["tiny", "base", "small", "medium", "large"], value="base", label="Whisper 模型"
        )
        language = gr.Dropdown(
            ["zh", "en", "auto"], value="zh", label="语音识别语言"
        )
        ocr_fps = gr.Slider(0.5, 5.0, value=1.0, step=0.5, label="OCR 抽帧频率 (fps)")
        ocr_lang = gr.Textbox(label="Tesseract 语言", value="chi_sim+eng", lines=1)
        ocr_similarity = gr.Slider(0.7, 0.98, value=0.9, step=0.01, label="OCR 去重相似度阈值")

        gr.Markdown("OCR 识别区域使用相对比例 (x、y、width、height)，范围 0-1。")
        with gr.Row():
            ocr_region_x = gr.Slider(0, 1, value=DEFAULT_OCR_REGION[0], step=0.01, label="OCR 区域 X")
            ocr_region_y = gr.Slider(0, 1, value=DEFAULT_OCR_REGION[1], step=0.01, label="OCR 区域 Y")
        with gr.Row():
            ocr_region_w = gr.Slider(0, 1, value=DEFAULT_OCR_REGION[2], step=0.01, label="OCR 区域 Width")
            ocr_region_h = gr.Slider(0, 1, value=DEFAULT_OCR_REGION[3], step=0.01, label="OCR 区域 Height")

    ocr_preview_section = gr.Accordion("OCR 区域预览（仅本地视频）", open=False)
    with ocr_preview_section:
        preview_time = gr.Slider(0, 10, value=0.0, step=0.1, label="预览时间点 (秒)")
        preview_image = gr.Image(label="OCR 区域预览", type="numpy")

    run_btn = gr.Button("开始处理", variant="primary")

    with gr.Row():
        audio_txt = gr.File(label="音频稿（txt）")
        subtitle_txt = gr.File(label="字幕稿（txt）")

    log_box = gr.Textbox(label="运行日志", lines=10)

    run_btn.click(
        run_action,
        inputs=[
            input_mode,
            video_path,
            bilibili_url,
            output_dir,
            model_size,
            language,
            ocr_fps,
            ocr_lang,
            ocr_similarity,
            ocr_region_x,
            ocr_region_y,
            ocr_region_w,
            ocr_region_h,
            do_transcribe,
            do_ocr,
        ],
        outputs=[audio_txt, subtitle_txt, log_box],
    )

    def _toggle_ocr_preview(enabled: bool):
        return gr.update(visible=enabled)

    do_ocr.change(_toggle_ocr_preview, inputs=[do_ocr], outputs=[ocr_preview_section])

    video_path.change(
        _on_video_selected,
        inputs=[video_path, ocr_region_x, ocr_region_y, ocr_region_w, ocr_region_h],
        outputs=[preview_time, preview_image],
    )

    preview_inputs = [
        video_path,
        preview_time,
        ocr_region_x,
        ocr_region_y,
        ocr_region_w,
        ocr_region_h,
    ]
    preview_time.change(
        _build_ocr_preview,
        inputs=preview_inputs,
        outputs=[preview_image],
    )
    for component in (ocr_region_x, ocr_region_y, ocr_region_w, ocr_region_h):
        component.change(
            _build_ocr_preview,
            inputs=preview_inputs,
            outputs=[preview_image],
        )


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port_env = os.getenv("GRADIO_SERVER_PORT")
    server_port = int(server_port_env) if server_port_env else None
    demo.launch(inbrowser=True, server_name=server_name, server_port=server_port)
