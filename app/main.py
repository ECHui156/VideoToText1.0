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
from src.export_utils import sanitize_filename, segments_to_plain_text
from src.llm_utils import call_llm_cleanup
from src.pipeline import run_pipeline


PROMPT_DEMO = """# Role
你是一位专业的文字编辑和排版专家，擅长处理口语化的“语音转写生肉稿件”，能将其转化为排版优雅、易于阅读的高质量文章。
# Task
请帮我整理我提供的语音转写文本。原始文本呈“一句一段”的碎片化格式，且缺乏标点、包含同音错别字。你需要根据语义将其重新排版为连贯、结构清晰的段落文本。
# Guidelines
1. **添加标点**：根据语境、停顿和语气，准确添加合适的标点符号（逗号、句号、问号、叹号等），确保文本阅读流畅。
2. **纠正错词**：智能识别并修正语音识别产生的同音/近音错别字（例如技术专有名词、日常用语错误等），但**严格禁止改变句子的原意**。
3. **语义分段**：打破原有的一句一段格式。根据上下文的逻辑连贯性、话题的自然切换点（如进入新阶段、新观点、新场景等）进行合理分段。不要分得太碎，也不要全篇不分段。
4. **保持原汁原味**：除了纠错和标点外，禁止随意增删原文本的实质内容，严格保留讲话者原本的口语风格、语气和表述逻辑。
# Output
直接输出整理后的正文即可，不需要任何额外的解释或寒暄。"""

PROMPT_DEMO_OCR = """# Role
你是一位资深的文字编辑与排版专家，极其擅长处理由视频 OCR（光学字符识别）导出的粗糙、碎片化字幕文本。你能精准地将杂乱无章的短句还原为流畅、通顺、结构清晰的专业文章。
# Task
请对用户提供的 OCR 字幕文本进行深度清理与排版重组。你需要去除所有的识别噪音与重复项，补充标点，并根据语义将“一句一行”的碎片文本重新合并为逻辑清晰的段落。
# Processing Rules
1. 【去重与去噪】：
    - 智能识别并删除连续重复出现的句子或短语（由于视频帧停留导致的重复识别）。
    - 彻底清理毫无意义的 OCR 乱码字符（如 `|`、`_`、`=`、`[` 等）、与解说内容无关的背景英文单词或字母碎片。
    - 仅保留真正的中文/英文解说核心内容。
2. 【智能标点】：
    - 原始文本通常缺乏标点。请务必根据上下文的语境、语气和断句节奏，自动补充正确的现代汉语标点符号（如逗号、句号、问号、顿号等），切勿仅用空格隔开。
3. 【语义分段】：
    - 打破原有的“一句一行”或“半句一行”的碎片格式，将连贯的话语合并成完整的段落。
    - 寻找“语义转折”、“话题切换”或“时间线过渡”作为自然分段的依据。
    - 控制段落长度：避免出现长篇大论不分段的情况，若同一话题篇幅过长，请根据子论点的切换进行适度切分，提升阅读体验。
4. 【坚守原意】：
    - 在执行上述清理和排版时，绝对禁止对讲话者的真实有效原话进行篡改、润色、缩写或总结。必须原汁原味地保留讲述者的原意与用词。
# Output Format
请严格按照以下两个模块进行输出：
### 整理后的文本
（在此输出经过排版整理、加好标点、分好段落的纯净文本）
### 整理说明
（极其简短地向用户汇报处理情况，例如：“已清理约 XX 处连续重复的乱码与无意义字符，补充了适当的标点，并根据话题走向将全文分为 X 个段落。”）"""


def _select_llm_input(audio_segments, subtitle_segments):
    if audio_segments:
        audio_text = segments_to_plain_text(audio_segments)
        if audio_text:
            return PROMPT_DEMO, audio_text, "音频稿"
    if subtitle_segments:
        subtitle_text = segments_to_plain_text(subtitle_segments)
        if subtitle_text:
            return PROMPT_DEMO_OCR, subtitle_text, "字幕稿"
    return "", "", ""


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


def _on_media_selected(video_path: str, audio_path: str, x: float, y: float, w: float, h: float):
    """
    当本地视频或本地音频被选择时调用。
    - 若选择音频：禁用并取消勾选 OCR，隐藏所有 OCR 相关控件。
    - 若选择视频：启用 OCR 选择（但不强制显示 OCR 选项），并返回预览。
    返回值顺序与绑定的 outputs 对应。
    """
    # 音频优先（若选择了音频文件，则视为纯音频输入）
    if audio_path:
        preview_time_update = gr.update(maximum=0, value=0)
        preview_image = None
        do_ocr_update = gr.update(value=False, interactive=False)
        hidden = gr.update(visible=False)
        return (
            preview_time_update,
            preview_image,
            do_ocr_update,
            hidden,
            hidden,
            hidden,
            hidden,
            hidden,
            hidden,
            hidden,
            hidden,
            hidden,
        )

    # 若选择视频，则生成预览并确保 OCR 控件可交互（visibility 由 do_ocr 控制）
    if video_path:
        try:
            duration = get_video_duration(video_path)
        except RuntimeError as exc:
            raise gr.Error(str(exc))
        preview_time = 0.0
        preview_image = _build_ocr_preview(video_path, preview_time, x, y, w, h)
        do_ocr_update = gr.update(interactive=True)
        noop = gr.update()
        return (
            gr.update(maximum=max(duration, 0.0), value=preview_time),
            preview_image,
            do_ocr_update,
            noop,
            noop,
            noop,
            noop,
            noop,
            noop,
            noop,
            noop,
            noop,
        )

    # 若都未选择，重置预览并保持 OCR 可交互但隐藏相关控件
    hidden = gr.update(visible=False)
    return (
        gr.update(maximum=0, value=0),
        None,
        gr.update(interactive=True),
        hidden,
        hidden,
        hidden,
        hidden,
        hidden,
        hidden,
        hidden,
        hidden,
        hidden,
    )


def run_action(
    video_path: str,
    audio_path: str,
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
    device: str,
    use_fp16: bool,
    use_llm: bool,
    llm_api_key: str,
    progress=gr.Progress(),
):
    if use_llm and not (llm_api_key or "").strip():
        raise gr.Error("已选择大模型整理，但未填写 API Key。")
    # 优先使用本地音频（若提供），否则本地视频或 Bilibili
    if audio_path:
        mode = "local"
        local_media_path = audio_path
        # 音频输入时强制关闭 OCR
        do_ocr = False
        ocr_region = None
    else:
        has_video = bool(video_path)
        has_url = bool(bilibili_url)
        if has_video and has_url:
            raise gr.Error("请只选择本地视频或填写 Bilibili 网址之一，不要同时使用。")
        if has_url:
            mode = "bilibili"
            local_media_path = None
        else:
            mode = "local"
            local_media_path = video_path
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
                local_video_path=local_media_path,
                bilibili_url=bilibili_url,
                output_dir=output_dir,
                model_size=model_size,
                do_transcribe=do_transcribe,
                do_ocr=do_ocr,
                device=("cuda" if (isinstance(device, str) and device.upper() == "GPU") else "cpu"),
                use_fp16=use_fp16,
                language=language,
                ocr_fps=ocr_fps,
                ocr_lang=ocr_lang,
                ocr_similarity=ocr_similarity,
                ocr_region=ocr_region,
                progress_cb=progress_cb,
            )
            llm_txt_path = ""
            if use_llm:
                prompt, llm_input, source_label = _select_llm_input(
                    result.audio_segments, result.subtitle_segments
                )
                if not llm_input:
                    q.put(("msg", "未生成整理稿（无可整理文本）"))
                else:
                    q.put(("msg", f"开始大模型整理（{source_label}）"))
                    llm_output = call_llm_cleanup(llm_api_key, prompt, llm_input)
                    output_dir_path = os.path.abspath(output_dir or "outputs")
                    os.makedirs(output_dir_path, exist_ok=True)
                    base_name = sanitize_filename(
                        os.path.splitext(os.path.basename(result.video_path))[0]
                    )
                    llm_txt_path = os.path.join(output_dir_path, f"{base_name}_整理稿.txt")
                    with open(llm_txt_path, "w", encoding="utf-8") as f:
                        f.write(llm_output)
                    q.put(("msg", f"输出整理稿: {llm_txt_path}"))
            q.put(("done", (result, llm_txt_path)))
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
            yield None, None, None, "\n".join(logs)
        elif typ == "done":
            result, llm_txt_path = payload
            # 合并并去重连续重复条目
            for l in (result.logs or []):
                if not logs or logs[-1] != l:
                    logs.append(l)
            yield (
                result.audio_txt_path,
                result.subtitle_txt_path,
                llm_txt_path or None,
                "\n".join(logs),
            )
            break
        elif typ == "error":
            err = payload
            logs.append(f"错误: {err}")
            yield None, None, None, "\n".join(logs)
            break


with gr.Blocks(title="视频转文字") as demo:
    gr.Markdown("# 视频转文字（本地版）\n支持本地视频或 Bilibili 网址输入，生成音频稿与字幕稿。")

    with gr.Accordion("提示词示范", open=False):
        gr.Markdown("### 音频稿排版整理提示词")
        gr.Textbox(
            label="可直接复制使用",
            value=PROMPT_DEMO,
            lines=14,
            interactive=False,
        )
        gr.Markdown("### OCR生成字幕稿排版整理提示词")
        gr.Textbox(
            label="可直接复制使用",
            value=PROMPT_DEMO_OCR,
            lines=22,
            interactive=False,
        )

    with gr.Row():
        video_path = gr.File(
            label="本地视频文件",
            file_types=[".mp4", ".mkv", ".flv", ".mov", ".avi"],
            type="filepath",
        )
        audio_path = gr.File(
            label="本地音频文件",
            file_types=[".mp3", ".wav", ".m4a", ".flac"],
            type="filepath",
        )
        bilibili_url = gr.Textbox(label="Bilibili 网址", placeholder="https://www.bilibili.com/video/...", lines=1)

    output_dir = gr.Textbox(label="输出目录", value="outputs", lines=1)

    with gr.Row():
        do_transcribe = gr.Checkbox(value=True, label="启用音频转写 (Whisper)")
        do_ocr = gr.Checkbox(value=False, label="启用字幕识别")
        use_fp16 = gr.Checkbox(value=False, label="使用 FP16（混合精度，加速但可能不兼容）")

    with gr.Accordion("大模型整理", open=False):
        use_llm = gr.Checkbox(value=False, label="使用大模型整理稿件")
        llm_api_key = gr.Textbox(
            label="API Key",
            type="password",
            lines=1,
            visible=True,
        )

    # 计算设备单独一行：把说明作为 Radio 的标签，并在右侧显示提示图标
    with gr.Row():
        # 使用 Markdown/HTML 显示标签与提示，确保提示能被渲染
        gr.Markdown('计算设备 <span title="GPU 通常比 CPU 更快，适合大模型与长音频；若没有 NVIDIA CUDA GPU，请选择 CPU" style="font-size:14px;">ⓘ</span>')
    with gr.Row():    
        device = gr.Radio([
            "CPU",
            "GPU",
        ], value="GPU", label=None)

    with gr.Accordion("高级设置", open=True):
        # 直接选择模型，不需要重复的强度偏好下拉
        with gr.Row():
            model_size = gr.Dropdown(
                ["tiny", "base", "small", "medium", "large", "large-v3", "large-v3-turbo"],
                value="large-v3",
                label="Whisper 模型",
            )
            model_info = gr.HTML('<span title="大模型（large / large-v3）通常提供最高识别精度；turbo 版本在速度/资源上做过优化，可能稍有精度差异" style="font-size:14px; margin-left:6px;">ⓘ</span>')
        language = gr.Dropdown(
            ["zh", "en", "auto"], value="zh", label="语音识别语言"
        )
        ocr_fps = gr.Slider(0.5, 5.0, value=1.0, step=0.5, label="OCR 抽帧频率 (fps)", visible=False)
        ocr_lang = gr.Textbox(label="Tesseract 语言", value="chi_sim+eng", lines=1, visible=False)
        ocr_similarity = gr.Slider(0.7, 0.98, value=0.9, step=0.01, label="OCR 去重相似度阈值", visible=False)

        ocr_region_note = gr.Markdown("OCR 识别区域使用相对比例 (x、y、width、height)，范围 0-1。", visible=False)
        with gr.Row():
            ocr_region_x = gr.Slider(0, 1, value=DEFAULT_OCR_REGION[0], step=0.01, label="OCR 区域 X", visible=False)
            ocr_region_y = gr.Slider(0, 1, value=DEFAULT_OCR_REGION[1], step=0.01, label="OCR 区域 Y", visible=False)
        with gr.Row():
            ocr_region_w = gr.Slider(0, 1, value=DEFAULT_OCR_REGION[2], step=0.01, label="OCR 区域 Width", visible=False)
            ocr_region_h = gr.Slider(0, 1, value=DEFAULT_OCR_REGION[3], step=0.01, label="OCR 区域 Height", visible=False)

        # 删除模型强度偏好选项（直接使用模型下拉）

    ocr_preview_section = gr.Accordion("OCR 区域预览（仅本地视频）", open=False)
    with ocr_preview_section:
        preview_time = gr.Slider(0, 10, value=0.0, step=0.1, label="预览时间点 (秒)")
        preview_image = gr.Image(label="OCR 区域预览", type="numpy")

    run_btn = gr.Button("开始处理", variant="primary")

    with gr.Row():
        audio_txt = gr.File(label="音频稿（txt）")
        subtitle_txt = gr.File(label="字幕稿（txt）")
        llm_txt = gr.File(label="整理稿（txt）")

    log_box = gr.Textbox(label="运行日志", lines=10)

    run_btn.click(
        run_action,
        inputs=[
            video_path,
            audio_path,
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
            device,
            use_fp16,
            use_llm,
            llm_api_key,
        ],
        outputs=[audio_txt, subtitle_txt, llm_txt, log_box],
    )

    def _toggle_ocr_preview(enabled: bool):
        # 控制 OCR 相关控件与预览的可见性（返回顺序须与 outputs 对应）
        v = gr.update(visible=enabled)
        return v, v, v, v, v, v, v, v, v

    do_ocr.change(
        _toggle_ocr_preview,
        inputs=[do_ocr],
        outputs=[
            ocr_preview_section,
            ocr_fps,
            ocr_lang,
            ocr_similarity,
            ocr_region_x,
            ocr_region_y,
            ocr_region_w,
            ocr_region_h,
            ocr_region_note,
        ],
    )

    video_path.change(
        _on_media_selected,
        inputs=[video_path, audio_path, ocr_region_x, ocr_region_y, ocr_region_w, ocr_region_h],
        outputs=[
            preview_time,
            preview_image,
            do_ocr,
            ocr_preview_section,
            ocr_fps,
            ocr_lang,
            ocr_similarity,
            ocr_region_x,
            ocr_region_y,
            ocr_region_w,
            ocr_region_h,
            ocr_region_note,
        ],
    )

    audio_path.change(
        _on_media_selected,
        inputs=[video_path, audio_path, ocr_region_x, ocr_region_y, ocr_region_w, ocr_region_h],
        outputs=[
            preview_time,
            preview_image,
            do_ocr,
            ocr_preview_section,
            ocr_fps,
            ocr_lang,
            ocr_similarity,
            ocr_region_x,
            ocr_region_y,
            ocr_region_w,
            ocr_region_h,
            ocr_region_note,
        ],
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
