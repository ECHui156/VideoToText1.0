import os
import re
import shutil
from difflib import SequenceMatcher
from typing import Callable, List, Optional, Tuple

import cv2
import pytesseract
from PIL import Image

from .models import Segment

_PUNCT_RE = re.compile(r"[，。！？,.!?\"“”'‘’·、:：;；【】\[\]（）()《》<>]+")

OCRRegion = Tuple[float, float, float, float]
DEFAULT_OCR_REGION: OCRRegion = (0.0, 0.6, 1.0, 0.4)


def configure_tesseract() -> None:
    env_path = os.getenv("TESSERACT_CMD") or os.getenv("TESSERACT_PATH")
    if env_path:
        pytesseract.pytesseract.tesseract_cmd = env_path


def ensure_tesseract() -> None:
    configure_tesseract()
    cmd = pytesseract.pytesseract.tesseract_cmd or "tesseract"
    if shutil.which(cmd) is None and not os.path.exists(cmd):
        raise RuntimeError("未检测到 Tesseract，请安装并配置 PATH，或设置 TESSERACT_CMD。")


def _clean_text(text: str) -> str:
    text = (text or "").replace("\x0c", "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_text(text: str) -> str:
    text = _clean_text(text)
    text = _PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", "", text)
    return text


def validate_ocr_region(region: OCRRegion) -> OCRRegion:
    if region is None:
        raise ValueError("OCR 区域不能为空。")
    if len(region) != 4:
        raise ValueError("OCR 区域必须包含 x、y、width、height 四个值。")

    x, y, width, height = region
    values = {"x": x, "y": y, "width": width, "height": height}
    normalized = {}
    for name, value in values.items():
        if value is None:
            raise ValueError(f"OCR 区域 {name} 不能为空。")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"OCR 区域 {name} 必须是数字。")
        if value < 0 or value > 1:
            raise ValueError(f"OCR 区域 {name} 必须在 0 到 1 之间。")
        normalized[name] = value

    if normalized["width"] <= 0:
        raise ValueError("OCR 区域 width 必须大于 0。")
    if normalized["height"] <= 0:
        raise ValueError("OCR 区域 height 必须大于 0。")
    if normalized["x"] + normalized["width"] > 1:
        raise ValueError("OCR 区域 x + width 不能超过 1。")
    if normalized["y"] + normalized["height"] > 1:
        raise ValueError("OCR 区域 y + height 不能超过 1。")

    return (
        normalized["x"],
        normalized["y"],
        normalized["width"],
        normalized["height"],
    )


def resolve_ocr_region(region: Optional[OCRRegion]) -> OCRRegion:
    if region is None:
        return DEFAULT_OCR_REGION
    return validate_ocr_region(region)


def _get_ocr_region_pixels(frame_shape, region: OCRRegion) -> Tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("无效的帧尺寸，无法计算 OCR 区域。")

    x, y, w, h = region
    x1 = int(width * x)
    y1 = int(height * y)
    x2 = int(width * (x + w))
    y2 = int(height * (y + h))

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def get_ocr_region_pixels(
    frame_shape, region: Optional[OCRRegion]
) -> Tuple[int, int, int, int]:
    return _get_ocr_region_pixels(frame_shape, resolve_ocr_region(region))


def draw_ocr_region(
    frame,
    region: Optional[OCRRegion],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
):
    if frame is None:
        return None
    x1, y1, x2, y2 = get_ocr_region_pixels(frame.shape, region)
    preview = frame.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), color, thickness)
    return preview


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件，无法获取时长。")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0 or frame_count <= 0:
        return 0.0
    return float(frame_count / fps)


def extract_frame_at_timestamp(video_path: str, timestamp_sec: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件，无法生成预览。")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

    if timestamp_sec is None:
        timestamp_sec = 0.0
    try:
        timestamp_sec = float(timestamp_sec)
    except (TypeError, ValueError):
        timestamp_sec = 0.0

    if duration > 0:
        timestamp_sec = min(max(timestamp_sec, 0.0), max(duration - 0.001, 0.0))
    else:
        timestamp_sec = max(timestamp_sec, 0.0)

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
    ret, frame = cap.read()
    if not ret and frame_count > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(frame_count - 1), 0))
        ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _ocr_frame(frame, lang: str, ocr_region: OCRRegion) -> str:
    # 仅截取指定区域以提升字幕识别准确率
    x1, y1, x2, y2 = _get_ocr_region_pixels(frame.shape, ocr_region)
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pil = Image.fromarray(thresh)
    text = pytesseract.image_to_string(pil, lang=lang, config="--psm 6")
    return _clean_text(text)


def ocr_subtitles_from_video(
    video_path: str,
    ocr_fps: float = 1.0,
    lang: str = "chi_sim+eng",
    similarity_threshold: float = 0.9,
    min_text_len: int = 2,
    ocr_region: Optional[OCRRegion] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> List[Segment]:
    ensure_tesseract()
    ocr_region = resolve_ocr_region(ocr_region)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件，OCR 失败。")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(video_fps / ocr_fps)), 1) if ocr_fps > 0 else 1
    frame_duration = 1.0 / ocr_fps if ocr_fps > 0 else 1.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    frame_index = 0
    segments: List[Segment] = []
    max_gap = max(frame_duration * 2.5, 1.0 / max(ocr_fps, 1.0))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % step != 0:
            frame_index += 1
            continue

        timestamp = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        text = _ocr_frame(frame, lang, ocr_region)
        if text and len(text) >= min_text_len:
            norm = _normalize_text(text)
            if segments:
                last = segments[-1]
                sim = SequenceMatcher(None, norm, _normalize_text(last.text)).ratio()
                gap = timestamp - last.end
                if norm and sim >= similarity_threshold and gap <= max_gap:
                    last.end = max(last.end, timestamp + frame_duration)
                    if len(text) > len(last.text):
                        last.text = text
                else:
                    segments.append(Segment(timestamp, timestamp + frame_duration, text))
            else:
                segments.append(Segment(timestamp, timestamp + frame_duration, text))

        frame_index += 1
        if progress_cb and total_frames:
            progress_cb(min(frame_index / total_frames, 1.0), "OCR处理中")

    cap.release()
    if progress_cb:
        progress_cb(1.0, "OCR完成")
    return segments
