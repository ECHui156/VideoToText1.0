from typing import Callable, List, Optional

import whisper

from .models import Segment

_MODEL_CACHE = {}


def _get_model(model_size: str):
    if model_size not in _MODEL_CACHE:
        _MODEL_CACHE[model_size] = whisper.load_model(model_size)
    return _MODEL_CACHE[model_size]


def transcribe_audio(
    audio_path: str,
    model_size: str,
    language: str,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> List[Segment]:
    if progress_cb:
        progress_cb(0.1, "加载 Whisper 模型")
    model = _get_model(model_size)
    if progress_cb:
        progress_cb(0.2, "开始语音转写")
    lang = None if language in ("auto", "", None) else language
    result = model.transcribe(audio_path, language=lang, fp16=False, verbose=False)
    segments = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        segments.append(
            Segment(float(seg.get("start", 0.0)), float(seg.get("end", 0.0)), text)
        )
    if progress_cb:
        progress_cb(1.0, "语音转写完成")
    return segments
