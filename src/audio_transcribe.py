from typing import Callable, List, Optional

import whisper

from .models import Segment

_MODEL_CACHE = {}


def _get_model(model_size: str, device: str = "cpu"):
    key = f"{model_size}:{device}"
    if key not in _MODEL_CACHE:
        # whisper.load_model accepts device argument
        _MODEL_CACHE[key] = whisper.load_model(model_size, device=device)
    return _MODEL_CACHE[key]


def transcribe_audio(
    audio_path: str,
    model_size: str,
    language: str,
    device: str = "cpu",
    use_fp16: bool = False,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> List[Segment]:
    if progress_cb:
        progress_cb(0.1, "加载 Whisper 模型")
    model = _get_model(model_size, device=device)
    if progress_cb:
        progress_cb(0.2, "开始语音转写")
    lang = None if language in ("auto", "", None) else language
    # Use fp16 when running on CUDA to accelerate and save memory
    # fp16 can trigger specialized CUDA kernels that may be incompatible with
    # some GPU builds. Expose control via `use_fp16`; default False to avoid
    # unexpected kernel errors on cutting-edge GPUs.
    fp16 = bool(use_fp16) and (isinstance(device, str) and device.startswith("cuda"))
    result = model.transcribe(audio_path, language=lang, fp16=fp16, verbose=False)
    segments = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        segments.append(
            Segment(float(seg.get("start", 0.0)), float(seg.get("end", 0.0)), text)
        )
    if progress_cb:
        progress_cb(1.0, "语音转写完成")
    return segments
