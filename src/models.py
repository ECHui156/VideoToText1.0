from dataclasses import dataclass
from typing import List


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class PipelineOutput:
    video_path: str
    audio_segments: List[Segment]
    subtitle_segments: List[Segment]
    audio_txt_path: str
    subtitle_txt_path: str
    logs: List[str]
