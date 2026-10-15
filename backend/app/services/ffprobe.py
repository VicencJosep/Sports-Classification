"""Metadatos de vídeo con ``ffprobe`` (ya instalado en el contenedor). Es la
única fuente de fps/duración del backend: el pipeline no los guarda."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from app.core.errors import UnsupportedMediaError

_BROWSER_CODECS = frozenset({"h264", "vp8", "vp9", "av1"})
_BROWSER_CONTAINERS = frozenset({"mp4", "webm"})


@dataclass(frozen=True)
class VideoProbe:
    container: str
    video_codec: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    start_time_s: float
    has_audio: bool

    @property
    def browser_playable(self) -> bool:
        return self.video_codec in _BROWSER_CODECS and self.container in _BROWSER_CONTAINERS


def _container_name(format_name: str, extension: str) -> str:
    names = {n.strip() for n in format_name.split(",")}
    ext = extension.lstrip(".").lower()
    if ext in names or (ext == "mkv" and "matroska" in names):
        return ext
    if "mp4" in names:
        return "mp4"
    if "webm" in names:
        return "webm"
    if "matroska" in names:
        return "mkv"
    return next(iter(sorted(names)), ext or "unknown")


def _fps(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(key)
        if value and value not in ("0/0", "0"):
            try:
                fps = float(Fraction(value))
            except (ValueError, ZeroDivisionError):
                continue
            if fps > 0:
                return fps
    return 0.0


def probe_video(path: Path, ffprobe_bin: str = "ffprobe", extension: str = "") -> VideoProbe:
    cmd = [
        ffprobe_bin, "-v", "error", "-show_entries",
        "stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,start_time,duration"
        ":format=format_name,duration,start_time",
        "-of", "json", str(path),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UnsupportedMediaError(f"ffprobe could not inspect the file: {exc}") from exc
    if completed.returncode != 0:
        raise UnsupportedMediaError("the file is not a readable video")
    try:
        info = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise UnsupportedMediaError("ffprobe returned unreadable output") from exc

    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise UnsupportedMediaError("the file has no video stream")
    fmt = info.get("format", {})

    fps = _fps(video)
    duration = float(video.get("duration") or fmt.get("duration") or 0.0)
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    if fps <= 0 or duration <= 0 or width <= 0 or height <= 0:
        raise UnsupportedMediaError("the video has no usable fps, duration or resolution")
    nb_frames = video.get("nb_frames")
    frame_count = int(nb_frames) if nb_frames and str(nb_frames).isdigit() else int(round(duration * fps))
    start_time = float(video.get("start_time") or fmt.get("start_time") or 0.0)

    return VideoProbe(
        container=_container_name(str(fmt.get("format_name", "")), extension),
        video_codec=str(video.get("codec_name", "unknown")),
        width=width,
        height=height,
        fps=round(fps, 3),
        frame_count=frame_count,
        duration_s=round(duration, 3),
        start_time_s=round(max(start_time, 0.0), 3),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )
