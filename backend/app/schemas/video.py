from __future__ import annotations

from pydantic import BaseModel, Field


class Video(BaseModel):
    """Vídeo subido, con los metadatos que ``ffprobe`` da de forma fiable."""

    id: str
    original_filename: str
    size_bytes: int
    container: str                      # "mp4" | "webm" | "mkv" | "mov" | "avi" | otro nombre de ffprobe
    video_codec: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    start_time_s: float = 0.0
    has_audio: bool = False
    browser_playable: bool
    uploaded_at: str
    content_url: str = Field(description="URL del fichero (soporta Range)")


class VideoRecord(Video):
    """Registro persistido: lo público más el nombre con el que está guardado."""

    stored_filename: str
    content_url: str = ""
