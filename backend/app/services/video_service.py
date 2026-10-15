from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from app.core.config import Settings
from app.core.errors import NotFoundError, UnsupportedMediaError
from app.repositories.records import VideoRepository
from app.repositories.video_storage import VideoStorage
from app.schemas import Video, VideoRecord
from app.schemas.common import new_id, utcnow_iso

from .ffprobe import probe_video

_MEDIA_TYPES = {
    "mp4": "video/mp4", "webm": "video/webm", "mkv": "video/x-matroska",
    "mov": "video/quicktime", "avi": "video/x-msvideo",
}


class VideoService:
    def __init__(self, settings: Settings, storage: VideoStorage, repo: VideoRepository) -> None:
        self.settings = settings
        self.storage = storage
        self.repo = repo

    def upload(self, original_filename: str, stream: BinaryIO) -> VideoRecord:
        name = Path(original_filename or "").name or "video"
        ext = Path(name).suffix.lower()
        if ext not in self.settings.allowed_extension_set:
            raise UnsupportedMediaError(
                f"unsupported extension {ext or '(none)'}; allowed: {', '.join(sorted(self.settings.allowed_extension_set))}"
            )
        staged, size = self.storage.stage_upload(stream, self.settings.max_upload_bytes)
        try:
            probe = probe_video(staged, self.settings.ffprobe_bin, extension=ext)
        except Exception:
            self.storage.discard(staged)
            raise
        video_id = new_id()
        stored_filename = f"source{ext}"   # el pipeline deja sus cachés como source.tracks.jsonl, etc.
        self.storage.commit(staged, video_id, stored_filename)
        record = VideoRecord(
            id=video_id,
            original_filename=name,
            stored_filename=stored_filename,
            size_bytes=size,
            container=probe.container,
            video_codec=probe.video_codec,
            width=probe.width,
            height=probe.height,
            fps=probe.fps,
            frame_count=probe.frame_count,
            duration_s=probe.duration_s,
            start_time_s=probe.start_time_s,
            has_audio=probe.has_audio,
            browser_playable=probe.browser_playable,
            uploaded_at=utcnow_iso(),
        )
        self.repo.save(record)
        return record

    def get(self, video_id: str) -> VideoRecord:
        record = self.repo.get(video_id)
        if record is None:
            raise NotFoundError(f"video {video_id} does not exist")
        return record

    def path(self, record: VideoRecord) -> Path:
        return self.storage.video_dir(record.id) / record.stored_filename

    def content(self, video_id: str) -> tuple[Path, str]:
        record = self.get(video_id)
        path = self.path(record)
        if not path.exists():
            raise NotFoundError(f"video file for {video_id} is missing")
        return path, _MEDIA_TYPES.get(record.container, "application/octet-stream")

    def to_public(self, record: VideoRecord) -> Video:
        data = record.model_dump(exclude={"stored_filename", "content_url"})
        return Video(**data, content_url=f"{self.settings.public_base_url}/api/videos/{record.id}/content")
