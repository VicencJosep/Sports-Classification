"""Almacenamiento físico de los vídeos subidos. Interfaz pequeña para poder
sustituir el disco local por otra cosa sin tocar rutas ni servicios."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO, Protocol

from app.core.errors import PayloadTooLargeError

_CHUNK = 4 * 1024 * 1024


class VideoStorage(Protocol):
    def video_dir(self, video_id: str) -> Path: ...
    def stage_upload(self, stream: BinaryIO, max_bytes: int) -> tuple[Path, int]: ...
    def commit(self, staged: Path, video_id: str, filename: str) -> Path: ...
    def discard(self, staged: Path) -> None: ...


class LocalVideoStorage:
    """``<root>/<video_id>/<filename>``; la subida se escribe por trozos en
    ``<tmp_dir>`` (mismo sistema de ficheros) y se mueve con un rename."""

    def __init__(self, root: Path, tmp_dir: Path) -> None:
        self.root = root
        self.tmp_dir = tmp_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def video_dir(self, video_id: str) -> Path:
        return self.root / video_id

    def stage_upload(self, stream: BinaryIO, max_bytes: int) -> tuple[Path, int]:
        staged = self.tmp_dir / f"upload-{uuid.uuid4().hex}.part"
        size = 0
        try:
            with open(staged, "wb") as out:
                while True:
                    chunk = stream.read(_CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise PayloadTooLargeError(f"video exceeds the maximum upload size ({max_bytes} bytes)")
                    out.write(chunk)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
        return staged, size

    def commit(self, staged: Path, video_id: str, filename: str) -> Path:
        target_dir = self.video_dir(video_id)
        target_dir.mkdir(parents=True, exist_ok=False)
        target = target_dir / filename
        try:
            os.replace(staged, target)
        except OSError:
            shutil.move(str(staged), str(target))
        return target

    def discard(self, staged: Path) -> None:
        staged.unlink(missing_ok=True)
