"""Repositorios de registros (vídeos, partidos, ejecuciones) en ficheros JSON.

Interfaces mínimas (``get``/``save``/``list_all``) para que PostgreSQL pueda
sustituirlas más adelante sin tocar los servicios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.schemas import AnalysisRun, MatchRecord, VideoRecord

from .json_store import read_json, write_json_atomic


class VideoRepository(Protocol):
    def get(self, video_id: str) -> VideoRecord | None: ...
    def save(self, record: VideoRecord) -> None: ...


class MatchRepository(Protocol):
    def get(self, match_id: str) -> MatchRecord | None: ...
    def save(self, record: MatchRecord) -> None: ...
    def list_all(self) -> list[MatchRecord]: ...


class RunStore(Protocol):
    def part_dir(self, match_id: str, part_id: str) -> Path: ...
    def get(self, match_id: str, part_id: str) -> AnalysisRun | None: ...
    def save(self, run: AnalysisRun) -> None: ...
    def list_all(self) -> list[AnalysisRun]: ...


class FileVideoRepository:
    """``<videos_dir>/<video_id>/video.json`` (junto al propio vídeo)."""

    def __init__(self, videos_dir: Path) -> None:
        self.videos_dir = videos_dir

    def get(self, video_id: str) -> VideoRecord | None:
        data = read_json(self.videos_dir / video_id / "video.json")
        return VideoRecord.model_validate(data) if data else None

    def save(self, record: VideoRecord) -> None:
        write_json_atomic(self.videos_dir / record.id / "video.json", record.model_dump())


class FileMatchRepository:
    """``<matches_dir>/<match_id>/match.json``."""

    def __init__(self, matches_dir: Path) -> None:
        self.matches_dir = matches_dir
        self.matches_dir.mkdir(parents=True, exist_ok=True)

    def get(self, match_id: str) -> MatchRecord | None:
        data = read_json(self.matches_dir / match_id / "match.json")
        return MatchRecord.model_validate(data) if data else None

    def save(self, record: MatchRecord) -> None:
        write_json_atomic(self.matches_dir / record.id / "match.json", record.model_dump())

    def list_all(self) -> list[MatchRecord]:
        records = []
        for path in self.matches_dir.glob("*/match.json"):
            data = read_json(path)
            if data:
                records.append(MatchRecord.model_validate(data))
        return sorted(records, key=lambda m: m.created_at, reverse=True)


class FileRunStore:
    """``<outputs_dir>/<match_id>/<part_id>/run.json``; en la misma carpeta van
    ``pipeline.log``, ``output/`` (``--output-dir`` del pipeline) y ``memory/``."""

    def __init__(self, outputs_dir: Path) -> None:
        self.outputs_dir = outputs_dir
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def part_dir(self, match_id: str, part_id: str) -> Path:
        return self.outputs_dir / match_id / part_id

    def get(self, match_id: str, part_id: str) -> AnalysisRun | None:
        data = read_json(self.part_dir(match_id, part_id) / "run.json")
        return AnalysisRun.model_validate(data) if data else None

    def save(self, run: AnalysisRun) -> None:
        write_json_atomic(self.part_dir(run.match_id, run.part_id) / "run.json", run.model_dump())

    def list_all(self) -> list[AnalysisRun]:
        runs = []
        for path in self.outputs_dir.glob("*/*/run.json"):
            data = read_json(path)
            if data:
                runs.append(AnalysisRun.model_validate(data))
        return sorted(runs, key=lambda r: r.created_at)
