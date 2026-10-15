"""Football Memory v1: el conocimiento estructurado de una parte de partido,
construido a partir de las salidas crudas del pipeline al terminar cada
ejecución (``ingest``) y consultado por la API.

* Eventos, acciones de balón y equipos se normalizan una vez y se guardan en
  ``memory/analysis.json`` (pequeño: ~2 MB por partido).
* El tracking se queda en los JSONL del pipeline (250-330 MB por partido) y
  se sirve por ventanas de tiempo con un índice de offsets (``memory/*.idx``).
"""

from __future__ import annotations

import math
from pathlib import Path

from app.adapters.telemetry import build_part_analysis
from app.adapters.tracklets import summarize_tracklets
from app.adapters.tracks import to_tracking_frame
from app.core.config import Settings
from app.core.errors import InvalidInputError, MalformedOutputError, NotReadyError
from app.repositories.json_store import read_json, write_json_atomic
from app.repositories.records import RunStore, VideoRepository
from app.repositories.tracking_index import TrackingIndex
from app.schemas import AnalysisRun, PartAnalysis, Tracklet, TrackingWindow, VideoRecord


class FootballMemory:
    def __init__(self, settings: Settings, runs: RunStore, videos: VideoRepository) -> None:
        self.settings = settings
        self.runs = runs
        self.videos = videos

    # --- rutas ---------------------------------------------------------------- #

    def output_dir(self, match_id: str, part_id: str) -> Path:
        return self.runs.part_dir(match_id, part_id) / "output"

    def memory_dir(self, match_id: str, part_id: str) -> Path:
        return self.runs.part_dir(match_id, part_id) / "memory"

    def _video_path(self, video: VideoRecord) -> Path:
        return self.settings.videos_dir / video.id / video.stored_filename

    def _tracking_sources(self, run: AnalysisRun, video: VideoRecord) -> tuple[Path | None, Path | None]:
        """(fichero base más rico, fichero de posesión)."""
        video_path = self._video_path(video)
        stem = video_path.stem
        tracks_2d = self.output_dir(run.match_id, run.part_id) / "tracks_2d.jsonl"
        tracks = video_path.with_name(f"{stem}.tracks.jsonl")
        possession = video_path.with_name(f"{stem}.possession.jsonl")
        base = tracks_2d if tracks_2d.exists() else tracks if tracks.exists() else None
        return base, possession if possession.exists() else None

    # --- ingesta --------------------------------------------------------------- #

    def ingest(self, run: AnalysisRun, video: VideoRecord) -> PartAnalysis:
        """Salidas crudas -> ``memory/``. Lanza MalformedOutputError si falta o
        no cuadra algo: el servicio de análisis lo convierte en estado ``error``."""
        out = self.output_dir(run.match_id, run.part_id)
        telemetry = read_json(out / "telemetry.json")
        if telemetry is None:
            raise MalformedOutputError("the pipeline finished without writing telemetry.json")
        teams = read_json(out / "teams.json")
        analysis = build_part_analysis(
            telemetry, teams, match_id=run.match_id, part_id=run.part_id, fps=video.fps, duration_s=video.duration_s
        )
        mem = self.memory_dir(run.match_id, run.part_id)
        mem.mkdir(parents=True, exist_ok=True)

        base, possession = self._tracking_sources(run, video)
        meta: dict = {"base": str(base) if base else None, "possession": str(possession) if possession else None}
        tracklets: list[Tracklet] = []
        if base is not None:
            TrackingIndex.build(base).save(mem / "base.idx")
            tracklets = summarize_tracklets(base, analysis.teams)
        if possession is not None:
            TrackingIndex.build(possession).save(mem / "possession.idx")
        write_json_atomic(mem / "tracking.json", meta)
        write_json_atomic(mem / "tracklets.json", [t.model_dump() for t in tracklets])

        # "Quién" hizo cada acción: el dorsal mayoritario de su tracklet, cuando se leyó alguno.
        jersey_of = {t.tracker_id: t.jersey for t in tracklets}
        for action in analysis.ball_actions:
            if action.tracker_id is not None:
                action.jersey = jersey_of.get(action.tracker_id)
        write_json_atomic(mem / "analysis.json", analysis.model_dump())
        return analysis

    # --- consulta -------------------------------------------------------------- #

    def _require_completed(self, match_id: str, part_id: str) -> AnalysisRun:
        run = self.runs.get(match_id, part_id)
        if run is None or run.status != "completed":
            status = run.status if run else "pending"
            raise NotReadyError(f"part {part_id} of match {match_id} has no completed analysis (status: {status})")
        return run

    def get_part_analysis(self, match_id: str, part_id: str) -> PartAnalysis:
        self._require_completed(match_id, part_id)
        data = read_json(self.memory_dir(match_id, part_id) / "analysis.json")
        if data is None:
            raise NotReadyError(f"analysis of part {part_id} is not ingested yet")
        return PartAnalysis.model_validate(data)

    def reingest(self, match_id: str, part_id: str) -> PartAnalysis:
        """Vuelve a normalizar las salidas crudas de una parte ya analizada (p.ej.
        tras cambiar los adaptadores). No toca el pipeline."""
        run = self._require_completed(match_id, part_id)
        video = self.videos.get(run.video_id)
        if video is None:
            raise NotReadyError(f"video {run.video_id} of part {part_id} is missing")
        return self.ingest(run, video)

    def get_tracklets(self, match_id: str, part_id: str) -> list[Tracklet]:
        self._require_completed(match_id, part_id)
        data = read_json(self.memory_dir(match_id, part_id) / "tracklets.json")
        return [Tracklet.model_validate(t) for t in (data or [])]

    def get_tracking_window(self, match_id: str, part_id: str, from_s: float, to_s: float) -> TrackingWindow:
        run = self._require_completed(match_id, part_id)
        if to_s < from_s:
            raise InvalidInputError("to_s must be >= from_s")
        if to_s - from_s > self.settings.tracking_max_window_s:
            raise InvalidInputError(f"window too large: max {self.settings.tracking_max_window_s:g} s")
        video = self.videos.get(run.video_id)
        if video is None:
            raise NotReadyError(f"video {run.video_id} of part {part_id} is missing")
        mem = self.memory_dir(match_id, part_id)
        meta = read_json(mem / "tracking.json") or {}
        base = Path(meta["base"]) if meta.get("base") else None
        possession = Path(meta["possession"]) if meta.get("possession") else None
        frames = []
        if base is not None and (mem / "base.idx").exists():
            from_frame = max(0, math.floor(from_s * video.fps))
            to_frame = math.ceil(to_s * video.fps)
            records = TrackingIndex.load(mem / "base.idx").read_window(base, from_frame, to_frame)
            possessor: dict[int, int | None] = {}
            if possession is not None and (mem / "possession.idx").exists():
                for rec in TrackingIndex.load(mem / "possession.idx").read_window(possession, from_frame, to_frame):
                    possessor[int(rec["frame_idx"])] = rec.get("possessor_tracker_id")
            frames = [to_tracking_frame(rec, possessor.get(int(rec["frame_idx"]))) for rec in records]
        return TrackingWindow(match_id=match_id, part_id=part_id, from_s=from_s, to_s=to_s, fps=video.fps, frames=frames)
