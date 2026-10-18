"""Composición de servicios (una vez por aplicación) y acceso desde las rutas."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from fastapi import Request

from app.core.config import Settings
from app.core.errors import ApiError
from app.repositories.records import FileMatchRepository, FileRunStore, FileVideoRepository
from app.repositories.video_storage import LocalVideoStorage
from app.schemas import AnalysisRun
from app.services.analysis_service import AnalysisService
from app.services.football_memory import FootballMemory
from app.services.highlights_service import HighlightsService
from app.services.match_service import MatchService
from app.services.pipeline import PipelineLauncher
from app.services.video_service import VideoService

logger = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    videos: VideoService
    matches: MatchService
    analysis: AnalysisService
    memory: FootballMemory
    highlights: HighlightsService

    def start_workers(self) -> None:
        self.analysis.start_worker()
        self.highlights.start_worker()

    def stop_workers(self) -> None:
        self.analysis.stop_worker()
        self.highlights.stop_worker()


def build_services(settings: Settings) -> Services:
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    video_repo = FileVideoRepository(settings.videos_dir)
    match_repo = FileMatchRepository(settings.matches_dir)
    runs = FileRunStore(settings.outputs_dir)
    gpu_lock = threading.Lock()
    videos = VideoService(settings, LocalVideoStorage(settings.videos_dir, settings.tmp_dir), video_repo)
    memory = FootballMemory(settings, runs, video_repo)
    analysis = AnalysisService(settings, match_repo, video_repo, runs, PipelineLauncher(settings), memory, gpu_lock)
    highlights = HighlightsService(settings, runs, gpu_lock)
    matches = MatchService(match_repo, runs, videos, analysis, highlights)

    def chain_highlights(run: AnalysisRun) -> None:
        try:
            highlights.start(run.match_id, run.part_id)
        except ApiError as exc:
            logger.warning("highlights not chained for %s/%s: %s", run.match_id, run.part_id, exc)

    analysis.on_completed = chain_highlights
    return Services(
        settings=settings, videos=videos, matches=matches, analysis=analysis, memory=memory, highlights=highlights,
    )


def get_services(request: Request) -> Services:
    return request.app.state.services
