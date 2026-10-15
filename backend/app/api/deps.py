"""Composición de servicios (una vez por aplicación) y acceso desde las rutas."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.core.config import Settings
from app.repositories.records import FileMatchRepository, FileRunStore, FileVideoRepository
from app.repositories.video_storage import LocalVideoStorage
from app.services.analysis_service import AnalysisService
from app.services.football_memory import FootballMemory
from app.services.match_service import MatchService
from app.services.pipeline import PipelineLauncher
from app.services.video_service import VideoService


@dataclass
class Services:
    settings: Settings
    videos: VideoService
    matches: MatchService
    analysis: AnalysisService
    memory: FootballMemory


def build_services(settings: Settings) -> Services:
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    video_repo = FileVideoRepository(settings.videos_dir)
    match_repo = FileMatchRepository(settings.matches_dir)
    runs = FileRunStore(settings.outputs_dir)
    videos = VideoService(settings, LocalVideoStorage(settings.videos_dir, settings.tmp_dir), video_repo)
    memory = FootballMemory(settings, runs, video_repo)
    analysis = AnalysisService(settings, match_repo, video_repo, runs, PipelineLauncher(settings), memory)
    matches = MatchService(match_repo, runs, videos, analysis)
    return Services(settings=settings, videos=videos, matches=matches, analysis=analysis, memory=memory)


def get_services(request: Request) -> Services:
    return request.app.state.services
