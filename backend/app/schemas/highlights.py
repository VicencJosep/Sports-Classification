"""Mejores jugadas de una parte ya analizada (``analytics/highlights``)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .common import AnalysisStatus


class HighlightsRun(BaseModel):
    """Estado de una generación de highlights sobre una parte (``highlights_run.json``)."""

    match_id: str
    part_id: str
    status: AnalysisStatus = "pending"
    force: bool = False                 # rehacer señales y veredictos del VLM (no reutilizar cachés)
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    exit_code: int | None = None
    pid: int | None = None
    command: list[str] = Field(default_factory=list)


class HighlightsStartRequest(BaseModel):
    force: bool = False


class HighlightsSummary(BaseModel):
    """Lo justo para la ficha del partido: si hay clips y si se están generando."""

    status: AnalysisStatus
    stage: str | None = None            # etapa en curso (signals, vlm 3/10, cut 2/5...) mientras procesa
    n_clips: int = 0
    error: str | None = None


class HighlightClip(BaseModel):
    id: str                             # hl-01, hl-02...
    rank: int
    label: str                          # goal | shot | play
    title: str
    description: str
    start_s: float
    end_s: float
    duration_s: float
    anchor_s: float                     # momento clave (tiro/gol) dentro del partido
    team_cluster: int | None
    final_score: float
    vlm_rank: int | None = None
    content_url: str | None             # None si el clip no se recortó


class HighlightsResponse(BaseModel):
    match_id: str
    part_id: str
    status: AnalysisStatus
    stage: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    generated_at: str | None = None
    vlm_model: str | None = None
    n_candidates: int = 0
    clips: list[HighlightClip] = Field(default_factory=list)
