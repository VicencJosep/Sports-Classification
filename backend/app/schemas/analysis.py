from __future__ import annotations

from pydantic import BaseModel, Field

from .common import AnalysisMode, AnalysisStatus


class AnalysisRun(BaseModel):
    """Estado de una ejecución del pipeline sobre una parte (``run.json``)."""

    match_id: str
    part_id: str
    video_id: str
    mode: AnalysisMode
    status: AnalysisStatus = "pending"
    stage: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    exit_code: int | None = None
    pid: int | None = None
    command: list[str] = Field(default_factory=list)
    pipeline_commit: str | None = None


class AnalysisStartRequest(BaseModel):
    mode: AnalysisMode = "full-match"


class PartAnalysisStatus(BaseModel):
    part_id: str
    status: AnalysisStatus
    stage: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


class AnalysisStatusResponse(BaseModel):
    match_id: str
    status: AnalysisStatus
    parts: list[PartAnalysisStatus]


# --- resultados normalizados (Football Memory) ------------------------------- #

class MatchEvent(BaseModel):
    id: str
    label: str                  # vocabulario canónico en snake_case: shot, goal, kick_off, throw_in, ...
    start_s: float
    end_s: float
    peak_confidence: float
    mean_confidence: float
    n_windows: int


class BallAction(BaseModel):
    id: str
    frame_idx: int
    timestamp_s: float
    tracker_id: int | None      # None cuando el pipeline no resolvió poseedor
    team_cluster: int | None
    jersey: int | None = None   # dorsal mayoritario leído para ese tracklet (si se leyó alguna vez)
    label: str                  # pass, drive, high_pass, header, throw_in, background, ...
    confidence: float
    probabilities: dict[str, float]
    is_background: bool
    spot_score: float | None


class TeamAssignments(BaseModel):
    clusters: list[int]
    assignments: dict[int, int]     # tracker_id -> cluster
    excluded: dict[int, str]        # tracker_id -> motivo


class Tracklet(BaseModel):
    """Resumen de un ``tracker_id`` a lo largo de la parte. Es un tracklet, no un
    jugador: el mismo jugador recibe varios ids a lo largo de un partido."""

    tracker_id: int
    team_cluster: int | None
    role: str                   # rol mayoritario
    jersey: int | None          # dorsal mayoritario (None si nunca se leyó)
    first_frame: int
    last_frame: int
    frame_count: int


class PartAnalysis(BaseModel):
    match_id: str
    part_id: str
    mode: AnalysisMode
    frame_count: int
    source_video_path: str | None
    events: list[MatchEvent]
    ball_actions: list[BallAction]
    teams: TeamAssignments
