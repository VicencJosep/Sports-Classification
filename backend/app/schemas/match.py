from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .common import AnalysisMode, AnalysisStatus, PartKind
from .highlights import HighlightsSummary
from .video import Video


class MatchPart(BaseModel):
    id: str
    kind: PartKind
    label: str
    mode: AnalysisMode
    status: AnalysisStatus
    stage: str | None = None            # etapa aproximada mientras está en "processing"
    error: str | None = None
    video: Video | None
    highlights: HighlightsSummary | None = None   # None: nunca se han generado


class Match(BaseModel):
    id: str
    name: str
    home_team: str | None
    away_team: str | None
    date: str | None                    # YYYY-MM-DD
    created_at: str
    status: AnalysisStatus
    error_message: str | None
    parts: list[MatchPart]
    duration_s: float | None


class MatchPartCreate(BaseModel):
    kind: PartKind
    video_id: str


class MatchCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    home_team: str | None = Field(default=None, max_length=100)
    away_team: str | None = Field(default=None, max_length=100)
    date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    parts: list[MatchPartCreate] = Field(min_length=1, max_length=2)

    @field_validator("parts")
    @classmethod
    def _parts_consistent(cls, parts: list[MatchPartCreate]) -> list[MatchPartCreate]:
        kinds = [p.kind for p in parts]
        if len(set(kinds)) != len(kinds):
            raise ValueError("duplicate part kind")
        if "full" in kinds and len(kinds) > 1:
            raise ValueError("a full-match part cannot be combined with halves")
        return parts


# --- registros persistidos -------------------------------------------------- #

class MatchPartRecord(BaseModel):
    id: str
    kind: PartKind
    label: str
    video_id: str
    mode: AnalysisMode = "full-match"


class MatchRecord(BaseModel):
    id: str
    name: str
    home_team: str | None
    away_team: str | None
    date: str | None
    created_at: str
    parts: list[MatchPartRecord]
