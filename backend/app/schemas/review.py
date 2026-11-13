"""Copia revisada del análisis de una parte (``review/analysis.json``).

Tiene la misma forma que ``memory/analysis.json`` (``PartAnalysis``) más lo que añade la corrección a mano:
quién ejecuta un evento, el lado (local/visitante) y si el elemento se ha tocado. Es de donde sale la hoja de
verdad (``tiempo,etiqueta,dorsal,equipo``) que leen ``scripts/annotation_csv.py`` y ``scripts/evaluate_match.py``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .analysis import BallAction, MatchEvent, PartAnalysis

Side = Literal["home", "away"]
ReviewStatus = Literal["edited", "added"]      # None = tal cual lo dejó el pipeline
ItemKind = Literal["action", "event"]


class ReviewedEvent(MatchEvent):
    tracker_id: int | None = None       # ejecutor (el pipeline no lo da para los eventos)
    team_cluster: int | None = None
    jersey: int | None = None
    side: Side | None = None
    review: ReviewStatus | None = None


class ReviewedBallAction(BallAction):
    side: Side | None = None
    review: ReviewStatus | None = None


class ReviewInfo(BaseModel):
    exists: bool = False                # False: aún no se ha corregido nada (no hay copia en disco)
    created_at: str | None = None
    updated_at: str | None = None
    n_edited: int = 0
    n_added: int = 0
    n_deleted: int = 0
    # El pipeline se volvió a ejecutar después de crear la copia: sus tracker_id ya no son los de la copia.
    stale: bool = False
    # Lado ya indicado para un tracklet: vale para todas sus acciones mientras no lleven uno propio.
    tracker_sides: dict[int, Side] = Field(default_factory=dict)


class ReviewedAnalysis(PartAnalysis):
    """Lo que enseña el visor: la copia revisada si existe; si no, la salida del pipeline con ``review.exists`` falso."""

    events: list[ReviewedEvent]
    ball_actions: list[ReviewedBallAction]
    review: ReviewInfo


class ReviewFile(PartAnalysis):
    """``review/analysis.json``: el análisis corregido más lo borrado (para deshacer y para auditar)."""

    events: list[ReviewedEvent]
    ball_actions: list[ReviewedBallAction]
    created_at: str
    updated_at: str
    based_on_run: str | None = None     # ``finished_at`` de la ejecución del pipeline de la que se copió
    tracker_sides: dict[int, Side] = Field(default_factory=dict)
    deleted_events: list[ReviewedEvent] = Field(default_factory=list)
    deleted_ball_actions: list[ReviewedBallAction] = Field(default_factory=list)


class ReviewItemPatch(BaseModel):
    """Campos a cambiar; los que no vienen en el cuerpo no se tocan. ``tracker_id: null`` = ejecutor desconocido
    y ``side: null`` = sin lado."""

    label: str | None = None
    time_s: float | None = Field(default=None, ge=0)
    tracker_id: int | None = Field(default=None, ge=0)
    side: Side | None = None


class ReviewItemCreate(BaseModel):
    kind: ItemKind
    label: str
    time_s: float = Field(ge=0)
    tracker_id: int | None = Field(default=None, ge=0)
    side: Side | None = None


class ReviewItemResult(BaseModel):
    kind: ItemKind
    id: str
    deleted: bool = False
    action: ReviewedBallAction | None = None
    event: ReviewedEvent | None = None
    review: ReviewInfo


class ReviewExportRequest(BaseModel):
    from_s: float = Field(default=0, ge=0)
    to_s: float | None = Field(default=None, ge=0)     # None = hasta el final


class ReviewExport(BaseModel):
    from_s: float
    to_s: float
    n_rows: int
    filename: str
    path: str       # dónde quedó guardada la hoja, relativa a la raíz del repo (para pasarla a evaluate_match)
    csv: str
