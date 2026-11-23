"""
core/team_possession/io_pipeline.py
===================================
Adaptador de **nuestro pipeline** a :class:`PositionalData`: lee el
``tracks_2d.jsonl`` que escribe ``core/homography.py::project_tracks`` (cada
track y el balón con ``field_xy`` en metros, origen en la esquina) y el
``teams.json`` de ``core/classification.py`` (``tracker_id`` -> ``team_cluster``).

Cada ``tracker_id`` con rol ``player`` o ``goalkeeper`` es un slot (como en el
ATD del reto, un slot es un tracklet, no una persona: las reentradas sin
resolver de BoT-SORT crean slots nuevos, y eso aquí no importa porque solo se
usa el equipo del slot). El clúster 0 se toma como local salvo que se indique
``home_cluster``. El balón sale de ``ball.field_xy`` (proyección con z=0: con
el balón en el aire la posición en el campo es incorrecta por construcción).

    data = load_tracks_2d("partido.tracks_2d.jsonl", "partido.teams.json", fps=25)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .frames import AWAY, HOME, NO_TEAM, PositionalData

_PLAYER_ROLES = ("player", "goalkeeper")


def _iter_jsonl(path: Path | str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_teams_json(path: Path | str) -> dict[int, int]:
    with open(path) as f:
        d = json.load(f)
    items = d["teams"] if isinstance(d, dict) and "teams" in d else d
    out = {}
    for t in items:
        c = t.get("team_cluster")
        if c is not None and c == c:  # descarta NaN
            out[int(t["tracker_id"])] = int(c)
    return out


def load_tracks_2d(tracks_2d_jsonl: Path | str, teams_json: Optional[Path | str] = None, fps: float = 25.0,
                   home_cluster: int = 0, pitch_length: float = 105.0, pitch_width: float = 68.0,
                   periods: Optional[Iterable[tuple[int, int, int]]] = None, name: str = "") -> PositionalData:
    """``periods``: ``[(primer_frame_idx, último_frame_idx, periodo)]``; por defecto todo es periodo 1."""
    team_of_tracker = load_teams_json(teams_json) if teams_json else {}
    records = list(_iter_jsonl(tracks_2d_jsonl))
    slots: list[int] = []
    slot_of: dict[int, int] = {}
    for rec in records:
        for t in rec.get("tracks", []):
            if t.get("role", "player") in _PLAYER_ROLES and "field_xy" in t:
                tid = int(t["tracker_id"])
                if tid not in slot_of:
                    slot_of[tid] = len(slots); slots.append(tid)
    T, N = len(records), len(slots)
    players = np.full((T, N, 2), np.nan, dtype=np.float32)
    ball = np.full((T, 2), np.nan, dtype=np.float32)
    frame_index = np.zeros(T, dtype=np.int64)
    for i, rec in enumerate(records):
        frame_index[i] = int(rec["frame_idx"])
        for t in rec.get("tracks", []):
            if t.get("role", "player") in _PLAYER_ROLES and "field_xy" in t:
                players[i, slot_of[int(t["tracker_id"])]] = t["field_xy"]
        b = rec.get("ball")
        if b and b.get("field_xy") is not None:
            ball[i] = b["field_xy"]
    team = np.full(N, NO_TEAM, dtype=np.int8)
    for tid, s in slot_of.items():
        c = team_of_tracker.get(tid)
        if c is not None:
            team[s] = HOME if c == home_cluster else AWAY
    period = np.ones(T, dtype=np.int16)
    if periods:
        for a, b, p in periods:
            period[(frame_index >= a) & (frame_index <= b)] = p
    order = np.argsort(frame_index, kind="stable")
    data = PositionalData(fps=fps, frame_index=frame_index[order], period=period[order],
                          players_xy=players[order], team=team, ball_xy=ball[order],
                          player_ids=[str(t) for t in slots], pitch_length=pitch_length, pitch_width=pitch_width,
                          name=name or Path(tracks_2d_jsonl).stem)
    data.validate()
    return data
