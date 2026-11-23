"""
core/team_possession/io_metrica.py
==================================
Cargadores de datos de Metrica Sports a :class:`PositionalData`:

* :func:`load_epts`: formato FIFA/EPTS (``*_metadata.xml`` + ``*_tracking.txt``),
  el mismo que el ATD del reto More than a Hack 2027 y que el partido 3 de
  ``metrica-sports/sample-data``. No necesita ``kloppy``.
* :func:`load_sample_csv`: los partidos 1 y 2 de ``sample-data`` (CSV con una
  fila por fotograma y columnas por jugador).
* :func:`labels_from_events`: verdad aproximada H/A/D a partir de los eventos
  de Metrica (``RawEventsData.csv`` o ``*_events.json``) con la semántica del
  reto: el equipo cambia con la primera acción con balón del rival; balón
  muerto desde ``BALL OUT`` / falta / gol hasta el siguiente ``SET PIECE``.

Coordenadas Metrica: normalizadas en [0, 1]; se pasan a metros con el tamaño
de campo (``FieldSize`` del XML, o el que se pase: el reto mide 105,3 x 68,0).
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .frames import AWAY, HOME, NO_TEAM, PositionalData
from .labels import A, D, H, X


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _find_all(root: ET.Element, name: str) -> list[ET.Element]:
    return [e for e in root.iter() if _strip_ns(e.tag) == name]


# ------------------------------------------------------------------- EPTS
def read_epts_metadata(metadata_xml: Path | str) -> dict:
    """Extrae lo que necesita el cargador: fps, campo, equipos, jugadores y
    especificaciones de formato (qué columnas hay en cada rango de fotogramas)."""
    root = ET.parse(str(metadata_xml)).getroot()
    fps_el = _find_all(root, "FrameRate")
    fps = float(fps_el[0].text) if fps_el else 25.0
    length = width = None
    for fs in _find_all(root, "FieldSize"):
        w = fs.find("Width"); h = fs.find("Height")
        if w is not None and h is not None:
            length, width = float(w.text), float(h.text)
    teams = [(t.get("id"), (t.findtext("Name") or t.get("id"))) for t in _find_all(root, "Team")]
    players = {}
    for p in _find_all(root, "Player"):
        players[p.get("id")] = {"team_id": p.get("teamId"), "name": p.findtext("Name") or p.get("id")}
    channel_player = {}
    for ch in _find_all(root, "PlayerChannel"):
        channel_player[ch.get("id")] = (ch.get("playerId"), ch.get("channelId"))
    specs = []
    for spec in _find_all(root, "DataFormatSpecification"):
        order: list[str] = []
        ball = False
        for e in spec.iter():
            tag = _strip_ns(e.tag)
            if tag == "PlayerChannelRef":
                pid, cid = channel_player[e.get("playerChannelId")]
                if cid == "x":
                    order.append(pid)
            elif tag == "BallChannelRef":
                ball = True
        specs.append({"start": int(spec.get("startFrame")), "end": int(spec.get("endFrame")),
                      "players": order, "ball": ball, "separator": spec.get("separator", ":")})
    params = {}
    for pp in _find_all(root, "ProviderParameter"):
        n = pp.findtext("Name"); v = pp.findtext("Value")
        if n and n not in params:
            params[n] = v
    return {"fps": fps, "pitch_length": length, "pitch_width": width, "teams": teams,
            "players": players, "specs": specs, "params": params}


def _periods_from_params(params: dict) -> list[tuple[int, int, int]]:
    """``[(start_frame, end_frame, period)]`` si los metadatos traen los límites."""
    names = {1: "first_half", 2: "second_half", 3: "first_extra_half", 4: "second_extra_half"}
    out = []
    for p, base in names.items():
        s = params.get(f"{base}_start") or params.get(f"period{p}_start_frame")
        e = params.get(f"{base}_end") or params.get(f"period{p}_end_frame")
        try:
            if s and e and int(s) > 0 and int(e) > int(s):
                out.append((int(s), int(e), p))
        except ValueError:
            continue
    return out


def load_epts(metadata_xml: Path | str, tracking_txt: Path | str, name: str = "",
              pitch_length: Optional[float] = None, pitch_width: Optional[float] = None,
              home_team_id: Optional[str] = None,
              periods: Optional[Iterable[tuple[int, int, int]]] = None) -> PositionalData:
    """Carga EPTS. ``periods``: ``[(start_frame, end_frame, period)]`` opcional;
    si no se da y los metadatos no lo traen, todo es periodo 1."""
    meta = read_epts_metadata(metadata_xml)
    L = pitch_length or meta["pitch_length"] or 105.0
    W = pitch_width or meta["pitch_width"] or 68.0
    team_ids = [t[0] for t in meta["teams"]]
    home_id = home_team_id or (team_ids[0] if team_ids else None)

    slots: list[str] = []
    slot_of: dict[str, int] = {}
    for spec in meta["specs"]:
        for pid in spec["players"]:
            if pid not in slot_of:
                slot_of[pid] = len(slots)
                slots.append(pid)
    N = len(slots)
    team = np.full(N, NO_TEAM, dtype=np.int8)
    for pid, i in slot_of.items():
        tid = meta["players"].get(pid, {}).get("team_id")
        if tid is None:
            continue
        team[i] = HOME if tid == home_id else AWAY

    specs = sorted(meta["specs"], key=lambda s: s["start"])
    spec_idx = 0
    frames: list[int] = []
    rows_p: list[np.ndarray] = []
    rows_b: list[tuple[float, float]] = []
    with open(tracking_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            fid = int(parts[0])
            while spec_idx + 1 < len(specs) and fid > specs[spec_idx]["end"]:
                spec_idx += 1
            spec = specs[spec_idx]
            cols = spec["players"]
            arr = np.full((N, 2), np.nan, dtype=np.float32)
            if len(parts) > 1 and parts[1]:
                for pid, tok in zip(cols, parts[1].split(";")):
                    xy = tok.split(",")
                    if len(xy) == 2 and xy[0] not in ("NaN", ""):
                        try:
                            arr[slot_of[pid]] = (float(xy[0]), float(xy[1]))
                        except ValueError:
                            pass
            bx = by = np.nan
            if spec["ball"] and len(parts) > 2 and parts[2]:
                xy = parts[2].split(",")
                if len(xy) >= 2 and xy[0] not in ("NaN", ""):
                    try:
                        bx, by = float(xy[0]), float(xy[1])
                    except ValueError:
                        pass
            frames.append(fid)
            rows_p.append(arr)
            rows_b.append((bx, by))

    players_xy = np.stack(rows_p) if rows_p else np.zeros((0, N, 2), dtype=np.float32)
    ball_xy = np.asarray(rows_b, dtype=np.float32).reshape(-1, 2)
    players_xy[..., 0] *= L; players_xy[..., 1] *= W
    ball_xy[:, 0] *= L; ball_xy[:, 1] *= W
    frame_index = np.asarray(frames, dtype=np.int64)

    period = np.zeros(len(frames), dtype=np.int16)
    plist = list(periods) if periods is not None else _periods_from_params(meta["params"])
    if plist:
        for s, e, p in plist:
            period[(frame_index >= s) & (frame_index <= e)] = p
    else:
        period[:] = 1

    data = PositionalData(fps=meta["fps"], frame_index=frame_index, period=period,
                          players_xy=players_xy, team=team, ball_xy=ball_xy, player_ids=slots,
                          pitch_length=L, pitch_width=W, name=name or Path(tracking_txt).stem)
    data.validate()
    return data


# ------------------------------------------------------------- CSV de ejemplo
def load_sample_csv(home_csv: Path | str, away_csv: Path | str, name: str = "",
                    pitch_length: float = 105.0, pitch_width: float = 68.0) -> PositionalData:
    """Partidos 1 y 2 de ``metrica-sports/sample-data``: dos CSV (local y visitante)
    con cabecera de 3 líneas y columnas ``Period, Frame, Time [s], P1x, P1y, ..., Ballx, Bally``."""
    def _read(path):
        with open(path, newline="") as f:
            r = csv.reader(f)
            next(r); next(r); header = next(r)
            rows = [row for row in r if row]
        names = [h for h in header[3:-2] if h]  # nombres en las columnas x
        T = len(rows)
        per = np.asarray([int(x[0]) for x in rows], dtype=np.int16)
        fr = np.asarray([int(x[1]) for x in rows], dtype=np.int64)
        vals = np.asarray([[np.nan if v in ("", "NaN") else float(v) for v in x[3:]] for x in rows],
                          dtype=np.float32)
        pxy = vals[:, :-2].reshape(T, -1, 2)
        bxy = vals[:, -2:]
        return per, fr, names, pxy, bxy

    per_h, fr_h, names_h, p_h, b_h = _read(home_csv)
    per_a, fr_a, names_a, p_a, b_a = _read(away_csv)
    assert np.array_equal(fr_h, fr_a), "los CSV local y visitante no tienen los mismos fotogramas"
    players_xy = np.concatenate([p_h, p_a], axis=1)
    ball_xy = np.where(np.isfinite(b_h), b_h, b_a)
    players_xy[..., 0] *= pitch_length; players_xy[..., 1] *= pitch_width
    ball_xy[:, 0] *= pitch_length; ball_xy[:, 1] *= pitch_width
    team = np.concatenate([np.full(p_h.shape[1], HOME, np.int8), np.full(p_a.shape[1], AWAY, np.int8)])
    ids = [f"Home_{n}" for n in names_h] + [f"Away_{n}" for n in names_a]
    data = PositionalData(fps=25.0, frame_index=fr_h, period=per_h, players_xy=players_xy,
                          team=team, ball_xy=ball_xy, player_ids=ids,
                          pitch_length=pitch_length, pitch_width=pitch_width,
                          name=name or Path(home_csv).stem.replace("_RawTrackingData_Home_Team", ""))
    data.validate()
    return data


# ------------------------------------------------------------------ eventos
def read_events_csv(path: Path | str, home_name: str = "Home") -> list[dict]:
    """Eventos del CSV de ejemplo a una lista homogénea de dicts."""
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            team = HOME if r["Team"] == home_name else (AWAY if r["Team"] else NO_TEAM)
            out.append({"team": team, "type": r["Type"].strip().upper(),
                        "subtype": (r.get("Subtype") or "").strip().upper(),
                        "period": int(r["Period"]), "start": int(r["Start Frame"]),
                        "end": int(r["End Frame"] or r["Start Frame"])})
    return out


def read_events_json(path: Path | str, home_team_id: Optional[str] = None) -> list[dict]:
    """Eventos del JSON de ejemplo (partido 3). ``home_team_id``: el primer
    equipo de los metadatos EPTS; si no se da, el primero que aparezca."""
    with open(path) as f:
        d = json.load(f)
    events = d["data"] if isinstance(d, dict) else d
    if home_team_id is None:
        for e in events:
            t = e.get("team") or {}
            if t.get("id"):
                home_team_id = t["id"]; break
    out = []
    for e in events:
        t = e.get("team") or {}
        team = NO_TEAM if not t.get("id") else (HOME if t["id"] == home_team_id else AWAY)
        subs = e.get("subtypes")
        if isinstance(subs, dict):
            subs = [subs]
        subtype = "-".join(s.get("name", "") for s in (subs or [])).upper()
        start = (e.get("start") or {}).get("frame")
        end = (e.get("end") or {}).get("frame") or start
        out.append({"team": team, "type": (e.get("type") or {}).get("name", "").upper(),
                    "subtype": subtype, "period": int(e.get("period") or 0),
                    "start": int(start), "end": int(end)})
    return out


ON_BALL = {"PASS", "CARRY", "RECOVERY", "SHOT", "SET PIECE"}
DEAD_STARTERS = {"BALL OUT", "FAULT RECEIVED"}


def labels_from_events(events: list[dict], frame_index: np.ndarray, period: np.ndarray) -> np.ndarray:
    """Verdad aproximada H/A/D por fotograma a partir de los eventos de Metrica.

    Reglas (semántica del reto): antes del primer ``SET PIECE`` de cada periodo
    y después del último evento, D. Un ``SET PIECE`` o acción con balón
    (``PASS``, ``CARRY``, ``RECOVERY``, ``SHOT``) pone en juego al equipo que la
    hace desde su fotograma inicial. ``BALL OUT`` y ``FAULT RECEIVED`` ponen D
    desde su fotograma final; un ``SHOT`` cuya subclase contiene ``GOAL`` u
    ``OUT`` pone D desde su fotograma final. ``BALL LOST`` y ``CHALLENGE`` no
    cambian nada (el cambio llega con la primera acción del rival).
    """
    T = len(frame_index)
    labels = np.full(T, D, dtype=np.int8)
    events = sorted(events, key=lambda e: (e["period"], e["start"]))
    marks: list[tuple[int, int, int]] = []  # (fotograma, periodo, código desde ahí)
    for e in events:
        typ, sub, team = e["type"], e["subtype"], e["team"]
        if typ in ON_BALL:
            if team in (HOME, AWAY):
                marks.append((e["start"], e["period"], H if team == HOME else A))
            if typ == "SHOT" and ("GOAL" in sub or "OUT" in sub):
                marks.append((max(e["end"], e["start"]), e["period"], D))
        elif typ in DEAD_STARTERS:
            marks.append((max(e["end"], e["start"]), e["period"], D))
    marks.sort()
    for per in np.unique(period):
        sel = np.flatnonzero(period == per)
        pm = [m for m in marks if m[1] == per]
        if len(sel) == 0 or not pm:
            continue
        fi = frame_index[sel]
        pos = np.searchsorted(fi, [m[0] for m in pm], side="left")
        lab = np.full(len(sel), D, dtype=np.int8)
        cur, prev = D, 0
        for p, m in zip(pos, pm):
            lab[prev:p] = cur
            cur, prev = m[2], p
        lab[prev:] = cur
        last_event_end = max(e["end"] for e in events if e["period"] == per)
        lab[int(np.searchsorted(fi, last_event_end, side="right")):] = D
        labels[sel] = lab
    return labels
