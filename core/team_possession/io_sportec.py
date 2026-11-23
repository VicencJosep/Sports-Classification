"""
core/team_possession/io_sportec.py
==================================
Cargador del dataset abierto IDSSE (Sportec / DFL, 7 partidos de Bundesliga
2022/23, CC-BY 4.0) a :class:`PositionalData`, con la **verdad por fotograma**
del balón: ``BallStatus`` (0 muerto, 1 en juego) y ``BallPossession`` (1 local,
2 visitante). Es la misma verdad que pide el reto (H/A/D), así que sirve para
medir ``spell_f1`` fuera del partido del reto.

Formato: ``DFL_04_03_positions_raw_observed_*.xml`` con un ``FrameSet`` por
persona y mitad (``GameSection``), cada uno con ``Frame N X Y S ...``; el
``FrameSet`` del balón (``TeamId="BALL"``) añade ``Z``, ``BallPossession`` y
``BallStatus``. Coordenadas en metros con origen en el centro: se desplazan a la
esquina. Los equipos y el rol local/visitante vienen de
``DFL_02_01_matchinformation_*.xml``.

Los ficheros pesan 350-420 MB; se leen con ``iterparse`` y conviene cachear el
resultado con :func:`save_npz` / :func:`load_npz`.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .frames import AWAY, HOME, NO_TEAM, PositionalData
from .labels import A, D, H, X

SECTION_PERIOD = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}


def read_matchinfo(path: Path | str) -> dict:
    root = ET.parse(str(path)).getroot()
    teams = {}
    person_team = {}
    for team in root.iter("Team"):
        tid = team.get("TeamId")
        teams[tid] = {"name": team.get("TeamName"), "role": team.get("Role")}
        for p in team.iter("Player"):
            person_team[p.get("PersonId")] = tid
    home = [t for t, v in teams.items() if v["role"] == "home"]
    guest = [t for t, v in teams.items() if v["role"] == "guest"]
    pitch = root.find(".//PitchSize")
    L = float(pitch.get("X")) if pitch is not None else 105.0
    W = float(pitch.get("Y")) if pitch is not None else 68.0
    return {"teams": teams, "person_team": person_team, "home": home[0] if home else None,
            "guest": guest[0] if guest else None, "pitch_length": L, "pitch_width": W}


def load_idsse(positions_xml: Path | str, matchinfo_xml: Path | str, name: str = "",
               fps: float = 25.0) -> PositionalData:
    info = read_matchinfo(matchinfo_xml)
    L, W = info["pitch_length"], info["pitch_width"]
    per_section: dict[str, dict] = {}
    slots: list[str] = []
    slot_of: dict[str, int] = {}

    def _section(sec: str) -> dict:
        if sec not in per_section:
            per_section[sec] = {"players": {}, "ball": None}
        return per_section[sec]

    cur_fs = None
    cur_frames: list = []
    for ev, el in ET.iterparse(str(positions_xml), events=("start", "end")):
        tag = el.tag
        if ev == "start":
            if tag == "FrameSet":
                cur_fs = dict(el.attrib)
                cur_frames = []
            continue
        if tag == "Frame" and cur_fs is not None:
            a = el.attrib
            if cur_fs.get("TeamId") == "BALL":
                cur_frames.append((int(a["N"]), float(a["X"]), float(a["Y"]), float(a.get("Z", "nan")),
                                   int(a.get("BallPossession", "0")), int(a.get("BallStatus", "-1"))))
            else:
                cur_frames.append((int(a["N"]), float(a["X"]), float(a["Y"])))
            el.clear()
        elif tag == "FrameSet":
            sec = cur_fs.get("GameSection", "firstHalf")
            s = _section(sec)
            if cur_fs.get("TeamId") == "BALL":
                s["ball"] = np.asarray(cur_frames, dtype=np.float64).reshape(-1, 6)
            elif cur_fs.get("TeamId") not in ("referee", "REFEREE", None):
                pid = cur_fs.get("PersonId")
                if pid not in slot_of:
                    slot_of[pid] = len(slots); slots.append(pid)
                s["players"][pid] = np.asarray(cur_frames, dtype=np.float64).reshape(-1, 3)
            cur_fs = None
            cur_frames = []
            el.clear()
        elif tag == "MetaData":
            ps = el.find("PitchSize")
            if ps is not None:
                L, W = float(ps.get("X")), float(ps.get("Y"))
            el.clear()

    N = len(slots)
    team = np.full(N, NO_TEAM, dtype=np.int8)
    for pid, i in slot_of.items():
        tid = info["person_team"].get(pid)
        if tid == info["home"]:
            team[i] = HOME
        elif tid == info["guest"]:
            team[i] = AWAY

    frames_all, period_all, players_all, ball_all, labels_all, z_all = [], [], [], [], [], []
    for sec in sorted(per_section, key=lambda s: SECTION_PERIOD.get(s, 9)):
        s = per_section[sec]
        ids = set()
        for arr in s["players"].values():
            ids.update(arr[:, 0].astype(np.int64).tolist())
        if s["ball"] is not None:
            ids.update(s["ball"][:, 0].astype(np.int64).tolist())
        fi = np.asarray(sorted(ids), dtype=np.int64)
        T = len(fi)
        pxy = np.full((T, N, 2), np.nan, dtype=np.float32)
        for pid, arr in s["players"].items():
            pos = np.searchsorted(fi, arr[:, 0].astype(np.int64))
            pxy[pos, slot_of[pid], 0] = arr[:, 1] + L / 2
            pxy[pos, slot_of[pid], 1] = arr[:, 2] + W / 2
        bxy = np.full((T, 2), np.nan, dtype=np.float32)
        bz = np.full(T, np.nan, dtype=np.float32)
        lab = np.full(T, X, dtype=np.int8)
        if s["ball"] is not None:
            b = s["ball"]
            pos = np.searchsorted(fi, b[:, 0].astype(np.int64))
            bxy[pos, 0] = b[:, 1] + L / 2
            bxy[pos, 1] = b[:, 2] + W / 2
            bz[pos] = b[:, 3]
            status = b[:, 5].astype(int)
            poss = b[:, 4].astype(int)
            code = np.where(status == 0, D, np.where(poss == 1, H, np.where(poss == 2, A, X)))
            lab[pos] = code.astype(np.int8)
        frames_all.append(fi); period_all.append(np.full(T, SECTION_PERIOD.get(sec, 0), np.int16))
        players_all.append(pxy); ball_all.append(bxy); labels_all.append(lab); z_all.append(bz)

    data = PositionalData(fps=fps, frame_index=np.concatenate(frames_all),
                          period=np.concatenate(period_all), players_xy=np.concatenate(players_all),
                          team=team, ball_xy=np.concatenate(ball_all), player_ids=slots,
                          ball_z=np.concatenate(z_all), pitch_length=L, pitch_width=W,
                          labels=np.concatenate(labels_all), name=name or Path(positions_xml).stem[-10:])
    data.validate()
    return data


# -------------------------------------------------------------------- caché
def save_npz(data: PositionalData, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, fps=data.fps, frame_index=data.frame_index, period=data.period,
        players_xy=data.players_xy, team=data.team, ball_xy=data.ball_xy,
        player_ids=np.asarray(data.player_ids, dtype=object),
        ball_z=np.zeros(0, np.float32) if data.ball_z is None else data.ball_z,
        pitch=np.asarray([data.pitch_length, data.pitch_width]),
        labels=np.full(0, 3, np.int8) if data.labels is None else data.labels, name=data.name)
    return path


def load_npz(path: Path | str) -> PositionalData:
    z = np.load(str(path), allow_pickle=True)
    ball_z = z["ball_z"]; labels = z["labels"]
    data = PositionalData(
        fps=float(z["fps"]), frame_index=z["frame_index"], period=z["period"],
        players_xy=z["players_xy"], team=z["team"], ball_xy=z["ball_xy"],
        player_ids=[str(p) for p in z["player_ids"]],
        ball_z=None if ball_z.size == 0 else ball_z,
        pitch_length=float(z["pitch"][0]), pitch_width=float(z["pitch"][1]),
        labels=None if labels.size == 0 else labels, name=str(z["name"]))
    data.validate()
    return data
