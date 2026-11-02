"""Lectura/escritura de secuencias en formato MOTChallenge (SoccerNet-Tracking,
SoccerTrack v2 `mot/`) para el benchmark de tracking.

Formato de fila (10 columnas, frame desde 1, caja en píxeles x,y,w,h):
    frame, id, x, y, w, h, conf, class_id, jersey, -1

En GT y en tracks ``id`` es la identidad; en detecciones es -1. ``class_id``
sigue ``core.tracking.ROLE_TO_CLASS_ID`` (0 = balón); ``jersey`` es -1 si no hay.
Ver docs/comparativa_benchmark.md §5.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np

from core.tracking import ROLE_TO_CLASS_ID

BALL_CLASS_ID = ROLE_TO_CLASS_ID["ball"]
# Identidad fija para el balón en los tracks (nuestro pipeline no lo trackea:
# una sola caja por frame, ver core/tracking.py::_resolve_ball).
BALL_TRACK_ID = 1_000_000

# gameinfo.ini de SoccerNet: "player team left;4", "goalkeepers team left;1",
# "referee;main", "ball;1", "other;..."
_GAMEINFO_ROLE = {"player": "player", "goalkeeper": "goalkeeper", "goalkeepers": "goalkeeper",
                  "referee": "referee", "ball": "ball", "other": "other"}


def list_sequences(root: Path) -> list[Path]:
    """Carpetas con ``seqinfo.ini`` bajo ``root`` (o ``root`` mismo)."""
    if (root / "seqinfo.ini").exists():
        return [root]
    return sorted(p.parent for p in root.glob("*/seqinfo.ini"))


def seq_info(seq_dir: Path) -> dict:
    cp = configparser.ConfigParser()
    cp.read(seq_dir / "seqinfo.ini")
    s = cp["Sequence"]
    return {
        "name": s["name"],
        "frame_rate": int(s.get("frameRate", 25)),
        "length": int(s["seqLength"]),
        "img_dir": seq_dir / s.get("imDir", "img1"),
        "img_ext": s.get("imExt", ".jpg"),
    }


def gt_roles(seq_dir: Path) -> dict[int, str]:
    """track id → rol, desde ``gameinfo.ini`` (vacío si no existe)."""
    path = seq_dir / "gameinfo.ini"
    if not path.exists():
        return {}
    cp = configparser.ConfigParser()
    cp.read(path)
    roles = {}
    for key, value in cp["Sequence"].items():
        if key.startswith("trackletid_"):
            kind = value.strip().split(";")[0].split()[0]
            roles[int(key.split("_")[1])] = _GAMEINFO_ROLE.get(kind, "other")
    return roles


def read_mot(path: Path) -> np.ndarray:
    """Filas MOT como array (N, 10) float; vacío si el fichero no existe o no tiene filas."""
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 10))
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    if rows.shape[1] < 10:  # MOT de 6-9 columnas: rellenar con -1
        rows = np.hstack([rows, -np.ones((len(rows), 10 - rows.shape[1]))])
    return rows[:, :10]


def write_mot(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            frame, tid, x, y, w, h, conf, cls, jersey = r
            f.write(f"{int(frame)},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},{int(cls)},{int(jersey)},-1\n")


def gt_as_detections(seq_dir: Path) -> np.ndarray:
    """GT como detecciones oráculo (conf 1, id -1) con la clase sacada de gameinfo.ini."""
    gt = read_mot(seq_dir / "gt" / "gt.txt")
    roles = gt_roles(seq_dir)
    dets = gt.copy()
    dets[:, 7] = [ROLE_TO_CLASS_ID[roles.get(int(i), "player")] for i in gt[:, 1]]
    dets[:, 1] = -1
    dets[:, 6] = 1.0
    dets[:, 8] = -1
    return dets
