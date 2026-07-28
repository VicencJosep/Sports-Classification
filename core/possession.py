"""
core/possession.py
===================
Resolución del "poseedor del balón" (jugador con la posesión en cada frame) a
partir del ``tracks.jsonl`` que produce ``core/tracking.py``.

Heurística: en cada frame, el poseedor es el ``tracker_id`` (de entre
``role in {"player", "goalkeeper"}``) cuyo ``anchor_xy`` está más cerca de
``ball.center_xy``, siempre que esa distancia no supere ``max_distance_px``.
Si no hay detección de balón en ese frame (hueco frecuente, ver docstring de
``core/tracking.py``) o ningún jugador está lo bastante cerca, se mantiene el
último poseedor conocido durante ``hysteresis_frames`` frames antes de caer a
``None`` — evita parpadeo de poseedor en los huecos cortos de detección del
balón, sin inventar posesión cuando el hueco es largo.

Es una utilidad independiente de cualquier tarea concreta (no solo la usa
``scripts/extract_ball_action_clips.py``): cualquier módulo que necesite saber
"quién tiene el balón" en un frame dado puede reutilizarla.

Uso rápido (standalone, desde la raíz del proyecto):
    python -m core.possession --tracks tracks.jsonl --output possession.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_DISTANCE_PX: float = 150.0
DEFAULT_HYSTERESIS_FRAMES: int = 10
_POSSESSOR_ROLES: frozenset[str] = frozenset({"player", "goalkeeper"})


def _iter_tracks_jsonl(path: Path | str) -> Iterator[dict]:
    """Lee un JSONL de tracking línea a línea (mismo patrón que core/homography.py)."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def resolve_possessor(
    record: dict,
    max_distance_px: float = DEFAULT_MAX_DISTANCE_PX,
) -> Optional[int]:
    """Devuelve el ``tracker_id`` más cercano al balón en este frame, o None.

    ``record`` es un dict con la forma de ``FrameTrackingResult.to_telemetry()``
    (``tracks``: lista de dicts con ``tracker_id``/``anchor_xy``/``role``;
    ``ball``: dict con ``center_xy`` o None). No aplica histéresis — eso lo
    gestiona ``iter_possession`` a lo largo de la secuencia.
    """
    ball = record.get("ball")
    if not ball:
        return None

    ball_xy = tuple(ball["center_xy"])
    best_id: Optional[int] = None
    best_dist = math.inf
    for track in record.get("tracks", []):
        if track.get("role") not in _POSSESSOR_ROLES:
            continue
        dist = _distance(tuple(track["anchor_xy"]), ball_xy)
        if dist < best_dist:
            best_dist = dist
            best_id = track["tracker_id"]

    if best_id is None or best_dist > max_distance_px:
        return None
    return best_id


def iter_possession(
    tracks_jsonl_path: Path | str,
    max_distance_px: float = DEFAULT_MAX_DISTANCE_PX,
    hysteresis_frames: int = DEFAULT_HYSTERESIS_FRAMES,
) -> Iterator[dict]:
    """Generador: cada registro de ``tracks_jsonl_path`` + ``possessor_tracker_id``.

    Aplica histéresis simple: si un frame no resuelve poseedor (balón no
    detectado o ningún jugador lo bastante cerca), se reutiliza el último
    poseedor conocido hasta ``hysteresis_frames`` frames consecutivos sin
    resolución; superado ese margen, ``possessor_tracker_id`` vuelve a None.
    """
    last_possessor: Optional[int] = None
    frames_since_resolved = 0

    for record in _iter_tracks_jsonl(tracks_jsonl_path):
        possessor = resolve_possessor(record, max_distance_px=max_distance_px)

        if possessor is not None:
            last_possessor = possessor
            frames_since_resolved = 0
        else:
            frames_since_resolved += 1
            if last_possessor is not None and frames_since_resolved <= hysteresis_frames:
                possessor = last_possessor
            else:
                last_possessor = None

        out_record = dict(record)
        out_record["possessor_tracker_id"] = possessor
        yield out_record


def ensure_possession(
    tracks_path: Path,
    max_distance_px: float = DEFAULT_MAX_DISTANCE_PX,
    hysteresis_frames: int = DEFAULT_HYSTERESIS_FRAMES,
) -> Path:
    """Corre iter_possession sobre tracks_path si el possession.jsonl no existe ya.

    Misma escritura atómica (``.tmp`` + rename) que ``core.tracking.
    ensure_tracking``, por el mismo motivo: un corte a mitad no debe dejar una
    caché parcial que una ejecución posterior reutilice como si fuera válida.

    Usado tanto por scripts/extract_ball_action_clips.py (que no pasa
    ``hysteresis_frames``, así que sigue usando el default de siempre) como por
    analytics/engine.py::analyze_full_match (Pase B del flujo de partido
    completo, que sí puede querer ajustarlo).
    """
    possession_path = tracks_path.with_suffix("").with_suffix(".possession.jsonl")
    if possession_path.exists():
        logger.info("possession.jsonl ya existe, reutilizando: %s", possession_path)
        return possession_path

    tmp_path = possession_path.with_suffix(possession_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        for record in iter_possession(
            tracks_path, max_distance_px=max_distance_px, hysteresis_frames=hysteresis_frames
        ):
            f.write(json.dumps(record) + "\n")
    tmp_path.rename(possession_path)
    return possession_path


# --------------------------------------------------------------------------- #
#  Ejecución standalone
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resuelve el poseedor del balón por frame sobre un tracks.jsonl"
    )
    parser.add_argument("--tracks", required=True, type=Path, help="tracks.jsonl de core/tracking.py")
    parser.add_argument("--output", required=True, type=Path, help="Salida: possession.jsonl")
    parser.add_argument("--max-distance-px", type=float, default=DEFAULT_MAX_DISTANCE_PX)
    parser.add_argument("--hysteresis-frames", type=int, default=DEFAULT_HYSTERESIS_FRAMES)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    n_resolved = 0
    n_total = 0
    with open(args.output, "w") as out_f:
        for record in iter_possession(
            args.tracks,
            max_distance_px=args.max_distance_px,
            hysteresis_frames=args.hysteresis_frames,
        ):
            n_total += 1
            if record["possessor_tracker_id"] is not None:
                n_resolved += 1
            out_f.write(json.dumps(record) + "\n")

    pct = (100.0 * n_resolved / n_total) if n_total else 0.0
    logger.info(
        "Posesión escrita en %s (%d/%d frames con poseedor resuelto, %.1f%%)",
        args.output, n_resolved, n_total, pct,
    )


if __name__ == "__main__":
    main()
