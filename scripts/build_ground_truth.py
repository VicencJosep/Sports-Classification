"""Construye un ``ground_truth/<partido>.gt.json`` (verdad de acciones de balón y eventos de un tramo de vídeo)
desde cualquiera de las tres fuentes de anotación que hay en el proyecto:

* ``--labels-ball``: SoccerNet Ball Action Spotting (``Labels-ball.json``, 7 partidos EFL; ms exactos);
* ``--labels-v2``: SoccerNet-v2 (``Labels-v2.json``, 500 partidos de TV: tiros, goles, córners, faltas,
  bandas...). Cada mitad lleva su desfase respecto al vídeo (``--half-offset 1=359.0 2=3554.0``: segundos de
  vídeo en los que arranca el minuto 0 de cada mitad);
* ``--csv``: la hoja de anotación manual (``tiempo,etiqueta,dorsal,equipo``, ver ``scripts/annotation_csv.py``),
  una o varias; el tramo anotado es el que va de la primera a la última fila (``--start-s/--end-s`` lo fijan).

El formato de salida es el que leen ``scripts/evaluate_match.py`` y ``scripts/evaluate_pipeline.py``: una
anotación por fila con la etiqueta original, la acción de balón canónica (``ball_action_label``, vocabulario de
``core/ball_actions.py``) y/o el evento (``event_label``, claves de ``core.ball_spotting.EVENT_SOURCES``), el
equipo, el instante en segundos de vídeo y el frame con el fps REAL del vídeo. Si se dan las cachés
``--tracks``/``--possession`` (``*.tracks.jsonl`` y ``*.possession.jsonl`` del pipeline) se añade el poseedor del
pipeline en ese frame y su caja; en el CSV manual el jugador es el que anotó el usuario (``dorsal`` =
``tracker_id``) y se marca ``tracker_from_annotation: true``.

Los JSON de ``ground_truth/`` van a git (``.gitignore``); los vídeos y cachés no.

Uso (raíz del proyecto; sin dependencias de GPU ni de cv2: el fps sale de ``--fps``, de un ``video.json`` del
backend junto al vídeo o de ``ffprobe``):

    # Middlesbrough, primeros 20 min, desde SoccerNet (regenera el GT histórico salvo sus retoques a mano)
    python -m scripts.build_ground_truth \\
        --labels-ball "datasets/SoccerNet/england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End/Labels-ball.json" \\
        --video ground_truth/2019-10-01_Middlesbrough-Preston_North_End-720p-trim.mp4 --max-minutes 20 \\
        --tracks ground_truth/2019-10-01_Middlesbrough-Preston_North_End-720p-trim.tracks.jsonl \\
        --possession ground_truth/2019-10-01_Middlesbrough-Preston_North_End-720p-trim.possession.jsonl \\
        --output output/gt_check/middlesbrough.gt.json

    # Espanyol 25-30 desde la hoja del usuario (fps 30 del video.json del backend)
    python -m scripts.build_ground_truth --csv output/annotations/espanyol_damm_Ground_Truth.csv \\
        --video backend/storage/videos/f0926c19277b28db/source.mp4 \\
        --output ground_truth/espanyol_damm_25-30.gt.json

    # Clásico desde Labels-v2 (TV; in-sample para T-DEED), con los desfases de cada mitad
    python -m scripts.build_ground_truth --labels-v2 "datasets/SoccerNet/spain_laliga/2016-2017/2017-04-23 - 21-45 Real Madrid 2 - 3 Barcelona/Labels-v2.json" \\
        --half-offset 1=359.0 2=3554.0 --video backend/storage/videos/573d9c1395b8f1c8/source.mp4 \\
        --output ground_truth/clasico_2017.gt.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

from scripts.annotation_csv import read_annotation_csv

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

# Labels-ball.json (SoccerNet Ball Action Spotting): etiqueta -> (acción de balón, evento)
BALL_LABELS: dict[str, tuple[Optional[str], Optional[str]]] = {
    "PASS": ("pass", None),
    "DRIVE": ("drive", None),
    "HIGH PASS": ("high_pass", None),
    "HEADER": ("header", None),
    "THROW IN": ("throw_in", "throw-in"),
    "CROSS": ("cross", None),
    "SHOT": ("shot", "shot"),
    "GOAL": ("goal", "goal"),
    "OUT": ("out", None),
    "FREE KICK": ("free_kick", "free-kick"),
    "PLAYER SUCCESSFUL TACKLE": ("player_successful_tackle", None),
    "BALL PLAYER BLOCK": ("ball_player_block", None),
}

# Labels-v2.json (SoccerNet-v2): etiqueta -> evento. No hay acciones de balón en v2.
V2_LABELS: dict[str, str] = {
    "Shots on target": "shot",
    "Shots off target": "shot",
    "Goal": "goal",
    "Corner": "corner",
    "Foul": "foul",
    "Direct free-kick": "free-kick",
    "Indirect free-kick": "free-kick",
    "Yellow card": "card",
    "Red card": "card",
    "Yellow->red card": "card",
    "Substitution": "substitution",
    "Kick-off": "kick-off",
    "Throw-in": "throw-in",
    "Penalty": "penalty",
    "Offside": "offside",
}

EVENT_WINDOW_S = 6.0          # ventana informativa alrededor del instante (como en el GT histórico)
BALL_ACTION_CLIP_FRAMES = 32


# --------------------------------------------------------------------------- #
#  Vídeo
# --------------------------------------------------------------------------- #

def video_meta(video: Optional[Path], fps: Optional[float]) -> dict:
    """``{fps, frame_count, duration_s}``: de ``--fps``, del ``video.json`` del backend junto al vídeo, o de
    ffprobe. Sin vídeo ni fps, falla: el frame depende del fps real."""
    meta: dict = {"fps": fps, "frame_count": None, "duration_s": None}
    if video is not None:
        vj = video.with_name("video.json")
        if vj.exists():
            d = json.loads(vj.read_text())
            meta["fps"] = meta["fps"] or float(d.get("fps") or 0) or None
            meta["frame_count"] = d.get("frame_count")
            meta["duration_s"] = d.get("duration_s")
        if (meta["fps"] is None or meta["frame_count"] is None) and video.exists():
            try:
                out = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                     "stream=r_frame_rate,nb_frames,duration", "-of", "json", str(video)],
                    capture_output=True, text=True, check=True,
                ).stdout
                st = json.loads(out)["streams"][0]
                num, den = st["r_frame_rate"].split("/")
                meta["fps"] = meta["fps"] or float(num) / float(den)
                if st.get("nb_frames", "N/A") != "N/A":
                    meta["frame_count"] = meta["frame_count"] or int(st["nb_frames"])
                if st.get("duration"):
                    meta["duration_s"] = meta["duration_s"] or float(st["duration"])
            except (OSError, subprocess.CalledProcessError, KeyError, ValueError, IndexError) as exc:
                logger.warning("ffprobe no pudo leer %s: %r", video, exc)
    if meta["fps"] is None:
        raise SystemExit("No sé el fps del vídeo: pasa --fps, o --video con video.json al lado o legible por ffprobe")
    if meta["duration_s"] is None and meta["frame_count"]:
        meta["duration_s"] = meta["frame_count"] / meta["fps"]
    return meta


# --------------------------------------------------------------------------- #
#  Fuentes
# --------------------------------------------------------------------------- #

def _game_time_to_half(game_time: str) -> Optional[int]:
    try:
        return int(game_time.split(" - ", 1)[0])
    except (ValueError, AttributeError):
        return None


def from_labels_ball(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows = []
    for a in data["annotations"]:
        action, event = BALL_LABELS.get(a["label"], (None, None))
        if a["label"] not in BALL_LABELS:
            logger.warning("Etiqueta de Labels-ball desconocida: %s", a["label"])
        rows.append({
            "game_time": a.get("gameTime"), "label_raw": a["label"], "ball_action_label": action,
            "event_label": event, "team": a.get("team") if a.get("team") in ("home", "away") else None,
            "visibility": a.get("visibility"), "t_s": int(a["position"]) / 1000.0, "tracker_id": None,
        })
    return rows


def from_labels_v2(path: Path, half_offsets: dict[int, float]) -> list[dict]:
    data = json.loads(path.read_text())
    rows = []
    skipped = Counter()
    for a in data["annotations"]:
        half = _game_time_to_half(a.get("gameTime", ""))
        if half not in half_offsets:
            skipped[f"mitad {half}"] += 1
            continue
        event = V2_LABELS.get(a["label"])
        if event is None:
            skipped[a["label"]] += 1
            continue
        rows.append({
            "game_time": a.get("gameTime"), "label_raw": a["label"], "ball_action_label": None, "event_label": event,
            "team": a.get("team") if a.get("team") in ("home", "away") else None, "visibility": a.get("visibility"),
            "t_s": int(a["position"]) / 1000.0 + half_offsets[half], "tracker_id": None,
        })
    if skipped:
        logger.info("Labels-v2: fuera %s", dict(skipped))
    return rows


def from_csv(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        parsed, warnings = read_annotation_csv(p)
        for w in warnings:
            logger.warning("%s: %s", p.name, w)
        for r in parsed:
            rows.append({
                "game_time": None, "label_raw": r["label_raw"], "ball_action_label": r["ball_action_label"],
                "event_label": r["event_label"], "team": r["team"], "visibility": "visible", "t_s": r["t_s"],
                "tracker_id": r["tracker_id"], "csv_line": r["line"], "csv_file": p.name,
            })
    return rows


# --------------------------------------------------------------------------- #
#  Poseedor del pipeline en el frame (cachés opcionales)
# --------------------------------------------------------------------------- #

def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def pipeline_possession(frames: set[int], tracks: Optional[Path], possession: Optional[Path]) -> dict[int, dict]:
    """``{frame_idx: {"tracker_id", "bbox_xyxy"}}`` para los frames pedidos (una pasada por cada JSONL)."""
    out: dict[int, dict] = {}
    if not frames:
        return out
    if possession is not None and possession.exists():
        for rec in _iter_jsonl(possession):
            fi = int(rec["frame_idx"])
            if fi in frames:
                out[fi] = {"tracker_id": rec.get("possessor_tracker_id"), "bbox_xyxy": None}
            if fi > max(frames):
                break
    if tracks is not None and tracks.exists():
        for rec in _iter_jsonl(tracks):
            fi = int(rec["frame_idx"])
            if fi in frames:
                entry = out.setdefault(fi, {"tracker_id": None, "bbox_xyxy": None})
                tid = entry.get("tracker_id")
                if tid is not None:
                    for tr in rec.get("tracks", []):
                        if tr.get("tracker_id") == tid:
                            entry["bbox_xyxy"] = tr.get("xyxy")
                            break
            if fi > max(frames):
                break
    return out


def _bbox_of(tracks: Optional[Path], frame_to_tracker: dict[int, int]) -> dict[int, Optional[list]]:
    """Caja del tracker anotado a mano en su frame (``tracks.jsonl``)."""
    out: dict[int, Optional[list]] = {}
    if tracks is None or not tracks.exists() or not frame_to_tracker:
        return out
    last = max(frame_to_tracker)
    for rec in _iter_jsonl(tracks):
        fi = int(rec["frame_idx"])
        if fi in frame_to_tracker:
            out[fi] = next((tr.get("xyxy") for tr in rec.get("tracks", []) if tr.get("tracker_id") == frame_to_tracker[fi]), None)
        if fi > last:
            break
    return out


# --------------------------------------------------------------------------- #
#  Construcción
# --------------------------------------------------------------------------- #

def build_annotations(rows: list[dict], fps: float, start_s: float, end_s: Optional[float],
                      tracks: Optional[Path] = None, possession: Optional[Path] = None) -> list[dict]:
    rows = sorted((r for r in rows if r["t_s"] >= start_s and (end_s is None or r["t_s"] <= end_s)),
                  key=lambda r: r["t_s"])
    for r in rows:
        r["frame_idx"] = int(round(r["t_s"] * fps))
    manual = {r["frame_idx"]: r["tracker_id"] for r in rows if r.get("tracker_id") is not None}
    auto = pipeline_possession({r["frame_idx"] for r in rows if r.get("tracker_id") is None}, tracks, possession)
    manual_bbox = _bbox_of(tracks, manual)
    out = []
    for k, r in enumerate(rows):
        fi = r["frame_idx"]
        if r.get("tracker_id") is not None:
            poss = {"tracker_id": r["tracker_id"], "bbox_xyxy": manual_bbox.get(fi), "from_annotation": True}
        else:
            poss = dict(auto.get(fi, {"tracker_id": None, "bbox_xyxy": None}), from_annotation=False)
        entry = {
            "annotation_id": k, "game_time": r.get("game_time"), "label_raw": r["label_raw"],
            "ball_action_label": r["ball_action_label"], "event_label": r["event_label"], "team": r["team"],
            "visibility": r.get("visibility") or "visible", "position_ms": int(round(r["t_s"] * 1000)),
            "timestamp_s": round(r["t_s"], 3), "frame_idx": fi,
            "windows": {
                "event_6s": {"start_s": round(max(0.0, r["t_s"] - EVENT_WINDOW_S / 2), 3),
                             "end_s": round(r["t_s"] + EVENT_WINDOW_S / 2, 3)},
                "ball_action_clip_32f": {"center_frame_idx": fi, "num_frames": BALL_ACTION_CLIP_FRAMES},
            },
            "possession": poss,
        }
        if r.get("csv_line") is not None:
            entry["csv"] = {"file": r["csv_file"], "line": r["csv_line"]}
        out.append(entry)
    return out


def summary(annotations: list[dict]) -> dict:
    return {
        "n_annotations": len(annotations),
        "ball_action_label_counts": dict(Counter(a["ball_action_label"] for a in annotations if a["ball_action_label"])),
        "event_label_counts": dict(Counter(a["event_label"] for a in annotations if a["event_label"])),
        "team_counts": dict(Counter(a["team"] for a in annotations if a["team"])),
        "n_with_tracker": sum(1 for a in annotations if a["possession"].get("tracker_id") is not None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_argument_group("fuente (una)")
    src.add_argument("--labels-ball", type=Path, help="Labels-ball.json de SoccerNet Ball Action Spotting")
    src.add_argument("--labels-v2", type=Path, help="Labels-v2.json de SoccerNet-v2")
    src.add_argument("--csv", type=Path, nargs="+", help="hoja(s) de anotación manual")
    parser.add_argument("--half-offset", nargs="*", default=[], metavar="MITAD=SEGUNDOS",
                        help="con --labels-v2: segundo de vídeo en que empieza cada mitad (p.ej. 1=359 2=3554)")
    parser.add_argument("--video", type=Path, help="vídeo al que se refieren los tiempos (fps real)")
    parser.add_argument("--fps", type=float, default=None, help="fps real si no hay video.json ni ffprobe")
    parser.add_argument("--max-minutes", type=float, default=None, help="solo los primeros N minutos")
    parser.add_argument("--start-s", type=float, default=None, help="tramo anotado (por defecto: todo / el del CSV)")
    parser.add_argument("--end-s", type=float, default=None)
    parser.add_argument("--tracks", type=Path, default=None, help="*.tracks.jsonl del pipeline (cajas)")
    parser.add_argument("--possession", type=Path, default=None, help="*.possession.jsonl del pipeline (poseedor)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sources = [s for s in (args.labels_ball, args.labels_v2, args.csv) if s]
    if len(sources) != 1:
        parser.error("indica exactamente una fuente: --labels-ball, --labels-v2 o --csv")

    meta = video_meta(args.video, args.fps)
    fps = float(meta["fps"])
    source: dict = {"video": str(args.video) if args.video else None}
    if args.labels_ball:
        rows = from_labels_ball(args.labels_ball)
        source.update(kind="labels_ball", path=str(args.labels_ball))
    elif args.labels_v2:
        offsets = {}
        for item in args.half_offset:
            h, s = item.split("=", 1)
            offsets[int(h)] = float(s)
        if not offsets:
            parser.error("--labels-v2 necesita --half-offset (p.ej. 1=359.0 2=3554.0)")
        rows = from_labels_v2(args.labels_v2, offsets)
        source.update(kind="labels_v2", path=str(args.labels_v2), half_offsets_s=offsets)
    else:
        rows = from_csv(args.csv)
        source.update(kind="csv", path=[str(p) for p in args.csv])

    start_s = args.start_s if args.start_s is not None else 0.0
    end_s = args.end_s
    if args.max_minutes is not None:
        end_s = args.max_minutes * 60.0 if end_s is None else min(end_s, args.max_minutes * 60.0)
    if source["kind"] == "csv" and rows:
        # El tramo anotado a mano es el que cubren las filas: fuera de él no hay verdad.
        if args.start_s is None:
            start_s = min(r["t_s"] for r in rows)
        if end_s is None:
            end_s = max(r["t_s"] for r in rows)

    annotations = build_annotations(rows, fps, start_s, end_s, args.tracks, args.possession)
    if not annotations:
        raise SystemExit("Ninguna anotación dentro del tramo")
    gt = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "source_video": str(args.video) if args.video else None,
        "source_tracks": str(args.tracks) if args.tracks else None,
        "source_possession": str(args.possession) if args.possession else None,
        "slice": {"start_s": start_s, "end_s": end_s, "max_minutes": args.max_minutes},
        "video": {"fps": fps, "frame_count": meta["frame_count"], "duration_s": meta["duration_s"]},
        "annotations": annotations,
        "summary": summary(annotations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gt, indent=2, ensure_ascii=False))
    s = gt["summary"]
    logger.info("%s: %d anotaciones [%s, %s] a %g fps; acciones %s; eventos %s; con jugador %d",
                args.output, s["n_annotations"], start_s, end_s, fps, s["ball_action_label_counts"],
                s["event_label_counts"], s["n_with_tracker"])


if __name__ == "__main__":
    main()
