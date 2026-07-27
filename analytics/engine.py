"""
analytics/engine.py
====================
Composición final del pipeline: une ``tracks.jsonl`` (``core/tracking.py``) +
``teams.json`` (``core/classification.py``) + posesión (``core/possession.py``,
calculada aquí) + predicciones de ``core/ball_actions.py`` (una por cambio de
posesión) + ``core/events.py`` (una por clip) en un único vídeo anotado y un
``telemetry.json``.

Alcance actual (decisión explícita — ver README § "Uso — composición final"):
CLIPS CORTOS ya recortados en torno a un evento, como los de
``datasets/clips_analisis/``, NO partidos completos de 90 min.
  - ``core/events.py`` clasifica el clip ENTERO como un solo evento — válido
    aquí porque el clip ya viene recortado; sobre un partido completo diluiría
    la señal (ver el propio docstring de ese módulo).
  - Este módulo decodifica el clip ENTERO a memoria (``_read_all_frames``) —
    viable para clips de segundos, NO para un partido completo.
El troceado de un partido completo en clips/ventanas queda fuera de alcance,
tal como ya lo estaba en ``core/events.py``.

Pipeline:
    tracks.jsonl ─┬─► core.possession.iter_possession ─► posesión por frame
                  │        │
                  │        ▼ (cambios de possessor_tracker_id)
                  │   core.ball_actions.BallActionClassifier.predict_window
                  │        (una vez por cambio, ventana T=32 centrada)
                  │
    source_video ─┼─► core.events.EventClassifier.predict (una vez, clip entero)
                  │
    teams.json ───┴─► utils.visualizers.TrackingVisualizer.annotate
                       (equipo + resalte de poseedor) ─► vídeo final + telemetry.json

Uso rápido (standalone, desde la raíz del proyecto):
    python -m analytics.engine --source clip.mp4 --tracks tracks.jsonl \
           --teams teams.json --output final.mp4 --telemetry telemetry.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import cv2
import numpy as np

if TYPE_CHECKING:
    from core.ball_actions import BallActionClassifier
    from core.events import EventClassifier

logger = logging.getLogger(__name__)

DEFAULT_MAX_POSSESSOR_DISTANCE_PX: float = 150.0
DEFAULT_HYSTERESIS_FRAMES: int = 10

_BANNER_BG: tuple[int, int, int] = (0, 0, 0)
_BANNER_FG: tuple[int, int, int] = (255, 255, 255)
_BANNER_ALPHA: float = 0.55
_BANNER_LINE_HEIGHT: int = 28


# --------------------------------------------------------------------------- #
#  Registros serializables (telemetría)
# --------------------------------------------------------------------------- #

@dataclass
class PossessionChangeAction:
    """Predicción de core.ball_actions en un cambio de posesión."""

    frame_idx: int
    tracker_id: int
    label: str
    confidence: float
    probabilities: dict[str, float]
    is_background: bool


@dataclass
class MatchTelemetry:
    source_video: str
    event: Optional[dict]              # EventPrediction.to_telemetry(), o None si no se pidió
    ball_actions: list[PossessionChangeAction]
    n_frames: int
    n_possession_changes: int

    def to_telemetry(self) -> dict:
        """Registro serializable (recursa ``ball_actions``, lista de dataclasses)."""
        return asdict(self)


# --------------------------------------------------------------------------- #
#  Posesión + acciones de balón por cambio de poseedor
# --------------------------------------------------------------------------- #

def _detect_possession_changes(possession_records: list[dict]) -> list[tuple[int, int]]:
    """``[(frame_idx, tracker_id), ...]`` por cada cambio de posesión resuelto
    (ignora huecos ``None`` — ``core.possession`` ya aplica histéresis ahí)."""
    changes: list[tuple[int, int]] = []
    last: Optional[int] = None
    for rec in possession_records:
        possessor = rec.get("possessor_tracker_id")
        if possessor is None:
            continue
        if possessor != last:
            changes.append((rec["frame_idx"], possessor))
        last = possessor
    return changes


def _bbox_for_tracker(record: dict, tracker_id: int) -> Optional[tuple[float, float, float, float]]:
    for t in record.get("tracks", []):
        if t["tracker_id"] == tracker_id:
            return tuple(t["xyxy"])
    return None


def _read_all_frames(source_video: Path) -> dict[int, np.ndarray]:
    """Decodifica el clip ENTERO a memoria, indexado por frame_idx real del
    vídeo (BGR). Solo viable para clips cortos — NO usar sobre un partido
    completo (ver docstring del módulo)."""
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {source_video}")
    frames: dict[int, np.ndarray] = {}
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames[idx] = frame
            idx += 1
    finally:
        cap.release()
    return frames


def _build_window(
    frames_by_idx: dict[int, np.ndarray], center_idx: int, num_frames: int
) -> list[np.ndarray]:
    """Hasta ``num_frames`` frames BGR consecutivos centrados en ``center_idx``,
    clampados a los límites del clip. Si el clip es más corto que
    ``num_frames``, se devuelven los disponibles — ``BallActionClassifier.
    predict_window`` ya rellena por debajo (repite el último) vía
    ``ensure_num_frames``."""
    available = sorted(frames_by_idx.keys())
    if not available:
        raise ValueError("No hay frames decodificados")
    lo, hi = available[0], available[-1]
    half = num_frames // 2
    start = max(lo, center_idx - half)
    end = min(hi, start + num_frames - 1)
    start = max(lo, end - num_frames + 1)
    return [frames_by_idx[i] for i in range(start, end + 1)]


def analyze_match(
    source_video: Path | str,
    tracks_jsonl_path: Path | str,
    ball_action_classifier: Optional["BallActionClassifier"] = None,
    event_classifier: Optional["EventClassifier"] = None,
    max_possessor_distance_px: float = DEFAULT_MAX_POSSESSOR_DISTANCE_PX,
    hysteresis_frames: int = DEFAULT_HYSTERESIS_FRAMES,
) -> tuple[MatchTelemetry, list[dict]]:
    """Analiza un clip ya trackeado: posesión + (opcional) acción de balón por
    cambio de poseedor + (opcional) evento del clip entero.

    Devuelve ``(telemetry, possession_records)`` — ``possession_records`` se
    reutiliza para el render (``possessor_tracker_id`` por frame), evita
    recalcular la posesión dos veces.

    ``ball_action_classifier``/``event_classifier`` son opcionales: si no se
    pasan, esa parte del análisis simplemente se omite (útil para probar el
    resto del pipeline sin cargar los modelos pesados de R3D-18/V-JEPA2).
    """
    from core.possession import iter_possession

    source_video = Path(source_video)
    tracks_jsonl_path = Path(tracks_jsonl_path)

    possession_records = list(
        iter_possession(tracks_jsonl_path, max_possessor_distance_px, hysteresis_frames)
    )
    changes = _detect_possession_changes(possession_records)

    ball_actions: list[PossessionChangeAction] = []
    if ball_action_classifier is not None and changes:
        records_by_frame = {r["frame_idx"]: r for r in possession_records}
        frames_by_idx = _read_all_frames(source_video)

        for frame_idx, tracker_id in changes:
            record = records_by_frame.get(frame_idx)
            bbox = _bbox_for_tracker(record, tracker_id) if record else None
            if bbox is None:
                logger.warning(
                    "Sin bbox para tracker_id=%s en frame %d, salto la ventana",
                    tracker_id, frame_idx,
                )
                continue
            window = _build_window(frames_by_idx, frame_idx, ball_action_classifier.num_frames)
            try:
                pred = ball_action_classifier.predict_window(window, bbox)
            except Exception:
                logger.exception("Error clasificando ventana de acción en frame %d", frame_idx)
                continue
            ball_actions.append(
                PossessionChangeAction(
                    frame_idx=frame_idx,
                    tracker_id=tracker_id,
                    label=pred.label,
                    confidence=pred.confidence,
                    probabilities=pred.probabilities,
                    is_background=pred.is_background,
                )
            )

    event: Optional[dict] = None
    if event_classifier is not None:
        try:
            event = event_classifier.predict(source_video).to_telemetry()
        except Exception:
            logger.exception("Error clasificando el evento del clip")

    telemetry = MatchTelemetry(
        source_video=str(source_video),
        event=event,
        ball_actions=ball_actions,
        n_frames=len(possession_records),
        n_possession_changes=len(changes),
    )
    return telemetry, possession_records


# --------------------------------------------------------------------------- #
#  Render final (equipo + posesión + banners de evento/acción)
# --------------------------------------------------------------------------- #

def _draw_banner(frame: np.ndarray, lines: Sequence[str], position: str = "top") -> np.ndarray:
    """Franja semitransparente con texto (arriba o abajo del frame) — para el
    evento (fijo durante todo el clip) y la acción de balón (la más reciente,
    persiste hasta el siguiente cambio de posesión)."""
    if not lines:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    banner_h = _BANNER_LINE_HEIGHT * len(lines) + 12
    y0 = 0 if position == "top" else max(0, h - banner_h)

    overlay = out.copy()
    cv2.rectangle(overlay, (0, y0), (w, y0 + banner_h), _BANNER_BG, -1)
    out = cv2.addWeighted(overlay, _BANNER_ALPHA, out, 1 - _BANNER_ALPHA, 0)

    for i, line in enumerate(lines):
        y = y0 + 20 + i * _BANNER_LINE_HEIGHT
        cv2.putText(
            out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _BANNER_FG, 1, cv2.LINE_AA
        )
    return out


def _event_banner_line(event: Optional[dict]) -> Optional[str]:
    if not event:
        return None
    tag = "fondo" if event.get("is_background") else "EVENTO"
    return f"[{tag}] {event['label']} ({event['confidence']:.0%})"


def _ball_action_banner_line(action: Optional[PossessionChangeAction]) -> Optional[str]:
    if action is None:
        return None
    tag = "fondo" if action.is_background else "ACCION"
    return f"[{tag}] Jugador #{action.tracker_id}: {action.label} ({action.confidence:.0%})"


def render_match_video(
    source_video: Path | str,
    tracks_jsonl_path: Path | str,
    output_path: Path | str,
    team_by_tracker_id: Optional[dict[int, int]],
    possession_records: list[dict],
    telemetry: MatchTelemetry,
    trace_length: int = 30,
) -> None:
    """Segunda pasada (post-hoc) sobre el clip: cajas por equipo + resalte del
    poseedor + banner de evento (fijo) + banner de acción de balón (la más
    reciente en cada frame) — todo en un único ``.mp4`` de salida.

    Reutiliza ``utils.visualizers.TrackingVisualizer``/``detections_from_record``
    para la parte de cajas/etiquetas (mismo motor que ``core/tracking.py
    --annotated-output`` y ``core/classification.py --annotated-output``);
    los banners son responsabilidad de este módulo, no de ``utils/visualizers.py``
    (que es una caja de herramientas de dibujo genérica, sin conocer eventos
    ni acciones de balón).
    """
    from utils.visualizers import TrackingVisualizer, detections_from_record

    def _iter_tracks_jsonl(path: Path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    records_by_frame: dict[int, dict] = {
        rec["frame_idx"]: rec for rec in _iter_tracks_jsonl(Path(tracks_jsonl_path))
    }
    possessor_by_frame: dict[int, Optional[int]] = {
        rec["frame_idx"]: rec.get("possessor_tracker_id") for rec in possession_records
    }
    if not records_by_frame:
        raise ValueError(f"tracks.jsonl vacío o sin registros: {tracks_jsonl_path}")

    # Acción de balón "vigente" en cada frame: la del último cambio de
    # posesión visto hasta ese punto (persiste hasta la siguiente).
    action_by_trigger_frame = {a.frame_idx: a for a in telemetry.ball_actions}
    sorted_trigger_frames = sorted(action_by_trigger_frame.keys())

    event_line = _event_banner_line(telemetry.event)

    import supervision as sv

    video_info = sv.VideoInfo.from_video_path(str(source_video))
    sorted_frame_indices = sorted(records_by_frame.keys())
    stride = (
        sorted_frame_indices[1] - sorted_frame_indices[0]
        if len(sorted_frame_indices) >= 2
        else 1
    )
    video_info.fps = max(1, round(video_info.fps / max(stride, 1)))

    visualizer = TrackingVisualizer(trace_length=trace_length)

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {source_video}")

    try:
        with sv.VideoSink(str(output_path), video_info) as sink:
            frame_idx = 0
            current_action: Optional[PossessionChangeAction] = None
            trigger_ptr = 0

            while True:
                rec = records_by_frame.get(frame_idx)
                if rec is None:
                    if not cap.grab():
                        break
                    frame_idx += 1
                    continue

                ok, frame = cap.read()
                if not ok:
                    break

                while (
                    trigger_ptr < len(sorted_trigger_frames)
                    and sorted_trigger_frames[trigger_ptr] <= frame_idx
                ):
                    current_action = action_by_trigger_frame[sorted_trigger_frames[trigger_ptr]]
                    trigger_ptr += 1

                persons, ball_dets = detections_from_record(rec)
                annotated = visualizer.annotate(
                    frame, persons, ball_dets, team_by_tracker_id,
                    possessor_tracker_id=possessor_by_frame.get(frame_idx),
                )

                banner_lines = [line for line in (event_line,) if line]
                annotated = _draw_banner(annotated, banner_lines, position="top")
                action_line = _ball_action_banner_line(current_action)
                annotated = _draw_banner(
                    annotated, [action_line] if action_line else [], position="bottom"
                )

                sink.write_frame(annotated)
                frame_idx += 1
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
#  Ejecución standalone
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Composición final: tracking + equipos + posesión + acciones de balón + evento"
    )
    parser.add_argument("--source", required=True, type=Path, help="Clip .mp4 (el mismo usado en todo el pipeline)")
    parser.add_argument("--tracks", required=True, type=Path, help="tracks.jsonl de core/tracking.py")
    parser.add_argument("--teams", default=None, type=Path, help="teams.json de core/classification.py (opcional)")
    parser.add_argument("--output", required=True, type=Path, help="Vídeo final anotado (.mp4)")
    parser.add_argument("--telemetry", default=None, type=Path, help="Si se indica, escribe también telemetry.json")
    parser.add_argument("--skip-ball-actions", action="store_true", help="No cargar/correr core.ball_actions")
    parser.add_argument("--skip-events", action="store_true", help="No cargar/correr core.events")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-possessor-distance-px", type=float, default=DEFAULT_MAX_POSSESSOR_DISTANCE_PX)
    parser.add_argument("--hysteresis-frames", type=int, default=DEFAULT_HYSTERESIS_FRAMES)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.source.exists():
        parser.error(f"No existe: {args.source}")
    if not args.tracks.exists():
        parser.error(f"No existe: {args.tracks}")

    team_by_tracker_id: Optional[dict[int, int]] = None
    if args.teams:
        if not args.teams.exists():
            parser.error(f"No existe: {args.teams}")
        teams_data = json.loads(args.teams.read_text())
        team_by_tracker_id = {t["tracker_id"]: t["team_cluster"] for t in teams_data["teams"]}

    ball_action_classifier = None
    if not args.skip_ball_actions:
        from core.ball_actions import BallActionClassifier

        ball_action_classifier = BallActionClassifier(device=args.device)

    event_classifier = None
    if not args.skip_events:
        from core.events import EventClassifier

        event_classifier = EventClassifier(device=args.device)

    telemetry, possession_records = analyze_match(
        source_video=args.source,
        tracks_jsonl_path=args.tracks,
        ball_action_classifier=ball_action_classifier,
        event_classifier=event_classifier,
        max_possessor_distance_px=args.max_possessor_distance_px,
        hysteresis_frames=args.hysteresis_frames,
    )

    logger.info(
        "Análisis: %d cambios de posesión, %d acciones de balón clasificadas, evento=%s",
        telemetry.n_possession_changes, len(telemetry.ball_actions),
        telemetry.event["label"] if telemetry.event else "(omitido)",
    )

    render_match_video(
        source_video=args.source,
        tracks_jsonl_path=args.tracks,
        output_path=args.output,
        team_by_tracker_id=team_by_tracker_id,
        possession_records=possession_records,
        telemetry=telemetry,
    )
    logger.info("Vídeo final escrito en %s", args.output)

    if args.telemetry:
        with open(args.telemetry, "w") as f:
            json.dump(telemetry.to_telemetry(), f, indent=2)
        logger.info("Telemetría escrita en %s", args.telemetry)


if __name__ == "__main__":
    main()
