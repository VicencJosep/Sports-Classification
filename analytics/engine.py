"""
analytics/engine.py
====================
Composición final del pipeline: une ``tracks.jsonl`` (``core/tracking.py``) +
``teams.json`` (``core/classification.py``) + posesión (``core/possession.py``,
calculada aquí) + predicciones de ``core/ball_actions.py`` (una por cambio de
posesión) + ``core/events.py`` en un único vídeo anotado y un ``telemetry.json``.

Dos flujos, cada uno con su función de análisis (el render se comparte):

* ``analyze_match`` (sin cambios respecto a la versión clip-corto original):
  CLIPS CORTOS ya recortados en torno a un evento, como los de
  ``datasets/clips_analisis/``. ``core/events.py`` clasifica el clip ENTERO
  como un solo evento (válido porque el clip ya viene recortado) y este módulo
  decodifica el clip entero a memoria (``_read_all_frames``) — viable para
  clips de segundos, no para un partido completo.
* ``analyze_full_match`` (nuevo): partidos completos de ~90 min. Pase A
  (tracking, cacheado vía ``core.tracking.ensure_tracking``) + Pase B
  (posesión + acciones de balón, streaming con buffer de frames ACOTADO —
  ``_analyze_possession_and_ball_actions``, sustituye a ``_read_all_frames``)
  + Pase C (eventos por ventana deslizante de duración fija —
  ``core/windowing.py`` + ``EventClassifier.predict_window`` + fusión de
  ventanas contiguas en ``EventSpan`` vía ``_merge_event_windows``). No incluye
  clasificación de equipos ni render — eso lo orquesta ``main.py``, igual que
  para el flujo de clip corto.

Pipeline (clip corto):
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

Para partidos completos, ver ``main.py --mode full-match`` (usa
``analyze_full_match`` de este módulo).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from core.windowing import Window, iter_windows

if TYPE_CHECKING:
    from core.ball_actions import BallActionClassifier
    from core.events import EventClassifier, EventPrediction

logger = logging.getLogger(__name__)

DEFAULT_MAX_POSSESSOR_DISTANCE_PX: float = 150.0
DEFAULT_HYSTERESIS_FRAMES: int = 10

# Posiciones fijas de las tarjetas de notificación (utils.visualizers.
# draw_notification_card) en la esquina superior izquierda: evento arriba,
# acción de balón debajo, cada una en su propio hueco fijo -- así no "saltan"
# de sitio cuando una aparece/desaparece independientemente de la otra.
_EVENT_CARD_TOP_LEFT: tuple[int, int] = (14, 14)
_ACTION_CARD_TOP_LEFT: tuple[int, int] = (14, 90)
_NOTIFICATION_CARD_WIDTH: int = 320
_NOTIFICATION_CARD_HEIGHT: int = 68


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


@dataclass
class EventSpan:
    """Tramo de evento fusionado a partir de ventanas deslizantes contiguas
    con la misma etiqueta no-fondo (ver ``_merge_event_windows``). El tramo es
    típicamente más ancho que el instante real del evento (~1 ventana de
    margen) — es el equivalente, para partido completo, del campo ``event``
    (singular) de ``MatchTelemetry`` para clip corto."""

    label: str
    start_s: float
    end_s: float
    peak_confidence: float
    mean_confidence: float
    n_windows: int

    def to_telemetry(self) -> dict:
        return asdict(self)


@dataclass
class FullMatchTelemetry:
    """Telemetría de ``analyze_full_match`` — mismo espíritu que
    ``MatchTelemetry`` pero con ``events`` como línea temporal (lista de
    ``EventSpan``) en vez de un único evento por clip. Igual que
    ``ball_actions``, se guardan los dataclasses tal cual (no ya convertidos a
    dict) para que ``render_match_video`` los pueda reutilizar directamente;
    ``to_telemetry()`` los serializa al final, igual que con ``ball_actions``.
    """

    source_video: str
    events: list[EventSpan]            # línea temporal completa del partido
    ball_actions: list[PossessionChangeAction]
    n_frames: int
    n_possession_changes: int

    def to_telemetry(self) -> dict:
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


def _aggregate_ball_action_probabilities(
    predictions: list["BallActionPrediction"],
) -> tuple[str, float, dict[str, float]]:
    if not predictions:
        raise ValueError("No hay predicciones de ball action para agregar")

    labels = list(predictions[0].probabilities.keys())
    combined = {
        label: sum(pred.probabilities.get(label, 0.0) for pred in predictions) / len(predictions)
        for label in labels
    }
    best_label = max(combined, key=combined.get)
    return best_label, float(combined[best_label]), combined


def _predict_ball_action_temporally(
    ball_action_classifier: "BallActionClassifier",
    frames_by_idx: dict[int, np.ndarray],
    records_by_frame: dict[int, dict],
    trigger_frame_idx: int,
    tracker_id: int,
    trigger_bbox: tuple[float, float, float, float],
    temporal_offsets_frames: tuple[int, ...],
) -> "BallActionPrediction":
    """Clasifica la ventana del trigger (opcionalmente promediando varias
    ventanas desplazadas, ver ``config.BallActionsConfig.temporal_offsets_frames``).

    El recorte replica EXACTAMENTE el de la extracción de dataset (ver
    ``scripts/extract_ball_action_clips.py::extract_positive_clips``): bbox del
    poseedor expandido con ``expand_with_ball_trajectory`` sobre los mismos
    frames que se van a clasificar. Sin esa expansión el modelo veía en
    inferencia un primer plano del jugador del que el balón desaparecía
    (medido: recorte 12.7× más pequeño en área que en entrenamiento, balón
    fuera de cuadro en el 92% de las ventanas) — el distribution shift contra
    el que avisa el docstring de ``core.ball_actions.compute_crop_box``.
    """
    from core.ball_actions import expand_with_ball_trajectory

    num_frames = ball_action_classifier.num_frames
    center_frames = tuple(trigger_frame_idx + offset for offset in temporal_offsets_frames)
    predictions = []

    for center_idx in center_frames:
        start, end = _window_bounds(frames_by_idx, center_idx, num_frames)
        window = [frames_by_idx[i] for i in range(start, end + 1)]
        record = records_by_frame.get(center_idx)
        bbox = _bbox_for_tracker(record, tracker_id) if record is not None else None
        if bbox is None:
            bbox = trigger_bbox
        bbox = expand_with_ball_trajectory(bbox, records_by_frame, start, end - start + 1)
        predictions.append(ball_action_classifier.predict_window(window, bbox))

    label, confidence, probabilities = _aggregate_ball_action_probabilities(predictions)
    return predictions[0].__class__(
        source=predictions[0].source,
        label=label,
        confidence=confidence,
        probabilities=probabilities,
        is_background=(label == "background"),
    )


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


def _window_bounds(
    frames_by_idx: dict[int, np.ndarray], center_idx: int, num_frames: int
) -> tuple[int, int]:
    """``(start, end)`` inclusivos de la ventana de ``num_frames`` centrada en
    ``center_idx`` y clampada a los frames disponibles.

    Si el clip/buffer es más corto que ``num_frames`` se devuelve el rango
    disponible — ``BallActionClassifier.predict_window`` ya rellena por debajo
    (repite el último frame) vía ``ensure_num_frames``.

    Único punto de verdad del rango: lo usan tanto los frames que se clasifican
    como la expansión del recorte por trayectoria del balón
    (``core.ball_actions.expand_with_ball_trajectory``), que DEBE cubrir
    exactamente esos mismos frames y no un rango teórico que podría diferir
    cuando la ventana se clampa al principio/final del buffer."""
    available = sorted(frames_by_idx.keys())
    if not available:
        raise ValueError("No hay frames decodificados")
    lo, hi = available[0], available[-1]
    half = num_frames // 2
    start = max(lo, center_idx - half)
    end = min(hi, start + num_frames - 1)
    start = max(lo, end - num_frames + 1)
    return start, end


def analyze_match(
    source_video: Path | str,
    tracks_jsonl_path: Path | str,
    ball_action_classifier: Optional["BallActionClassifier"] = None,
    event_classifier: Optional["EventClassifier"] = None,
    max_possessor_distance_px: float = DEFAULT_MAX_POSSESSOR_DISTANCE_PX,
    hysteresis_frames: int = DEFAULT_HYSTERESIS_FRAMES,
    ball_action_temporal_offsets_frames: tuple[int, ...] = (0,),
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
            try:
                pred = _predict_ball_action_temporally(
                    ball_action_classifier=ball_action_classifier,
                    frames_by_idx=frames_by_idx,
                    records_by_frame=records_by_frame,
                    trigger_frame_idx=frame_idx,
                    tracker_id=tracker_id,
                    trigger_bbox=bbox,
                    temporal_offsets_frames=ball_action_temporal_offsets_frames,
                )
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
#  Partido completo: posesión + acciones de balón en streaming (Pase B) +
#  eventos por ventana deslizante fusionados en tramos (Pase C)
# --------------------------------------------------------------------------- #

def _analyze_possession_and_ball_actions(
    source_video: Path,
    tracks_jsonl_path: Path,
    ball_action_classifier: Optional["BallActionClassifier"],
    max_possessor_distance_px: float,
    hysteresis_frames: int,
    temporal_offsets_frames: tuple[int, ...],
) -> tuple[list[PossessionChangeAction], dict[int, Optional[int]], int]:
    """Pase B del flujo de partido completo: posesión (generador de
    ``core.possession``, O(1) de estado) + acción de balón por cambio de
    posesión — sin decodificar el vídeo entero (``_read_all_frames``).

    Si hay ``ball_action_classifier``, mantiene un buffer de frames ACOTADO
    (podado a medida que avanza el pase) alimentado en el mismo recorrido
    secuencial que la posesión; si no lo hay, ni siquiera abre el vídeo (solo
    lee el ``possession.jsonl``). Junto al buffer de frames se mantiene, con la
    misma cota de poda, un buffer de los ``records`` de posesión: el recorte
    necesita el bbox del balón en toda la ventana, no solo en el frame del
    trigger (ver ``_predict_ball_action_temporally``).

    Devuelve ``(ball_actions, possessor_by_frame, n_frames)``.
    ``possessor_by_frame`` (``frame_idx -> tracker_id | None``) es más ligero
    que la lista completa de ``possession_records`` que usa ``analyze_match``
    y es lo único de posesión que necesita ``render_match_video`` — apto para
    partidos de 90 min.
    """
    from core.possession import ensure_possession

    possession_path = ensure_possession(
        tracks_jsonl_path, max_possessor_distance_px, hysteresis_frames
    )

    def _iter_possession_jsonl():
        with open(possession_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    possessor_by_frame: dict[int, Optional[int]] = {}
    ball_actions: list[PossessionChangeAction] = []
    n_frames = 0
    last_possessor: Optional[int] = None

    if ball_action_classifier is None:
        # Sin acciones de balón que clasificar, no hace falta decodificar vídeo.
        for record in _iter_possession_jsonl():
            possessor_by_frame[record["frame_idx"]] = record.get("possessor_tracker_id")
            n_frames += 1
        return ball_actions, possessor_by_frame, n_frames

    num_frames = ball_action_classifier.num_frames
    half = num_frames // 2
    min_offset = min(temporal_offsets_frames) if temporal_offsets_frames else 0
    max_offset = max(temporal_offsets_frames) if temporal_offsets_frames else 0
    frame_buffer: dict[int, np.ndarray] = {}
    # Records de posesión de los mismos frames que `frame_buffer` (de aquí sale
    # el bbox del balón para expandir el recorte). Se poda con la misma cota:
    # sin podarlo serían ~135k records con ~10 tracks cada uno en un partido de
    # 90 min, creciendo linealmente hasta el final del pase.
    records_buffer: dict[int, dict] = {}
    pending: list[tuple[int, int, tuple[float, float, float, float]]] = []

    def _flush_ready(up_to_idx: float) -> None:
        while pending and pending[0][0] + max_offset + half <= up_to_idx:
            trig_idx, tracker_id, bbox = pending.pop(0)
            try:
                pred = _predict_ball_action_temporally(
                    ball_action_classifier=ball_action_classifier,
                    frames_by_idx=frame_buffer,
                    records_by_frame=records_buffer,
                    trigger_frame_idx=trig_idx,
                    tracker_id=tracker_id,
                    trigger_bbox=bbox,
                    temporal_offsets_frames=temporal_offsets_frames,
                )
            except Exception:
                logger.exception(
                    "Error clasificando ventana de acción en frame %d", trig_idx
                )
                continue
            ball_actions.append(
                PossessionChangeAction(
                    frame_idx=trig_idx,
                    tracker_id=tracker_id,
                    label=pred.label,
                    confidence=pred.confidence,
                    probabilities=pred.probabilities,
                    is_background=pred.is_background,
                )
            )

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {source_video}")

    try:
        for record in _iter_possession_jsonl():
            frame_idx = record["frame_idx"]
            ok, frame = cap.read()
            if not ok:
                break

            possessor = record.get("possessor_tracker_id")
            possessor_by_frame[frame_idx] = possessor
            n_frames += 1
            frame_buffer[frame_idx] = frame
            records_buffer[frame_idx] = record

            if possessor is not None and possessor != last_possessor:
                bbox = _bbox_for_tracker(record, possessor)
                if bbox is not None:
                    pending.append((frame_idx, possessor, bbox))
                else:
                    logger.warning(
                        "Sin bbox para tracker_id=%s en frame %d, salto la ventana",
                        possessor, frame_idx,
                    )
            if possessor is not None:
                last_possessor = possessor

            _flush_ready(frame_idx)

            min_needed = (
                min(pending_item[0] + min_offset - half for pending_item in pending)
                if pending
                else (frame_idx + min_offset - half)
            )
            for old_idx in [i for i in frame_buffer if i < min_needed]:
                del frame_buffer[old_idx]
            for old_idx in [i for i in records_buffer if i < min_needed]:
                del records_buffer[old_idx]
    finally:
        cap.release()

    _flush_ready(float("inf"))
    return ball_actions, possessor_by_frame, n_frames


def _finalize_span(
    label: str, start_s: float, end_s: float, confidences: list[float]
) -> EventSpan:
    return EventSpan(
        label=label,
        start_s=start_s,
        end_s=end_s,
        peak_confidence=max(confidences),
        mean_confidence=sum(confidences) / len(confidences),
        n_windows=len(confidences),
    )


def _merge_event_windows(
    window_predictions: list[tuple[Window, "EventPrediction"]]
) -> list[EventSpan]:
    """Funde ventanas deslizantes contiguas/solapadas con la misma etiqueta
    no-fondo en tramos de evento (``EventSpan``).

    Regla: una ventana con ``is_background=True`` (cualquiera de las 7 clases
    ``background_*``) nunca abre ni extiende un tramo — simplemente cierra el
    que esté abierto. Con solape (``event_stride_s < event_window_s``), dos
    ventanas consecutivas de la misma etiqueta se solapan por construcción,
    así que basta comprobar que la nueva ventana empieza antes de que acabe el
    tramo actual (``window.start_s <= current_end``).

    Limitación conocida y aceptada para v1: el tramo resultante es más ancho
    que el instante real del evento (~1 ventana de margen), y una ventana mal
    clasificada entre dos iguales puede partir un evento en dos tramos — si
    esto resulta un problema en la práctica, un suavizado por voto mayoritario
    sobre las predicciones crudas antes de fusionar es la mejora natural.
    """
    spans: list[EventSpan] = []
    current_label: Optional[str] = None
    current_start = 0.0
    current_end = 0.0
    current_confidences: list[float] = []

    for window, pred in window_predictions:
        extends_current = (
            current_label is not None
            and not pred.is_background
            and pred.label == current_label
            and window.start_s <= current_end
        )
        if extends_current:
            current_end = max(current_end, window.end_s)
            current_confidences.append(pred.confidence)
            continue

        if current_label is not None:
            spans.append(
                _finalize_span(current_label, current_start, current_end, current_confidences)
            )
            current_label = None

        if not pred.is_background:
            current_label = pred.label
            current_start = window.start_s
            current_end = window.end_s
            current_confidences = [pred.confidence]

    if current_label is not None:
        spans.append(
            _finalize_span(current_label, current_start, current_end, current_confidences)
        )

    return spans


def analyze_full_match(
    source_video: Path | str,
    detector_name: str = "hybrid",
    ball_action_classifier: Optional["BallActionClassifier"] = None,
    event_classifier: Optional["EventClassifier"] = None,
    max_possessor_distance_px: float = DEFAULT_MAX_POSSESSOR_DISTANCE_PX,
    hysteresis_frames: int = DEFAULT_HYSTERESIS_FRAMES,
    event_window_s: float = 6.0,
    event_stride_s: float = 3.0,
    min_window_fraction: float = 0.5,
    ball_action_temporal_offsets_frames: tuple[int, ...] = (0,),
    device: Optional[str] = None,
) -> tuple[FullMatchTelemetry, Path, dict[int, Optional[int]]]:
    """Analiza un partido completo (~90 min): Pase A (tracking, cacheado) +
    Pase B (posesión + acciones de balón, streaming) + Pase C (eventos por
    ventana deslizante). NO incluye clasificación de equipos
    (``core.classification.assign_teams``) ni render (``render_match_video``)
    — se orquestan aparte, desde ``main.py``, igual que para
    ``analyze_match``/``render_match_video`` en el flujo de clip corto.

    Devuelve ``(telemetry, tracks_jsonl_path, possessor_by_frame)`` —
    ``tracks_jsonl_path`` y ``possessor_by_frame`` se reutilizan para el
    render, con el mismo propósito con el que ``analyze_match`` devuelve
    ``possession_records``.
    """
    from core.tracking import ensure_tracking

    source_video = Path(source_video)

    # Pase A: tracking (cacheado junto al vídeo — ver core.tracking.ensure_tracking)
    tracks_path = ensure_tracking(source_video, detector_name=detector_name, device=device)

    # Pase B: posesión + acciones de balón (streaming, buffer de frames acotado)
    ball_actions, possessor_by_frame, n_frames = _analyze_possession_and_ball_actions(
        source_video=source_video,
        tracks_jsonl_path=tracks_path,
        ball_action_classifier=ball_action_classifier,
        max_possessor_distance_px=max_possessor_distance_px,
        hysteresis_frames=hysteresis_frames,
        temporal_offsets_frames=ball_action_temporal_offsets_frames,
    )

    # Pase C: eventos por ventana deslizante de duración fija -> tramos fusionados
    events: list[EventSpan] = []
    if event_classifier is not None:
        import supervision as sv

        video_info = sv.VideoInfo.from_video_path(str(source_video))
        fps = float(video_info.fps or 25)
        window_predictions: list[tuple[Window, "EventPrediction"]] = []
        for window in iter_windows(video_info.total_frames, fps, event_window_s, event_stride_s):
            if window.duration_s < event_window_s * min_window_fraction:
                continue
            try:
                pred = event_classifier.predict_window(source_video, window.start_s, window.end_s)
            except Exception:
                logger.exception(
                    "Error clasificando ventana de evento [%.2f, %.2f]s",
                    window.start_s, window.end_s,
                )
                continue
            window_predictions.append((window, pred))
        events = _merge_event_windows(window_predictions)

    telemetry = FullMatchTelemetry(
        source_video=str(source_video),
        events=events,
        ball_actions=ball_actions,
        n_frames=n_frames,
        n_possession_changes=len(ball_actions),
    )
    return telemetry, tracks_path, possessor_by_frame


# --------------------------------------------------------------------------- #
#  Render final (equipo + posesión + banners de evento/acción)
# --------------------------------------------------------------------------- #

NotificationContent = tuple[str, str, str]  # (título, subtítulo, etiqueta-para-color)


def _event_notification_content(event: Optional[dict]) -> Optional[NotificationContent]:
    """Contenido de la tarjeta de evento para el flujo de clip corto (fijo
    durante todo el clip, ``telemetry.event``) — None para "fondo" (tipo de
    plano sin evento) o si no se pidió clasificar eventos."""
    if not event or event.get("is_background"):
        return None
    label = event["label"]
    return label.upper().replace("_", " "), f"{event['confidence']:.0%} confianza", label


def _event_span_notification_content(span: Optional[EventSpan]) -> Optional[NotificationContent]:
    """Igual que ``_event_notification_content``, para el tramo de evento
    vigente del flujo de partido completo (``analyze_full_match``)."""
    if span is None:
        return None
    return span.label.upper().replace("_", " "), f"{span.peak_confidence:.0%} confianza", span.label


def _ball_action_notification_content(
    action: Optional[PossessionChangeAction],
) -> Optional[NotificationContent]:
    if action is None or action.is_background:
        return None
    return (
        action.label.upper().replace("_", " "),
        f"Jugador #{action.tracker_id} · {action.confidence:.0%}",
        action.label,
    )


def render_match_video(
    source_video: Path | str,
    tracks_jsonl_path: Path | str,
    output_path: Path | str,
    team_by_tracker_id: Optional[dict[int, int]],
    possessor_by_frame: dict[int, Optional[int]],
    telemetry: MatchTelemetry | FullMatchTelemetry,
    trace_length: int = 30,
    event_spans: Optional[list[EventSpan]] = None,
    action_highlight_frames: int = 60,
) -> None:
    """Segunda pasada (post-hoc) sobre el vídeo: cajas por equipo + resalte del
    poseedor + "notificación" visual de acción de balón (anillo verde en el
    jugador + balón especial durante un drive) + banner de evento + banner de
    acción de balón — todo en un único ``.mp4`` de salida.

    ``action_highlight_frames`` (config.AnalyticsConfig.action_highlight_frames,
    ~0.6s a 25fps por defecto): a diferencia del banner de acción de balón de
    antes (que persistía "hasta la siguiente"), tanto el resalte visual como el
    propio banner ahora se apagan pasados estos frames desde el disparador —
    se leen como una notificación puntual, no como un estado permanente.

    ``event_spans`` es opcional: si es ``None`` (flujo de clip corto, vía
    ``analyze_match``), el banner de evento es fijo durante todo el clip
    (``telemetry.event``, comportamiento sin cambios respecto a antes). Si se
    pasa (flujo de partido completo, vía ``analyze_full_match``), el banner
    cambia a lo largo del vídeo según el ``EventSpan`` vigente en cada frame —
    mismo patrón "vigente hasta el siguiente disparador" que ya usa el banner
    de acción de balón, salvo que aquí sí se apaga al pasar ``span.end_s``
    (los eventos son mucho más esporádicos que los cambios de posesión; dejar
    un banner de "GOL" fijo minutos después de que pase sería engañoso).

    Reutiliza ``utils.visualizers.TrackingVisualizer``/``detections_from_record``
    para la parte de cajas/etiquetas (mismo motor que ``core/tracking.py
    --annotated-output`` y ``core/classification.py --annotated-output``);
    los banners son responsabilidad de este módulo, no de ``utils/visualizers.py``
    (que es una caja de herramientas de dibujo genérica, sin conocer eventos
    ni acciones de balón).
    """
    from utils.visualizers import (
        TrackingVisualizer,
        action_color,
        detections_from_record,
        draw_notification_card,
    )

    def _iter_tracks_jsonl(path: Path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    records_by_frame: dict[int, dict] = {
        rec["frame_idx"]: rec for rec in _iter_tracks_jsonl(Path(tracks_jsonl_path))
    }
    if not records_by_frame:
        raise ValueError(f"tracks.jsonl vacío o sin registros: {tracks_jsonl_path}")

    # Acción de balón "vigente" en cada frame: la del último cambio de
    # posesión visto hasta ese punto (persiste hasta la siguiente).
    action_by_trigger_frame = {a.frame_idx: a for a in telemetry.ball_actions}
    sorted_trigger_frames = sorted(action_by_trigger_frame.keys())

    import supervision as sv

    video_info = sv.VideoInfo.from_video_path(str(source_video))
    source_fps = float(video_info.fps or 25)  # frame_idx de tracks.jsonl está en este espacio
    sorted_frame_indices = sorted(records_by_frame.keys())
    stride = (
        sorted_frame_indices[1] - sorted_frame_indices[0]
        if len(sorted_frame_indices) >= 2
        else 1
    )
    video_info.fps = max(1, round(video_info.fps / max(stride, 1)))

    static_event_content: Optional[NotificationContent] = None
    event_by_trigger_frame: dict[int, EventSpan] = {}
    sorted_event_trigger_frames: list[int] = []
    if event_spans is not None:
        event_by_trigger_frame = {
            int(round(span.start_s * source_fps)): span for span in event_spans
        }
        sorted_event_trigger_frames = sorted(event_by_trigger_frame.keys())
    else:
        static_event_content = _event_notification_content(getattr(telemetry, "event", None))

    visualizer = TrackingVisualizer(trace_length=trace_length)

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {source_video}")

    try:
        with sv.VideoSink(str(output_path), video_info) as sink:
            frame_idx = 0
            current_action: Optional[PossessionChangeAction] = None
            trigger_ptr = 0
            current_event_span: Optional[EventSpan] = None
            event_trigger_ptr = 0

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
                if (
                    current_action is not None
                    and frame_idx - current_action.frame_idx > action_highlight_frames
                ):
                    current_action = None

                if event_spans is not None:
                    while (
                        event_trigger_ptr < len(sorted_event_trigger_frames)
                        and sorted_event_trigger_frames[event_trigger_ptr] <= frame_idx
                    ):
                        current_event_span = event_by_trigger_frame[
                            sorted_event_trigger_frames[event_trigger_ptr]
                        ]
                        event_trigger_ptr += 1
                    if (
                        current_event_span is not None
                        and frame_idx / source_fps > current_event_span.end_s
                    ):
                        current_event_span = None

                show_action = current_action is not None and not current_action.is_background
                persons, ball_dets = detections_from_record(rec)

                event_content = (
                    _event_span_notification_content(current_event_span)
                    if event_spans is not None
                    else static_event_content
                )

                # Resalte de jugador/balón: la acción de balón manda si está
                # activa (tracker_id real y concreto); si no, y hay un evento
                # activo, se usa el poseedor del frame como aproximación de
                # "quién está implicado" (core.events clasifica la ventana
                # entera, no identifica jugador).
                highlight_tracker_id: Optional[int] = None
                highlight_label: Optional[str] = None
                if show_action:
                    highlight_tracker_id = current_action.tracker_id
                    highlight_label = current_action.label
                elif event_content is not None:
                    possessor = possessor_by_frame.get(frame_idx)
                    if possessor is not None:
                        highlight_tracker_id = possessor
                        highlight_label = event_content[2]

                annotated = visualizer.annotate(
                    frame, persons, ball_dets, team_by_tracker_id,
                    possessor_tracker_id=possessor_by_frame.get(frame_idx),
                    highlight_tracker_id=highlight_tracker_id,
                    highlight_label=highlight_label,
                )

                if event_content is not None:
                    title, subtitle, color_label = event_content
                    annotated = draw_notification_card(
                        annotated, title, subtitle, action_color(color_label),
                        top_left=_EVENT_CARD_TOP_LEFT,
                        width=_NOTIFICATION_CARD_WIDTH, height=_NOTIFICATION_CARD_HEIGHT,
                    )

                action_content = _ball_action_notification_content(current_action)
                if action_content is not None:
                    title, subtitle, color_label = action_content
                    annotated = draw_notification_card(
                        annotated, title, subtitle, action_color(color_label),
                        top_left=_ACTION_CARD_TOP_LEFT,
                        width=_NOTIFICATION_CARD_WIDTH, height=_NOTIFICATION_CARD_HEIGHT,
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
    parser.add_argument("--events-backend", choices=["vjepa2", "videoarm", "direct", "hybrid"], default="vjepa2",
                        help="Clasificador de eventos: 'vjepa2' (core.events, in-process, por "
                             "defecto) o los de core.events_vllm (requieren el servidor vLLM "
                             "arrancado): 'videoarm', 'direct', 'hybrid' (el mejor medido, ver README)")
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
        # Ambos backends cumplen el mismo contrato (predict/predict_window ->
        # EventPrediction), así que analyze_match no distingue cuál recibe.
        if args.events_backend in ("videoarm", "direct", "hybrid"):
            from core.events_vllm import DirectVLMEventClassifier, HybridEventClassifier, VideoARMEventClassifier

            event_classifier = {
                "videoarm": VideoARMEventClassifier,
                "direct": DirectVLMEventClassifier,
                "hybrid": HybridEventClassifier,
            }[args.events_backend]()
        else:
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

    possessor_by_frame = {
        r["frame_idx"]: r.get("possessor_tracker_id") for r in possession_records
    }
    render_match_video(
        source_video=args.source,
        tracks_jsonl_path=args.tracks,
        output_path=args.output,
        team_by_tracker_id=team_by_tracker_id,
        possessor_by_frame=possessor_by_frame,
        telemetry=telemetry,
    )
    logger.info("Vídeo final escrito en %s", args.output)

    if args.telemetry:
        with open(args.telemetry, "w") as f:
            json.dump(telemetry.to_telemetry(), f, indent=2)
        logger.info("Telemetría escrita en %s", args.telemetry)


if __name__ == "__main__":
    main()
