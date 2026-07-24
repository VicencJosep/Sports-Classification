"""
utils/visualizers.py
=====================
Anotadores gráficos (roboflow/supervision) para dibujar sobre el frame
original los resultados de ``core.tracking.PlayerTracker``: cajas + etiqueta
(ID de tracking, rol, dorsal) coloreadas por ``tracker_id`` (cada jugador
conserva su color entre frames), estela de trayectoria por jugador, y una
marca aparte para el balón.

También soporta colorear por EQUIPO en vez de por ``tracker_id`` (rojo/azul,
gris para quien no tenga equipo asignado), usando el resultado de
``core/classification.py`` (``TeamClusteringResult.teams``). El balón se
anota con un annotator distinto porque nunca tiene ``tracker_id`` propio (se
resuelve por mejor detección en cada frame, ver
``core/tracking.py::PlayerTracker._resolve_ball``).

Este módulo NO importa nada de ``core.tracking`` — trabaja directamente sobre
``sv.Detections`` o sobre los registros ya serializados de ``tracks.jsonl``,
para poder renderizar vídeo tanto en vivo (durante el tracking, ver
``core/tracking.py --annotated-output``) como a posteriori, sin re-ejecutar
el detector (ver ``render_annotated_video``, usada por
``core/classification.py --annotated-output`` para el vídeo coloreado por
equipo, que no puede conocerse hasta que se ha visto el vídeo entero).

Uso rápido (standalone, desde la raíz del proyecto):
    python -m core.tracking --source partido.mp4 --annotated-output tracks.mp4
    python -m core.classification --source partido.mp4 --tracks tracks.jsonl \
           --output teams.json --annotated-output teams.mp4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import supervision as sv

# Paleta fija para el modo "por equipo": índice 0/1 = equipo 0/1, índice 2 =
# sin equipo asignado (árbitro, portero, tracklet excluido por
# core/classification.py — ver TeamClusteringResult.excluded).
_TEAM_PALETTE: sv.ColorPalette = sv.ColorPalette(colors=[sv.Color.RED, sv.Color.BLUE, sv.Color.GREY])
_TEAM_UNASSIGNED_IDX: int = 2
_TEAM_LABELS: dict[int, str] = {0: "Equipo 0", 1: "Equipo 1"}

# class_id placeholder para el balón: BoxAnnotator con color_lookup=CLASS (su
# default) exige que class_id no sea None, pero aquí el balón siempre se
# pinta con un color FIJO (sv.Color.WHITE, no una paleta), así que el valor
# concreto es irrelevante -- ver TrackingVisualizer._ball_annotator.
_BALL_CLASS_ID_PLACEHOLDER: int = 0


class TrackingVisualizer:
    """Dibuja sobre un frame BGR un conjunto de detecciones ya trackeadas.

    Args:
        trace_length: nº de frames de estela por tracker_id (0 desactiva la estela).
    """

    def __init__(self, trace_length: int = 30) -> None:
        self._box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self._label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.TRACK, text_position=sv.Position.TOP_LEFT,
            smart_position=True,  # separa etiquetas solapadas (jugadores muy juntos)
        )
        self._trace_annotator = (
            sv.TraceAnnotator(trace_length=trace_length, color_lookup=sv.ColorLookup.TRACK)
            if trace_length > 0
            else None
        )
        # Anotadores aparte para el modo "por equipo": la PALETA de colores
        # se fija en el constructor de cada annotator (aquí _TEAM_PALETTE),
        # no se puede compartir instancia con los de arriba (paleta genérica
        # por tracker_id) y solo cambiar el índice de color en cada llamada.
        self._team_box_annotator = sv.BoxAnnotator(color=_TEAM_PALETTE)
        self._team_label_annotator = sv.LabelAnnotator(
            color=_TEAM_PALETTE, text_position=sv.Position.TOP_LEFT, smart_position=True,
        )
        self._team_trace_annotator = (
            sv.TraceAnnotator(trace_length=trace_length, color=_TEAM_PALETTE)
            if trace_length > 0
            else None
        )
        # El balón no tiene tracker_id ni equipo -> color fijo siempre.
        self._ball_annotator = sv.BoxAnnotator(color=sv.Color.WHITE, thickness=2)

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        ball_detections: sv.Detections,
        team_by_tracker_id: Optional[dict[int, int]] = None,
    ) -> np.ndarray:
        """Devuelve una COPIA anotada de ``frame`` (no lo modifica in-place).

        Si se indica ``team_by_tracker_id`` (``{tracker_id: team_cluster}``,
        de ``core/classification.py::TeamClusteringResult.teams``), colorea
        cajas/etiquetas/estela por EQUIPO (rojo/azul; gris para cualquier
        ``tracker_id`` no presente en el dict — árbitros, porteros, tracks
        excluidos) en vez de por ``tracker_id``.
        """
        annotated = frame.copy()
        persons = detections

        if len(persons) > 0:
            if team_by_tracker_id is not None:
                custom_lookup = np.array(
                    [
                        team_by_tracker_id.get(int(tid), _TEAM_UNASSIGNED_IDX)
                        for tid in persons.tracker_id
                    ],
                    dtype=int,
                )
                box_annotator = self._team_box_annotator
                label_annotator = self._team_label_annotator
                trace_annotator = self._team_trace_annotator
            else:
                custom_lookup = None
                box_annotator = self._box_annotator
                label_annotator = self._label_annotator
                trace_annotator = self._trace_annotator

            if trace_annotator is not None:
                annotated = trace_annotator.annotate(
                    scene=annotated, detections=persons, custom_color_lookup=custom_lookup
                )
            annotated = box_annotator.annotate(
                scene=annotated, detections=persons, custom_color_lookup=custom_lookup
            )
            labels = [
                self._person_label(persons, i, team_by_tracker_id) for i in range(len(persons))
            ]
            annotated = label_annotator.annotate(
                scene=annotated, detections=persons, labels=labels, custom_color_lookup=custom_lookup
            )

        if len(ball_detections) > 0:
            annotated = self._ball_annotator.annotate(
                scene=annotated, detections=ball_detections
            )

        return annotated

    @staticmethod
    def _person_label(
        persons: sv.Detections, idx: int, team_by_tracker_id: Optional[dict[int, int]]
    ) -> str:
        roles = persons.data.get("role", ["player"] * len(persons))
        jerseys = persons.data.get("jersey", [None] * len(persons))
        tracker_id = int(persons.tracker_id[idx])
        role = roles[idx]

        if team_by_tracker_id is not None:
            team = team_by_tracker_id.get(tracker_id)
            # Con equipo asignado, "player" es redundante con el color -> se omite
            # para dejar sitio en escenas con muchos jugadores muy juntos (córners,
            # áreas...). Sin equipo (árbitro/portero/excluido) sí interesa el rol.
            label = (
                f"#{tracker_id} · {_TEAM_LABELS[team]}"
                if team is not None
                else f"#{tracker_id} {role} · sin equipo"
            )
        else:
            label = f"#{tracker_id} {role}"

        if jerseys[idx] is not None:
            label += f" ({jerseys[idx]})"
        return label


# --------------------------------------------------------------------------- #
#  Renderizado a posteriori (a partir de tracks.jsonl ya escrito, sin
#  re-ejecutar el detector)
# --------------------------------------------------------------------------- #

def _iter_tracks_jsonl(path: Path) -> Iterator[dict]:
    """Lee un JSONL de tracking línea a línea (mismo patrón que
    core/homography.py, core/possession.py, core/classification.py)."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _detections_from_record(record: dict) -> tuple[sv.Detections, sv.Detections]:
    """Reconstruye (personas, balón) como ``sv.Detections`` a partir de un
    registro de ``tracks.jsonl`` (formato ``FrameTrackingResult.
    to_telemetry()``) — para renderizar un vídeo anotado a partir de
    telemetría ya escrita, sin tener que re-ejecutar el tracking."""
    tracks = record.get("tracks", [])
    if tracks:
        persons = sv.Detections(
            xyxy=np.array([t["xyxy"] for t in tracks], dtype=float),
            confidence=np.array([t["confidence"] for t in tracks], dtype=float),
            class_id=np.array([t["class_id"] for t in tracks], dtype=int),
            tracker_id=np.array([t["tracker_id"] for t in tracks], dtype=int),
        )
        persons.data["role"] = np.array([t.get("role") for t in tracks], dtype=object)
        persons.data["jersey"] = np.array([t.get("jersey") for t in tracks], dtype=object)
    else:
        persons = sv.Detections.empty()

    ball = record.get("ball")
    if ball:
        ball_dets = sv.Detections(
            xyxy=np.array([ball["xyxy"]], dtype=float),
            confidence=np.array([ball["confidence"]], dtype=float),
            class_id=np.array([_BALL_CLASS_ID_PLACEHOLDER], dtype=int),
        )
    else:
        ball_dets = sv.Detections.empty()

    return persons, ball_dets


def render_annotated_video(
    source_video: Path | str,
    tracks_jsonl_path: Path | str,
    output_path: Path | str,
    team_by_tracker_id: Optional[dict[int, int]] = None,
    trace_length: int = 30,
) -> None:
    """Segunda pasada (post-hoc) sobre un vídeo ya trackeado: dibuja cajas,
    etiquetas y estelas sobre cada frame a partir de ``tracks_jsonl_path``
    (de ``core/tracking.py``) y escribe un ``.mp4`` anotado — sin volver a
    correr el detector, solo decodifica el vídeo original.

    Si se pasa ``team_by_tracker_id`` (de ``core/classification.py::
    TeamClusteringResult``), colorea por equipo en vez de por ``tracker_id``
    — pensado exactamente para este caso: la asignación de equipo no se
    conoce hasta que se ha visto el vídeo entero (K-Means sobre todos los
    tracklets), así que no puede pintarse en la misma pasada que el tracking.

    Solo decodifica los frames que tienen registro en ``tracks_jsonl_path``
    (avanza con ``cap.grab()``, sin decodificar, en los demás — mismo patrón
    que ``core/classification.py::assign_teams``). El fps de salida se ajusta
    automáticamente si el `tracks.jsonl` viene de un `--stride > 1` (inferido
    de la separación entre `frame_idx` consecutivos).
    """
    records_by_frame: dict[int, dict] = {
        rec["frame_idx"]: rec for rec in _iter_tracks_jsonl(Path(tracks_jsonl_path))
    }
    if not records_by_frame:
        raise ValueError(f"tracks.jsonl vacío o sin registros: {tracks_jsonl_path}")

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

                persons, ball_dets = _detections_from_record(rec)
                annotated = visualizer.annotate(frame, persons, ball_dets, team_by_tracker_id)
                sink.write_frame(annotated)
                frame_idx += 1
    finally:
        cap.release()
