"""
utils/visualizers.py
=====================
Anotadores gráficos (roboflow/supervision) para dibujar sobre el frame
original los resultados de ``core.tracking.PlayerTracker``: cajas + etiqueta
(ID de tracking, rol, dorsal) coloreadas por ``tracker_id`` (cada jugador
conserva su color entre frames), estela de trayectoria por jugador, y una
marca aparte para el balón.

El balón se anota con un annotator distinto porque nunca tiene
``tracker_id`` propio (se resuelve por mejor detección en cada frame, ver
``core/tracking.py::PlayerTracker._resolve_ball``): colorear por ``TRACK``
sobre detecciones sin tracker_id lanzaría un error en supervision.

Uso rápido (standalone, desde la raíz del proyecto):
    python -m core.tracking --source partido.mp4 --annotated-output tracks.mp4
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from core.tracking import FrameTrackingResult


class TrackingVisualizer:
    """Dibuja sobre un frame BGR los resultados de un ``FrameTrackingResult``.

    Args:
        trace_length: nº de frames de estela por tracker_id (0 desactiva la estela).
    """

    def __init__(self, trace_length: int = 30) -> None:
        self._box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self._label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.TRACK, text_position=sv.Position.TOP_LEFT
        )
        self._trace_annotator = (
            sv.TraceAnnotator(trace_length=trace_length, color_lookup=sv.ColorLookup.TRACK)
            if trace_length > 0
            else None
        )
        # El balón no tiene tracker_id -> color fijo, no color_lookup=TRACK.
        self._ball_annotator = sv.BoxAnnotator(color=sv.Color.WHITE, thickness=2)

    def annotate(self, frame: np.ndarray, result: FrameTrackingResult) -> np.ndarray:
        """Devuelve una COPIA anotada de ``frame`` (no lo modifica in-place)."""
        annotated = frame.copy()
        persons = result.detections

        if len(persons) > 0:
            if self._trace_annotator is not None:
                annotated = self._trace_annotator.annotate(scene=annotated, detections=persons)
            annotated = self._box_annotator.annotate(scene=annotated, detections=persons)
            labels = [self._person_label(persons, i) for i in range(len(persons))]
            annotated = self._label_annotator.annotate(
                scene=annotated, detections=persons, labels=labels
            )

        if len(result.ball_detections) > 0:
            annotated = self._ball_annotator.annotate(
                scene=annotated, detections=result.ball_detections
            )

        return annotated

    @staticmethod
    def _person_label(persons: sv.Detections, idx: int) -> str:
        roles = persons.data.get("role", ["player"] * len(persons))
        jerseys = persons.data.get("jersey", [None] * len(persons))
        label = f"#{int(persons.tracker_id[idx])} {roles[idx]}"
        if jerseys[idx] is not None:
            label += f" ({jerseys[idx]})"
        return label
