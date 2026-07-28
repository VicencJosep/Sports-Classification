"""
core/windowing.py
==================
Generación de ventanas temporales de longitud fija sobre un vídeo largo (p.ej.
un partido completo de 90 min), para alimentar en modo ventana deslizante a
clasificadores pensados originalmente para un clip corto ya recortado
(``core/events.py``).

Lógica pura: sin I/O de vídeo, sin ML. ``iter_windows`` no toma decisiones de
política (p.ej. si se descarta la última ventana por ser demasiado corta) —
eso es responsabilidad de la capa de composición (``analytics/engine.py``),
igual que ``core/possession.py::resolve_possessor`` no aplica histéresis (eso
lo hace ``iter_possession`` un nivel por encima).

Las ventanas se dimensionan por duración real (segundos), no por nº de frames:
la densidad de muestreo de ``core/events.py`` depende de cuántos segundos
reales cubre la ventana (ver ``config.EventsConfig.window_s``), no solo de
cuántos frames tiene.

Uso rápido:
    from core.windowing import iter_windows
    for window in iter_windows(total_frames=142887, fps=25.0, window_s=6.0, stride_s=3.0):
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Window:
    """Una ventana temporal sobre un vídeo, en frames y en segundos."""

    index: int
    start_frame: int  # inclusive
    end_frame: int  # exclusivo
    start_s: float
    end_s: float

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def iter_windows(
    total_frames: int,
    fps: float,
    window_s: float,
    stride_s: float,
) -> Iterator[Window]:
    """Ventanas de duración ``window_s`` cada ``stride_s`` segundos sobre
    ``[0, total_frames)`` a ``fps`` frames/segundo.

    La última ventana se clampa a ``total_frames`` (no se rellena) si no
    encaja exacta, así que puede quedar más corta que ``window_s`` — el
    llamador decide si la descarta (ver ``config.FullMatchConfig.
    min_window_fraction``). No genera nada si ``total_frames <= 0``.
    """
    if total_frames <= 0:
        return
    if fps <= 0:
        raise ValueError(f"fps debe ser positivo: {fps}")
    if window_s <= 0 or stride_s <= 0:
        raise ValueError(f"window_s y stride_s deben ser positivos: {window_s}, {stride_s}")

    window_frames = max(1, round(window_s * fps))
    stride_frames = max(1, round(stride_s * fps))

    index = 0
    start_frame = 0
    while start_frame < total_frames:
        end_frame = min(start_frame + window_frames, total_frames)
        yield Window(
            index=index,
            start_frame=start_frame,
            end_frame=end_frame,
            start_s=start_frame / fps,
            end_s=end_frame / fps,
        )
        index += 1
        start_frame += stride_frames
