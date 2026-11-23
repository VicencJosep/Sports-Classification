"""
core/team_possession/frames.py
==============================
Representación común de datos posicionales para la posesión de equipo.

Cualquier fuente (EPTS de Metrica, XML de Sportec, nuestro ``tracks_2d.jsonl``)
se convierte a un :class:`PositionalData` y a partir de ahí el resto del paquete
(``features``, ``control``, ``decode``) no sabe de dónde vienen los datos.

Convenciones:

* Coordenadas en **metros con origen en la esquina** del campo: ``x`` en
  ``[0, pitch_length]`` a lo largo, ``y`` en ``[0, pitch_width]`` a lo ancho.
  Es la misma convención que ``field_xy`` de ``core/homography.py``.
* Un *slot* es una columna de ``players_xy``: puede ser un jugador con
  identidad (Sportec, nuestro tracking) o un tracklet anónimo (ATD de Metrica,
  46 slots que cambian de identidad). Solo se exige que cada slot tenga equipo.
* Posiciones ausentes = ``NaN``. El balón falta con frecuencia (41 % en el
  reto) y eso es parte del problema, no un error de carga.
* Las etiquetas usan los códigos del scorer oficial: H=0, A=1, D=2, X=3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

HOME, AWAY, NO_TEAM = 0, 1, -1


@dataclass
class PositionalData:
    """Posiciones de jugadores y balón por fotograma, más metadatos."""

    fps: float
    frame_index: np.ndarray  # [T] int64: índice de fotograma de la fuente, creciente
    period: np.ndarray  # [T] int16: 1, 2, ... (0 si se desconoce)
    players_xy: np.ndarray  # [T, N, 2] float32 en metros, NaN si ausente
    team: np.ndarray  # [N] int8: HOME / AWAY / NO_TEAM
    ball_xy: np.ndarray  # [T, 2] float32 en metros, NaN si ausente
    player_ids: list[str] = field(default_factory=list)
    ball_z: Optional[np.ndarray] = None  # [T] float32 en metros si la fuente lo da
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    labels: Optional[np.ndarray] = None  # [T] int8 H/A/D/X si hay verdad
    name: str = ""

    @property
    def n_frames(self) -> int:
        return int(self.players_xy.shape[0])

    @property
    def n_slots(self) -> int:
        return int(self.players_xy.shape[1])

    def validate(self) -> None:
        T, N = self.n_frames, self.n_slots
        assert self.players_xy.shape == (T, N, 2), self.players_xy.shape
        assert self.ball_xy.shape == (T, 2), self.ball_xy.shape
        assert self.team.shape == (N,), self.team.shape
        assert self.frame_index.shape == (T,) and self.period.shape == (T,)
        assert np.all(np.diff(self.frame_index) > 0), "frame_index debe ser creciente"
        if self.labels is not None:
            assert self.labels.shape == (T,), self.labels.shape
        if self.ball_z is not None:
            assert self.ball_z.shape == (T,), self.ball_z.shape

    def slice(self, start: int, stop: int) -> "PositionalData":
        """Sub-secuencia por posición (no por frame_index), sin copiar los slots."""
        return PositionalData(
            fps=self.fps,
            frame_index=self.frame_index[start:stop],
            period=self.period[start:stop],
            players_xy=self.players_xy[start:stop],
            team=self.team,
            ball_xy=self.ball_xy[start:stop],
            player_ids=self.player_ids,
            ball_z=None if self.ball_z is None else self.ball_z[start:stop],
            pitch_length=self.pitch_length,
            pitch_width=self.pitch_width,
            labels=None if self.labels is None else self.labels[start:stop],
            name=self.name,
        )

    def period_runs(self) -> list[tuple[int, int]]:
        """Tramos ``[start, stop)`` de posiciones con el mismo periodo, en orden.

        Todo lo temporal (velocidades, decodificación) se hace dentro de un
        periodo y nunca a través del descanso, igual que el scorer oficial hace
        con sus ``valid_runs``.
        """
        if self.n_frames == 0:
            return []
        breaks = np.flatnonzero(self.period[1:] != self.period[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [self.n_frames]))
        return [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]

    def iter_periods(self) -> Iterator[tuple[int, "PositionalData"]]:
        for start, stop in self.period_runs():
            yield int(self.period[start]), self.slice(start, stop)

    def ball_observed(self) -> np.ndarray:
        """[T] bool: el balón tiene posición en ese fotograma."""
        return np.isfinite(self.ball_xy).all(axis=1)

    def slot_observed(self) -> np.ndarray:
        """[T, N] bool: el slot tiene posición en ese fotograma."""
        return np.isfinite(self.players_xy).all(axis=2)

    def summary(self) -> str:
        obs = self.ball_observed().mean() if self.n_frames else 0.0
        per_frame = self.slot_observed().sum(axis=1)
        parts = [
            f"{self.name or 'posiciones'}: {self.n_frames} fotogramas a {self.fps:g} fps,",
            f"{self.n_slots} slots ({(self.team == HOME).sum()} local,",
            f"{(self.team == AWAY).sum()} visitante), balón visible {100 * obs:.1f} %,",
            f"jugadores por fotograma {per_frame.mean():.1f} (máx {per_frame.max() if len(per_frame) else 0}),",
            f"periodos {sorted(set(self.period.tolist()))}",
        ]
        if self.labels is not None:
            counts = {k: int((self.labels == v).sum()) for k, v in (("H", 0), ("A", 1), ("D", 2), ("X", 3))}
            parts.append(f"etiquetas {counts}")
        return " ".join(parts)


def resample(data: PositionalData, step: int) -> PositionalData:
    """Submuestrea cada ``step`` fotogramas (p. ej. 25 fps -> 5 fps con step=5).

    Mantiene ``frame_index`` original para poder volver a 25 fps con
    :func:`expand_labels`. Las etiquetas se submuestrean igual (sin votar).
    """
    if step <= 1:
        return data
    sel = slice(0, None, step)
    return PositionalData(
        fps=data.fps / step,
        frame_index=data.frame_index[sel],
        period=data.period[sel],
        players_xy=data.players_xy[sel],
        team=data.team,
        ball_xy=data.ball_xy[sel],
        player_ids=data.player_ids,
        ball_z=None if data.ball_z is None else data.ball_z[sel],
        pitch_length=data.pitch_length,
        pitch_width=data.pitch_width,
        labels=None if data.labels is None else data.labels[sel],
        name=data.name,
    )


def expand_labels(sub_frame_index: np.ndarray, sub_labels: np.ndarray,
                  full_frame_index: np.ndarray, fill: int = 3) -> np.ndarray:
    """Devuelve etiquetas por fotograma completo a partir de las submuestreadas.

    Cada fotograma completo toma la etiqueta del último fotograma submuestreado
    que no lo supera (escalón hacia delante); los anteriores al primero reciben
    ``fill`` (por defecto X).
    """
    pos = np.searchsorted(sub_frame_index, full_frame_index, side="right") - 1
    out = np.full(len(full_frame_index), fill, dtype=np.int8)
    ok = pos >= 0
    out[ok] = sub_labels[pos[ok]]
    return out
