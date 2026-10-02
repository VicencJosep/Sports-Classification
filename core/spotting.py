"""
core/spotting.py
=================
De probabilidades por ventana deslizante a eventos PUNTUALES
``{label, t_s, score}``: curva de score por clase -> máximos locales ->
supresión temporal de no-máximos (NMS).

Entrada: las filas de la caché de ``analytics.engine.ensure_event_windows``
(``start_s``, ``end_s``, ``probabilities`` de todas las clases). Salida: una
lista de ``SpottedEvent`` que ``scripts/evaluate_spotting.py`` compara 1 a 1
contra el ground truth.

Alternativa a ``analytics.engine._merge_event_windows`` (etiqueta ganadora
por ventana + fusión de tramos contiguos), que en partido completo tiene dos
problemas medidos sobre Hull–Sheffield W. (96 min, 21 tiros):

1. Un evento solo existe si su clase GANA en alguna ventana: 4 de los 5
   tiros perdidos quedaron bajo ventanas ganadas por foul/corner, aunque
   "shot" pudiera tener probabilidad alta en ellas.
2. No hay un mando que cambie precisión por recall: la confianza de la
   etiqueta ganadora no separa aciertos de fallos (precisión 33-36% con
   cualquier umbral).

Aquí cada clase tiene su propia curva y su propio umbral, así que se puede
elegir el punto de trabajo: umbral bajo para proponer candidatos a un
verificador (recall), o alto para emitir eventos directamente (precisión).

Convención gol ⊂ tiro: SoccerNet-v2 y ``Labels-ball.json`` anotan el tiro que
acaba en gol ADEMÁS del gol (0,6-3,5 s antes). Por eso la curva de "shot"
suma P(shot) + P(goal) (``CURVE_COMPONENTS``): un gol también produce su
evento de tiro.

Lógica pura: sin I/O de vídeo ni ML, igual que ``core/windowing.py``.

Uso rápido:
    from core.spotting import SpottingParams, spot_events
    events = spot_events(rows, {"shot": SpottingParams(min_score=0.2, nms_s=8.0)})
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

# Clases cuya curva es la suma de varias probabilidades del clasificador.
# El resto de clases usan solo su propia probabilidad.
CURVE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "shot": ("shot", "goal"),
}


@dataclass(frozen=True)
class SpottedEvent:
    """Evento puntual detectado en la curva de una clase."""

    label: str
    t_s: float     # instante estimado del evento (centro de ventana + offset_s)
    score: float   # valor de la curva en el pico

    def to_telemetry(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SpottingParams:
    """Parámetros de detección de picos para una clase."""

    min_score: float = 0.0   # un pico por debajo no se emite
    nms_s: float = 8.0       # radio de supresión: dentro de ±nms_s de un pico
                              # más alto no se emite otro
    offset_s: float = 0.0    # corrección "instante real - centro de ventana",
                              # medida sobre el GT (evaluate_spotting la reporta)


def class_curve(
    rows: Sequence[Mapping],
    label: str,
    components: Optional[Mapping[str, tuple[str, ...]]] = None,
) -> tuple[list[float], list[float]]:
    """Curva ``(tiempos, scores)`` de ``label``: un punto por ventana, en su
    centro, con la suma de las probabilidades de ``components[label]`` (o
    solo la de ``label`` si no aparece ahí)."""
    parts = (components if components is not None else CURVE_COMPONENTS).get(label, (label,))
    times = [(r["start_s"] + r["end_s"]) / 2 for r in rows]
    scores = [sum(float(r["probabilities"].get(p, 0.0)) for p in parts) for r in rows]
    return times, scores


def find_peaks(
    times: Sequence[float], scores: Sequence[float], min_score: float, nms_s: float
) -> list[tuple[float, float]]:
    """Máximos locales de la curva con score >= ``min_score``, tras NMS
    greedy: se recorren de mayor a menor score y se descarta cualquiera a
    ``nms_s`` segundos o menos de uno ya aceptado. Devuelve ``[(t, score)]``
    ordenado por tiempo.

    Como un pico solo puede ser suprimido por otro MÁS alto, filtrar por
    ``min_score`` antes o después del NMS da el mismo resultado -- por eso se
    puede calcular una vez con ``min_score=0`` y barrer umbrales después.
    """
    n = len(scores)
    candidates = [
        (scores[i], times[i])
        for i in range(n)
        if scores[i] >= min_score
        and (i == 0 or scores[i] >= scores[i - 1])
        and (i == n - 1 or scores[i] >= scores[i + 1])
    ]
    candidates.sort(reverse=True)

    kept: list[tuple[float, float]] = []
    for score, t in candidates:
        if all(abs(t - kt) > nms_s for kt, _ in kept):
            kept.append((t, score))
    return sorted(kept)


def spot_events(
    rows: Sequence[Mapping],
    params: Mapping[str, SpottingParams],
    components: Optional[Mapping[str, tuple[str, ...]]] = None,
) -> list[SpottedEvent]:
    """Eventos puntuales de todas las clases de ``params``, ordenados por tiempo."""
    events: list[SpottedEvent] = []
    for label, p in params.items():
        times, scores = class_curve(rows, label, components)
        events.extend(
            SpottedEvent(label=label, t_s=t + p.offset_s, score=score)
            for t, score in find_peaks(times, scores, p.min_score, p.nms_s)
        )
    events.sort(key=lambda e: e.t_s)
    return events
