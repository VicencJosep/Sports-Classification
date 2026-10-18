"""Selección automática de las mejores jugadas de un partido ya analizado.

Tres etapas (ver ``pipeline.generate_highlights``):

1. ``signals``: rejilla de 0,5 s con todo lo que el análisis ya dejó (curvas
   de T-DEED, poseedor/equipo, posición en el campo) más la excitación del
   audio, que es la única señal que delata un gol cuando T-DEED no lo ve.
2. ``candidates``: anclas (tiros, goles, picos de audio) -> jugada completa
   hacia atrás (misma posesión) -> puntuación por reglas.
3. ``vlm``: un modelo de visión-lenguaje mira los frames de cada candidato,
   decide si merece highlight, dónde empieza la jugada y cuándo vuelve el
   partido tras la celebración. ``cut`` recorta con ffmpeg.
"""

from .pipeline import HighlightInputs, generate_highlights

__all__ = ["HighlightInputs", "generate_highlights"]
