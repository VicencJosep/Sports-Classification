"""
core/team_possession
====================
Posesión de **equipo** por fotograma (local / visitante / balón muerto) a
partir de posiciones de jugadores y balón, con el balón ausente a menudo.

Complementa a ``core/possession.py`` (poseedor individual por cercanía en
píxeles): este paquete trabaja en metros, decide por *control* y no por
cercanía, detecta balón muerto y decodifica tramos coherentes. Nació para el
reto More than a Hack 2027 de Metrica Sports y está pensado para usarse igual
sobre nuestro ``tracks_2d.jsonl``.

Módulos: ``frames`` (representación común), ``io_metrica`` / ``io_sportec`` /
``io_pipeline`` (cargadores), ``features`` (rasgos por fotograma), ``control``
(evidencia de control por reglas), ``decode`` (suavizado y Viterbi),
``labels`` (códigos H/A/D/X) y ``spell_metrics`` (scorer oficial del reto).
Guía de uso: ``docs/uso/team_possession.md``.
"""
