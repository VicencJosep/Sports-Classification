# Uso — composición final (`analytics/engine.py`)

> Sección sacada del README el 2026-11-09, sin cambios de contenido.

Une todo lo anterior en un único vídeo anotado + `telemetry.json`: tracking (cajas/IDs) +
equipos (color) + posesión (calculada aquí a partir de `tracks.jsonl`, no hace falta correr
`core/possession.py` aparte) + acción de balón (`core/ball_actions.py`, una vez por cada cambio
de poseedor) + evento (`core/events.py`, una vez para todo el clip).

```bash
python -m core.tracking --source clip.mp4 --output tracks.jsonl --detector hybrid --device cuda
python -m core.classification --source clip.mp4 --tracks tracks.jsonl --output teams.json --device cuda

python -m analytics.engine --source clip.mp4 --tracks tracks.jsonl --teams teams.json \
       --output final.mp4 --telemetry telemetry.json --device cuda
# --teams es opcional (sin él, las cajas se colorean por tracker_id en vez de por equipo)
# --skip-ball-actions / --skip-events omiten esa parte del análisis (y no cargan el modelo pesado
# correspondiente) si solo quieres probar el resto del pipeline más rápido
```

**Alcance actual, decisión explícita**: clips cortos ya recortados en torno a un evento (los de
`datasets/clips_analisis/`), NO partidos completos. `core/events.py` clasifica el clip **entero**
como un solo evento (válido aquí porque ya viene recortado — sobre un partido de 90 min diluiría
la señal, ver el propio docstring de ese módulo) y este módulo decodifica el clip entero a memoria
(viable para clips de segundos, no para un partido completo). El troceado de un partido en
clips/ventanas queda fuera de alcance por ahora.

**Qué se ve en el vídeo final**:
- Cajas + etiqueta coloreadas por equipo (o por `tracker_id` si no se pasa `--teams`), como en
  `core/classification.py --annotated-output`.
- Un contorno **amarillo** grueso sobre el poseedor del balón en cada frame, con `(POSESION)` en
  su etiqueta.
- Un banner superior semitransparente con el evento del clip (fijo durante todo el vídeo), p.ej.
  `[EVENTO] shot (81%)` — o `[fondo] ...` si el clasificador determina que es un plano sin evento
  real (ver taxonomía de `core/events.py`).
- Un banner inferior con la última acción de balón clasificada (`[ACCION] Jugador #9: pass (77%)`),
  que se actualiza en cada cambio de poseedor y persiste hasta el siguiente.

✅ **Verificado visualmente en ejecución real** (17 jul 2026,
`datasets/clips_analisis/shots_off_target/shotoff104.mp4`): las cuatro capas (equipo, posesión,
acción, evento) renderizan correctamente y a la vez sobre el mismo frame. Nota de diseño
verificada en el proceso: `sv.LabelAnnotator`/los banners de este módulo dibujan texto con
`cv2.putText` (fuentes Hershey), que **no soporta emoji/Unicode fuera de Latin-1** — usar solo
texto ASCII en cualquier etiqueta nueva (de ahí `(POSESION)` en vez de un emoji de balón).

`telemetry.json` (`MatchTelemetry.to_telemetry()`):

```json
{
  "source_video": "clip.mp4",
  "event": {"label": "shot", "confidence": 0.81, "is_background": false, "probabilities": {}},
  "ball_actions": [
    {"frame_idx": 12, "tracker_id": 9, "label": "pass", "confidence": 0.77,
     "probabilities": {}, "is_background": false}
  ],
  "n_frames": 75,
  "n_possession_changes": 1
}
```
