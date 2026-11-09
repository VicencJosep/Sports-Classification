# Uso — módulo de tracking (standalone)

> Sección sacada del README el 2026-11-09, sin cambios de contenido.

```bash
python -m core.tracking --source partido.mp4 --output tracks.jsonl \
       --detector hybrid --device cuda --stride 1 --max-frames 300
```

python -m core.tracking --source clips_analisis/shots_off_target/shotoff101.mp4 \
       --output tracks.jsonl --annotated-output tracks.mp4 \
       --detector hybrid --device cuda


`--detector` acepta `hybrid` (por defecto: personas de SoccerMaster + balón de YOLO11x),
`soccermaster` (solo personas, con rol y dorsal) o `yolo` (fallback ligero, sin
checkpoints de Soccer_Master ni roles).

Genera un JSONL con un registro por frame:

```json
{
  "frame_idx": 120,
  "timestamp_s": 4.0,
  "tracks": [
    {"tracker_id": 7, "xyxy": [512.1, 300.4, 560.8, 420.2],
     "anchor_xy": [536.4, 420.2], "confidence": 0.91, "class_id": 3,
     "role": "player", "jersey": 10}
  ],
  "ball": {"xyxy": [880.0, 500.1, 902.3, 521.9],
           "center_xy": [891.1, 511.0], "confidence": 0.42}
}
```

`class_id` sigue `ROLE_TO_CLASS_ID` (`ball=0, goalkeeper=1, other=2, player=3, referee=4`);
`role`/`jersey` solo se rellenan con detectores `hybrid`/`soccermaster` (con YOLO puro,
`jersey` es siempre `null`).

### Vídeo anotado (`--annotated-output`)

```bash
python -m core.tracking --source partido.mp4 --output tracks.jsonl \
       --annotated-output tracks.mp4 --detector hybrid --device cuda
```

Con `--annotated-output` se escribe además un `.mp4` con lo que devuelve
`utils/visualizers.py::TrackingVisualizer`: caja + etiqueta (`#tracker_id rol
(dorsal)`) coloreadas por `tracker_id` (cada jugador mantiene su color entre
frames, vía `sv.ColorLookup.TRACK`), estela de trayectoria de los últimos 30
frames por jugador, y una caja blanca aparte para el balón (que nunca tiene
`tracker_id` propio, así que se colorea aparte por diseño — ver docstring del
módulo). Con `--stride > 1` el `.mp4` de salida ajusta su fps para que el
vídeo se siga viendo a velocidad real pese a saltarse frames.

Notas de diseño de `core/tracking.py`:

- **Personas → ByteTrack** (IDs persistentes entre frames); **balón → mejor detección por frame** (ByteTrack pierde objetos pequeños y rápidos; la trayectoria se suavizará por interpolación en `homography.py`/`analytics`).
- `anchor_xy` es el **bottom-center** de la caja: el punto de apoyo del jugador con el suelo, que es el que debe proyectarse con la matriz de homografía.
- `process_video()` es un **generador**, pensado para encadenarse a colas asíncronas sin cargar el vídeo en memoria.
- El balón de SoccerMaster casi nunca se detecta (dataset GSR centrado en personas); de ahí el `HybridDetector` por defecto, que combina personas de SoccerMaster con balón de YOLO11x. Ese YOLO corre a 1280 px (`core.tracking.DEFAULT_BALL_IMGSZ`, la resolución nativa de un 720p): a los 640 por defecto de Ultralytics el balón medía 3-5 px y se detectaba en el 54% de los frames frente al 70% a 1280 (medido el 11-oct-2026 en un juvenil de YouTube; +20 ms/frame). Hay además un YOLO11s de una sola clase ajustado con SoccerNet Tracking (`models/ball/yolo11s_snt_ball.pt`, `core.tracking.DEFAULT_BALL_MODEL_PATH`, se activa con `--ball-model`): en los clips de test de SoccerNet su mejor detección por frame acierta el 78% frente al 62% de COCO, pero en el partido completo del Espanyol (YouTube) salta a botas, farolas y árboles del fondo (23% de saltos de >80 px entre frames seguidos frente al 5% de COCO), así que NO es el de por defecto hasta tener coherencia temporal al elegir el balón. Con COCO a 1280 ese partido pasó de balón en el 19% a 50% de los frames de juego y de poseedor en el 34% al 65%. Se reentrena con `scripts/ball_detector/prepare_soccernet_tracking_ball.py` + `train_ball_yolo.py`; números en `output/ball_experiments/CONCLUSIONES.md`. Ese detector también dispara sobre el escudo del marcador y otros gráficos fijos (en el juvenil de YouTube el escudo ganaba al balón real en el 45-80% de los frames), así que `PlayerTracker._resolve_ball` pasa antes por `core.tracking.StaticBallFilter`: una detección clavada en el mismo píxel en el 70% de los últimos 750 frames se descarta (un balón parado 20 s con la cámara quieta también, pero entonces no hay posesión que perder). Los primeros ~15-25 s de cada vídeo no se filtran por falta de historial.
