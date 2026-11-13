# Backend — Football Intelligence API

FastAPI que hace de puente entre el pipeline de análisis de fútbol de este repo (`python -m main`) y el frontend
React (`../frontend_Sports-Classification`). El proceso de la API **no importa nada de ML**: lanza el pipeline como
subproceso con el intérprete de `/opt/venv` y adapta los ficheros que deja (`telemetry.json`, `teams.json`,
`*.tracks.jsonl`, `*.possession.jsonl`, `tracks_2d.jsonl`) a un dominio estable.

```
React ──HTTP/JSON + vídeo (Range)──► FastAPI (routers) ─► services ─► Football Memory ─► repositorios en fichero
                                                            │                 ▲
                                                            ▼                 │ adapters/ (JSON de Python -> Pydantic)
                                                  AnalysisService ── subproceso ──► python -m main (sin cambios)
```

## Arranque (dentro del contenedor `Sports-Classification`)

El pipeline solo corre en el contenedor, así que la API vive allí también, en su propio entorno (`backend/.venv`,
no toca `/opt/venv` ni `requirements.txt`). La primera vez:

```bash
docker exec -w /app/backend Sports-Classification sh -c 'python3 -m venv .venv && .venv/bin/pip install -r requirements.txt'
```

Arrancar (el contenedor usa la red del host, así que queda en `http://localhost:8100` sin publicar puertos):

```bash
docker exec -d -w /app/backend Sports-Classification sh -c '.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8100 > /tmp/uvicorn.log 2>&1'
curl http://localhost:8100/api/health        # {"status":"ok"}
# documentación interactiva (Swagger): http://localhost:8100/api/docs  (ReDoc: /api/redoc)
# con el frontend en marcha también sale por su puerto: http://localhost:5173/api/docs
```

Parar: `docker exec Sports-Classification pkill -f "uvicorn app.main:app"`. No uses `--reload` con un análisis en
marcha: reiniciar el servidor marca la ejecución como interrumpida (hay que relanzarla; las cachés del pipeline
hacen que el reintento sea barato).

Configuración: variables `BACKEND_*` o `backend/.env` (ver `.env.example`). El puerto por defecto es 8100 porque el
8000 está ocupado en esta máquina.

## API

| Ruta | Qué hace |
|---|---|
| `GET /api/health` | `{"status":"ok"}` |
| `POST /api/videos/upload` (multipart `file`) | Valida con `ffprobe`, guarda en `storage/videos/<id>/source.<ext>` y devuelve metadatos (fps, frames, duración…) |
| `GET /api/videos/{id}` · `GET /api/videos/{id}/content` | Metadatos · el fichero, con `Range` (206) para el `<video>` del navegador. Cada respuesta a un `Range` lleva como mucho `BACKEND_VIDEO_RANGE_CHUNK_BYTES` (2 MB; 0 = sin tope): el reproductor pide el trozo siguiente cuando lo necesita y no queda ninguna conexión abierta con el resto del fichero |
| `GET /api/videos/{id}/thumbnail?t=<s>&h=<px>` | Un fotograma como JPEG pequeño (3-4 KB, `ffmpeg`), para la tira de miniaturas del editor de clips. `t` se redondea a décimas; se guarda en `videos/<id>/thumbs/` y lleva `Cache-Control: immutable` |
| `POST /api/matches` · `GET /api/matches` · `GET /api/matches/{id}` | Partido = metadatos del usuario + 1 parte (`full`) o 2 (`first-half`, `second-half`), cada una ligada a un vídeo |
| `POST /api/matches/{id}/analysis` | Encola el pipeline para cada parte (uno a la vez: la GPU es compartida). Cuerpo opcional `{"mode": "full-match" \| "clip"}` |
| `GET /api/matches/{id}/analysis/status` | Estado por parte (`pending`, `processing` + etapa, `completed`, `error` + motivo) |
| `GET /api/matches/{id}/parts/{part}/analysis` | Eventos, acciones de balón y clusters de equipo normalizados |
| `POST /api/matches/{id}/parts/{part}/reingest` | Re-normaliza las salidas crudas de una parte completada (tras cambiar adaptadores), sin pipeline |
| `GET /api/matches/{id}/parts/{part}/review` | El análisis tal como lo enseña el visor: la **copia revisada** si existe; si no, la salida del pipeline (`review.exists: false`). Misma forma que `/analysis` más `side`, `review` (`edited`/`added`) y, en los eventos, `tracker_id` |
| `PATCH /api/matches/{id}/parts/{part}/review/items/{item}` | Corrige una acción (`act-…`) o un evento (`evt-…`): `label`, `time_s`, `tracker_id` (o `null`), `side` (`home`/`away`/`null`); solo cambia los campos que vienen. La primera corrección crea la copia |
| `POST …/review/items` · `DELETE …/review/items/{item}` · `POST …/review/items/{item}/restore` | Añade lo que el pipeline no detectó (`kind`: `action`/`event`) · borra lo que detectó de más · deshace el borrado |
| `POST /api/matches/{id}/parts/{part}/review/export` | Hoja de verdad `tiempo,etiqueta,dorsal,equipo` de un tramo (`{"from_s": 600, "to_s": 3600}`): la devuelve y la deja en `review/ground_truth_<desde>-<hasta>.csv` |
| `GET /api/matches/{id}/parts/{part}/tracklets` | Resumen por tracklet (rol y dorsal mayoritarios, equipo, primer/último frame) |
| `GET /api/matches/{id}/parts/{part}/tracking?from_s=&to_s=` | Frames de tracking de una ventana (máx. 120 s) |
| `POST /api/matches/{id}/parts/{part}/highlights` | Encola la generación de las mejores jugadas de una parte ya analizada (`analytics/highlights`; cuerpo opcional `{"force": true}` para no reutilizar cachés). También se encadena sola al acabar el análisis si `POST .../analysis` llevó `{"highlights": true}` |
| `GET /api/matches/{id}/parts/{part}/highlights` | Estado de la generación (`pending`, `processing` + etapa, `completed`, `error`) y, al terminar, los clips (título, inicio/fin, etiqueta, `content_url`) |
| `GET /api/matches/{id}/parts/{part}/highlights/{clip}/content` | El clip MP4, con `Range` (206) |

Convenciones: tiempos en segundos desde el inicio del vídeo de la parte (`*_s`), `frame_idx` desde 0
(`timestamp_s = frame_idx / fps`), fechas ISO-8601 UTC; coordenadas en píxeles del frame original y `field_xy` en
metros (origen en una esquina, campo 105 × 68); etiquetas en snake_case (`kick_off`, `throw_in`); estados
`pending | processing | completed | error`.

No hay `/players` ni `/statistics`: el pipeline no tiene identidad de jugador (solo tracklets) ni calcula métricas.

## Almacenamiento (`backend/storage/`, fuera de git)

```
videos/<video_id>/   video.json, source.mp4, las cachés que el pipeline deja junto al vídeo (*.tracks.jsonl, ...) y
                     thumbs/ (miniaturas ya servidas; se puede borrar)
matches/<match_id>/  match.json
outputs/<match_id>/<part_id>/  run.json (estado), pipeline.log, output/ (--output-dir), memory/ (normalizado + índices),
                               review/ (copia revisada: analysis.json, edits.jsonl y las hojas exportadas)
tmp/                 subidas en curso
```

El análisis se lanza con `--skip-render` (el frontend dibuja las capas sobre el vídeo original; `final.mp4` no se
reproduce en navegadores y pesa ~5 GB) y con homografía (`BACKEND_SKIP_HOMOGRAPHY=false`). La asignación de equipos
(`core/classification.py`, PRTReID) necesita el paquete `prtreid` en `/opt/venv`; si no está instalado, pon
`BACKEND_SKIP_CLASSIFICATION=true` (el pipeline corre con `--skip-classification` y la API devuelve clusters vacíos).

## Copia revisada (verdad anotada desde el visor)

El editor de clips del frontend corrige lo que detectó el pipeline: tipo, ejecutor, equipo e instante de una acción de
balón o de un evento; borra lo que sobra y añade lo que falta. `memory/analysis.json` no se toca: la primera
corrección lo copia a `review/analysis.json` y todas las siguientes se guardan en esa copia (escritura atómica), con
cada operación apuntada en `review/edits.jsonl` (qué había y qué quedó). Lo borrado se conserva en `deleted_*` dentro
de la copia, para deshacer y para contar falsos positivos.

- El equipo (`side`) se indica una vez por tracklet: vale para todas sus acciones que no lleven uno propio.
- La hoja exportada es la de `scripts/annotation_csv.py`; lo que no se ha tocado dentro del tramo cuenta como
  correcto. Un evento con una acción de la misma jugada a menos de 1,5 s (el saque de banda de T-DEED sale como
  acción y como evento) es una sola fila; `dorsal` es el `tracker_id` de esa ejecución del pipeline.
- Evaluar contra ella: `python -m scripts.evaluate_match --csv backend/storage/outputs/<id>/full/review/ground_truth_10m00-60m00.csv --backend-match <id>`.
- Si el partido se vuelve a analizar, los `tracker_id` cambian y la copia deja de valer (`review.stale: true`).

## Uso por un túnel SSH

El frontend se suele abrir por un túnel (reenvío de puertos de VS Code), que es lento y reparte mal el ancho de banda:
unas pocas descargas de vídeo en paralelo lo atascan y todo lo demás (guardar una corrección de 1 KB, cargar la
página) se queda esperando. Por eso nada debe bajar vídeo salvo el reproductor: el tracking se sirve por ventanas y
las miniaturas las corta el backend (`/thumbnail`); el vídeo sale en trozos de 2 MB y las coordenadas del tracking
van redondeadas (décima de píxel, centímetro de campo). Para ver si el túnel está atascado, las colas de envío hacia él:
`ss -tn state established '( sport = :5173 )'` (columna Send-Q; varias conexiones con ~4 MB = vídeo esperando).

## Tests

```bash
docker exec -w /app/backend Sports-Classification .venv/bin/python -m pytest -q
```

Usan un pipeline falso (`tests/fake_pipeline.py`) que deja recortes de salidas reales, así que no necesitan GPU.
