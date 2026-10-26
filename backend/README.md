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
| `GET /api/videos/{id}` · `GET /api/videos/{id}/content` | Metadatos · el fichero, con `Range` (206) para el `<video>` del navegador |
| `POST /api/matches` · `GET /api/matches` · `GET /api/matches/{id}` | Partido = metadatos del usuario + 1 parte (`full`) o 2 (`first-half`, `second-half`), cada una ligada a un vídeo |
| `POST /api/matches/{id}/analysis` | Encola el pipeline para cada parte (uno a la vez: la GPU es compartida). Cuerpo opcional `{"mode": "full-match" \| "clip"}` |
| `GET /api/matches/{id}/analysis/status` | Estado por parte (`pending`, `processing` + etapa, `completed`, `error` + motivo) |
| `GET /api/matches/{id}/parts/{part}/analysis` | Eventos, acciones de balón y clusters de equipo normalizados |
| `POST /api/matches/{id}/parts/{part}/reingest` | Re-normaliza las salidas crudas de una parte completada (tras cambiar adaptadores), sin pipeline |
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
videos/<video_id>/   video.json, source.mp4 y las cachés que el pipeline deja junto al vídeo (*.tracks.jsonl, ...)
matches/<match_id>/  match.json
outputs/<match_id>/<part_id>/  run.json (estado), pipeline.log, output/ (--output-dir), memory/ (normalizado + índices)
tmp/                 subidas en curso
```

El análisis se lanza con `--skip-render` (el frontend dibuja las capas sobre el vídeo original; `final.mp4` no se
reproduce en navegadores y pesa ~5 GB) y con homografía (`BACKEND_SKIP_HOMOGRAPHY=false`). La asignación de equipos
(`core/classification.py`, PRTReID) necesita el paquete `prtreid` en `/opt/venv`; si no está instalado, pon
`BACKEND_SKIP_CLASSIFICATION=true` (el pipeline corre con `--skip-classification` y la API devuelve clusters vacíos).

## Tests

```bash
docker exec -w /app/backend Sports-Classification .venv/bin/python -m pytest -q
```

Usan un pipeline falso (`tests/fake_pipeline.py`) que deja recortes de salidas reales, así que no necesitan GPU.
