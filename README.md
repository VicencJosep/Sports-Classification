# Sports-Classification ⚽

Análisis automático de partidos de fútbol a partir de vídeo (retransmisión de TV o cámara única): eventos y acciones de
balón con su instante, tracking de jugadores y balón, equipos, posesión, homografía, goles y mejores jugadas. Una API
(`backend/`) lanza el pipeline y sirve los resultados al frontend (`../frontend_Sports-Classification`).

## Flujo de un partido completo

```
partido.mp4
  ├─ core/ball_spotting.py    T-DEED: eventos y acciones de balón con instante (tiro, gol, pase, centro, córner…)
  ├─ core/tracking.py         personas con rol y dorsal (SoccerMaster) + balón (YOLO11x) → BoT-SORT (GMC + ReID OSNet)
  ├─ core/possession.py       poseedor por frame: da el jugador de cada acción
  ├─ analytics/engine.py      composición → telemetry.json (fuente única de verdad)
  ├─ core/classification.py   equipos (PRTReId + K-Means) → teams.json
  ├─ core/homography.py       campo (PnLCalib) → tracks_2d.jsonl en metros
  ├─ analytics/goals.py       goles confirmados por marcador o saque de centro
  └─ render                   final.mp4

analytics/highlights/         mejores jugadas: reglas + SoccerHigh + VLM → clips
backend/                      FastAPI: lanza `python -m main` como subproceso y adapta sus ficheros
automation/                   publicación en YouTube Studio (MolmoWeb + Playwright)
```

Los cachés (`*.tracks.jsonl`, `*.possession.jsonl`, `*.tdeed.<modelo>.npz`) se guardan junto al vídeo; los entregables
van a `--output-dir`.

## Uso

Todo se ejecuta dentro del contenedor `Sports-Classification` (el repo está montado en `/app`):

```bash
# Partido completo
docker exec -w /app Sports-Classification python -m main --source partido.mp4 --output-dir output/mipartido --mode full-match

# Clip corto ya recortado en torno a un evento
docker exec -w /app Sports-Classification python -m main --source clip.mp4 --output-dir output/miclip --mode clip

# Mejores jugadas de un partido ya analizado
docker exec -w /app Sports-Classification python -m analytics.highlights --help

# Tests
docker exec -w /app Sports-Classification python -m pytest tests -q
```

Opciones principales de `main` (`python -m main --help` para todas): `--detector {hybrid,soccermaster,yolo}` (por
defecto `hybrid`), `--event-source {tdeed,mixed,vjepa2}` (por defecto `tdeed`), `--thresholds {auto,fixed}`,
`--skip-classification`, `--skip-homography`, `--skip-scoreboard`, `--skip-render`, `--publish`. Los valores por
defecto de cada módulo están en `config.py`.

## Instalación (Docker)

Imagen CUDA 12.8 + cuDNN sobre Ubuntu 24.04; no se usan entornos virtuales en el host.

```bash
docker build -t sports-analytics .
docker run -d --name Sports-Classification --gpus all --network host --restart unless-stopped \
  -v "$(pwd)":/app -w /app sports-analytics
```

`requirements.txt` se instala en el build. `yolo11x.pt` (Ultralytics) y los modelos de Hugging Face se descargan solos
en la primera ejecución, estos últimos a `models/` (`HF_HOME=/app/models`). Los pesos que hay que colocar a mano están
en la guía de cada módulo (`docs/uso/`). Solo para `automation/` hace falta además
`docker exec -it Sports-Classification playwright install --with-deps chromium`. El backend tiene su propio entorno
(`backend/README.md`).

## Problemas conocidos

Las versiones fijadas, los incidentes de dependencias sin pin y los parches a `Soccer_Master` están en
`docs/dependencias_y_problemas.md` (los comentarios del código que citan «README § Problemas conocidos» se refieren a
ese fichero).

## Estructura

| Carpeta | Contenido |
|---|---|
| `core/` | Módulos del pipeline: tracking, posesión, equipos, homografía, T-DEED (`ball_spotting`), R3D-18 (`ball_actions`), V-JEPA2 + SVC (`events`, alternativa a T-DEED), ReID (`osnet`) |
| `analytics/` | `engine.py` (composición y telemetría), `goals.py`, `highlights/` |
| `backend/` | API FastAPI y sus tests (`backend/README.md`) |
| `automation/` | Publicación en YouTube Studio |
| `scripts/` | Arneses de evaluación y utilidades; `benchmark/` (comparativa con el estado del arte); `shot_benchmark/` y `ball_detector/` (experimentos de tiros, panorámica y detector de balón) |
| `training/` | Entrenamiento del R3D-18 de acciones de balón |
| `tests/` | Tests (pytest) |
| `Soccer_Master/` | Repo de terceros vendorizado: detector de personas por defecto. Solo se versionan los ficheros parcheados; los pesos están en `Soccer_Master/sn-gamestate/pretrained_models` |
| `docs/` | Documentación (tabla siguiente) |

No se versionan: `datasets/`, `models/`, `output/`, `temp/`, `runs/`.

## Documentación

| Para… | Leer |
|---|---|
| Usar un módulo suelto | `docs/uso/`: `tracking`, `clasificacion_equipos`, `homografia`, `ball_spotting_tdeed`, `acciones_balon_r3d18`, `eventos_vjepa2`, `engine`, `highlights`, `publicacion_youtube` |
| Backend y API | `backend/README.md` |
| Saber qué toca ahora y qué está descartado | `docs/hoja_de_ruta.md` |
| Datasets y particiones train/val/eval | `docs/datasets.md`, `docs/particion_datasets.md` |
| Cifras frente al estado del arte | `docs/comparativa_benchmark.md` (un fichero por resultado en `docs/comparativa_benchmark/`) |
| Estado del arte | `docs/estado_arte_tracking.md`, `docs/estado_arte_highlights.md` |
| Mejores jugadas | `docs/highlights_evaluacion.md`, `docs/highlights_soccerhigh.md` |
| Problemas de dependencias y parches a `Soccer_Master` | `docs/dependencias_y_problemas.md` |
| Material cerrado | `docs/archivo/` (estudio de detección de tiros, bitácora completa) |

## Referencias

- [roboflow/supervision](https://github.com/roboflow/supervision) y [Ultralytics](https://docs.ultralytics.com/models/yolo11/): detección, BoT-SORT, anotadores
- [SoccerMaster](https://github.com/haolinyang-hlyang/SoccerMaster/): detección de personas con rol y dorsal
- [T-DEED](https://github.com/arturxe2/T-DEED): eventos y acciones de balón (GPL-3.0)
- [PnLCalib](https://github.com/mguti97/PnLCalib): calibración de campo
- [V-JEPA2](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256): embeddings de vídeo para `core/events.py`
- [SoccerNet](https://www.soccer-net.org/): datos y taxonomía de eventos
