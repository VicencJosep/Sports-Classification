# Sports-Classification ⚽

Sistema de **Sports Analytics para fútbol** que procesa vídeos de partidos con tres objetivos:

1. **Métricas tácticas deterministas** (bottom-up): tracking de jugadores y balón, homografía del campo, posesión, velocidades y heatmaps, usando **YOLO11x** + **roboflow/supervision** (ByteTrack).
2. **Motor de consultas semánticas** (top-down): orquestador **VideoARM / Symphony** para que un entrenador pregunte en lenguaje natural sobre el vídeo (*"¿cuántas llegadas al área tuvimos en la segunda parte?"*).
3. **Publicación automática en RRSS**: el VLM **MolmoWeb-4B** navega la web con **Playwright** y publica clips + métricas.

## Flujo de datos

```
                        ┌──────────────── BOTTOM-UP (batch) ────────────────┐
 partido.mp4 ──► core/tracking.py ──► core/classification.py ──► core/homography.py
                 (YOLO11x+ByteTrack)   (equipos: PRTReId+K-Means; (px → metros 2D)
                                        dorsal ViT-OCR pendiente)
                        │
                 core/events.py  (V-JEPA2 + SVC, eventos: gol/penalti/falta/...)
                        │
                        ▼
              analytics/engine.py ──► telemetry.json  ◄── fuente única de verdad
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              ▼                                                       ▼
   orquestrator/agent.py                                automation/molmo_publisher.py
   (VideoARM/Symphony: Q&A                              (MolmoWeb-4B + Playwright:
    en lenguaje natural sobre                            clips y métricas a RRSS)
    JSON + .mp4)
```

## Estructura del proyecto

```
Sports-Classification/
├── config.py                  # Configuraciones globales (paths, hiperparámetros, API keys)
├── main.py                    # ⏳ Orquestador del pipeline principal (batch processing) — vacío
├── core/
│   ├── tracking.py            # ✅ YOLO11x + SoccerMaster + ByteTrack (genera sv.Detections con IDs)
│   ├── possession.py          # ✅ Poseedor del balón por frame (jugador más cercano al balón)
│   ├── homography.py          # ⏳ Calibración del campo, matriz H, interpolación 2D
│   ├── classification.py      # ✅ Asignación de equipo (PRTReId + K-Means, 2 clusters); dorsal (ViT OCR) pendiente
│   ├── events.py              # ✅ Inferencia de eventos (V-JEPA2 + SVC) sobre un clip .mp4
│   └── ball_actions.py        # ✅ Inferencia de pase/regate/entrada (R3D-18) sobre ventana T=32
├── scripts/
│   ├── download_ball_action_data.py    # ✅ Descarga SoccerNet (Ball Action Spotting + labels públicas)
│   └── extract_ball_action_clips.py    # ✅ tracking+posesión -> clips T=32 etiquetados
├── training/
│   └── train_ball_actions.py  # ✅ Fine-tuning R3D-18 sobre los clips extraídos
├── analytics/
│   └── engine.py              # ⏳ Velocidades, posesión, heatmaps, JSON builder
├── orquestrator/
│   └── agent.py               # ⏳ VideoARM / Symphony (Q&A sobre el .mp4)
├── automation/
│   └── molmo_publisher.py     # ⏳ MolmoWeb-4B + Playwright (publicación RRSS)
├── utils/
│   ├── visualizers.py         # ✅ TrackingVisualizer (cajas/IDs/estelas o por equipo, roboflow/supervision)
│   └── legacy_inference_demo.py  # Demo original YOLO+supervision (referencia)
├── svc_classifier_more_penalties.joblib  # Clasificador SVC entrenado (16 clases, ver abajo)
├── Dockerfile
└── requirements.txt
```

✅ implementado · ⏳ pendiente (desarrollo iterativo, módulo a módulo)

## Instalación (Docker)

El proyecto se ejecuta en contenedor (imagen CUDA 12.8 + cuDNN sobre Ubuntu 24.04); no se usan
entornos virtuales en el host.

```bash
docker build -t sports-analytics .

docker run -d --gpus all \
  --name sports-analytics \
  -v "$(pwd)":/app -w /app \
  sports-analytics

docker exec -it sports-analytics bash
```

docker run -d --name Sports-Classification --gpus all -it --rm -v $(pwd):/app sports-analytics

Dentro del contenedor, `requirements.txt` ya está instalado (se hace en el build). Para
Playwright (solo necesario para `automation/`):

```bash
docker exec -it sports-analytics playwright install --with-deps chromium
```

Los pesos `yolo11x.pt` se descargan automáticamente en la primera ejecución (Ultralytics). El
modelo `facebook/vjepa2-vitl-fpc64-256` (usado por `core/events.py`) se descarga de Hugging Face
en la primera ejecución (~1.2 GB) y requiere `transformers` reciente (V-JEPA2 se integró en la
librería en 2025; si `from transformers import VJEPA2Model` falla, `pip install -U transformers`
dentro del contenedor). Con `TRANSFORMERS_CACHE=/app/models` (ya seteado en el Dockerfile) las
descargas persisten en el volumen montado entre reinicios del contenedor.

## Uso — módulo de tracking (standalone)

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
- El balón de SoccerMaster casi nunca se detecta (dataset GSR centrado en personas); de ahí el `HybridDetector` por defecto, que combina personas de SoccerMaster con balón de YOLO11x.

## Uso — clasificación de equipos (`core/classification.py`, PRTReId + K-Means)

Asigna cada `tracker_id` de un vídeo ya trackeado a uno de 2 equipos, por apariencia. Puerto del
algoritmo real de `Soccer_Master/sn-gamestate/sn_gamestate/team/tracklet_team_clustering_api.py`
(`TrackletTeamClustering`) fuera de tracklab/Hydra/pandas — no es una reimplementación desde cero.

```bash
python -m core.tracking --source partido.mp4 --output tracks.jsonl --detector hybrid --device cuda

python -m core.classification --source partido.mp4 --tracks tracks.jsonl --output teams.json --device cuda
# --embedder stub corre el mismo pipeline sin GPU ni prtreid instalado (color medio del
# crop en vez de un embedding aprendido) — útil para probar el resto del pipeline en seco.
```

Algoritmo (idéntico al original, verificado línea a línea contra su código fuente):

1. Recorta cada detección con `role=="player"` de `tracks.jsonl` (goalkeeper/referee/ball quedan
   fuera del clustering, igual que en `TrackletTeamClustering`).
2. Extrae un embedding de apariencia de 256-d por crop vía **PRTReId** (BPBReID + backbone
   HRNet32, entrenado en SoccerNet) — o, con `--embedder stub`, el color medio BGR del crop.
3. Promedia los embeddings de cada `tracker_id` (media, no mediana ni "frame más nítido" —
   así lo hace el original) y separa esas medias con `KMeans(n_clusters=2, random_state=0)`
   (sin normalizar antes, igual que el original).

`teams.json` (un único objeto, no JSONL — es un resultado a nivel de vídeo completo):

```json
{
  "source_video": "partido.mp4", "tracks_jsonl": "tracks.jsonl",
  "embedder": "prtreid", "embedding_dim": 256, "n_clusters": 2, "random_state": 0,
  "teams": [{"tracker_id": 7, "team_cluster": 0}, {"tracker_id": 12, "team_cluster": 1}],
  "excluded": [{"tracker_id": 3, "reason": "never_player"}, {"tracker_id": 45, "reason": "no_valid_crops"}],
  "n_frames_processed": 812, "n_crops_embedded": 5123, "n_crops_skipped_degenerate": 2
}
```

Join trivial para `analytics/engine.py`: `{t["tracker_id"]: t["team_cluster"] for t in teams["teams"]}`.

### Vídeo anotado por equipo (`--annotated-output`)

```bash
python -m core.classification --source partido.mp4 --tracks tracks.jsonl --output teams.json \
       --annotated-output teams.mp4 --device cuda
```

Escribe un `.mp4` con cajas + etiqueta coloreadas por **equipo** (rojo/azul; gris para quien no
tenga equipo asignado — árbitros, porteros, tracks excluidos) en vez de por `tracker_id` como hace
`core/tracking.py --annotated-output`. Es una **segunda pasada** sobre el vídeo (decodifica de
nuevo, barato, sin re-ejecutar el detector ni PRTReId) — necesaria porque el equipo de cada
jugador no se conoce hasta que se ha visto el vídeo entero (K-Means necesita todos los tracklets a
la vez). La reutiliza `utils/visualizers.py::render_annotated_video`, que lee directamente
`tracks.jsonl` reconstruyendo las detecciones — no depende de `core.tracking` en absoluto, así que
también sirve para volver a renderizar sin tener el pipeline de tracking cargado en memoria.

✅ **Verificado visualmente en ejecución real** (17 jul 2026, mismo clip) contra un frame extraído
del vídeo resultante: dos grupos de jugadores bien separados por color, los 4 árbitros del clip
correctamente en gris con su rol visible, y un jugador lejos del resto del grupo agrupado
correctamente con su equipo por apariencia (no por cercanía en el campo — confirma que usa el
embedding real).

✅ **Verificado en ejecución real** (17 jul 2026, `datasets/clips_analisis/corner/corner118.mp4`,
`--device cuda`) de punta a punta con `--embedder prtreid`. `prtreid`/`bpbreid` no están
vendorizados ni se distribuyen por PyPI — se instalan vía git sin versión fijada (ver
`requirements.txt` y § "Problemas conocidos" más abajo, que documenta los cinco incidentes reales
que hubo que resolver para llegar hasta aquí: dos imports rotos parcheados con stubs en
`sys.modules`, un dataset que había que registrar a mano, y un `torch.load` que había que forzar a
`weights_only=False`). Si tras reconstruir la imagen algo vuelve a fallar al cargar el modelo o
extraer embeddings, empieza por ahí — usa `--embedder stub` mientras tanto para no bloquear el
resto del pipeline.

Explícitamente fuera de alcance (ver roadmap): etiquetado izquierda/derecha de equipo (necesita
`core/homography.py`, que a su vez depende de PnLCalib, sin vendorizar/verificar todavía) y OCR de
dorsales (la otra mitad de este módulo en el roadmap).

## Uso — módulo de eventos (`core/events.py`, standalone)

Clasifica un clip .mp4 cualquiera (sin necesidad de etiqueta ni nombre especial):
extrae embeddings de V-JEPA2 (`facebook/vjepa2-vitl-fpc64-256`, 64 frames muestreados
uniformemente por timestamp) y los pasa por un `Pipeline(StandardScaler + SVC)` ya
entrenado (`svc_classifier_more_penalties.joblib`).

```bash
# Un único clip -> resultado por stdout
python -m core.events --source clip.mp4

# Carpeta de clips -> un JSONL con una predicción por línea
python -m core.events --source clips_analisis/goal/ --output events.jsonl
# --classifier / --model-id / --device permiten sobreescribir los defaults
```

Cada predicción (`EventPrediction.to_telemetry()`):

```json
{
  "video_path": "clips_analisis/goal/corner100.mp4",
  "label": "goal",
  "confidence": 0.874,
  "probabilities": {"card": 0.01, "corner": 0.02, "...": "...", "goal": 0.874, "...": "..."},
  "is_background": false
}
```

El clasificador reconoce **16 clases**: 9 de evento (`card`, `corner`, `foul`, `goal`,
`kick-off`, `penalty`, `shot`, `substitution`, `throw-in`) y 7 de "fondo" — planos sin
evento (primeros planos, público, cámara principal...) que evitan que el modelo dispare
eventos falsos durante callejones del partido. La taxonomía es deliberadamente más
gruesa que la de `clips_analisis/`/`dataset_vjepa_backup/` (que sí distinguen
`shots_on_target`/`shots_off_target` y `yellow`/`red_card`): se fusionaron a propósito
en `shot`/`card` porque esa frontera era difusa para el modelo y perjudicaba la
precisión — es una decisión de diseño, no una limitación pendiente.

**Limitación de diseño importante**: `sample_frames` reparte los 64 frames por igual a
lo largo de TODA la duración del vídeo de entrada. El módulo espera un **clip ya
recortado en torno a un único evento** (unos pocos segundos, como los de
`clips_analisis/`), no un partido completo — si le pasas 90 minutos sin trocear, la
señal del evento se diluye entre miles de frames irrelevantes y la predicción no será
útil. Trocear un partido completo en clips (sliding window / detección de escenas) es
tarea de una capa superior (`analytics/engine.py` o `main.py`), pendiente de implementar.

## Uso — acciones de balón: pase/regate/cabeceo/saque (`core/ball_actions.py`, R3D-18)

Complementa a `core/events.py` con un paradigma distinto: en vez de clasificar un clip
ya recortado en torno a un evento discreto (gol/falta/córner...), este pipeline
clasifica ventanas **densas** de `T=32` frames centradas en el jugador que posee el
balón en cada instante — pensado para cubrir `Pass`, `Drive` (regate), `High Pass`
(pase elevado), `Header` (cabeceo), `Throw In` (saque de banda) y una clase
`Background`. `Shot`/`Goal` ya los cubre `core/events.py`, así que no se duplican aquí.

Otras clases de `Labels-ball.json` quedan **fuera de esta primera fase** (fácil de
reintroducir: descomentar su línea en `scripts/extract_ball_action_clips.py::BALL_LABEL_TO_CLASS`
y añadirla a `DEFAULT_CLASSES` en `training/train_ball_actions.py`):
- `Player Successful Tackle`: muy minoritaria (3-18 anotaciones/partido frente a
  500-700+ de pass/drive) y, al revisar los clips extraídos, resultaban confusos.
- `Cross`: subtipo de pase definido por posición en el campo (banda → área), no por
  técnica corporal — mismo riesgo de confusión con `High Pass`/`Pass`.
- `Ball Player Block`: acción defensiva, probablemente confundible con tackle.
- `Free Kick`/`Goal`: demasiado raras (21/13 anotaciones en total en los 7 partidos).
- `Out`/`Shot`: no son técnicas del poseedor sino resultados/eventos de otro tipo
  (y Shot ya lo cubre `core/events.py`).

Backbone: **R3D-18** (`torchvision.models.video.r3d_18`, fine-tuning sobre
Kinetics-400) — se eligió frente a V-JEPA2/ViViT porque estas acciones ocurren miles
de veces por partido (sliding-window denso) y R3D-18 es mucho más barato de correr a
ese volumen.

### Origen de los datos: dos corpus de SoccerNet distintos

**Importante**: `Labels-ball.json` (Ball Action Spotting, la única fuente con estas
etiquetas) **no pertenece a las 6 ligas** ya descargadas en
`/mnt/datasets/SoccerNet` (England EPL, Champions League, Ligue 1, Bundesliga, Serie A,
La Liga, 2014-2017). Es un corpus aparte: **9 partidos de la EFL Championship inglesa,
temporada 2019-2020** (4 train / 1 valid / 2 test con ground truth pública + 2 challenge
sin GT), descargable solo con password NDA de SoccerNet (viene empaquetado junto al
vídeo). Como consecuencia, `Labels-cameras.json` (tipo de plano) tampoco existe para
esos 9 partidos — solo cubre 200 de los 500 partidos del corpus original (las 6 ligas
ya descargadas) — así que el filtrado por cámara no se puede aplicar a los clips
positivos en sí (mitigado porque esas anotaciones ya están hechas sobre jugadas
en vivo por construcción del propio dataset). Para `Background` sí se puede seguir
aprovechando `Labels-cameras.json`, tomando muestras del corpus de 6 ligas además de los
huecos sin anotación dentro de los propios partidos EFL (ver más abajo).

Taxonomía real de `Labels-ball.json` (12 clases, timestamp puntual, no rango; total
medido sobre los 7 partidos descargados entre paréntesis): `Pass` (4985), `Drive`
(4300), `High Pass` (761), `Header` (713), `Out` (551), `Throw In` (362), `Cross`
(261), `Ball Player Block` (223), `Shot` (169), `Player Successful Tackle` (74),
`Free Kick` (21), `Goal` (13). De momento se usan `Pass`, `Drive`, `High Pass`,
`Header` y `Throw In` (ver más arriba para el resto).

### 1. Descarga (`scripts/download_ball_action_data.py`)

```bash
# Público (sin NDA): Labels-v2.json + Labels-cameras.json de las 6 ligas ya descargadas
python -m scripts.download_ball_action_data --tasks labels-v2,cameras

# Ball Action Spotting (requiere NDA, ver https://www.soccer-net.org/data):
export SOCCERNET_NDA_PASSWORD="..."
python -m scripts.download_ball_action_data --tasks ball

# Validar parámetros sin descargar nada:
python -m scripts.download_ball_action_data --tasks all --dry-run
```

### 2. Poseedor del balón (`core/possession.py`)

Utilidad reusable e independiente de este pipeline concreto: dado un `tracks.jsonl` de
`core/tracking.py`, resuelve el `tracker_id` más cercano al balón en cada frame (con
histéresis para no parpadear en los huecos de detección del balón, ya documentados como
frecuentes en `core/tracking.py`).

```bash
python -m core.possession --tracks tracks.jsonl --output possession.jsonl
```

### 3. Extracción de clips (`scripts/extract_ball_action_clips.py`)

Por partido: corre `core.tracking` + `core.possession` sobre el vídeo (cacheando
`*.tracks.jsonl`/`*.possession.jsonl` junto al vídeo), localiza el bbox del poseedor en
el frame de cada anotación de `Labels-ball.json`, y recorta una ventana de 32 frames con
un único bbox estático (expandido `padding_factor`× y recortado a los límites del
frame) — evita el jitter de recortar frame a frame sobre detecciones ruidosas.

Los partidos de Ball Action Spotting llegan en `.zip` sin descomprimir (uno por split:
`train.zip`/`valid.zip`/`test.zip`) y, a diferencia del corpus de 6 ligas, cada partido es
**un único vídeo** (`720p.mp4`, sin dividir en mitades) — hay que descomprimirlos antes de
extraer clips:

```bash
cd datasets/SoccerNet && for z in spotting-ball-2024/*.zip; do unzip -o "$z" -d .; done
```

```bash
# --match-dir depende de dónde hayas descargado con --local-dir (ver sección 1)
python -m scripts.extract_ball_action_clips --match-dir datasets/SoccerNet/england_efl/2019-2020/<partido> \
       --output dataset_ball_actions/ --background-per-half 40
```

### 4. Entrenamiento (`training/train_ball_actions.py`)

Fine-tuning de R3D-18 sobre `dataset_ball_actions/<clase>/*.mp4`, con split train/val
**por partido** (no por clip, para no filtrar información entre clips de la misma
jugada) y balanceo de clases configurable (`WeightedRandomSampler` por defecto, o
`CrossEntropyLoss(weight=...)` con `--balance loss`).

**Nota de dependencias**: la lectura de los clips (aquí y en `core/ball_actions.py`)
usa `av` (`core.ball_actions.read_clip_frames`), no `torchvision.io.read_video` —
`torchvision` 0.26 (la versión pineada en `requirements.txt`) eliminó por completo
`read_video`/`VideoReader` de `torchvision.io` (solo quedan funciones de imagen). Si
alguna vez ves `ImportError: cannot import name 'read_video' from 'torchvision.io'`,
es este mismo problema — no reintroducir ese import.

```bash
python -m training.train_ball_actions --data-dir dataset_ball_actions/ \
       --epochs 15 --checkpoint models/r3d18_ball_actions.pt
```

### 5. Inferencia (`core/ball_actions.py`)

Mismo estilo que `core/events.py`: clasifica un clip ya recortado, o vía
`BallActionClassifier.predict_window(frames, possessor_bbox)` una ventana en vivo con el
mismo recorte que la extracción offline (`compute_crop_box`, compartido entre ambos para
no divergir entre entrenamiento e inferencia).

```bash
python -m core.ball_actions --source dataset_ball_actions/pass/algun_clip.mp4
```

## Roadmap de desarrollo

| Orden | Módulo | Descripción |
|-------|--------|-------------|
| 1 | `core/tracking.py` | ✅ Detección + tracking |
| 2 | `core/classification.py` | ✅ Asignación de equipo (PRTReId + K-Means); dorsal (ViT OCR) queda pendiente |
| 3 | `core/homography.py` | Keypoints del campo → matriz H (ver `Soccer_Master/sn_calibration` como referencia) |
| 4 | `analytics/engine.py` | Métricas + `telemetry.json` |
| 5 | `core/events.py` | ✅ Clasificación de eventos (V-JEPA2 + SVC) sobre clips ya recortados; falta el troceado de partido completo → clips |
| 6 | `utils/visualizers.py` | ✅ Vídeo anotado (`--annotated-output` en `core/tracking.py`); falta `main.py` (orquestación batch multi-módulo) |
| 7 | `core/possession.py` / `core/ball_actions.py` | ✅ Poseedor del balón + clasificación Pass/Drive/High Pass/Header/Throw In/Background (R3D-18; Tackle/Cross/Ball Player Block diferidas, ver arriba); datos descargados y extraídos, falta entrenar |
| 8 | `orquestrator/agent.py` | Q&A semántico |
| 9 | `automation/molmo_publisher.py` | Publicación RRSS |

## Problemas conocidos: `Soccer_Master` y dependencias sin pin

Sección viva — si vuelves a esto en otra sesión (de Claude Code o no), léela **antes**
de tocar `Soccer_Master/`. Documenta una ronda real de depuración (17 jul 2026) que costó
varias iteraciones porque los síntomas parecían no estar relacionados entre sí.

### Causa raíz común

`requirements.txt` solo fijaba mínimos (`torch>=2.1.0`, `transformers>=4.44.0`), nunca
versiones exactas. Cada `docker build` nuevo resuelve `pip install` contra lo que esté
disponible **ese día**, así que el entorno no es reproducible entre reconstrucciones: algo
que funcionaba hace semanas puede romperse hoy sin que nadie haya tocado una línea de este
repo. Vía web confirmé que hasta el propio "Quick Start" de `Soccer_Master/README.md`
(el repo de terceros vendorizado en `Soccer_Master/SoccerMaster/`) tiene el mismo problema:
instala `transformers` con `pip install git+https://github.com/huggingface/transformers`,
sin fijar commit. O sea, ni siquiera seguir esas instrucciones al pie de la letra habría
dado un entorno reproducible.

**Importante — ese Quick Start (conda, Python 3.10.16, `torch==2.4.1+cu121`,
`pip install -e .` de `sn-gamestate`/`tracklab`/`sam2`, `flash-attn`, `qwen-vl-utils`) es
para el pipeline GSR completo** (tracklab + SAM2 + Qwen2.5-VL: calibración de cámara, ReID,
OCR de dorsales vía VLM). **No es lo que usa este proyecto.** El propio
`Soccer_Master/SoccerMaster/infer_mp4.py` (script que ya funcionaba antes en este repo)
documenta en su docstring que ese pipeline completo está roto/incompleto en este checkout
(faltan `sn_gamestate/configs/dataset/soccernet_gs.yaml` y `tracklab/wrappers/datasets/`), y
por eso existe como alternativa ligera que solo usa `models/multi_task.py` + SigLIP2 + el
head DETR — exactamente lo que envuelve `core/tracking.py:SoccerMasterDetector` y
`core/events.py`. **No instales el stack conda/tracklab/SAM2/Qwen para arreglar esto** —
es una pista falsa, resuelve un problema distinto.

### Incidentes concretos (17 jul 2026) y su fix

| # | Síntoma | Fichero | Causa | Fix |
|---|---------|---------|-------|-----|
| 1 | `RuntimeError: The NVIDIA driver on your system is too old (found version 12080)` al primer `.to("cuda")` | — (dependencia) | `pip install torch` sin pin resolvió a torch 2.13.0, cuyo runtime CUDA por defecto es **13.0** (`requires_dist: cuda-toolkit==13.0.3`); el driver del host (570.211.01) solo soporta hasta CUDA 12.8 | Pin en `requirements.txt`: `torch==2.11.0+cu128` + `torchvision==0.26.0+cu128` vía `--extra-index-url https://download.pytorch.org/whl/cu128` |
| 2 | `AttributeError: 'SiglipVisionModel' object has no attribute 'vision_model'` | `Soccer_Master/SoccerMaster/models/soccer_master.py`, `VisionBackbone.__init__` | En `transformers` viejo, `SiglipVisionModel` anidaba el transformer en `.vision_model`. En `transformers` 5.x, `SiglipVisionModel` **es** el transformer directamente | `siglip_vision_model = getattr(model, "vision_model", model)` (compatible con ambas APIs) |
| 3 | `IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)` dentro de `SiglipAttention.forward` (`q_proj(...).view(...).transpose(1,2)`) | mismo fichero, `ResidualAttentionBlock.forward` | En `transformers` viejo, llamar a una capa del encoder devolvía `(hidden_states, attn_weights)`; el código hacía `[0]` para desempaquetar. En `transformers` 5.x, `SiglipEncoderLayer.forward` devuelve el tensor **directamente** — el `[0]` heredado ya no desempaqueta una tupla, **recorta la dimensión de batch del tensor**, y la forma va quedando corrompida capa a capa hasta que revienta unas capas más adelante (por eso el traceback apunta a un sitio que parece no tener relación con el cambio real) | Helper `_encoder_layer_hidden_states()` que solo indexa `[0]` si el resultado es efectivamente una tupla |

Ambos fixes de la tabla están **verificados línea a línea contra el código fuente real
instalado** de `transformers` (`site-packages/transformers/models/siglip/modeling_siglip.py`),
no son parches a ciegas.

### Extensión CUDA `MultiScaleDeformableAttention` (Deformable DETR) — ya blindada, no tocar

`Soccer_Master/SoccerMaster/models/deformable_detr/ops/` contiene una extensión CUDA que se
compila aparte (`ops/setup.py` / `ops/make.sh`), **no** vía `pip install -r requirements.txt`.
Ya existe:
- Un `.so` compilado (13 jul, para cpython-312): `ops/build/lib.linux-x86_64-cpython-312/MultiScaleDeformableAttention.cpython-312-x86_64-linux-gnu.so`.
- Un fallback defensivo a PyTorch puro (16 jul, ver `ops/functions/ms_deform_attn_func.py` y
  `ops/modules/ms_deform_attn.py`; los `.orig` junto a cada uno son las versiones previas al
  parche): si la extensión no importa o el tensor no está en CUDA, usa
  `ms_deform_attn_core_pytorch` en vez de fallar.

Si veis `ImportError: MultiScaleDeformableAttention` o un error de símbolos indefinidos tras
cambiar de versión de `torch` (el `.so` está enlazado contra el ABI de una versión concreta),
**no es grave**: cae solo al fallback (más lento en esa operación puntual, pero corre). Si
hace falta rendimiento y tocara recompilar:
```bash
cd Soccer_Master/SoccerMaster/models/deformable_detr/ops
python setup.py build install   # necesita nvcc; la imagen Docker (-devel) ya lo trae
```

### Inventario de cambios sobre el `Soccer_Master` original (para git / `.gitignore`)

`Soccer_Master/` es un repo de terceros vendorizado (copia local de
[haolinyang-hlyang/SoccerMaster](https://github.com/haolinyang-hlyang/SoccerMaster/), que a su
vez trae dentro copias de `sn-gamestate`, `tracklab` y `sam2`). Va ignorado en git salvo estos
ficheros — el `.gitignore` del proyecto ya tiene el bloque `Soccer_Master/*` + excepciones en
cascada para que solo estos suban:

| Fichero | Tipo de cambio | Qué es |
|---|---|---|
| `Soccer_Master/SoccerMaster/models/soccer_master.py` | **Modificado** (2 parches, esta sesión) | `VisionBackbone.__init__`: `model.vision_model` → `getattr(model, "vision_model", model)`. `ResidualAttentionBlock.forward`: `self.encoder(...)[0]` → helper `_encoder_layer_hidden_states(...)`. Ver tabla de incidentes #2 y #3 más arriba. |
| `Soccer_Master/SoccerMaster/models/deformable_detr/ops/functions/ms_deform_attn_func.py` | **Modificado** (sesión previa, 16 jul) | Import de `MultiScaleDeformableAttention` envuelto en try/except (`MSDA = None` si no compila). |
| `Soccer_Master/SoccerMaster/models/deformable_detr/ops/modules/ms_deform_attn.py` | **Modificado** (sesión previa, 16 jul) | Fallback a `ms_deform_attn_core_pytorch` si `MSDA is None` o el tensor no está en CUDA. |
| `Soccer_Master/SoccerMaster/infer_mp4.py` | **Añadido** (sesión previa, 13 jul) | Script de inferencia standalone sobre un `.mp4`, alternativa al pipeline tracklab/SAM2/Qwen (roto en este checkout, ver docstring del propio fichero). No es upstream. |
| `Soccer_Master/SoccerMaster/eval_captions.py` | **Añadido** (sesión previa, 13 jul) | Evalúa el head `VideoCaption` (retrieval SigLIP-style) contra `clips_analisis/`. No es upstream. |
| `Soccer_Master/SoccerMaster/eval_classification_head.py` | **Añadido** (sesión previa, 13 jul) | Evalúa el head `CaptionClassification` (softmax, 23 clases fijas) contra `clips_analisis/`. No es upstream. |

Todo lo demás dentro de `Soccer_Master/` (incluyendo `sam2/`, `sn-gamestate/`, `tracklab/`,
checkpoints en `pretrained_models/`, y los artefactos compilados `ops/build/`, `ops/dist/`,
`*.egg-info/`) es upstream sin tocar o binario regenerable/descargable — no va a git. Los
`.orig` junto a los dos ficheros parcheados de `deformable_detr/ops/` (versiones previas al
parche) tampoco van a git; una vez esto esté en un commit, el propio `git diff`/`git blame`
cumple esa función y los `.orig` se pueden borrar si molestan.

Si en el futuro se parchea algo más dentro de `Soccer_Master/`, hay que añadir tanto la fila a
esta tabla como la línea `!ruta/al/fichero` correspondiente en `.gitignore` (respetando la
cascada: cada carpeta intermedia necesita su propia línea `!carpeta/` + `carpeta/*` antes de
poder excepcionar un fichero nieto — si no, la excepción no hace nada porque git no baja a mirar
dentro de una carpeta ya ignorada).

### Versiones fijadas actualmente (RTX 4090, driver 570.211.01, dentro del contenedor)

- `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128` — confirmado: soluciona el incidente #1 (se probó de verdad, el contenedor llegó más allá del punto donde crasheaba).
- `transformers==5.14.1` — versión contra la que se diagnosticaron y parchearon los incidentes #2 y #3. Ambos **confirmados solucionados en ejecución real** (`python -m core.tracking` completo, `--detector hybrid --device cuda`, sin errores).

### Dependencias solo-git sin pin: `prtreid`/`bpbreid` (`core/classification.py`)

Segunda instancia del mismo patrón de riesgo que `transformers`/`torch` sin pin (arriba): PRTReId
(el extractor de embeddings de apariencia que usa `core/classification.py` para asignar equipos)
depende de `prtreid`/`bpbreid`, dos paquetes que **no se distribuyen por PyPI** — la única
instalación documentada por sus propios autores es `pip install git+https://github.com/VlSomers/prtreid`
(sin commit fijado). Un `docker build` en un día distinto puede traer una versión distinta sin
ningún aviso. Si `core/classification.py` con `--embedder prtreid` falla al cargar el modelo o al
extraer embeddings tras reconstruir la imagen, sospecha esto primero — usa `--embedder stub`
mientras tanto para no bloquearte, y compara contra el `cfg`/versión que funcionó la última vez.

**Incidente concreto ya visto (17 jul 2026)**: `from prtreid.scripts.main import build_config`
importa en cascada `prtreid.utils.visualization`, que llama a `matplotlib.cm.get_cmap('hsv')` —
eliminada en `matplotlib` 3.9 (existía, deprecada, hasta 3.8.x). Como `prtreid`/`torchreid` no
pinean `matplotlib` en sus propios requisitos, `pip install` resolvía la última (3.11.1) y
`core/classification.py` reventaba con `AttributeError: module 'matplotlib.cm' has no attribute
'get_cmap'` **al importar `prtreid`, no al usar nada de plotting nuestro** — mismo patrón exacto
que los incidentes de `transformers`/Siglip: una dependencia transitiva sin pin, no algo que
hiciéramos mal. Fix: `matplotlib<3.9` fijado en `requirements.txt`.

Justo detrás, mismo `import prtreid` en cascada, apareció un segundo incidente de naturaleza
distinta: `prtreid.data.data_augmentation.random_occlusion` importa `skimage` a nivel de módulo
(aumento de datos de entrenamiento, que no usamos en inferencia) pero `scikit-image` no está en
el empaquetado de `prtreid` — no es un conflicto de versión, es una dependencia real que falta
del todo (`ModuleNotFoundError: No module named 'skimage'`). Fix: `scikit-image` añadido a
`requirements.txt`.

Un tercer incidente, más profundo, en el mismo fichero: tras resolver `skimage`, ese mismo
`random_occlusion.py` hace `from albumentations import (DualTransform, functional)`, y
`functional` como submódulo importable directamente desde la raíz de `albumentations` es una API
de versiones MUY anteriores a las reorganizaciones internas por tipo de efecto (blur/crops/
geometric/...) que tiene el paquete desde hace varias versiones mayores — **ninguna versión
razonablemente reciente de `albumentations` la expone así**, y bajar a una versión lo bastante
vieja para que exista arriesga romper otra cosa en la cadena de dependencias (numpy 2.x, compat
con Python 3.12, etc.). Confirmado en ejecución real (17 jul 2026):
`ImportError: cannot import name 'functional' from 'albumentations'`.

Esta vez el fix NO es un pin en `requirements.txt` — es código en
`core/classification.py::_stub_prtreid_random_occlusion()`: inyecta un módulo vacío (con un
`RandomOcclusion` dummy, porque otros módulos de `prtreid` sí necesitan que ese nombre exista) en
`sys.modules` **antes** de que `import prtreid` intente cargar el fichero real, así que el import
roto ni se ejecuta. `random_occlusion.py` es aumento de datos para ENTRENAMIENTO — no lo usamos,
solo llamamos a `FeatureExtractor` para inferencia — así que perdernos ese fichero no quita
funcionalidad real. La técnica (pre-poblar `sys.modules` con un stub antes del import que lo
dispara) está verificada de forma aislada, sin `prtreid` instalado, antes de aplicarla. Vive en
nuestro propio código (no en `site-packages`), así que sobrevive a cualquier reconstrucción de la
imagen — a diferencia de parchear el paquete instalado directamente.

Un cuarto incidente, ya no de import sino de **configuración**: una vez `import prtreid` funciona,
`prtreid.scripts.main.build_config(config=yacs_cfg)` llama internamente a
`compute_parts_num_and_names(cfg)`, que necesita ALGÚN dataset registrado bajo el nombre que
digan `cfg.data.sources`/`targets` (`"SoccerNet"` en nuestro `cfg`, copiado del `prtreid.yaml` de
`sn-gamestate`). El propio `prtreid` **no** registra ningún dataset por defecto — solo
`sn-gamestate` lo hace, con su propia `sn_gamestate.reid.prtreid_dataset.ReidDataset` (fuertemente
acoplada a `tracklab`: necesita un `TrackingDataset` completo con splits de train/query/gallery
de ground truth, inviable de replicar solo para inferencia). Confirmado en ejecución real
(17 jul 2026): `ValueError: Invalid dataset name. Received "SoccerNet", but expected to be one of
[...]`. Fix, en `core/classification.py::_register_prtreid_soccernet_dataset()`: `prtreid` SÍ trae
su propia clase `SoccerNet` (`prtreid.data.datasets.image.soccernet.SoccerNet`), importada pero no
registrada por defecto — la registramos nosotros bajo el nombre `"SoccerNet"`. No hace falta que
sea idéntica a la `ReidDataset` de `sn-gamestate`: con `masks.dir="pose_on_img_crops"` (nuestro
`cfg`), tanto una como la otra devuelven `None` en `get_masks_config` (ninguna reconoce esa
clave), así que el resultado final es el mismo da igual cuál se registre. Solo se registra la
clase (nunca se instancia), así que no dispara ningún escaneo de disco de un dataset que no
tenemos.

Un quinto incidente, ya sin relación con imports: una vez `build_config` puede resolver el
dataset, intenta leer el propio checkpoint (`load_checkpoint(cfg.model.load_weights)`, dentro de
`prtreid.utils.torchtools`) llamando a `torch.load(fpath, map_location=...)` sin especificar
`weights_only`. Desde PyTorch 2.6 el valor por defecto de ese argumento cambió de `False` a
`True`, y el checkpoint publicado de PRTReId (un pickle con objetos `numpy.core.multiarray.scalar`
de una versión de numpy antigua) no pasa esa comprobación de seguridad por defecto. Confirmado en
ejecución real (17 jul 2026): `_pickle.UnpicklingError: Weights only load failed ... Unsupported
global: GLOBAL numpy.core.multiarray.scalar`. Fix, en
`core/classification.py::_patch_torch_load_weights_only_false()`: parchea `torch.load` para que
use `weights_only=False` por defecto, con ámbito acotado al proceso (no toca `site-packages` ni
desactiva la comprobación de seguridad para el resto del proyecto) — aceptable porque el
checkpoint viene de una URL de Zenodo fija que ya verificamos por MD5 antes de usar, no es un
pickle arbitrario de origen desconocido.

✅ **Con estos cinco fixes, el pipeline completo de `core/classification.py --embedder prtreid`
corre de punta a punta** (verificado en ejecución real, 17 jul 2026, sobre
`datasets/clips_analisis/corner/corner118.mp4`: 23 tracklets → 11 vs 12 por equipo, 5 excluidos
como `never_player`, 778 crops embebidos, 0 degenerados). De paso se confirmó empíricamente algo
que en el plan original quedaba marcado como suposición sin verificar: PRTReId normaliza los
crops con estadísticas estándar de ImageNet (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
0.225]`) tras un resize a 256×128 — se ve en el log de `FeatureExtractor` al construirse.

Nota menor, no un bug: `build_config` crea al final un directorio numérico vacío en la raíz del
proyecto (`os.makedirs(cfg.data.save_dir)`, con `cfg.project.job_id` sin fijar explícitamente por
nosotros) — es un directorio de logs de `prtreid` que no usamos para nada; se puede borrar sin
miedo (`rmdir <número>`) y no debería ir a git.

**Patrón a esperar**: `import prtreid` arrastra un árbol de imports grande (data augmentation,
visualización, engines de entrenamiento...) que no necesitamos para inferencia pero que igualmente
tiene que resolver en el momento de importar. Es plausible que aparezcan más
`ModuleNotFoundError`/`ImportError`/`AttributeError` de este estilo la primera vez que se
reconstruya la imagen desde cero — no asumas que el próximo será alguno de los tres de arriba
otra vez; sigue el traceback hasta el import que falla y decide caso por caso si el fix es un pin
en `requirements.txt` (si es una API que existe en OTRA versión del paquete) o un stub en
`sys.modules` como este (si es un fichero de `prtreid` que no necesitamos para nada y cuya API
esperada ya no existe en ninguna versión razonable).

Tercera instancia del mismo patrón, ya autodocumentada en su propio fichero: `core/homography.py`
depende de una copia externa de [PnLCalib](https://github.com/mguti97/PnLCalib) que tampoco está
vendorizada ni verificada end-to-end en este repo (ver el aviso al principio de ese fichero).

Puntero cruzado para quien llegue a depurar cualquiera de las tres: `torch`/`transformers`
(pineados en `requirements.txt`, incidentes documentados arriba), `prtreid`/`bpbreid` (sin pin,
esta sección), `PnLCalib` (sin vendorizar, docstring de `core/homography.py`).

### Si algo se rompe otra vez tras un `docker build`

1. Comprobar que `requirements.txt` sigue con las versiones pineadas de arriba — si alguien
   las soltó "para probar algo", ese suele ser el problema.
2. Si aparece un `AttributeError`/`IndexError` nuevo **dentro de `Soccer_Master/SoccerMaster/models/`**,
   sospechar el mismo patrón: `transformers` cambió otra API interna de Siglip/DETR entre la
   versión pineada y la recién instalada. Comparar el traceback contra el código fuente
   REAL instalado (`site-packages/transformers/models/siglip/modeling_siglip.py` o el módulo
   que corresponda) en vez de asumir que el código vendorizado de Soccer_Master tiene razón
   — casi siempre es el vendored code el que quedó desactualizado, no `transformers`.
3. No instalar el stack conda/Python 3.10/tracklab/SAM2/Qwen del Quick Start oficial — es
   para el pipeline GSR completo, que este proyecto no usa (ver más arriba).
4. Los parches viven directamente en los ficheros vendorizados (no hay un `.patch`/diff
   aparte), y esos ficheros son propiedad de `root` en el host aunque el contenedor corra
   como root — si una sesión de Claude Code sin acceso root al host necesita tocarlos, tiene
   que dárselos un comando para que lo ejecutéis vosotros dentro del contenedor (como se hizo
   aquí), no puede escribirlos directamente desde fuera.

## Referencias

- [roboflow/supervision](https://github.com/roboflow/supervision) — ByteTrack, anotadores
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [SoccerMaster](https://github.com/haolinyang-hlyang/SoccerMaster/) — calibración de campo y multi-task (copia local en `../world_cup/Soccer_Master/`)
- [V-JEPA2 en transformers](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) — embeddings de vídeo para `core/events.py`
- SoccerNet — taxonomía de eventos para `core/events.py`
