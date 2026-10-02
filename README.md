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
                    │                   dorsal ViT-OCR pendiente)
                    ▼
             core/possession.py ──► core/ball_actions.py   core/events.py
             (poseedor por frame)   (R3D-18, por cambio     (V-JEPA2+SVC, evento
                    │                de poseedor)            del clip entero)
                    └───────────────────────┬───────────────────────┘
                                             ▼
                    analytics/engine.py ──► vídeo final anotado + telemetry.json
                                          (fuente única de verdad — hoy: clips
                                           cortos ya recortados, no partido completo)
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
│   ├── homography.py          # ✅ Calibración del campo (PnLCalib) → matriz H → tracks_2d.jsonl + radar
│   ├── classification.py      # ✅ Asignación de equipo (PRTReId + K-Means, 2 clusters); dorsal (ViT OCR) pendiente
│   ├── events.py              # ✅ Inferencia de eventos (V-JEPA2 + SVC) sobre un clip .mp4
│   ├── spotting.py            # ✅ Probabilidades por ventana -> eventos puntuales (picos + NMS por clase)
│   └── ball_actions.py        # ✅ Inferencia de pase/regate/entrada (R3D-18) sobre ventana T=32
├── scripts/
│   ├── download_ball_action_data.py    # ✅ Descarga SoccerNet (Ball Action Spotting + labels públicas)
│   ├── extract_ball_action_clips.py    # ✅ tracking+posesión -> clips T=32 etiquetados
│   ├── scan_event_windows.py           # ✅ Escaneo de partido completo -> caché de probabilidades por ventana
│   ├── evaluate_spotting.py            # ✅ Eventos puntuales vs GT SoccerNet (±tol, 1 a 1, barrido de umbral)
│   └── verify_shot_candidates.py       # ✅ Candidatos a tiro de V-JEPA2 -> verificador VLM (Qwen3.5-9B)
├── training/
│   └── train_ball_actions.py  # ✅ Fine-tuning R3D-18 sobre los clips extraídos
├── analytics/
│   └── engine.py              # ✅ Composición final: equipo+posesión+ball_actions+events -> vídeo + telemetry.json
├── orquestrator/
│   └── agent.py               # ⏳ VideoARM / Symphony (Q&A sobre el .mp4)
├── automation/
│   └── molmo_publisher.py     # ✅ MolmoWeb-4B + Playwright (subida a YouTube Studio guiada
│                              #    por coordenadas del VLM; `main.py --publish`)
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

Explícitamente fuera de alcance (ver roadmap): etiquetado izquierda/derecha de equipo (ya es
posible — `core/homography.py` da la posición en metros, solo falta implementarlo) y OCR de
dorsales (la otra mitad de este módulo en el roadmap).

## Uso — homografía / calibración de campo (`core/homography.py`)

Convierte el tracking de píxeles a **metros sobre un campo canónico de 105x68**, que es lo que
hace comparables las posiciones entre frames, entre cámaras y entre partidos.

### Dependencias (no versionadas, hay que traerlas a mano)

```bash
# 1) Código de PnLCalib (solo el código; no lo tocamos, por eso va entero al .gitignore)
git clone https://github.com/mguti97/PnLCalib.git models/homography/PnLCalib

# 2) Pesos (releases v1.0.0 del mismo repo) -> models/homography/weights/
mkdir -p models/homography/weights && cd models/homography/weights
curl -L -o SV_kp    https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp
curl -L -o SV_lines https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines
```

### Ejecución

```bash
# Integrado en el pipeline (etapa entre tracking y analytics)
python -m main --source partido.mp4 --output-dir output/partido --mode full-match --radar

# Standalone (requiere tracks.jsonl previo)
python -m core.homography --source partido.mp4 --output tracks_2d.jsonl --radar-video radar.mp4
```

Salidas: `<video>.homography.jsonl` (caché de matrices H, junto al vídeo como tracks/possession),
`tracks_2d.jsonl` (tracking + `field_xy` en metros) y opcionalmente `radar.mp4` (vista cenital).

### Sistema de coordenadas — ojo con esto

PnLCalib trabaja **centrado** (X ∈ [-52.5, 52.5], Y ∈ [-34, 34]): su `keypoint_world_coords_2D`
aplica `[x - 52.5, y - 34]`. Este módulo NO propaga esa convención — traslada a **origen en la
esquina** (X ∈ [0, 105], Y ∈ [0, 68]) antes de devolver nada. Todo lo que consuma `field_xy` debe
asumir esquina. Si algún día los puntos salen todos en el cuadrante negativo, este es el motivo.

### Rendimiento (medido en la RTX 4090 del contenedor)

Es la etapa más cara del pipeline y es GPU-bound: son dos HRNet-W48 a 960x540 por frame. El solve
de la homografía en CPU (RANSAC) es 0.5 ms/frame, despreciable.

| | batch=1 | batch=4 | batch=8 |
|---|---|---|---|
| fp32 | 58 ms/frame | 41 ms/frame (6.1 GiB) | 42 ms/frame (11.6 GiB) |
| **fp16** | 43 ms/frame | **26 ms/frame (4.7 GiB)** | 24 ms/frame (8.7 GiB) |

Defaults: `half=True`, `batch_size=4`, `stride=5`. El parámetro que manda es `stride`: a 25 fps un
partido de 90 min son 135k frames, o sea ~1 h a stride=1 y ~12 min a stride=5. Una calibración
cada 0.2 s sobra para el paneo suave de una cámara de retransmisión.

### Qué esperar de la calidad (medido sobre `ground_truth/`, 20 min de partido real)

- **74% de los frames** acaban con homografía (el resto: repeticiones, planos cortos, pocas líneas
  visibles). Se marcan con `homography_ok: false` en vez de arrastrar una H de otro plano.
- **95% de los jugadores proyectados caen dentro del campo**, 99.3% con 5 m de margen.
- Error de reproyección mediano ~1.9 px.

⚠️  **La velocidad instantánea NO es usable todavía**: mediana 5.1 m/s, con 21% de muestras por
encima de 12 m/s (físicamente imposible). No es culpa de la calibración — se comprobó que los
pares de frames que comparten la MISMA H tienen peor velocidad (5.51 m/s) que los que cruzan H
distintas (4.20), así que el jitter no viene de la homografía. Viene de arriba: el `anchor_xy` del
tracker se mueve 3.7 px/frame de mediana y la perspectiva amplifica eso a metros en el fondo del
campo. **Antes de publicar cualquier métrica física hace falta suavizado temporal en el espacio de
campo** (Kalman/savgol sobre `field_xy` por `tracker_id`), que es el siguiente paso natural.

### Proyecciones descartadas

`project_tracks` no escribe `field_xy` cuando el punto no es proyectable (3.4% de los casos):
`w < 0` (el punto cae detrás del plano de cámara y la división proyectiva devuelve un punto
reflejado de aspecto plausible) o `w ≈ 0` (sobre la línea del horizonte, la homografía diverge y
proyecta a cientos de km). Se verificó que estos descartes **no tienen sesgo de profundidad**
(mediana de `y_img` 356.8 en descartados vs 358.5 en aceptados), así que no se está perdiendo
sistemáticamente al equipo del fondo: son detecciones fuera del campo (banquillo, público).

El balón se proyecta asumiendo z=0 y se marca con `field_xy_assumes_ground: true` — cuando está
en el aire la proyección es incorrecta por construcción.

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

## Uso — eventos vía orquestador (`core/events_vllm.py`, VideoARM + Qwen3.5-9B/vLLM)

Backend **alternativo** a `core/events.py`: en vez de embeddings de V-JEPA2 + un SVC
entrenado, un orquestador agéntico ([VideoARM](https://github.com/MILVLG/videoarm))
razona sobre el clip llamando a sus propias herramientas (`scene_snapper`,
`clip_analyzer`) contra un VLM local — **Qwen3.5-9B servido con vLLM**. No sustituye al
otro: los dos conviven y se eligen con `--events-backend`.

Los dos cumplen el **mismo contrato** (`predict` / `predict_window` → `EventPrediction`),
así que `analytics/engine.py` los consume de forma duck-typed y no distingue cuál recibe.
Diferencias que sí se notan:

| | `vjepa2` (`core/events.py`) | `videoarm` (`core/events_vllm.py`) |
|---|---|---|
| Modelo | V-JEPA2 + SVC, **en este proceso** | Qwen3.5-9B en **otro contenedor**, por HTTP |
| GPU | usa `--device` de este proceso | no usa la GPU de este proceso |
| Clases | 16 (9 eventos + 7 subtipos de fondo) | 10 (9 eventos + `background`) |
| Entrenamiento | requiere el `.joblib` entrenado | zero-shot, solo prompt |
| Requisito | ninguno | el servidor vLLM arrancado |

### 1. Servir Qwen3.5-9B con vLLM

```bash
./scripts/serve_events_vllm.sh --detach     # arranca en segundo plano
docker logs -f vllm-events                  # seguir la carga (tarda unos minutos)
curl http://localhost:8010/v1/models        # comprobar que está vivo
docker rm -f vllm-events                    # parar
```

Decisiones que lleva ese script dentro, todas revisables por variable de entorno:

- **Contenedor, no venv.** vLLM resuelve su propio torch/CUDA y no convive con los pines
  de `requirements.txt` (`torch==2.11.0+cu128`, `transformers==5.14.1`). La imagen oficial
  `vllm/vllm-openai:v0.28.0` lo aísla del todo — mismo motivo que el intérprete aparte de
  MolmoWeb-4B, pero sin necesitar root en el host. `Qwen3_5ForConditionalGeneration` ya
  está en el registro de modelos de las releases estables (verificado en v0.26.0 y
  v0.28.0): **no** hace falta compilar la rama `main`.
- **`--quantization fp8`.** En bf16 el checkpoint ocupa 19,3 GB y no cabe junto a la caché
  KV en una 4090 de 24 GB **compartida con otros procesos del host**. FP8 (W8A8, al vuelo
  desde el bf16) lo deja en ~10 GB. La 4090 es Ada (compute capability 8.9), que tiene FP8
  nativo; el único checkpoint precuantizado del Hub es NVFP4, que exigiría Blackwell.
  `QUANTIZATION="" ./scripts/serve_events_vllm.sh` sirve en bf16 si algún día sobra VRAM.
- **`--tool-call-parser qwen3_xml`.** VideoARM invoca sus herramientas por function
  calling, así que el servidor tiene que saber parsearlas. Qwen3.5 **no** emite el JSON de
  Hermes sino un XML anidado (`<tool_call><function=...><parameter=...>`), que es lo que
  espera este parser. Con el parser equivocado el agente no consigue llamar a ninguna
  herramienta y se queda sin evidencia que mirar.
- **Puerto 8010**, no el 8000 por defecto de vLLM: con `--network host` se comparte el
  espacio de puertos del host, donde 8000/8001 ya los ocupa otro proyecto. Si se cambia,
  cambiar también `config.VideoARMEventsConfig.base_url`.
- **`HF_HOME=/models`**, replicando el `Dockerfile` del proyecto: los ~19 GB de pesos caen
  en `models/hub/`, junto a V-JEPA2 y MolmoWeb-4B, no en el `~/.cache` del host. El caché de
  `torch.compile` se monta aparte en `models/vllm-cache/`: sin eso vive dentro del contenedor y
  cada reinicio recompila (24 s vs 5 s medidos).

### La VRAM hay que repartirla: `--gpu-memory-utilization` es el parámetro delicado

vLLM reserva su fracción de VRAM **de forma permanente**, no solo mientras atiende peticiones.
Con una sola 4090 compartida entre este servidor y el resto del pipeline, el margen es estrecho
por los dos lados:

- Con `0.80`, `python -m main --events-backend videoarm` **muere con CUDA OOM** al cargar
  SoccerMaster para el tracking (que pide ~3 GB) — no queda nada libre.
- Con `0.55`, el que no arranca es vLLM: `ValueError: No available memory for the cache blocks`.
  Los pesos en fp8 ya son 10,8 GB, y encima va la caché del encoder de visión.
- Y entre medias aparece un tercer límite, propio de **Qwen3.5 como modelo híbrido**: sus capas
  Mamba reservan un bloque de caché **por secuencia en decodificación**, así que con el defecto
  de vLLM (`max_num_seqs=256`) y la VRAM justa el arranque muere con
  `max_num_seqs (256) exceeds available Mamba cache blocks (92)`. El script fija
  **`--max-num-seqs 16`**: aquí no hay concurrencia que aprovechar —`core/events_vllm.py`
  clasifica los clips de uno en uno— y bajarlo es justo lo que libera bloques para poder
  reservar menos VRAM.

Por eso el script usa **`0.65`** por defecto, que deja convivir a las dos etapas. Súbelo solo si
este servidor va a tener la GPU para él —clasificando clips ya troceados, sin tracking en
paralelo— y recuerda que en esta máquina hay **otros procesos ajenos al proyecto** usando la
GPU: comprueba `nvidia-smi` antes de dar por bueno un número.

La alternativa, si el reparto no cuadra, es serializar: el tracking ya se cachea en
`*.tracks.jsonl` (`core.tracking.ensure_tracking`), así que se puede correr el tracking con el
servidor parado y arrancarlo después para la etapa de eventos.

### 2. Clasificar

```bash
# Un clip suelto
python -m core.events_vllm --source datasets/clips_analisis/goal/goal1.mp4

# Una carpeta -> JSONL con una predicción por línea (mismo formato que core.events)
python -m core.events_vllm --source datasets/clips_analisis/corner/ --output events.jsonl

# Solo una ventana de un partido completo (recorta con ffmpeg y clasifica ese trozo)
python -m core.events_vllm --source partido.mp4 --start-s 120 --end-s 126
# --base-url / --model-id / --max-iterations / --total-frames-limit sobreescriben defaults
```

La salida es el mismo `EventPrediction.to_telemetry()` que `core/events.py`, con
`probabilities` sobre las 10 etiquetas en vez de 16.

### 3. En el pipeline principal

```bash
python -m main --source partido.mp4 --output-dir output/mipartido --mode full-match \
       --events-backend videoarm
```

`--events-backend` (`vjepa2` por defecto) existe también en `analytics/engine.py`. Con
`videoarm`, `--device` no afecta al clasificador de eventos: el modelo no vive en este
proceso.

### Qué esperar de la calidad (medido, 12 clips de `clips_analisis/`)

Con la configuración por defecto de arriba, sobre 2 clips de cada una de 6 clases
(`goal`, `corner`, `foul`, `substitution`, `yellow_card`, `kick-off`):

| | max_iterations=4 (sin parche) | **max_iterations=1 + parche (por defecto)** |
|---|---|---|
| Respuestas ilegibles | 9 / 12 | **0 / 12** |
| Aciertos | 3 / 12 | **3 / 12** |
| Tiempo | 94 s | **79 s** (~6,6 s por clip) |

El salto real está en la columna de ilegibles: con `max_iterations=4` nueve de las doce
"predicciones" no eran predicciones, sino el fallback por búsqueda de subcadena de
`_parse_response` sobre un tool call mal formado — el 3/12 de esa columna es literalmente
suerte. Con la configuración por defecto las doce salidas son distribuciones que el modelo ha
emitido de verdad.

Dicho eso, **3/12 zero-shot no compite con el SVC entrenado** de `core/events.py`. Ese primer
eval (12 clips, 2 por clase de 6) era demasiado pequeño para sacar nada en firme, así que se
repitió con un eval MUCHO más serio: el pool COMPLETO de `goal`/`shots_on_target`/
`shots_off_target` en `clips_analisis/` (61 clips: 33 goal + 9 + 19 shot, sin submuestrear
nada), comparando cuatro configuraciones sobre el mismo set.

### ¿Aporta valor VideoARM? (medido sobre el pool completo, 61 clips goal+shot)

Antes de responder eso hacía falta aislar la variable: ¿el valor (si lo hay) viene del modelo
(Qwen3.5-9B) o de la capa de orquestación de VideoARM (tool-calling, navegación coarse-to-fine,
memoria HM3)? Para eso se construyó `DirectVLMEventClassifier` — el MISMO modelo, hablando con
el MISMO servidor vLLM, pero SIN pasar por VideoARM: un único mensaje con N frames y la pregunta
de clasificación, sin tools. Comparado contra `VideoARMEventClassifier` y el SVC entrenado:

| Backend | Total | `goal` | `shot` | `shot`→`background` | Latencia/clip |
|---|---|---|---|---|---|
| SVC entrenado (`core/events.py`) | 27/61 (44%) | 16/33 (48%) | 11/28 (39%) | 11/28 | 0,8 s |
| VideoARM + Qwen3.5-9B, sin forzar `clip_analyzer` | 30/61 (49%) | 19/33 (58%) | 11/28 (39%) | 11/28 | 10,2 s |
| Directo (mismo modelo, sin VideoARM) | 31/61 (51%) | **82%** | 14% | 19/28 | 4,0 s |
| **Híbrido v1** (directo + VideoARM sin forzar `clip_analyzer`) | 40/61 (66%) | 91% | 36% | — | ~8,8 s |
| **Híbrido v2** (directo + `clip_analyzer` forzado, config actual) | **42/61 (69%)** | 85% | **50%** | — | ~9 s |

Respuesta corta: **sí aporta valor, pero no donde se esperaba, y no solo.** Ninguno de los
backends individuales gana en las dos clases a la vez: el directo arrasa en `goal` (evento
visualmente inequívoco, no necesita navegación) pero falla `shot` sistemáticamente hacia
`background` (68% de los tiros reales, sin herramienta para "mirar más de cerca"); VideoARM,
en cambio, es donde `shot` mejora — y ahí hubo una segunda ronda de iteración que merece
contarse porque el resultado fue contraintuitivo:

**Se encontró un bug real**: `clip_analyzer` manda hasta 50 frames sueltos por llamada (no en
mosaico, a diferencia de `scene_snapper`), pero el servidor limita `--limit-mm-per-prompt` a
16 -- la llamada fallaba con **400 en silencio** (`_execute_tool` de VideoARM solo atrapa
`TypeError`, no errores HTTP), así que el agente se quedaba sin esa evidencia sin enterarse.
Se corrigió bajando `frame_analysis_max_frames` a 16 (`VIDEOARM_FRAME_ANALYSIS_MAX_FRAMES`).
De paso se midió que, incluso arreglado ese bug, el agente solo elegía `clip_analyzer` como
segunda herramienta el 46% de las veces (repetía `scene_snapper`) -- así que también se forzó
la iteración 2 a esa función concreta vía `tool_choice={"type":"function","function":{"name":
"clip_analyzer"}}` (sí soportado por vLLM), y se sustituyó la pregunta que el agente le hacía
(genérica, "describe el evento") por una fija centrada en la disyuntiva real: ¿hay un intento
de tiro, y entra o no en la red?

Con esos dos arreglos, VideoARM solo en `shot` sube de 39% a 57%. Pero de paso `goal` se
desploma de 58% a 33% -- la pregunta fija enmarca todo como "¿es un SHOT...?" con el gol como
cláusula subordinada, y ese sesgo contamina hasta los goles inequívocos. Se probó la corrección
obvia: dar a las tres opciones (`goal`/`shot`/ninguno) el mismo peso gramatical con un a/b/c
explícito. **Empeoró todo** (`shot` 57%->36%, `goal` solo recupera a 42%) -- el modelo de 9B
no maneja bien la instrucción más larga/con más opciones, mismo patrón que la vez anterior que
se probaron definiciones largas por etiqueta en el prompt principal.

La resolución fue mirar el HÍBRIDO, no VideoARM aislado: el híbrido con la pregunta "sesgada"
da 69%, con la balanceada 59% -- **la pregunta sesgada es la mejor, pese a hundir el `goal`
standalone de VideoARM**, porque en el híbrido VideoARM casi nunca ve un gol claro (el directo
ya se lo queda antes de escalar). Lo que importa no es que VideoARM sea bueno en las dos
clases, es que sea lo mejor posible en el residuo que de verdad le llega -- mayoritariamente
`shot`. Config final: pregunta "sesgada" + `clip_analyzer` forzado (ver `_CLIP_ANALYZER_QUESTION`
en `core/events_vllm.py`, con el razonamiento completo en su comentario).

También se comprobó y descartó filtrar el enrutado del híbrido por la confianza que reporta el
propio modelo: en clips de `shot` mal clasificados como `goal` (falsos positivos, 5/33) la
confianza es 0,90-0,98, IGUAL que en los goles reales acertados. Mal calibrado, sin señal ahí.

**Importante — varianza de ejecución a ejecución.** El 69% de arriba sale de combinar dos JSONL
calculados por separado (directo de una pasada, VideoARM de otra). Al correr el híbrido REAL de
punta a punta en una sola pasada sobre los mismos 61 clips: **61%** (`goal` 85%, `shot` 32%,
frente al 50% de la combinación). El enrutado escaló el mismo número de clips (29/61) las dos
veces, así que la diferencia está en lo que responde VideoARM para esos clips concretos, no en
el enrutado. Es esperable en parte: el agente elige él mismo qué rango de frames analizar en
`clip_analyzer`, así que dos pasadas no ven necesariamente la misma evidencia aunque la pregunta
esté fijada; y vLLM con `temperature=0.0` bajo batching continuo no es estrictamente
determinista bit a bit. **Conclusión honesta: el número real está en un rango, no es un punto
fijo** -- algo entre 61% y 69% según la tirada, siempre claramente por encima de los tres
backends individuales (44-51%), pero sin la precisión de dos cifras que sugeriría un solo run.
Esto también es relevante para el propio experimento: la diferencia entre la pregunta "sesgada"
(69%/61%) y la balanceada a/b/c (59%, un solo run) podría estar parcialmente inflada por esta
misma varianza -- no se ha vuelto a correr la balanceada una segunda vez para comprobarlo.

Con 61 clips (un solo split, sin validación cruzada) esto no es un veredicto cerrado, pero el
salto (44-51% -> 66% -> 69%) es demasiado grande para ser ruido de muestra, y se ha conseguido
DOS veces sin tocar el tamaño del modelo -- ambas veces el techo resultó ser un bug o una
elección de prompt, no capacidad del modelo. Eso pesa en contra de que **escalar a un modelo
más grande (Qwen3.8-27B) sea la palanca con más retorno ahora mismo** -- sigue habiendo
estructura barata sin explorar (ver más abajo) y en esta 4090 de 24 GB el 27B solo cabría
cuantizado a int4 (comunidad, sin garantía de calidad), ocupando la tarjeta entera.

Ideas para seguir sin tocar el modelo, sin probar todavía:
- `shot` en la rama VideoARM del híbrido sigue en 50% -- mirar clip a clip qué le sigue
  fallando (¿rango de frames mal elegido por el agente? ¿el ángulo de cámara no deja ver si
  entra?) antes de seguir puliendo el texto de la pregunta a ciegas.
- Subir `total_frames_limit`/`frame_analysis_max_frames` juntos con
  `--limit-mm-per-prompt` en el servidor, para no perder el instante exacto del disparo.
- Probar si dar al híbrido un tercer nivel -- confiar también en `shot` del directo cuando
  venga con alta confianza, no reconsultar siempre -- cambia el balance latencia/precisión.

### Tres incompatibilidades VideoARM ↔ Qwen3.5 (medidas, no teóricas)

VideoARM está escrito contra o3 + GPT-4.1. Servirlo con Qwen3.5 destapa tres cosas que
merecen quedar escritas, porque las tres se manifiestan como "el backend no clasifica nada"
y ninguna da un error obvio:

**1. Qwen3.5 es un modelo *thinking* y VideoARM nunca manda `max_tokens`.** Su
`call_openai_model_with_tools` acepta el argumento pero no lo mete en el payload, así que el
techo real de generación es `max_model_len` entero. Con el modo thinking por defecto, un clip
de 4,6 s tardó **4 min 30 s**, casi todo cadena de razonamiento. Con
`--default-chat-template-kwargs '{"enable_thinking": false}'` (lo que hace el script) el mismo
clip baja a **~5 s**. A escala de partido completo la diferencia es entre inviable y viable.

**2. Fuera de la primera iteración, Qwen3.5 escribe mal sus propias llamadas a tools.**
VideoARM usa `tool_choice="required"` solo en la iteración 1; a partir de la 2 pasa a `"auto"`.
Con `required` el servidor restringe la gramática y el tool call sale bien formado; con `auto`,
sin restringir, el modelo emite `<=clip_analyzer` en vez de `<function=clip_analyzer>`. El
parser no lo reconoce, así que VideoARM lo interpreta como "no ha llamado a ninguna tool" y se
traga ese texto como **respuesta final**. Medido sobre 12 clips con `max_iterations=4`: **9
respuestas ilegibles**. De ahí que `max_iterations` valga **1** por defecto — no por ahorrar,
sino para quedarse solo en el tramo restringido y dejar que la respuesta la redacte
`_force_final_answer`, que va sin tools.

**3. `_force_final_answer` manda un mensaje `system` al final, y Qwen3.5 lo rechaza.**
La plantilla de chat del modelo responde `400 {"message": "System message must be at the
beginning."}`, VideoARM se come el error y devuelve el literal *"Maximum iterations reached
without a conclusive answer."*. Como con `max_iterations=1` ese es el camino que da **siempre**
la respuesta, sin arreglarlo el backend no clasifica ni un clip. `core/events_vllm.py` lo parchea
con una subclase mínima (`_build_agent_class`) que manda esa misma instrucción con rol `user`;
se hace por subclase y no forkeando el paquete para que actualizar `videoarm` siga siendo un
`pip install -U`.

### Sobre el audio (`audio_transcriber`)

VideoARM tiene una tercera herramienta, `audio_transcriber`, que apunta a `whisper-1` —
detrás de nuestro servidor vLLM no hay ningún `/v1/audio/transcriptions`. **No hace falta
desactivarla**: el propio agente hace `ffprobe` del stream de audio *antes* de llamar a la
API y, si no hay, devuelve `{"status": "no_audio"}` sin tocar la red y marca
`video_has_audio=False`, lo que inyecta un "Do not call audio_transcriber" en las
iteraciones siguientes. Por eso `predict_window` recorta **sin** pista de audio (`-an`):
garantiza ese camino, y como mucho se desperdicia una iteración.

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

## Uso — detección puntual de tiros y acciones de balón (`core/ball_spotting.py`, T-DEED)

En partido completo (`--mode full-match`), los tiros/goles y el *instante* de cada acción de balón
salen de [T-DEED](https://github.com/arturxe2/T-DEED) (GPL-3.0, ganador de SoccerNet Ball Action
Spotting 2024), que da una probabilidad por frame para 12 clases (PASS, DRIVE, HEADER, HIGH PASS,
OUT, CROSS, THROW IN, SHOT, BALL PLAYER BLOCK, PLAYER SUCCESSFUL TACKLE, FREE KICK, GOAL).

### Instalación (`models/` no se versiona)

```bash
git clone https://github.com/arturxe2/T-DEED.git models/tdeed/T-DEED
pip install gdown
# Pesos (carpeta de Google Drive enlazada en el README de T-DEED):
gdown 1uCWam8EhAXt3z6QQAfhRfzHITUcml438 -O models/tdeed/SoccerNetBall_challenge2/checkpoint_best.pt
gdown 1b7REo4_Cota8SWPSmtdSEfbVNXNgziX6 -O models/tdeed/SoccerNetBall_test/checkpoint_best.pt
```

`SoccerNetBall_challenge2` (RegNetY-800MF, entrenado con 6 de los 7 partidos EFL) es el de
producción (`config.BallSpottingConfig.model_name`). `SoccerNetBall_test` (RegNetY-200MF, split
train) solo sirve para evaluar limpio en Middlesbrough, Stoke y Reading.

### Uso

```bash
# Standalone: lista los tiros/goles de un partido (escanea y cachea junto al vídeo, ~9-15 min)
python -m core.ball_spotting --source partido.mp4 --classes SHOT GOAL

# En el pipeline (valores por defecto):
python -m main --mode full-match --source partido.mp4 --output-dir output/x \
    --shot-backend tdeed --ball-actions-trigger tdeed --ball-actions-label combined
```

* `--shot-backend tdeed|events`: `tdeed` (por defecto) saca shot/goal de T-DEED; el resto de
  eventos sigue saliendo de `--events-backend`. `events` es el comportamiento anterior.
* `--ball-actions-trigger tdeed|possession`: cuándo hay una acción de balón. Con `tdeed` (por
  defecto) cada acción detectada por T-DEED; el jugador sale siempre de `core/possession.py`.
* `--ball-actions-label combined|classifier|spotter`: quién pone la etiqueta (R3D-18 y T-DEED
  combinados, solo el R3D-18, o solo T-DEED).

### Qué esperar (medido, partido completo)

| Tiros | AP ±3 s | P / R |
|---|---|---|
| V-JEPA2 + SVC, etiqueta ganadora por ventana (antes) | 31,6% | 29% / 73% |
| V-JEPA2 + SVC, picos por clase (`core/spotting.py`) | 34,5% | 26% / 70% |
| T-DEED `SoccerNetBall_test`, 3 partidos EFL no vistos (69 tiros) | 75,6% | 74% / 70% |
| T-DEED `SoccerNetBall_challenge2`, 6 partidos SoccerNet-v2 de otras ligas (130 tiros) | 78,0% | 65% / 84% (umbral 0,3) |

| Acciones de balón, `ground_truth/` (367 acciones, ±1 s) | P | R | F1 |
|---|---|---|---|
| Disparo por cambio de posesión + etiqueta R3D-18 (antes) | 28% | 57% | 38% |
| Disparo T-DEED + etiqueta R3D-18 | 80% | 80% | 80% |
| Disparo T-DEED + etiqueta combinada (por defecto) | 87% | 88% | 88% |
| Disparo T-DEED + etiqueta T-DEED | 90% | 92% | 91% |

Un verificador VLM sobre los candidatos (Qwen3.5-9B con frames o vídeo, con y sin razonamiento;
VideoARM; Qwen3-Omni-30B) NO separa tiros de centros (AUC 0,49-0,63 sobre 150 candidatos
difíciles): queda en el código (`ShotVerifier`, `scripts/verify_shot_candidates.py`) pero no se
usa. Los scripts de todas estas mediciones están en `scripts/shot_benchmark/`.

## Uso — composición final (`analytics/engine.py`)

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

## Uso — publicación en YouTube Studio (`automation/molmo_publisher.py`)

Último eslabón del pipeline: sube el `final.mp4` de `analytics/engine.py` a YouTube Studio
sin escribir un solo selector CSS. El bucle del agente es:

```
page.screenshot()  ->  MolmoWeb-4B ("point to the CREAR button")  ->  (x, y)  ->  page.mouse.click()
```

### Por qué un VLM y no selectores

YouTube Studio es una SPA de Polymer con Shadow DOM y clases ofuscadas que cambian entre
despliegues: cualquier XPath que escribamos hoy se rompe solo. MolmoWeb-4B, en cambio, es un
**agente web** (arquitectura Molmo 2: Qwen3-8B + SigLIP 2): se le da un objetivo en lenguaje
natural y responde con `{"thought": ..., "action": {"name": "click", "x": 34.6, "y": 8.7}}`,
en **porcentaje** del ancho/alto de la imagen. Cuando YouTube rediseñe la UI hay que retocar
objetivos en lenguaje natural (`StudioPrompts`), no selectores.

Ojo, que es fácil equivocarse aquí (yo lo hice en la primera versión): **no** es el Molmo de
2024 con *pointing* suelto (`point to X` → `<point x=… y=…>`). Cambia todo lo que lo rodea:

| | Molmo 2024 (`Molmo-7B-D-0924`) | MolmoWeb-4B (Molmo 2) |
|---|---|---|
| Carga | `AutoModelForCausalLM` | `AutoModelForImageTextToText` |
| Preproceso | `processor.process(...)` | `processor.apply_chat_template(...)` |
| Generación | `model.generate_from_batch(...)` | `model.generate(...)` |
| Prompt | `"point to the X"` | `"molmo_web_think: "` + plantilla Jinja (GOAL / PREVIOUS STEPS / CURRENTLY ACTIVE PAGE) |
| Respuesta | `<point x="52.3" y="14.7">` | `{"thought":…, "action":{"name":"click","x":…,"y":…}}` |
| Coordenadas | % de la imagen | **% de la imagen (igual)** |

`parse_molmo_response()` entiende los dos dialectos, así que `--model-id` puede apuntar a
cualquiera de las dos familias. Y hay dos detalles que sí cambian el diseño:

**Es un modelo de trayectoria**, no un localizador sin memoria: `_TrajectoryPointer` guarda
los pasos dados y los reinyecta como `PREVIOUS STEPS`, que es lo que le permite interpretar la
pantalla siguiente sabiendo que ya pulsó "CREAR".

**Es un agente completo, y a veces la respuesta correcta no es un clic.** Su espacio de
acciones incluye `goto`, `scroll`, `keyboard_type`, `send_msg_to_user`. Cuando devuelve una de
esas, no es que no haya encontrado el elemento: ha decidido que tocaba otra cosa.
`describe_non_click_action()` separa ambos casos en el log porque el arreglo es distinto
(contexto/prompt vs. timing). Pasó de verdad en el banco de pruebas: con la URL por defecto
(`studio.youtube.com`) sobre una página que era otra cosa, respondió
`goto(url='https://studio.youtube.com')` — decisión correcta para el estado que le habíamos
descrito. Por eso `predict_and_click` le pasa ahora el `page.url` y el `page.title()` reales:
**si le mientes sobre dónde está, razona sobre la mentira**.

El porcentaje está verificado contra su propio dataset de grounding
(`allenai/MolmoWeb-SyntheticGround`): en imagen de 1288×712, `bbox [257.4, 38.0, 693.4, 74.0]`
(centro 36.9% / 7.9%) ↔ respuesta del modelo `x=34.6, y=8.7`. Dentro de la caja, en las 5
muestras comprobadas. Y su agente oficial lo dice explícito en `convert_action_json_to_action_obj`:
*"Coordinates are expected as percentages (0-100) and converted to pixels using the screenshot
dimensions"*, con el recorte a `[1, dim-2]` que replica `Point.to_viewport_px`.

Dos detalles no obvios del mapeo de coordenadas, y ambos son fuente de clics desplazados:

- Las capturas se toman **siempre** con `full_page=False`. Una captura de página completa
  produce coordenadas fuera del viewport que `page.mouse.click` no puede alcanzar.
- El porcentaje se multiplica por el **viewport en píxeles CSS**, nunca por el tamaño en
  píxeles de la captura (que va multiplicado por el `deviceScaleFactor` de la pantalla).
  Normalizar a fracción [0, 1] es lo que desacopla ambas escalas (`Point.to_viewport_px`).

Las dos únicas excepciones a "todo por VLM" están justificadas: el selector de ficheros es
una ventana **nativa del SO** (no sale en la captura, se intercepta con `expect_file_chooser`),
y el chequeo de "¿se cerró el diálogo?" usa el tag del web component `ytcp-uploads-dialog`,
que es un nombre de elemento, no una clase generada por el bundler.

### Sesión de Google (una vez): display virtual + VNC

El login **no** se automatiza: es justo lo que Google protege con 2FA y CAPTCHAs. Se hace a
mano una vez y `launch_persistent_context` guarda las cookies en `automation/chrome_session/`
(en `.gitignore` — son credenciales de sesión) para todas las ejecuciones siguientes.

Pero este host es **headless**: `DISPLAY` vacío, sin sockets en `/tmp/.X11-unix`, sesión `tty`.
`headless=False` no tiene dónde dibujar. Por eso `automation/vnc_session.sh` levanta un display
virtual (Xvfb) + servidor VNC dentro del contenedor:

```bash
# dentro del contenedor
bash automation/vnc_session.sh start     # imprime la contraseña VNC generada
```

Desde tu portátil, túnel SSH contra la IP del contenedor en la red bridge (no hace falta
publicar puertos ni reiniciar nada) y cliente VNC a `localhost:5900`:

```bash
ssh -L 5900:172.17.0.3:5900 <usuario>@<host>
```

Y ya en el contenedor, con el display puesto:

```bash
export DISPLAY=:99
python -m automation.molmo_publisher --login-only
```

Requiere en el contenedor (ya instalado): `apt-get install -y xvfb x11vnc` y
`playwright install --with-deps chromium`.

Dos trampas que nos costaron un rato y están resueltas dentro del script, por si vuelves a
tocarlo:

- **`pkill -f "Xvfb :99"` se mata a sí mismo**: el patrón casa con la propia línea de comandos
  del shell que lo ejecuta. De ahí los patrones tipo `"[X]vfb :99"`.
- **Zombis**: el PID 1 del contenedor es `tail -f /dev/null`, que no recolecta hijos huérfanos.
  Cuando `x11vnc` muere deja un `<defunct>` que `pgrep` sigue listando, y el script se creía
  arrancado con el puerto 5900 sin escuchar a nadie. `alive()` filtra por estado, y `status`
  comprueba el puerto de verdad, no el proceso.

### Ejecución

```bash
# Ensayo en seco: pide coordenadas a Molmo y vuelca capturas con el punto marcado,
# SIN hacer un solo clic. Es la forma de validar prompts antes de tocar nada.
python -m automation.molmo_publisher --video output/miclip/final.mp4 \
    --dry-run --debug-dir output/miclip/publish_debug

# Subida real (privada por defecto)
python -m automation.molmo_publisher --video output/miclip/final.mp4 \
    --title "Test Clip Short" --visibility private --output output/miclip/publish.json
```

Integrado en el pipeline, encadenado al render del clip:

```bash
python -m main --source clip.mp4 --output-dir output/miclip --mode clip \
    --publish --publish-title "Test Clip Short"
```

`--publish` solo aplica a `--mode clip` (publica `final.mp4`) y escribe `publish.json` en
`--output-dir`. Si la publicación falla **no** tumba la ejecución: el análisis
(`telemetry.json` + `final.mp4`) ya está en disco y es el entregable de verdad.

Visibilidad limitada a `private`/`unlisted` a propósito: esto es un banco de pruebas, no
hay opción `public`. Todo lo configurable vive en `config.PublisherConfig` (viewport,
pausas, modelo, perfil de Chromium).

### Qué se rompe y cómo se depura

Las pausas antes de cada captura (`settle_ms`, `modal_settle_ms`, `upload_settle_ms`) no son
paranoia: Studio anima modales, y una captura tomada a mitad de la transición le da al VLM
una UI a medio pintar (opacidad/escala intermedias) con la que las coordenadas salen
desplazadas. `--debug-dir` guarda cada paso con una cruz roja en el punto predicho, que es
lo que distingue los dos modos de fallo: si la cruz apunta a otro elemento el problema es
el **prompt**; si apunta a una UI a medio dibujar, es el **timing**.

Un aviso sobre el **idioma de la UI**: Studio se muestra en el idioma de la cuenta, y los
prompts citan la etiqueta española con su traducción inglesa al lado (`"CREAR" (Create)`)
precisamente para no depender de eso. La pantalla de login sin sesión sale en inglés
(`hl=en`), así que hasta el primer login real no sabremos en qué idioma queda el asistente.

Requiere `playwright install --with-deps chromium` en el contenedor (ver § Instalación) y
GPU libre para el VLM: ~16 GB en bfloat16. Los autores **recomiendan float32**, que no cabe
en una 4090 de 24 GB (el modelo lleva Qwen3-8B de base); si las coordenadas salen erráticas,
`config.PublisherConfig.dtype` es la primera sospechosa.

### El VLM corre en OTRO intérprete (`/opt/venv-molmo`)

Esto es lo más importante que hay que saber antes de tocar nada aquí. El checkpoint se guardó
con **transformers 4.57.3** y su repo oficial pinea `transformers>=4.51,<5`. Este repo fija
`transformers==5.14.1` porque lo exige la API de Siglip de `Soccer_Master`. Son incompatibles
y ninguna de las dos versiones es negociable.

Con transformers 5.x el código remoto del checkpoint falla **en cadena** (los tres, medidos el
17 ago 2026): `KeyError: 'default'` en `ROPE_INIT_FUNCTIONS` → `TypeError: Unexpected keyword
argument image_use_col_tokens` en `ProcessorMixin` → `cache_position` a `None` en
`prepare_inputs_for_generation`. Se llegó a parchear los dos primeros y funcionaban, pero se
descartó esa vía: cada shim reimplementa a mano un trozo del delta 4.x→5.x, y el siguiente
fallo de ese tipo puede no ser una excepción ruidosa sino **coordenadas mal calculadas en
silencio**.

Solución: el VLM vive en `/opt/venv-molmo` y se le habla por tubería.

```
proceso principal  (/opt/venv, transformers 5.14.1)
├─ YouTubeStudioPublisher ──► Playwright ──► Chromium
└─ RemoteMolmoPointer ─┐          monta el prompt y parsea la respuesta AQUÍ
                       │  JSON por líneas (stdin/stdout), PNG en base64
                       ▼
   subproceso  (/opt/venv-molmo, transformers 4.57.3)
   └─ molmo_server.py ──► MolmoWebPointer ──► MolmoWeb-4B en GPU
```

Dos decisiones que mantienen esto barato de mantener:

- **El servidor importa nuestro propio `MolmoWebPointer`.** El repo está montado igual para
  ambos intérpretes y `molmo_publisher.py` solo importa stdlib a nivel de módulo (torch y
  transformers son imports perezosos dentro del constructor). Cero código duplicado entre venvs.
- **El prompt se monta y se parsea en el lado principal**; el subproceso solo hace
  imagen+texto → texto. Así la lógica que tiene tests (plantilla, historial, parser) vive en un
  único sitio y el otro lado no tiene nada que testear.

Crear el venv (una vez):

```bash
/opt/venv/bin/python -m venv /opt/venv-molmo
/opt/venv-molmo/bin/pip install "transformers==4.57.3" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" \
    accelerate einops safetensors pillow jinja2 numpy
```

Ojo: `python -m venv --system-site-packages` **no** sirve para reutilizar el torch del venv
principal — hereda del Python del sistema, no del venv padre, y te quedas sin torch. Hay que
instalarlo entero (~3 GB), que además es lo que queríamos: aislamiento de verdad.

## Roadmap de desarrollo

| Orden | Módulo | Descripción |
|-------|--------|-------------|
| 1 | `core/tracking.py` | ✅ Detección + tracking |
| 2 | `core/classification.py` | ✅ Asignación de equipo (PRTReId + K-Means); dorsal (ViT OCR) queda pendiente |
| 3 | `core/homography.py` | ✅ Keypoints+líneas del campo (PnLCalib, 2 HRNet-W48) → matriz H → `tracks_2d.jsonl` (`field_xy` en metros) + `radar.mp4`. Falta CONSUMIRLA: posesión en metros, métricas físicas, zonas |
| 4 | `analytics/engine.py` | ✅ Composición final (equipo + posesión + acción de balón + evento) en un vídeo + `telemetry.json`, sobre clips cortos ya recortados; partidos completos quedan pendientes |
| 5 | `core/events.py` | ✅ Clasificación de eventos (V-JEPA2 + SVC) sobre clips ya recortados; falta el troceado de partido completo → clips |
| 6 | `utils/visualizers.py` | ✅ Vídeo anotado (`--annotated-output` en `core/tracking.py`); falta `main.py` (orquestación batch multi-módulo) |
| 7 | `core/possession.py` / `core/ball_actions.py` | ✅ Poseedor del balón + clasificación Pass/Drive/High Pass/Header/Throw In/Background (R3D-18, ya entrenado y probado en ejecución real vía `analytics/engine.py`; Tackle/Cross/Ball Player Block diferidas, ver arriba) |
| 8 | `orquestrator/agent.py` | Q&A semántico |
| 9 | `automation/molmo_publisher.py` | ✅ Publicación en YouTube Studio (MolmoWeb-4B apunta, Playwright clica; sesión persistente en `automation/chrome_session/`, integrado en `main.py --publish`). Escrito y probado con dobles de `Page`/`Pointer`; **falta la primera pasada real contra la UI de YouTube** para calibrar prompts y pausas. Otras redes (X/Instagram) pendientes |

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

Tercera instancia del mismo patrón, ya resuelta: `core/homography.py` depende de
[PnLCalib](https://github.com/mguti97/PnLCalib), que ahora se clona en
`models/homography/PnLCalib/` (ignorado por
git, igual que `Soccer_Master/`) y está verificado end-to-end. Dio exactamente el fallo esperado
del patrón: `utils/utils_optimize.py` importa `shapely`, que no estaba instalado —
`ModuleNotFoundError` en el primer `import utils.utils_calib`. Añadido a `requirements.txt`.

Reutiliza `core.tracking._repo_import_context` sin cambios: PnLCalib define un paquete namespace
`utils` que choca con el `utils/` regular del proyecto, que es literalmente el mismo problema que
ese context manager ya resolvía para `Soccer_Master`.

Puntero cruzado para quien llegue a depurar cualquiera de las tres: `torch`/`transformers`
(pineados en `requirements.txt`, incidentes documentados arriba), `prtreid`/`bpbreid` (sin pin,
esta sección), `PnLCalib` (`shapely`, esta sección).

### A vigilar (no rompe todavía): `scikit-learn` sin techo y el `.joblib` de `core/events.py`

Visto en `analytics/engine.py` (17 jul 2026), solo como warning, no error: `InconsistentVersionWarning:
Trying to unpickle estimator ... from version 1.7.2 when using version 1.9.0`. El clasificador de
eventos (`svc_classifier_more_penalties.joblib`) se serializó con scikit-learn 1.7.2; `requirements.txt`
solo fija `scikit-learn>=1.4.0` (sin techo), así que un `docker build` en otro momento puede traer
una versión bastante más nueva. Por ahora sigue funcionando (mismo patrón de riesgo que
`transformers`/`prtreid` sin pin, todavía sin materializarse en rotura real) — si en algún momento
`core/events.py` empieza a fallar al cargar el `.joblib`, o las predicciones se vuelven raras sin
ningún error visible, fijar `scikit-learn==1.7.2` (o reentrenar/re-serializar el clasificador con
la versión que esté instalada) es el primer sitio a mirar.

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
