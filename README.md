# Sports-Classification ⚽

Sistema de **Sports Analytics para fútbol** que procesa vídeos de partidos con tres objetivos:

1. **Métricas tácticas deterministas** (bottom-up): tracking de jugadores y balón, homografía del campo, posesión, velocidades y heatmaps, usando **YOLO11x** + **roboflow/supervision** (ByteTrack).
2. **Motor de consultas semánticas** (top-down): orquestador **VideoARM / Symphony** para que un entrenador pregunte en lenguaje natural sobre el vídeo (*"¿cuántas llegadas al área tuvimos en la segunda parte?"*).
3. **Publicación automática en RRSS**: el VLM **MolmoWeb-4B** navega la web con **Playwright** y publica clips + métricas.

## Flujo de datos

```
                        ┌──────────────── BOTTOM-UP (batch) ────────────────┐
 partido.mp4 ──► core/tracking.py ──► core/classification.py ──► core/homography.py
                 (YOLO11x+ByteTrack)   (equipos HSV/K-Means,      (px → metros 2D)
                                        dorsales ViT-OCR)
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
│   ├── homography.py          # ⏳ Calibración del campo, matriz H, interpolación 2D
│   ├── classification.py      # ⏳ Filtro HSV / K-Means de equipos + ViT OCR dorsales
│   └── events.py              # ✅ Inferencia de eventos (V-JEPA2 + SVC) sobre un clip .mp4
├── analytics/
│   └── engine.py              # ⏳ Velocidades, posesión, heatmaps, JSON builder
├── orquestrator/
│   └── agent.py               # ⏳ VideoARM / Symphony (Q&A sobre el .mp4)
├── automation/
│   └── molmo_publisher.py     # ⏳ MolmoWeb-4B + Playwright (publicación RRSS)
├── utils/
│   ├── visualizers.py         # ✅ TrackingVisualizer (cajas/IDs/estelas, roboflow/supervision)
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

## Roadmap de desarrollo

| Orden | Módulo | Descripción |
|-------|--------|-------------|
| 1 | `core/tracking.py` | ✅ Detección + tracking |
| 2 | `core/classification.py` | Asignación de equipo (HSV/K-Means sobre crops) + dorsal (ViT OCR) |
| 3 | `core/homography.py` | Keypoints del campo → matriz H (ver `Soccer_Master/sn_calibration` como referencia) |
| 4 | `analytics/engine.py` | Métricas + `telemetry.json` |
| 5 | `core/events.py` | ✅ Clasificación de eventos (V-JEPA2 + SVC) sobre clips ya recortados; falta el troceado de partido completo → clips |
| 6 | `utils/visualizers.py` | ✅ Vídeo anotado (`--annotated-output` en `core/tracking.py`); falta `main.py` (orquestación batch multi-módulo) |
| 7 | `orquestrator/agent.py` | Q&A semántico |
| 8 | `automation/molmo_publisher.py` | Publicación RRSS |

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
- `transformers==5.14.1` — versión contra la que se diagnosticaron y parchearon los incidentes #2 y #3. El incidente #2 se confirmó solucionado en ejecución real; el #3 tiene el fix escrito pero **pendiente de confirmar** en una ejecución (última vez visto en el traceback, aún no reintentado). Si lo ejecutas y funciona, sería buen momento para borrar este matiz. Si falla de otra forma, hay un cuarto incidente por documentar aquí.

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
