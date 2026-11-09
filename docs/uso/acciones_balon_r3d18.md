# Uso — acciones de balón: pase/regate/cabeceo/saque (`core/ball_actions.py`, R3D-18)

> Sección sacada del README el 2026-11-09, sin cambios de contenido.

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
