# Sports-Classification/basketball 🏀

Pipeline de baloncesto, aislado del pipeline de fútbol de la raíz del repo. No comparte
Dockerfile, requirements.txt ni entorno con `Sports-Classification/` — ver notas de
aislamiento más abajo. Basado en
[Basketball-Scoring-Analysis](https://github.com/Ahmed-M0hamed/Basketball-Scoring-Analysis)
(MIT).

## Modelos

- Jugadores/balón/árbitros: Roboflow Universe `basketball-player-detection-3-ycjdo/4`
  (workspace `roboflow-jvuqo`)
- Keypoints de cancha: Roboflow Universe `basketball-court-detection-2/14`
  (workspace `roboflow-jvuqo`)
- Clasificación de equipo: zero-shot con `patrickjohncyh/fashion-clip` (Hugging Face)

Los dos modelos de Roboflow se cargan con `inference.get_model(...)`: la librería descarga
los pesos una vez (autenticando con tu API key) y luego corre `.infer()` **localmente**
(onnxruntime), no como llamadas HTTP por frame a un endpoint hosted. Ver
`infering/infer.py`.

## Configuración

```bash
cp .env.example .env   # y rellena ROBOFLOW_API_KEY (https://app.roboflow.com/settings/api)
```

`.env` está en `.gitignore`; nunca se hardcodea la key en el código.

## Build & run (imagen propia, no toca el Dockerfile/requirements.txt de fútbol)

```bash
cd basketball
docker build -t basketball-analytics .

docker run -d --name basketball-analytics --gpus all \
  -u "$(id -u):$(id -u)" \
  -v "$(pwd)":/app -w /app \
  --env-file .env \
  basketball-analytics

docker exec -it basketball-analytics bash
```

`-u "$(id -u):$(id -u)"` evita que los ficheros que genera el contenedor (outputs/, caché de
modelos, `__pycache__`) queden con dueño `root` en el host.

## Descargar los datasets originales de Roboflow

```bash
docker exec -it basketball-analytics python3 scripts/download_datasets.py
```

Descarga a `basketball/datasets/` (gitignored).

## Correr el pipeline

```bash
docker exec -it basketball-analytics python3 main.py
```

Lee `inputs/extended_input.mp4`, corre inferencia real (`presaved=False` ya activado en
`main.py`) y escribe `outputs/out_5.mp4`. Imprime tiempo por etapa (detección
jugador/balón, keypoints de cancha, clasificación de equipo).

## Estado tras el smoke test (2026-07-20)

Ver el resumen que Claude dio en el chat para versiones exactas, incompatibilidades
resueltas y coste/tiempo medido. En corto:

- Pipeline end-to-end funcional con inferencia real (los 3 modelos), ByteTrack, homografía
  y vídeo de salida.
- `onnxruntime` (CPU-only) instalado — los dos modelos de Roboflow corren en CPU, no GPU.
  Para acelerar, instalar `onnxruntime-gpu` en vez de `onnxruntime`.
- El clip de smoke test (`inputs/extended_input.mp4`, ver `inputs/NOTICE.txt`) es un plano
  cenital de calle, no de retransmisión — produjo pocas detecciones. Para validar precisión
  real hace falta un clip con ángulo de cámara elevado/lateral tipo TV.
- Se corrigieron 3 bugs de robustez en el código vendorizado (asumían que siempre hay al
  menos una detección/evento por frame): `scoring_system/find_events.py`, `draw/draw.py`,
  `visualizations/utils.py::check_which_team_offencing`,
  `visualizations/game_momentum.py::get_momuntums`.
