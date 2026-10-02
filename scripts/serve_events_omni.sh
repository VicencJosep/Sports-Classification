#!/usr/bin/env bash
# scripts/serve_events_omni.sh
# =============================
# Arranca el servidor vLLM que consume core/events_omni.py
# (Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit). Backend de eventos alternativo a
# core/events.py (V-JEPA2 + SVC) y a core/events_vllm.py (rama videoARM,
# VideoARM + Qwen3.5-9B) -- este es el que manda el clip entero (vídeo+audio)
# nativamente, sin orquestador ni muestreo de frames.
#
# Imagen y receta reutilizadas de /media/dd1/Projects/Pol/
# native_video_understanding (Dockerfile.vllm-audio + avlib.py: SERVERS
# ["local"]), que ya sirve este mismo checkpoint en este mismo host -- por
# eso el puerto por defecto es el mismo, 8002, para que
# core/events_omni.py::DEFAULT_URL no tenga que cambiar entre proyectos.
#
#   ./scripts/serve_events_omni.sh          # arranca (attached, Ctrl-C para parar)
#   ./scripts/serve_events_omni.sh --detach # en segundo plano
#
# Comprobar que está vivo:   curl http://localhost:8002/v1/models
# Ver el log (si --detach):  docker logs -f vllm-events-omni
# Parar (si --detach):       docker rm -f vllm-events-omni
set -euo pipefail

MODEL_ID="${MODEL_ID:-cyankiwi/Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit}"
# Imagen ya construida en este host (vllm/vllm-openai:v0.26.0 + vllm[audio],
# ver Dockerfile.vllm-audio de native_video_understanding) -- no se reconstruye
# aquí para no duplicarla; si no existe, construirla desde ese Dockerfile.
IMAGE="${VLLM_IMAGE:-vllm-audio:local}"
PORT="${PORT:-8002}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-events-omni}"

# La GPU (1x RTX 4090, 24GB) es compartida con otros procesos del host y NO
# siempre está libre -- comprobar `nvidia-smi` / `docker ps` antes de lanzar
# esto. A 2026-09-07 el checkpoint AWQ-4bit + activaciones + caché KV cabe a
# --gpu-memory-utilization 0.95 en una GPU por lo demás libre (receta medida
# en native_video_understanding/README.md), pero aquí conviven potencialmente
# con el tracking del propio pipeline (SoccerMaster, ~3GB) -- si
# `python -m main` va a correr en paralelo, bajar esto (ver el mismo
# problema, resuelto con 0.65, en serve_events_vllm.sh de la rama videoARM).
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
# 8192, no 16384: es la cuantización AWQ-4bit local (el checkpoint bf16
# completo, Qwen/Qwen3-Omni-30B-A3B-Thinking sin cuantizar, es el que usa
# 16384 en el otro host de native_video_understanding, con más VRAM libre).
# Los clips de clips_analisis/ duran ~5-6s -- de sobra dentro de 8192 incluso
# con audio + razonamiento, sin necesitar el contexto largo.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# Un vídeo + hasta 32 imágenes sueltas (por si en el futuro se reutiliza este
# mismo servidor para prompts de solo-imagen) + 1 pista de audio por prompt.
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 32, \"video\": 1, \"audio\": 1\}}"
# El servidor separa la traza de razonamiento (Qwen3-Omni-Thinking) de la
# respuesta final -- ver core/events_omni.py::DEFAULT_ENABLE_THINKING (activo
# por defecto en el cliente; esto solo habilita que el PARSER del servidor
# sepa separarla si el cliente la pide).
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

# Se reemplaza cualquier contenedor previo con el mismo nombre en vez de
# fallar: relanzar el servidor tras un cambio de flags es lo habitual.
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

# Ojo: NADA de --rm. El contenedor tiene que sobrevivir a su propia muerte
# para poder leer `docker logs ${CONTAINER_NAME}` -- si vLLM aborta al
# arrancar (OOM de VRAM, arquitectura no soportada...) con --rm se borra y el
# error se pierde.
DOCKER_ARGS=(--name "${CONTAINER_NAME}" --gpus all --ipc=host --shm-size 16G -p "${PORT}:8000")
if [[ "${1:-}" == "--detach" ]]; then
    DOCKER_ARGS+=(-d)
    shift
fi

# Reusa el caché de HuggingFace del usuario del host -- el checkpoint (~20GB)
# ya está descargado ahí por native_video_understanding, evita re-bajarlo.
exec docker run "${DOCKER_ARGS[@]}" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    "${IMAGE}" \
    --model "${MODEL_ID}" \
    --served-model-name "${MODEL_ID}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
    --reasoning-parser "${REASONING_PARSER}" \
    "$@"
