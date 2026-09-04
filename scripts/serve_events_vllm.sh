#!/usr/bin/env bash
# scripts/serve_events_vllm.sh
# =============================
# Arranca el servidor vLLM que consume core/events_vllm.py (VideoARM +
# Qwen3.5-9B). Es el "backend" del clasificador de eventos alternativo a
# core/events.py (V-JEPA2 + SVC).
#
# Por qué un contenedor y no un venv: vLLM resuelve su propio torch/CUDA, que
# choca con los pines de este repo (torch==2.11.0+cu128, transformers==5.14.1
# en requirements.txt). La imagen oficial vllm/vllm-openai lo aisla por
# completo -- mismo espíritu que el intérprete aparte de MolmoWeb-4B
# (config.PublisherConfig.venv_python), pero sin necesitar root en el host.
#
#   ./scripts/serve_events_vllm.sh          # arranca (attached, Ctrl-C para parar)
#   ./scripts/serve_events_vllm.sh --detach # en segundo plano
#
# Comprobar que está vivo:   curl http://localhost:8010/v1/models
# Ver el log (si --detach):  docker logs -f vllm-events
# Parar (si --detach):       docker rm -f vllm-events
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.28.0}"
# 8010, no el 8000 por defecto de vLLM: con --network host se comparte el
# espacio de puertos del host, y ahí 8000/8001 ya los ocupa otro proyecto
# (repscan_takedown_agent). Si se cambia, cambiar también
# config.VideoARMEventsConfig.base_url.
PORT="${PORT:-8010}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-events}"

# La GPU (1x RTX 4090, 24 GB) se comparte con otros procesos del host, así que
# NO se puede reservar entera. Y ojo: vLLM reserva su fracción PARA SIEMPRE, no
# solo mientras atiende. Con 0.80 no queda VRAM para el tracking y
# `python -m main --events-backend videoarm` muere con CUDA OOM al cargar
# SoccerMaster (que pide ~3 GB) -- por eso el defecto es 0.65, que deja sitio a
# las dos etapas. Es un margen estrecho por los dos lados y no conviene tocarlo
# a ciegas: por debajo de ~0.60 vLLM ni arranca ("No available memory for the
# cache blocks" -- los pesos en fp8 ya son 10,8 GB y encima va la caché del
# encoder de visión). Subirlo solo si este servidor va a tener la GPU para él
# (p.ej. clasificando clips ya troceados, sin tracking en paralelo).
# Ver README § "Servir Qwen3.5-9B con vLLM".
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Qwen3.5 es híbrido (capas Mamba), y CADA secuencia en decodificación reserva
# un bloque de caché Mamba. Con el defecto de vLLM (max_num_seqs=256) y una
# reserva de VRAM ajustada, el arranque muere con
#   "max_num_seqs (256) exceeds available Mamba cache blocks (92)".
# Aquí no hay concurrencia que aprovechar: core/events_vllm.py clasifica los
# clips DE UNO EN UNO (predict_many es un generador secuencial), así que 16
# sobra y libera bloques para poder bajar GPU_MEMORY_UTILIZATION.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
# fp8 (W8A8) cuantiza al vuelo desde el checkpoint bf16: 19.3 GB -> ~10 GB.
# En bf16 el modelo NO cabe junto a la caché KV en 24 GB compartidos. La 4090
# es Ada (compute capability 8.9), que sí tiene FP8 nativo -- NVFP4 (el único
# checkpoint precuantizado que hay en el Hub) exigiría Blackwell, y GGUF/MLX
# no sirven aquí. Poner QUANTIZATION="" para servir en bf16 si algún día hay
# GPU libre de sobra y se quiere descartar la cuantización como sospechosa.
QUANTIZATION="${QUANTIZATION:-fp8}"

# VideoARM llama a sus tools por function calling, así que el servidor tiene
# que saber parsearlas. Qwen3.5 NO emite el JSON de Hermes sino un XML anidado
# (<tool_call><function=nombre><parameter=x>…), que es justo lo que espera el
# parser qwen3_xml (ver la plantilla de chat del propio checkpoint).
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"

# Qwen3.5 es un modelo "thinking": si no se le dice lo contrario, su plantilla
# de chat abre <think> y razona sin límite. Y VideoARM NUNCA manda max_tokens
# (su call_openai_model_with_tools acepta el argumento pero no lo mete en el
# payload), así que el techo real de generación es max_model_len entero: un
# clip de 4,6 s tardó 4,5 min medidos, casi todo cadena de razonamiento.
# Apagándolo el mismo clip baja a ~5 s (12 clips en 79 s) -- ver README.
# ENABLE_THINKING=true lo reactiva si se quiere medir la diferencia en calidad,
# que NO se ha medido: se apagó por tiempo, no porque razonar sea peor.
ENABLE_THINKING="${ENABLE_THINKING:-false}"
# 32 frames (VIDEOARM_TOTAL_FRAMES_LIMIT) se envían en mosaicos de 2x3 -> ~6
# imágenes por prompt; 16 deja margen para el clip_analyzer sin permitir que
# una llamada desbocada reviente el contexto.
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 16\}}"

# Se reemplaza cualquier contenedor previo con el mismo nombre en vez de
# fallar: relanzar el servidor tras un cambio de flags es lo habitual.
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

# Ojo: NADA de --rm. El contenedor tiene que sobrevivir a su propia muerte para
# poder leer `docker logs ${CONTAINER_NAME}` -- si vLLM aborta al arrancar (OOM
# de VRAM, arquitectura no soportada...) con --rm se borra y el error se pierde.
DOCKER_ARGS=(--name "${CONTAINER_NAME}" --gpus all --ipc=host --network host)
if [[ "${1:-}" == "--detach" ]]; then
    DOCKER_ARGS+=(-d)
    shift
fi

# HF_HOME=/models replica la convención del Dockerfile del proyecto
# (ENV HF_HOME=/app/models): los pesos caen en models/hub/, junto a
# V-JEPA2 y MolmoWeb-4B, y no en ~/.cache del host.
# El caché de torch.compile vive en /root/.cache/vllm DENTRO del contenedor:
# sin montarlo, cada arranque vuelve a compilar y capturar los CUDA graphs
# (varios minutos). Montado, solo se paga la primera vez.
mkdir -p "${PROJECT_ROOT}/models/vllm-cache"

exec docker run "${DOCKER_ARGS[@]}" \
    -e HF_HOME=/models \
    -v "${PROJECT_ROOT}/models:/models" \
    -v "${PROJECT_ROOT}/models/vllm-cache:/root/.cache/vllm" \
    "${IMAGE}" \
    --model "${MODEL_ID}" \
    --served-model-name "${MODEL_ID}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    ${QUANTIZATION:+--quantization "${QUANTIZATION}"} \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --default-chat-template-kwargs "{\"enable_thinking\": ${ENABLE_THINKING}}" \
    --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
    "$@"
