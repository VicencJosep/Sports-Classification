#!/usr/bin/env bash
# Descarga UN partido de SoccerTrack v2 (https://huggingface.co/datasets/atomscott/soccertrack-v2,
# cámara panorámica fija 4K, 12 clases de acción de balón) a datasets/soccertrack-v2/.
#
# El dataset es de acceso restringido: hay que aceptar sus condiciones en la web con una cuenta
# de Hugging Face y poner un token de lectura en .env (no se versiona):
#     HF_TOKEN=hf_xxx
#
# Uso (desde el HOST, raíz del proyecto):
#     ./scripts/download_soccertrack_v2.sh              # partido 117093 (~7,3 GB)
#     ./scripts/download_soccertrack_v2.sh 118575       # otro partido
#     WITH_GSR=1 ./scripts/download_soccertrack_v2.sh   # + posiciones por frame (~5,5 GB más)
#
# Se ejecuta dentro del contenedor porque datasets/ es de root.
set -euo pipefail
cd "$(dirname "$0")/.."

MATCH="${1:-117093}"
TOKEN="$(grep -h '^HF_TOKEN=' .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"'"'"' ' || true)"
if [ -z "$TOKEN" ]; then
    echo "Falta HF_TOKEN=... en .env (token de lectura de Hugging Face con acceso al dataset)" >&2
    exit 1
fi

PATTERNS=(
    "README.md" "CHANGES_v1_*.md"
    "videos/${MATCH}/*"
    "bas/${MATCH}/*"
    "ball/${MATCH}_*"
    "mot/${MATCH}.txt" "mot/README.md"
    "raw/${MATCH}/*"
)
[ "${WITH_GSR:-0}" = "1" ] && PATTERNS+=("gsr/${MATCH}/*")

INCLUDES=()
for p in "${PATTERNS[@]}"; do INCLUDES+=(--include "$p"); done

docker exec -e HF_TOKEN="$TOKEN" -w /app Sports-Classification \
    hf download atomscott/soccertrack-v2 --repo-type dataset \
    --local-dir datasets/soccertrack-v2 "${INCLUDES[@]}"

echo "Descargado en datasets/soccertrack-v2/:"
docker exec -w /app Sports-Classification du -sh datasets/soccertrack-v2/*
