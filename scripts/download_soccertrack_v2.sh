#!/usr/bin/env bash
# Descarga partidos de SoccerTrack v2 (https://huggingface.co/datasets/atomscott/soccertrack-v2,
# cámara panorámica fija 4K, 12 clases de acción de balón) a datasets/soccertrack-v2/.
#
# El dataset es de acceso restringido: hay que aceptar sus condiciones en la web con una cuenta
# de Hugging Face y poner un token de lectura en .env (no se versiona):
#     HF_TOKEN=hf_xxx
#
# Uso (desde el HOST, raíz del proyecto):
#     ./scripts/download_soccertrack_v2.sh                  # partido 117093 (~7,3 GB)
#     ./scripts/download_soccertrack_v2.sh 118575 128057    # otros partidos
#     WITH_GSR=1 ./scripts/download_soccertrack_v2.sh       # + posiciones por frame (~5,5 GB más por partido)
#     ALL=1 WITH_GSR=1 ./scripts/download_soccertrack_v2.sh # los 10 partidos (~130 GB)
#
# Desde el host se ejecuta dentro del contenedor porque datasets/ es de root. Dentro del contenedor
# llama a hf directamente, así que también se puede dejar desatendido (sigue aunque se cierre la sesión):
#     docker exec -d -w /app -e ALL=1 -e WITH_GSR=1 Sports-Classification \
#         bash -c 'bash scripts/download_soccertrack_v2.sh > output/soccertrack_download.log 2>&1'
set -euo pipefail
if [ -f /.dockerenv ]; then cd /app; else cd "$(dirname "$0")/.."; fi

ALL_MATCHES=(117092 117093 118575 118576 118577 118578 128057 128058 132831 132877)
if [ "${ALL:-0}" = "1" ]; then
    MATCHES=("${ALL_MATCHES[@]}")
elif [ $# -gt 0 ]; then
    MATCHES=("$@")
else
    MATCHES=(117093)
fi
TOKEN="$(grep -h '^HF_TOKEN=' .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"'"'"' ' || true)"
if [ -z "$TOKEN" ]; then
    echo "Falta HF_TOKEN=... en .env (token de lectura de Hugging Face con acceso al dataset)" >&2
    exit 1
fi

PATTERNS=("README.md" "CHANGES_v1_*.md" "mot/README.md")
for m in "${MATCHES[@]}"; do
    PATTERNS+=("videos/${m}/*" "bas/${m}/*" "ball/${m}_*" "mot/${m}.txt" "raw/${m}/*")
    if [ "${WITH_GSR:-0}" = "1" ]; then PATTERNS+=("gsr/${m}/*"); fi
done

INCLUDES=()
for p in "${PATTERNS[@]}"; do INCLUDES+=(--include "$p"); done
HF=(hf download atomscott/soccertrack-v2 --repo-type dataset --local-dir datasets/soccertrack-v2 "${INCLUDES[@]}")

echo "Partidos: ${MATCHES[*]} (GSR: ${WITH_GSR:-0})"
if [ -f /.dockerenv ]; then
    HF_TOKEN="$TOKEN" "${HF[@]}"
    echo "Descargado en datasets/soccertrack-v2/:"
    du -sh datasets/soccertrack-v2/*
else
    docker exec -e HF_TOKEN="$TOKEN" -w /app Sports-Classification "${HF[@]}"
    echo "Descargado en datasets/soccertrack-v2/:"
    docker exec -w /app Sports-Classification du -sh datasets/soccertrack-v2/*
fi
