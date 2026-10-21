#!/usr/bin/env bash
# Experimento barato de la hoja de ruta §3.6: escanear con el T-DEED afinado en panorámica (SNB_pano1)
# los partidos de TV y juveniles, para compararlo con el de producción (SoccerNetBall_challenge2).
#
# Corre DENTRO del contenedor (sobrevive a la sesión que lo lanza):
#   docker exec -d -w /app Sports-Classification bash scripts/shot_benchmark/run_pano1_tv.sh
#
# 1. Espera a que la GPU compartida tenga >= MIN_FREE_MIB libres (el escaneo pide ~6,5 GB con batch 1).
# 2. Escanea las dos cabezas de T-DEED (tdeed_scan_heads.py) en: Middlesbrough completo (AP de tiros, 25 tiros),
#    el recorte de 20 min de ground_truth/ (acciones), Mallorca, Espanyol y Clásico.
# 3. Lanza el informe (pano1_tv_report.sh): tiros en TV, acciones contra el GT y P(SHOT/GOAL) en los goles.
# Salidas: output/pano1_tv/ (chain.log, scan.log, informes) y output/shot_benchmark/tdeed/SNB_pano1_heads_tv/*.npz.
set -u
cd /app
OUT=/app/output/pano1_tv
mkdir -p "$OUT"
LOG="$OUT/chain.log"
MIN_FREE_MIB=${MIN_FREE_MIB:-8000}
MAX_WAIT_S=${MAX_WAIT_S:-259200}     # 3 días
POLL_S=${POLL_S:-120}
MODEL=SNB_pano1
TAG=_tv

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

log "== inicio; espero >= ${MIN_FREE_MIB} MiB libres (sondeo cada ${POLL_S} s, máximo ${MAX_WAIT_S} s)"
waited=0
while :; do
    free=$(free_mib)
    if [ -n "$free" ] && [ "$free" -ge "$MIN_FREE_MIB" ]; then
        log "VRAM libre: ${free} MiB -> arranco el escaneo"
        break
    fi
    if [ "$waited" -ge "$MAX_WAIT_S" ]; then
        log "ABANDONO: ${waited} s esperando y la GPU sigue ocupada (libre: ${free:-?} MiB)"
        exit 1
    fi
    if [ $((waited % 3600)) -eq 0 ]; then
        log "esperando... VRAM libre ${free:-?} MiB (llevo ${waited} s)"
    fi
    sleep "$POLL_S"
    waited=$((waited + POLL_S))
done

cd /app/models/tdeed/T-DEED
PYTHONPATH=/app python -u /app/scripts/shot_benchmark/tdeed_scan_heads.py "$MODEL" \
    "/app/datasets/SoccerNet/england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End/720p.mp4" \
    /app/ground_truth/2019-10-01_Middlesbrough-Preston_North_End-720p-trim.mp4 \
    /app/backend/storage/videos/8e759934bde2156e/source.mp4 \
    /app/backend/storage/videos/f0926c19277b28db/source.mp4 \
    /app/backend/storage/videos/573d9c1395b8f1c8/source.mp4 \
    --out-tag "$TAG" --batch-size 1 > "$OUT/scan.log" 2>&1
rc=$?
log "escaneo terminado (rc=$rc); npz en output/shot_benchmark/tdeed/${MODEL}_heads${TAG}/"
cd /app
if [ "$rc" -ne 0 ]; then
    log "el escaneo falló: ver $OUT/scan.log"
    exit "$rc"
fi

bash scripts/shot_benchmark/pano1_tv_report.sh > "$OUT/report.log" 2>&1
log "informe terminado (rc=$?): $OUT/report.log y $OUT/*.txt"
