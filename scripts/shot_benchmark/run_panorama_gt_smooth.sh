#!/usr/bin/env bash
# Control del experimento de suavizado: cámara virtual 796x448 guiada por el balón del DATASET pero con una media
# móvil de SMOOTH segundos (3 por defecto; la de siempre es 1 s), escaneada con T-DEED y comparada con la de 1 s.
# Si el AP de tiros cae también aquí, T-DEED depende de la dinámica de la cámara, no solo del encuadre.
#   docker exec -d -w /app -e SMOOTH=3 Sports-Classification bash output/panorama_ball/experiments/run_panorama_gt_smooth.sh [MATCH]
set -u
cd /app
MATCH="${1:-117093}"; SMOOTH="${SMOOTH:-3}"; WAIT_FOR="${WAIT_FOR:-}"
TAG="s${SMOOTH}"
RUN=output/panorama_ball; VEXP=output/panorama_views/experiments
V=datasets/soccertrack-v2/views/$MATCH
MIN_FREE_MB="${MIN_FREE_MB:-7500}"
export PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
status() { echo "$(date '+%F %T') $*" >> "$RUN/STATUS"; }
if [ -n "$WAIT_FOR" ]; then until grep -q "$WAIT_FOR" "$RUN/STATUS"; do sleep 60; done; fi
status "cámara dataset $MATCH suavizado ${SMOOTH}s: render"
for h in 1st 2nd; do
    out="$V/${h}_vcam796$TAG.mp4"
    [ -f "$out" ] || nice -n 10 python -u "$VEXP/panorama_vcam.py" "$MATCH" "$h" 0 -1 --size 796x448 --smooth-s "$SMOOTH" \
        --out "$out" > "$RUN/render_${MATCH}_${h}_gt$TAG.log" 2>&1 || status "render $h falló"
done
videos=()
for h in 1st 2nd; do [ -f "$V/${h}_vcam796$TAG.mp4" ] && videos+=("/app/$V/${h}_vcam796$TAG.mp4"); done
for attempt in 1 2 3 4 5 6; do
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$MIN_FREE_MB" ]; do sleep 60; done
    (cd models/tdeed/T-DEED && python -u "/app/$VEXP/tdeed_scan_heads.py" SoccerNetBall_challenge2 "${videos[@]}" \
        --out-tag _views --batch-size 1) >> output/panorama_views/scan.log 2>&1 && break
    status "escaneo cortado (intento $attempt/6), reintento en 10 min"; sleep 600
done
python -u "$VEXP/panorama_views_eval.py" "$MATCH" --views vcam796 "vcam796$TAG" > "output/panorama_views/report_${MATCH}_gt_$TAG.txt" 2>&1 \
    || status "el informe de vistas falló"
status "FIN cámara dataset $MATCH suavizado ${SMOOTH}s"
