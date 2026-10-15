#!/usr/bin/env bash
# Prueba de punta a punta sin el balón del dataset: pista de nuestro detector (panorama_ball_eval.py track) ->
# cámara virtual 796x448 -> T-DEED -> AP de tiros frente a la cámara guiada por el balón del dataset.
#   docker exec -d -w /app Sports-Classification bash output/panorama_ball/experiments/run_panorama_yolo_cam.sh [MATCH]
# MODE (tiles1024), TH (0,1) y METHOD (prox|top1|kalman) eligen cómo se construye la pista; TAG (p. ej. _t03) distingue
# variantes en los nombres de salida (vcam796yolo$TAG). Requiere las detecciones de run_panorama_ball.sh.
set -u
cd /app
MATCH="${1:-117093}"; MODE="${MODE:-tiles1024}"; TH="${TH:-0.1}"; METHOD="${METHOD:-prox}"; TAG="${TAG:-}"
SMOOTH="${SMOOTH:-1}"            # segundos de media móvil de la cámara (panorama_vcam.py --smooth-s)
WAIT_FOR="${WAIT_FOR:-}"         # si se indica, espera a que esa línea aparezca en STATUS (encadenar tras otra variante)
RUN=output/panorama_ball; EXP=$RUN/experiments; VEXP=output/panorama_views/experiments
V=datasets/soccertrack-v2/views/$MATCH
MIN_FREE_MB="${MIN_FREE_MB:-7500}"
export PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
status() { echo "$(date '+%F %T') $*" >> "$RUN/STATUS"; }

if [ -n "$WAIT_FOR" ]; then until grep -q "$WAIT_FOR" "$RUN/STATUS"; do sleep 60; done; fi
status "cámara YOLO $MATCH ($MODE, θ=$TH, $METHOD, suavizado ${SMOOTH}s, tag '$TAG'): informe de detección + pista"
[ -f "$RUN/report_$MATCH.txt" ] || python -u "$EXP/panorama_ball_eval.py" report "$MATCH" > "$RUN/report_$MATCH.txt" 2>&1 || status "el informe de detección falló"
for h in 1st 2nd; do
    python -u "$EXP/panorama_ball_eval.py" track "$MATCH" "$h" "$MODE" --th "$TH" --method "$METHOD" --tag "$TAG" >> "$RUN/track_$MATCH.log" 2>&1 || { status "track $h falló"; continue; }
    out="$V/${h}_vcam796yolo$TAG.mp4"
    [ -f "$out" ] || nice -n 10 python -u "$VEXP/panorama_vcam.py" "$MATCH" "$h" 0 -1 --size 796x448 --smooth-s "$SMOOTH" \
        --camera "$RUN/${MATCH}_${h}_${MODE}_cam$TAG.npz" --out "$out" > "$RUN/render_${MATCH}_${h}_yolo$TAG.log" 2>&1 || status "render $h falló"
done
status "cámara YOLO $MATCH$TAG: render hecho"

videos=()
for h in 1st 2nd; do [ -f "$V/${h}_vcam796yolo$TAG.mp4" ] && videos+=("/app/$V/${h}_vcam796yolo$TAG.mp4"); done
for attempt in 1 2 3 4 5 6; do
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$MIN_FREE_MB" ]; do sleep 60; done
    (cd models/tdeed/T-DEED && python -u "/app/$VEXP/tdeed_scan_heads.py" SoccerNetBall_challenge2 "${videos[@]}" \
        --out-tag _views --batch-size 1) >> output/panorama_views/scan.log 2>&1 && break
    status "escaneo cortado (intento $attempt/6), reintento en 10 min"; sleep 600
done
python -u "$VEXP/panorama_views_eval.py" "$MATCH" --views vcam796 vcam796yolo "vcam796yolo$TAG" > "output/panorama_views/report_${MATCH}_yolo_cam$TAG.txt" 2>&1 \
    || status "el informe de vistas falló"
status "FIN cámara YOLO $MATCH$TAG"
