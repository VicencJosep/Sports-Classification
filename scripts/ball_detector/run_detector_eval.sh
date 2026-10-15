#!/usr/bin/env bash
# Detector de balón entrenado (yolo11s_pano) sobre la panorámica completa de los partidos de validación, desatendido:
#   docker exec -d -w /app Sports-Classification bash output/ball_panorama/experiments/run_detector_eval.sh [MATCH...]
# Ventanas de 1024 px a resolución nativa, paso 5 frames (igual que run_panorama_ball.sh con YOLO11x), un proceso por
# partido. Salida: output/panorama_ball/<match>_<half>_tiles1024_pano.jsonl y output/ball_panorama/report_<match>.txt
# (tiles1024 = YOLO11x COCO de run_panorama_ball.sh si existe, tiles1024_pano = el nuevo). Luego la cámara -> T-DEED con
# MODE=tiles1024_pano en run_panorama_yolo_cam.sh. NUNCA 128057 ni 132831 hasta fijar umbral y pista.
set -u
cd /app
MATCHES=("$@"); [ ${#MATCHES[@]} -eq 0 ] && MATCHES=(117093 132877)
WEIGHTS="${WEIGHTS:-output/ball_panorama/runs/yolo11s_pano/weights/best.pt}"; TAG="${TAG:-_pano}"
RUN=output/ball_panorama; EVAL=$RUN/experiments/panorama_ball_eval.py
export PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1
status() { echo "$(date '+%F %T') $*" >> "$RUN/STATUS"; }

one_match() {
    local m=$1
    for h in 1st 2nd; do
        for attempt in 1 2 3; do
            if nice -n 5 python -u "$EVAL" detect "$m" "$h" --stride 5 --modes tiles1024 --weights "$WEIGHTS" --tag "$TAG" \
                >> "$RUN/detect_${m}_${h}.log" 2>&1; then
                status "detector $m $h: hecho"; break
            fi
            status "detector $m $h: cortado (intento $attempt/3), reintento en 5 min"; sleep 300
        done
    done
    python -u "$EVAL" report "$m" --modes tiles1024 "tiles1024$TAG" > "$RUN/report_$m.txt" 2>&1 || status "informe $m falló"
}

status "detector $WEIGHTS ($TAG) sobre ${MATCHES[*]}: inicio"
for m in "${MATCHES[@]}"; do one_match "$m" & done
wait
python -u "$EVAL" report "${MATCHES[@]}" --modes tiles1024 "tiles1024$TAG" > "$RUN/report_val.txt" 2>&1 || status "informe conjunto falló"
status "FIN detector sobre ${MATCHES[*]}"
