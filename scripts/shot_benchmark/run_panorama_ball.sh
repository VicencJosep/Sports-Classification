#!/usr/bin/env bash
# Detección de balón (YOLO11x COCO) sobre la panorámica completa de un partido de SoccerTrack v2, desatendida:
#   docker exec -d -w /app Sports-Classification bash output/panorama_ball/experiments/run_panorama_ball.sh [MATCH]
# MODES="pipeline640 tiles1024" (por defecto; "full" = frame entero a imgsz nativo, redundante con tiles y lento).
# Reanudable (el JSONL de cada modo guarda el último frame hecho). Luego: panorama_ball_eval.py report MATCH
set -u
cd /app
MATCH="${1:-117093}"
MODES="${MODES:-pipeline640 tiles1024}"
RUN=output/panorama_ball
export PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1
status() { echo "$(date '+%F %T') $*" >> "$RUN/STATUS"; }
status "balón $MATCH: inicio (paso 5 frames, modos $MODES)"
for h in 1st 2nd; do
    for attempt in 1 2 3; do
        if nice -n 5 python -u "$RUN/experiments/panorama_ball_eval.py" detect "$MATCH" "$h" --stride 5 --modes $MODES >> "$RUN/detect_${MATCH}_${h}.log" 2>&1; then
            status "balón $MATCH $h: hecho"; break
        fi
        status "balón $MATCH $h: cortado (intento $attempt/3), reintento en 5 min"; sleep 300
    done
done
status "FIN balón $MATCH"
