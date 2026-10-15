#!/usr/bin/env bash
# Pasa un checkpoint de T-DEED por las vistas de cámara virtual de unos partidos de SoccerTrack v2 y saca los informes
# de tiros y por clase del modelo y del base (SoccerNetBall_challenge2) con el mismo evaluador, para compararlos.
# Desatendido:
#   docker exec -d -w /app Sports-Classification bash output/panorama_ball/experiments/run_tdeed_eval.sh
# Variables: MODEL (SNB_pano1), MATCHES ("128057 132831" = evaluación, ver docs/particion_datasets.md),
# VIEWS ("vcam796 vcam796vl1"), TAG (eval: sufijo de los informes), BASE (SoccerNetBall_challenge2), MIN_FREE_MB (8000).
# Los vídeos ya escaneados por un modelo se saltan (tdeed_scan_heads.py). Escaneo con lote 1: ~3 min por mitad y vista.
# Salida en output/panorama_views/: report_<modelo>_<TAG>.txt (tiros), report_<modelo>_<TAG>_<partido>.txt y
# report_<modelo>_classes_<TAG>.txt (12 clases), para MODEL y BASE; progreso en output/panorama_ball/STATUS.
set -u
cd /app
MODEL="${MODEL:-SNB_pano1}"; MATCHES="${MATCHES:-128057 132831}"; VIEWS="${VIEWS:-vcam796 vcam796vl1}"; TAG="${TAG:-eval}"
BASE="${BASE:-SoccerNetBall_challenge2}"; MIN_FREE_MB="${MIN_FREE_MB:-8000}"
RUN=/app/output/panorama_ball; LOG=/app/output/tdeed_finetune; REP=/app/output/panorama_views; mkdir -p "$LOG"
export PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
status() { echo "$(date '+%F %T') $*" >> "$RUN/STATUS"; }
gpu_wait() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$MIN_FREE_MB" ]; do sleep 60; done; }

videos=()
for m in $MATCHES; do for h in 1st 2nd; do for v in $VIEWS; do
    f="/app/datasets/soccertrack-v2/views/$m/${h}_$v.mp4"
    if [ -f "$f" ]; then videos+=("$f"); else status "evaluación $MODEL: falta $f"; fi
done; done; done
status "evaluación T-DEED $MODEL y $BASE en [$MATCHES], vistas [$VIEWS]: escaneo (${#videos[@]} vídeos por modelo)"
cd models/tdeed/T-DEED
for model in "$MODEL" "$BASE"; do
    ok=0
    for attempt in 1 2 3; do
        gpu_wait
        python -u /app/output/panorama_views/experiments/tdeed_scan_heads.py "$model" "${videos[@]}" --out-tag _views --batch-size 1 \
            >> "$LOG/scan_${model}_$TAG.log" 2>&1 && { ok=1; break; }
        status "evaluación: escaneo de $model cortado (intento $attempt/3), reintento en 5 min"; sleep 300
    done
    [ "$ok" = 1 ] || { status "evaluación: FALLÓ el escaneo de $model (ver $LOG/scan_${model}_$TAG.log)"; exit 1; }
done
cd /app
for model in "$MODEL" "$BASE"; do
    TDEED_MODEL="$model" python -u scripts/shot_benchmark/panorama_views_eval.py $MATCHES --views $VIEWS \
        > "$REP/report_${model}_$TAG.txt" 2>&1 || status "evaluación: el informe de tiros de $model falló"
    for m in $MATCHES; do
        TDEED_MODEL="$model" python -u scripts/shot_benchmark/panorama_views_eval.py $m --views $VIEWS \
            > "$REP/report_${model}_${TAG}_$m.txt" 2>&1 || status "evaluación: el informe de tiros de $model en $m falló"
    done
    TDEED_MODEL="$model" python -u scripts/shot_benchmark/panorama_class_eval.py $MATCHES --views $VIEWS \
        > "$REP/report_${model}_classes_$TAG.txt" 2>&1 || status "evaluación: el informe por clases de $model falló"
done
status "FIN evaluación $MODEL en [$MATCHES]: $REP/report_${MODEL}_$TAG.txt, report_${MODEL}_classes_$TAG.txt (y los de $BASE)"
