#!/usr/bin/env bash
# Afinado de T-DEED con nuestra cámara virtual, desatendido:
#   docker exec -d -w /app Sports-Classification bash output/panorama_ball/experiments/run_tdeed_finetune.sh
# 1) espera a WAIT_FOR en output/panorama_ball/STATUS (los renders vcam796vl1 de los partidos de train);
# 2) frames + etiquetas + splits (tdeed_finetune_data.py; reanudable); 3) indexa los clips (--store);
# 4) entrena NAME desde el checkpoint SoccerNetBall_challenge2 (mejor mAP en validación 117093+132877); si el proceso
#    se corta (p. ej. OOM porque otro proceso ocupa la GPU) espera GPU libre y reanuda desde last.pt (hasta ATTEMPTS veces);
# 5) escanea con NAME las vistas vcam796 y vcam796vl1 de validación y saca los informes por clase y de tiros.
# Variables: NAME (SNB_pano1), TRAIN, VAL, EPOCHS (8), LR (1e-4), EPOCH_FRAMES (200000), BATCH (2), ACC_GRAD (2),
# WORKERS (8), EXTRA (p. ej. "--freeze-backbone --no-mixup"), WAIT_FOR ("FIN TRAIN de día"; vacío = no esperar),
# MIN_FREE_MB (20000), ATTEMPTS (4). Prueba de humo 2026-10-11 (lote 2 x acc 2 = 1 clip por pasada): pico 18,7 GB de VRAM,
# ~1,2 s por clip de 100 frames y 1,5-4 min por mitad de validación al medir mAP; con 2 clips por pasada (ACC_GRAD 1) no hay
# medida: esa prueba murió por OOM con la GPU compartida. 2026-10-11: el primer intento real murió por OOM en el paso 1
# porque otro proceso tomó 8 GB de la GPU mientras se evaluaba el modelo base; de ahí los reintentos.
set -u
cd /app
NAME="${NAME:-SNB_pano1}"; TRAIN="${TRAIN:-118575 118576 118577 118578 128058}"; VAL="${VAL:-117093 132877}"
EPOCHS="${EPOCHS:-8}"; LR="${LR:-1e-4}"; EPOCH_FRAMES="${EPOCH_FRAMES:-200000}"; BATCH="${BATCH:-2}"; ACC_GRAD="${ACC_GRAD:-2}"
WORKERS="${WORKERS:-8}"; EXTRA="${EXTRA:-}"; WAIT_FOR="${WAIT_FOR-FIN TRAIN de día}"; MIN_FREE_MB="${MIN_FREE_MB:-20000}"
ATTEMPTS="${ATTEMPTS:-4}"
RUN=/app/output/panorama_ball; LOG=/app/output/tdeed_finetune; mkdir -p "$LOG"   # absolutas: más abajo se hace cd al repo
export PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
status() { echo "$(date '+%F %T') $*" >> "$RUN/STATUS"; }
gpu_wait() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$MIN_FREE_MB" ]; do sleep 60; done; }

if [ -n "$WAIT_FOR" ]; then until grep -q "$WAIT_FOR" "$RUN/STATUS"; do sleep 60; done; fi
status "afinado T-DEED $NAME: datos (frames + etiquetas) de train [$TRAIN] y val [$VAL]"
python -u scripts/shot_benchmark/tdeed_finetune_data.py --train $TRAIN --val $VAL --jobs 4 > "$LOG/data_$NAME.log" 2>&1 \
    || { status "afinado $NAME: FALLÓ la preparación de datos"; exit 1; }
cd models/tdeed/T-DEED
status "afinado $NAME: indexando clips"
python -u /app/scripts/shot_benchmark/tdeed_finetune.py "$NAME" --store > "$LOG/store_$NAME.log" 2>&1 \
    || { status "afinado $NAME: FALLÓ el indexado"; exit 1; }
ok=0
for attempt in $(seq 1 "$ATTEMPTS"); do
    gpu_wait
    status "afinado $NAME: entrenando (intento $attempt/$ATTEMPTS; $EPOCHS épocas, lr $LR, $EPOCH_FRAMES frames/época, lote $BATCH x $ACC_GRAD $EXTRA)"
    python -u /app/scripts/shot_benchmark/tdeed_finetune.py "$NAME" --epochs "$EPOCHS" --lr "$LR" --epoch-frames "$EPOCH_FRAMES" \
        --batch-size "$BATCH" --acc-grad "$ACC_GRAD" --workers "$WORKERS" --resume $EXTRA >> "$LOG/train_$NAME.log" 2>&1 && { ok=1; break; }
    status "afinado $NAME: entrenamiento cortado (intento $attempt/$ATTEMPTS, ver $LOG/train_$NAME.log); reanudación en 5 min"
    sleep 300
done
[ "$ok" = 1 ] || { status "afinado $NAME: FALLÓ el entrenamiento tras $ATTEMPTS intentos"; exit 1; }
status "afinado $NAME: entrenado: $(grep '^FIN' "$LOG/train_$NAME.log" | tail -1)"

videos=()
for m in $VAL; do for h in 1st 2nd; do for v in vcam796 vcam796vl1; do
    f="/app/datasets/soccertrack-v2/views/$m/${h}_$v.mp4"; [ -f "$f" ] && videos+=("$f")
done; done; done
status "afinado $NAME: escaneo de validación (${#videos[@]} vídeos)"
for attempt in 1 2 3; do
    gpu_wait
    python -u /app/output/panorama_views/experiments/tdeed_scan_heads.py "$NAME" "${videos[@]}" --out-tag _views --batch-size 1 \
        >> "$LOG/scan_$NAME.log" 2>&1 && break
    status "afinado $NAME: escaneo cortado (intento $attempt/3), reintento en 5 min"; sleep 300
done
cd /app
TDEED_MODEL="$NAME" python -u scripts/shot_benchmark/panorama_views_eval.py $VAL --views vcam796 vcam796vl1 > "output/panorama_views/report_${NAME}_val.txt" 2>&1 \
    || status "afinado $NAME: el informe de tiros falló"
TDEED_MODEL="$NAME" python -u scripts/shot_benchmark/panorama_class_eval.py $VAL --views vcam796 vcam796vl1 > "output/panorama_views/report_${NAME}_classes_val.txt" 2>&1 \
    || status "afinado $NAME: el informe por clases falló"
status "FIN afinado $NAME: output/panorama_views/report_${NAME}_val.txt y report_${NAME}_classes_val.txt"
