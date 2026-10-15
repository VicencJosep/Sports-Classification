#!/usr/bin/env bash
# Experimentos de balón desatendidos (11-oct-2026), en orden y sin pisarse en la GPU compartida:
#   1. zoom_test.py: ¿mejora el balón a 1920/2560 px? (techo de una cámara virtual)   [GPU ligera, minutos]
#   2. distance_vs_ball.py sobre Mallorca y Espanyol: ¿falla el balón cuando los jugadores están lejos?  [CPU]
#   3. POST /analysis del partido "Espanyol · balón a 1280 px" (vídeo enlazado, tracking nuevo)  [GPU, ~3-4 h]
#   4. prepare_soccernet_tracking_ball.py: dataset YOLO desde SoccerNet Tracking  [CPU, en paralelo]
#   5. al acabar 3: distance_vs_ball.py --compare (posesión 640 vs 1280 del Espanyol)
#   6. train_ball_yolo.py cuando la GPU tenga sitio, y eval_ball_detectors.py
#   7. REPORT.md con todo
# Reanudable: cada paso se salta si su salida ya existe. Estado en output/ball_experiments/STATUS.
#   docker exec -d -w /app Sports-Classification bash scripts/ball_detector/run_unattended.sh
set -u
cd /app
export PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
PY=/opt/venv/bin/python
OUT=output/ball_experiments
mkdir -p "$OUT"
STATUS=$OUT/STATUS
status() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
jget() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }

IDS=$OUT/espanyol_1280.json
MATCH_1280=$(jget $IDS match_1280); VIDEO_1280=$(jget $IDS video_1280); VIDEO_OLD=$(jget $IDS video_old); MALLORCA=$(jget $IDS mallorca_video)
V_OLD=backend/storage/videos/$VIDEO_OLD; V_NEW=backend/storage/videos/$VIDEO_1280; V_MAL=backend/storage/videos/$MALLORCA
RUN_JSON=backend/storage/outputs/$MATCH_1280/full/run.json

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0; }
wait_gpu() {  # wait_gpu <MiB libres necesarios> <tope s>
    local need=$1 limit=$2 t=0 used free
    while :; do
        used=$(gpu_used); free=$((24564 - used))
        [ "$free" -ge "$need" ] && return 0
        [ "$t" -ge "$limit" ] && { status "GPU: solo $free MiB libres tras ${limit}s de espera, sigo igualmente"; return 1; }
        sleep 60; t=$((t + 60))
    done
}
run_status() { grep -o '"status": "[a-z]*"' "$RUN_JSON" 2>/dev/null | head -1 | cut -d'"' -f4; }

status "=== inicio (match 1280 = $MATCH_1280, vídeo = $VIDEO_1280) ==="

# 1. Zoom test (antes de ocupar la GPU con el pipeline)
if [ ! -f $OUT/zoom_test/result_espanyol.json ]; then
    wait_gpu 5000 1800
    $PY scripts/ball_detector/zoom_test.py --video $V_MAL/source.mp4 --start-frame 15000 --n 1000 --out $OUT/zoom_test/result_mallorca.json > $OUT/zoom_test_mallorca.log 2>&1 \
        && status "1. zoom test Mallorca hecho" || status "1. zoom test Mallorca FALLÓ (ver zoom_test_mallorca.log)"
    $PY scripts/ball_detector/zoom_test.py --video $V_OLD/source.mp4 --start-frame 60000 --n 1000 --out $OUT/zoom_test/result_espanyol.json > $OUT/zoom_test_espanyol.log 2>&1 \
        && status "1. zoom test Espanyol hecho" || status "1. zoom test Espanyol FALLÓ (ver zoom_test_espanyol.log)"
else
    status "1. zoom test ya hecho"
fi

# 2. Distancia a cámara vs balón (CPU, en paralelo)
(
    [ -f $OUT/distance_vs_ball_mallorca.txt ] || $PY scripts/ball_detector/distance_vs_ball.py --possession $V_MAL/source.possession.jsonl --fps 25 --out $OUT/distance_vs_ball_mallorca.txt > $OUT/distance_mallorca.log 2>&1
    [ -f $OUT/distance_vs_ball_espanyol.txt ] || $PY scripts/ball_detector/distance_vs_ball.py --possession $V_OLD/source.possession.jsonl --fps 30 --out $OUT/distance_vs_ball_espanyol.txt > $OUT/distance_espanyol.log 2>&1
    status "2. distancia vs balón (Mallorca, Espanyol) hecho"
) &
DIST_PID=$!

# 3. Re-análisis del Espanyol con el balón a 1280 px (el backend lanza main.py con el código nuevo)
case "$(run_status)" in
    completed) status "3. análisis $MATCH_1280 ya completado" ;;
    processing|pending) status "3. análisis $MATCH_1280 ya en marcha ($(run_status))" ;;
    *)
        # (el contenedor no tiene curl: urllib)
        code=$($PY -c "import json,sys,urllib.request as u
req=u.Request('http://127.0.0.1:8100/api/matches/$MATCH_1280/analysis', data=b'{\"mode\":\"full-match\"}', headers={'Content-Type':'application/json'}, method='POST')
try:
    r=u.urlopen(req, timeout=30); open('$OUT/start_analysis.json','wb').write(r.read()); print(r.status)
except Exception as e: print('ERROR', e)")
        status "3. POST /api/matches/$MATCH_1280/analysis -> HTTP $code"
        ;;
esac

# 4. Dataset SoccerNet Tracking (CPU, en paralelo con el pipeline)
if [ ! -f datasets/soccernet_tracking_ball/data.yaml ]; then
    ( nice -n 5 $PY scripts/ball_detector/prepare_soccernet_tracking_ball.py > $OUT/prepare_dataset.log 2>&1 \
        && status "4. dataset SoccerNet Tracking listo" || status "4. dataset FALLÓ (ver prepare_dataset.log)" ) &
    PREP_PID=$!
else
    status "4. dataset ya existe"; PREP_PID=""
fi

# 5. Esperar al análisis (tope 9 h) y comparar cobertura 640 vs 1280
t=0
while :; do
    st=$(run_status)
    case "$st" in completed|error) break ;; esac
    [ "$t" -ge 32400 ] && { status "5. tope de espera del análisis alcanzado (estado: ${st:-sin run.json})"; break; }
    sleep 120; t=$((t + 120))
done
status "5. análisis Espanyol 1280 terminado con estado: ${st:-desconocido}"
wait $DIST_PID 2>/dev/null
if [ -f $V_NEW/source.possession.jsonl ]; then
    $PY scripts/ball_detector/distance_vs_ball.py --possession $V_OLD/source.possession.jsonl --compare $V_NEW/source.possession.jsonl --fps 30 --out $OUT/espanyol_640_vs_1280.txt > $OUT/compare.log 2>&1 \
        && status "5. comparación 640 vs 1280 hecha" || status "5. comparación FALLÓ (ver compare.log)"
else
    status "5. sin possession.jsonl nuevo: no hay comparación (¿falló el tracking? mira pipeline.log del partido)"
fi

# 6. Entrenamiento + evaluación
[ -n "${PREP_PID}" ] && wait $PREP_PID 2>/dev/null
if [ -f models/ball/yolo11s_snt_ball.pt ]; then
    status "6. entrenamiento ya hecho (models/ball/yolo11s_snt_ball.pt existe)"
elif [ -f datasets/soccernet_tracking_ball/data.yaml ]; then
    wait_gpu 7000 21600
    status "6. entrenamiento: inicio (GPU usada: $(gpu_used) MiB)"
    $PY scripts/ball_detector/train_ball_yolo.py > $OUT/train.log 2>&1 && status "6. entrenamiento hecho" || status "6. entrenamiento FALLÓ (ver train.log)"
    wait_gpu 5000 3600
    $PY scripts/ball_detector/eval_ball_detectors.py --out $OUT/eval \
        --model coco_yolo11x_640=yolo11x.pt:32:640 --model coco_yolo11x_1280=yolo11x.pt:32:1280 \
        --model snt_yolo11s_1280=models/ball/yolo11s_snt_ball.pt:0:1280 \
        --slice $V_MAL/source.mp4:15000 --slice $V_OLD/source.mp4:60000 > $OUT/eval.log 2>&1 \
        && status "6. evaluación hecha" || status "6. evaluación FALLÓ (ver eval.log)"
else
    status "6. sin dataset: no se entrena"
fi
if [ -f models/ball/yolo11s_snt_ball.pt ] && [ ! -f $OUT/eval.txt ]; then
    wait_gpu 5000 3600
    $PY scripts/ball_detector/eval_ball_detectors.py --out $OUT/eval \
        --model coco_yolo11x_640=yolo11x.pt:32:640 --model coco_yolo11x_1280=yolo11x.pt:32:1280 \
        --model snt_yolo11s_1280=models/ball/yolo11s_snt_ball.pt:0:1280 > $OUT/eval.log 2>&1 \
        && status "6. evaluación hecha" || status "6. evaluación FALLÓ (ver eval.log)"
fi

# 7. Informe
{
    echo "# Experimentos de balón (desatendidos, $(date '+%F'))"
    echo; echo "## 1. Zoom: ¿ayudaría una cámara virtual? (YOLO11x COCO, frame entero reescalado)"
    for f in $OUT/zoom_test/result_mallorca.json $OUT/zoom_test/result_espanyol.json; do [ -f $f ] && { echo; echo '```'; cat $f; echo '```'; }; done
    echo; echo "## 2. ¿Falla el balón cuando los jugadores están lejos? (tracking original, balón a 640 px)"
    for f in $OUT/distance_vs_ball_mallorca.txt $OUT/distance_vs_ball_espanyol.txt; do [ -f $f ] && { echo; echo "### $(basename $f)"; echo '```'; cat $f; echo '```'; }; done
    echo; echo "## 3. Espanyol: balón a 640 px (A) frente a 1280 px (B), mismo vídeo"
    [ -f $OUT/espanyol_640_vs_1280.txt ] && { echo '```'; cat $OUT/espanyol_640_vs_1280.txt; echo '```'; } || echo "(sin comparación: ver STATUS)"
    echo; echo "## 4. Detector ajustado con SoccerNet Tracking"
    [ -f datasets/soccernet_tracking_ball/stats.json ] && { echo '```'; cat datasets/soccernet_tracking_ball/stats.json; echo '```'; }
    [ -f $OUT/runs/yolo11s_snt_ball/results.csv ] && { echo; echo "Últimas épocas (results.csv):"; echo '```'; head -1 $OUT/runs/yolo11s_snt_ball/results.csv | cut -c1-160; tail -3 $OUT/runs/yolo11s_snt_ball/results.csv | cut -c1-160; echo '```'; }
    [ -f $OUT/eval.txt ] && { echo; echo '```'; cat $OUT/eval.txt; echo '```'; }
    echo; echo "## Estado"; echo '```'; cat $STATUS; echo '```'
} > $OUT/REPORT.md
status "=== FIN: informe en $OUT/REPORT.md ==="
