#!/usr/bin/env bash
# Ajuste de BoT-SORT (NMS previo, buffer de perdidos, ReID sin puerta de IoU) en 12
# secuencias del split de TRAIN de SoccerNet-Tracking (docs/comparativa_benchmark.md §7.10):
# detección híbrida (GPU) -> variantes de PlayerTracker -> TrackEval + id_analysis.
# Dentro del contenedor, desde la raíz del proyecto:
#   bash scripts/benchmark/run_snmot_train_tuning.sh
set -euo pipefail
SEQS=datasets/SoccerNet/tracking/extracted/train
B=output/benchmark/snmot_train
LIMIT=${LIMIT:-12}
mkdir -p "$B/tracks" "$B/results"
python -m scripts.benchmark.detect_mot --seqs "$SEQS" --detector hybrid --limit "$LIMIT" --out "$B/dets/hybrid"
# solo las secuencias con detecciones (las $LIMIT primeras): una carpeta con enlaces
SUB="$B/seqs"; mkdir -p "$SUB"
for d in $(ls "$B/dets/hybrid" | sed 's/\.txt$//'); do ln -sfn "/app/$SEQS/$d" "$SUB/$d"; done
VARIANTS="ours_botsort ours_botsort_nms ours_botsort_osnet ours_botsort_osnet_nms ours_botsort_osnet_nms_buf60 ours_botsort_osnet_nms_buf60_lost ours_botsort_osnet_nms_lost ours_botsort_osnet_nms_buf125_lost"
run() {
    TR=$1
    python -m scripts.benchmark.track_mot --seqs "$SUB" --dets "$B/dets/hybrid" --tracker "$TR" --out "$B/tracks/hybrid__$TR" \
      && python -m scripts.benchmark.eval_mot --seqs "$SUB" --tracks "$B/tracks/hybrid__$TR" --mode persons --out "$B/results/hybrid__$TR.persons.json"
}
N=0
for TR in $VARIANTS; do
    run "$TR" > "$B/run_hybrid__$TR.log" 2>&1 &
    N=$((N + 1)); if [ $((N % 4)) -eq 0 ]; then wait; fi
done
wait
python -m scripts.benchmark.id_analysis --seqs "$SUB" --tracks $(for TR in $VARIANTS; do echo "$B/tracks/hybrid__$TR"; done) \
  --gap 5 --window 10 --out "$B/results/id_analysis_gap5_w10.json" > "$B/id_analysis.txt" 2>&1
echo done
