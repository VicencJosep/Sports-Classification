#!/usr/bin/env bash
# Benchmark de tracking T1 (docs/comparativa_benchmark.md §5.2) sobre el test de
# SoccerNet-Tracking: detección (GPU, reanudable) -> nuestro tracker -> TrackEval.
# Dentro del contenedor, desde la raíz del proyecto:
#   bash scripts/benchmark/run_snmot.sh [detector]   # hybrid por defecto
set -euo pipefail

DET=${1:-hybrid}
SEQS=datasets/SoccerNet/tracking/extracted/test
B=output/benchmark/snmot_test

python -m scripts.benchmark.detect_mot --seqs "$SEQS" --detector "$DET" --out "$B/dets/$DET"
for TR in ours ul_bytetrack ul_botsort; do
    python -m scripts.benchmark.track_mot --seqs "$SEQS" --dets "$B/dets/$DET" --tracker "$TR" --out "$B/tracks/${DET}__$TR"
    python -m scripts.benchmark.eval_mot --seqs "$SEQS" --tracks "$B/tracks/${DET}__$TR" --mode persons \
        --out "$B/results/${DET}__$TR.persons.json"
done
python -m scripts.benchmark.eval_mot --seqs "$SEQS" --tracks "$B/tracks/${DET}__ours" --mode all \
    --out "$B/results/${DET}__ours.all.json"
