#!/usr/bin/env bash
# Comparativa de variantes del VLM (Qwen3.5-9B en :8010) sobre el banco de 150 candidatos.
#   docker exec -d -w /app Sports-Classification bash output/shot_benchmark/experiments/run_vlm_bench.sh
set -u; cd /app
OUT=output/shot_benchmark; export PYTHONPATH=/app
status() { echo "$(date '+%F %T') $*" >> "$OUT/STATUS"; }
for v in frames_json video_json frames_think video_think; do
    status "vlm_bench: $v"
    python -u "$OUT/experiments/vlm_bench.py" "$v" --workers 6 >> "$OUT/vlm_bench/$v.log" 2>&1 || status "vlm_bench $v falló"
done
python "$OUT/experiments/vlm_bench.py" report > "$OUT/vlm_bench/report_9b.txt" 2>&1
status "FIN vlm_bench 9B"
