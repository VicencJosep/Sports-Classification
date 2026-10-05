#!/usr/bin/env bash
# VideoARM (Qwen3.5-9B) y Qwen3-Omni-30B como VERIFICADORES sobre los 447 candidatos de V-JEPA2 de los 3 partidos
# EFL limpios (output/shot_benchmark/vlm_clean/candidates.json), para la tabla de partido completo.
# Desde el HOST (necesita docker); espera a que acabe el render de demo (usa la GPU) antes de arrancar vLLM.
set -u
cd /media/dd3/ner-training/Sports-Classification
OUT=output/shot_benchmark
inc() { docker exec -w /app -e PYTHONPATH=/app -e VLM_BENCH_DIR=$OUT/vlm_clean Sports-Classification bash -c "$1"; }
status() { inc "echo \"\$(date '+%F %T') $*\" >> $OUT/STATUS"; }
run() { inc "python -u $OUT/experiments/vlm_bench.py $1 --workers $2 >> $OUT/vlm_clean/$1.log 2>&1" || status "vlm_clean $1 falló"; }
wait_ready() { for i in $(seq 1 180); do curl -sf "http://localhost:$1/v1/models" >/dev/null && return 0; [ "$(docker inspect -f '{{.State.Running}}' "$2")" != "true" ] && return 1; sleep 5; done; return 1; }

until grep -q "FIN render demo" "$OUT/STATUS"; do sleep 60; done
status "vlm_clean: arrancando 9B para VideoARM"
GPU_MEMORY_UTILIZATION=0.70 LIMIT_MM_PER_PROMPT='{"image": 16, "video": 1}' ./scripts/serve_events_vllm.sh --detach >/dev/null
if wait_ready 8010 vllm-events; then status "vlm_clean: videoarm (447 candidatos)"; run videoarm 1; else status "el 9B no arrancó"; fi
docker stop vllm-events >/dev/null
./scripts/serve_events_omni.sh --detach >/dev/null
if wait_ready 8002 vllm-events-omni; then status "vlm_clean: omni_video"; run omni_video 2; else status "Omni no arrancó"; fi
docker stop vllm-events-omni >/dev/null
status "FIN vlm_clean"
