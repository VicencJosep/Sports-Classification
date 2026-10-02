#!/usr/bin/env bash
# Parte 2 de la comparativa: VideoARM (sobre el 9B en :8010) y luego Qwen3-Omni-30B (:8002, GPU entera).
# Se lanza desde el HOST (necesita docker). Todo lo que escribe en output/ lo hace DENTRO del
# contenedor: output/ es de root y el usuario del host no puede escribir ahí.
set -u
cd /media/dd3/ner-training/Sports-Classification
OUT=output/shot_benchmark
inc() { docker exec -w /app -e PYTHONPATH=/app Sports-Classification bash -c "$1"; }
status() { inc "echo \"\$(date '+%F %T') $*\" >> $OUT/STATUS"; }
run() { inc "python -u $OUT/experiments/vlm_bench.py $1 --workers $2 >> $OUT/vlm_bench/$1.log 2>&1" || status "vlm_bench $1 falló"; }
wait_ready() {  # $1 puerto, $2 contenedor
    for i in $(seq 1 180); do
        curl -sf "http://localhost:$1/v1/models" >/dev/null && return 0
        [ "$(docker inspect -f '{{.State.Running}}' "$2")" != "true" ] && return 1
        sleep 5
    done
    return 1
}

if ! curl -sf http://localhost:8010/v1/models >/dev/null; then
    docker rm -f vllm-events-omni >/dev/null 2>&1
    GPU_MEMORY_UTILIZATION=0.70 LIMIT_MM_PER_PROMPT='{"image": 16, "video": 1}' ./scripts/serve_events_vllm.sh --detach >/dev/null
fi
if wait_ready 8010 vllm-events; then
    status "vlm_bench: videoarm"
    run videoarm 1
else
    status "el 9B no arrancó, se salta videoarm"
fi

status "vlm_bench: parando 9B, arrancando Omni-30B"
docker stop vllm-events >/dev/null
./scripts/serve_events_omni.sh --detach >/dev/null
if wait_ready 8002 vllm-events-omni; then
    status "vlm_bench: omni_video"
    run omni_video 2
else
    status "Omni no arrancó"
fi
docker stop vllm-events-omni >/dev/null
inc "python $OUT/experiments/vlm_bench.py report > $OUT/vlm_bench/report_all.txt 2>&1"
status "FIN vlm_bench parte 2"
