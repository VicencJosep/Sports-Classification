#!/usr/bin/env bash
# Reintento del Omni-30B como verificador sobre los 447 candidatos limpios (+ diagnóstico con razonamiento).
# El Omni reserva el 95 % de la GPU: espera a que termine la prueba panorámica y a que la GPU esté vacía.
set -u
cd /media/dd3/ner-training/Sports-Classification
OUT=output/shot_benchmark
inc() { docker exec -w /app -e PYTHONPATH=/app -e VLM_BENCH_DIR=$OUT/vlm_clean Sports-Classification bash -c "$1"; }
status() { inc "echo \"\$(date '+%F %T') $*\" >> $OUT/STATUS"; }
until grep -q "FIN panorama$" "$OUT/STATUS" && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 60; done
status "omni_clean: arrancando Omni-30B"
./scripts/serve_events_omni.sh --detach >/dev/null
for i in $(seq 1 180); do curl -sf http://localhost:8002/v1/models >/dev/null && break; [ "$(docker inspect -f '{{.State.Running}}' vllm-events-omni)" != "true" ] && break; sleep 5; done
if curl -sf http://localhost:8002/v1/models >/dev/null; then
    status "omni_clean: omni_video (447 candidatos)"
    inc "python -u $OUT/experiments/vlm_bench.py omni_video --workers 2 >> $OUT/vlm_clean/omni_video.log 2>&1" || status "omni_clean falló"
    status "omni_clean: diagnóstico con razonamiento"
    inc "python -u $OUT/experiments/omni_diag.py > $OUT/vlm_clean/omni_diag.log 2>&1" || status "omni_diag falló"
else
    status "omni_clean: Omni no arrancó (ver docker logs vllm-events-omni)"
fi
docker stop vllm-events-omni >/dev/null 2>&1
status "FIN omni_clean"
