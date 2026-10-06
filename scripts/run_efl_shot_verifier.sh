#!/usr/bin/env bash
# Segunda etapa del benchmark de tiros: verificador VLM sobre los candidatos
# de V-JEPA2 en los 7 EFL (cachés de run_efl_shot_benchmark.sh) y evaluación
# con --verdicts, todo a ficheros en output/shot_benchmark/.
#
# Requiere el servidor vLLM (scripts/serve_events_vllm.sh --detach) y un
# contenedor con --network host: desde la red bridge el cortafuegos del host
# corta bridge -> host:8010 (172.17.0.1:8010 da timeout). Desde el HOST:
#
#   docker exec -d -w /app Sports-Classification bash scripts/run_efl_shot_verifier.sh
#
# (Sports-Classification se recreó el 2026-09-29 con --network host; si vuelve
# a estar en bridge, usar uno efímero: docker run -d --rm --network host
# -v "$PWD:/app" -w /app sports-analytics bash scripts/run_efl_shot_verifier.sh)
#
# Un proceso por partido en paralelo (vLLM agrupa las peticiones). Relanzarlo
# es seguro: cada partido solo verifica los candidatos que no tengan veredicto.

set -u
shopt -s nullglob
cd "$(dirname "$0")/.."

OUT=output/shot_benchmark
EFL=datasets/SoccerNet/england_efl/2019-2020
CACHE_NAME=720p.event_windows.vjepa2.w6s3.jsonl
MIN_SCORE="${MIN_SCORE:-0.02}"
BASE_URL="${BASE_URL:-http://localhost:8010/v1}"
mkdir -p "$OUT"

status() { echo "$(date '+%F %T') $*" >> "$OUT/STATUS"; }

python -c "import openai" 2>/dev/null || pip install -q openai

caches=("$EFL"/*/"$CACHE_NAME")
status "verificador: ${#caches[@]} partidos, θ=$MIN_SCORE"
for i in "${!caches[@]}"; do
    python -m scripts.verify_shot_candidates --min-score "$MIN_SCORE" --base-url "$BASE_URL" \
        --cache "${caches[$i]}" > "$OUT/verify_$i.log" 2>&1 &
done
wait
status "verificador terminado (errores: $(grep -c 'Error verificando' "$OUT"/verify_*.log | awk -F: '{s+=$2} END {print s}'))"

if python -m scripts.evaluate_spotting --cache "${caches[@]}" --verdicts --list-missed \
        --json-out "$OUT/vjepa2_vlm_efl.json" > "$OUT/vjepa2_vlm_efl.txt" 2>&1; then
    status "evaluación escrita en $OUT/vjepa2_vlm_efl.txt"
else
    status "la evaluación falló, ver $OUT/vjepa2_vlm_efl.txt"
fi
status "FIN verificador"
