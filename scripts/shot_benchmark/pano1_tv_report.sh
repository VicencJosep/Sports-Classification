#!/usr/bin/env bash
# Informe del experimento §3.6 (tras run_pano1_tv.sh). Dentro del contenedor, cwd /app:
#   docker exec -w /app Sports-Classification bash scripts/shot_benchmark/pano1_tv_report.sh
# Produce en output/pano1_tv/:
#   tdeed_eval_<modelo>.txt   AP de tiros en Middlesbrough completo (25 tiros), base y afinado
#   telemetry_<modelo>.json   telemetría del recorte de 20 min regenerada desde la caché (respot, umbrales auto)
#   actions_<modelo>.txt      acciones/eventos/equipo contra ground_truth/ (evaluate_match), afinado con delta frente a base
#   goals.txt                 P(SHOT), P(GOAL) y picos en juveniles y Clásico, base frente a afinado
set -u
cd /app
OUT=/app/output/pano1_tv
mkdir -p "$OUT"
FINE=SNB_pano1
BASE=SoccerNetBall_challenge2
SCAN="output/shot_benchmark/tdeed/${FINE}_heads_tv"
GT_VIDEO=ground_truth/2019-10-01_Middlesbrough-Preston_North_End-720p-trim.mp4
GT_JSON=ground_truth/2019-10-01_Middlesbrough-Preston_North_End.gt.json

echo "== 1. Tiros en TV (Middlesbrough completo; tdeed_eval.py, ±3 s y ±1 s)"
for model in "$BASE" "${FINE}_heads_tv"; do
    cat scripts/shot_benchmark/common.py scripts/shot_benchmark/tdeed_eval.py | python - "$model" > "$OUT/tdeed_eval_${model}.txt" 2>&1
    echo "-- $model"; grep -i "middles\|AP\|clean\|limpio" "$OUT/tdeed_eval_${model}.txt" | head -12
done

echo "== 2. Acciones y eventos en el recorte de 20 min (respot con umbrales automáticos, etiqueta del spotter)"
src="$SCAN/ground_truth [2019-10-01_Middlesbrough-Preston_North_End-720p-trim].npz"
dst="ground_truth/2019-10-01_Middlesbrough-Preston_North_End-720p-trim.tdeed.${FINE}.npz"
if [ -f "$src" ]; then
    cp -f "$src" "$dst"
else
    echo "falta $src"; exit 1
fi
for model in "$BASE" "$FINE"; do
    python -m scripts.respot_telemetry --source "$GT_VIDEO" --model-name "$model" --thresholds auto \
        --output "$OUT/telemetry_${model}.json" > "$OUT/respot_${model}.log" 2>&1 || { echo "respot $model falló"; tail -5 "$OUT/respot_${model}.log"; }
done
python -m scripts.evaluate_match --gt "$GT_JSON" --telemetry "$OUT/telemetry_${BASE}.json" \
    --teams backend/storage/outputs/6565edb583a4a434/full/output/teams.json > "$OUT/actions_${BASE}.txt" 2>&1
python -m scripts.evaluate_match --gt "$GT_JSON" --telemetry "$OUT/telemetry_${FINE}.json" \
    --teams backend/storage/outputs/6565edb583a4a434/full/output/teams.json \
    --compare "$OUT/telemetry_${BASE}.json" > "$OUT/actions_${FINE}.txt" 2>&1
grep -A6 "ACCIONES" "$OUT/actions_${FINE}.txt"; grep -A8 "EVENTOS" "$OUT/actions_${FINE}.txt"

echo "== 3. Goles y picos en juveniles y Clásico"
PYTHONPATH=/app python scripts/shot_benchmark/pano1_tv_report.py | tee "$OUT/goals.txt"
