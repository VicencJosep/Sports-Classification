#!/usr/bin/env bash
# Prueba "sin recorte": embeddings V-JEPA2 de los 1.149 candidatos con el frame entero
# (squash 256x256 y wide 256x448) y la misma cabeza leave-one-match-out que con el recorte.
# Desatendido y reanudable:
#   docker exec -d -w /app Sports-Classification bash output/shot_benchmark/experiments/run_nocrop.sh
set -u
cd /app
OUT=output/shot_benchmark
EXP=$OUT/experiments
export PYTHONPATH=/app OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
status() { echo "$(date '+%F %T') $*" >> "$OUT/STATUS"; }

for v in squash wide; do
    status "sin recorte: extracción $v"
    python -u - "$v" < "$EXP/extract_cand_emb_nocrop.py" >> "$OUT/extract_emb_$v.log" 2>&1 \
        || status "extracción $v falló, ver $OUT/extract_emb_$v.log"
done

n_ok=0
for v in squash wide; do
    [ "$(ls "$OUT/candidate_embeddings_$v"/*.npz 2>/dev/null | grep -vc '\.tmp\.npz')" -ge 7 ] && n_ok=$((n_ok + 1))
done
if [ "$n_ok" -eq 2 ]; then
    status "sin recorte: evaluación"
    cat "$EXP/common.py" "$EXP/emb_eval_nocrop.py" | python - > "$OUT/emb_eval_nocrop.txt" 2>&1 \
        || status "evaluación sin recorte falló, ver $OUT/emb_eval_nocrop.txt"
else
    status "faltan embeddings sin recorte, no se evalúa"
fi
status "FIN sin recorte"
