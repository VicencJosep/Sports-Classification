#!/usr/bin/env bash
# Cadena tras el verificador VLM (se lanza con docker exec -d, no depende de ninguna sesión):
# 1) espera "FIN verificador" en STATUS, 2) combinación V-JEPA2+VLM, 3) embeddings V-JEPA2
# de los 1.149 candidatos (reanudable por partido), 4) cabeza sobre embeddings (leave-one-match-out).
# Relanzar tras un corte: docker exec -d -w /app Sports-Classification bash output/shot_benchmark/experiments/run_after_verifier.sh
set -u
cd /app
OUT=output/shot_benchmark
EXP=$OUT/experiments
export PYTHONPATH=/app OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
status() { echo "$(date '+%F %T') $*" >> "$OUT/STATUS"; }

status "experimentos: esperando al verificador"
until grep -q "FIN verificador" "$OUT/STATUS"; do sleep 60; done

status "experimentos: combinación V-JEPA2 + VLM"
cat "$EXP/common.py" "$EXP/combine.py" | python - > "$OUT/combine_vlm.txt" 2>&1 \
    || status "combinación falló, ver $OUT/combine_vlm.txt"

status "experimentos: extracción de embeddings de candidatos"
python -u "$EXP/extract_cand_emb.py" >> "$OUT/extract_emb.log" 2>&1 \
    || status "extracción falló, ver $OUT/extract_emb.log"

if [ "$(ls "$OUT"/candidate_embeddings/*.npz 2>/dev/null | grep -vc '\.tmp\.npz')" -ge 7 ]; then
    status "experimentos: evaluación sobre embeddings"
    cat "$EXP/common.py" "$EXP/emb_eval.py" | python - > "$OUT/emb_eval.txt" 2>&1 \
        || status "evaluación de embeddings falló, ver $OUT/emb_eval.txt"
else
    status "faltan embeddings, no se evalúa"
fi
status "FIN experimentos"
