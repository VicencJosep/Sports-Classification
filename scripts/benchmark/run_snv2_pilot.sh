#!/usr/bin/env bash
# Piloto del test oficial de SoccerNet-v2 (17 eventos) para la cabeza SoccerNet de T-DEED, limpia para todos los
# checkpoints (preentrenaron solo con train+val). Para cada partido de la lista: espera a que estén las dos mitades
# (las copia otro proceso desde /mnt, solo visible en el host), puntúa con `core.ball_spotting.TDEEDSpotter`
# (caché <mitad>.tdeed.<modelo>.npz junto al vídeo, con probs_sn) y al final evalúa con `eval_snv2.py` (tight + loose).
# Reanudable: salta cachés hechas. Uso (host): bash scripts/benchmark/run_snv2_pilot.sh [modelos...]
set -u -o pipefail
cd "$(dirname "$0")/../.."
LIST=output/benchmark/sn_v2_pilot20.txt
OUT=output/benchmark/snv2
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(SoccerNetBall_test SoccerNetBall_challenge2)
dx() { docker exec -w /app Sports-Classification "$@"; }
dxi() { docker exec -i -w /app Sports-Classification "$@"; }  # con stdin (para tee)
dx mkdir -p "$OUT"
LOG=$OUT/run_pilot.log; STATUS=$OUT/STATUS
st() { echo "$(date '+%F %T') $*" | dxi tee -a "$STATUS" >/dev/null; }
st "inicio: modelos [${MODELS[*]}], lista $LIST"
for m in "${MODELS[@]}"; do
  while IFS= read -r g; do
    [ -z "$g" ] && continue
    for h in 1 2; do
      v="datasets/SoccerNet/$g/${h}_720p.mkv"
      c="datasets/SoccerNet/$g/${h}_720p.tdeed.$m.npz"
      dx test -s "$c" && continue
      n=0
      until dx test -s "$v" && ! dx test -e "$v.part"; do sleep 60; n=$((n+1)); [ $n -eq 1 ] && st "esperando copia de $g/$h"; done
      st "puntuando $m $g/$h"
      dx python -c "import sys; from pathlib import Path; from core.ball_spotting import TDEEDSpotter; TDEEDSpotter(sys.argv[1]).ensure_scores(Path(sys.argv[2]))" "$m" "$v" 2>&1 | dxi tee -a "$LOG" >/dev/null || st "FALLÓ $m $g/$h"
    done
  done < "$LIST"
  st "evaluando $m"
  dx python -m scripts.benchmark.eval_snv2 --model "$m" --games "$LIST" --tag pilot20 --metrics tight loose 2>&1 | dxi tee -a "$LOG" >/dev/null || st "FALLÓ eval $m"
done
st "fin"
