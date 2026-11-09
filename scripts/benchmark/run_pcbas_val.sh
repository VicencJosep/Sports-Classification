#!/usr/bin/env bash
# Evaluación de los spotters sobre el split de validación de SN-PCBAS-2026 (FOOTPASS: game_18, game_24, game_47, 1080p,
# 2023/24; metraje nuevo para todos los modelos). Fase A (cabe junto al piloto de SN-v2): T-DEED SoccerNetBall_test y
# AdaSpot big/small. Fase B: espera a que el piloto de SN-v2 termine (GPU libre) y puntúa el 800MF de producción (challenge2).
# Tras cada modelo evalúa con eval_bas --dataset pcbas (at1, at2, tight) tal cual y con --merge-high-pass. Reanudable.
# Uso (host): bash scripts/benchmark/run_pcbas_val.sh
set -u -o pipefail
cd "$(dirname "$0")/../.."
VID=datasets/SoccerNet/pcbas-2026/videos
GAMES=(game_18 game_24 game_47)
OUT=output/benchmark/bas_pcbas
dx() { docker exec -w /app Sports-Classification "$@"; }
dxi() { docker exec -i -w /app Sports-Classification "$@"; }
dx mkdir -p "$OUT"
LOG=$OUT/run_val.log; STATUS=$OUT/STATUS
st() { echo "$(date '+%F %T') $*" | dxi tee -a "$STATUS" >/dev/null; }
evalm() {  # evalm <modelo>
  st "evaluando $1"
  dx python -m scripts.benchmark.eval_bas --model "$1" --dataset pcbas --metrics at1 at2 tight 2>&1 | dxi tee -a "$LOG" >/dev/null || st "FALLÓ eval $1"
  dx python -m scripts.benchmark.eval_bas --model "$1" --dataset pcbas --metrics at1 at2 tight --merge-high-pass 2>&1 | dxi tee -a "$LOG" >/dev/null || st "FALLÓ eval merged $1"
}
tdeed() {  # tdeed <checkpoint>
  for g in "${GAMES[@]}"; do
    dx test -s "$VID/$g.tdeed.$1.npz" && continue
    st "puntuando T-DEED $1 $g"
    dx python -c "import sys; from pathlib import Path; from core.ball_spotting import TDEEDSpotter; TDEEDSpotter(sys.argv[1]).ensure_scores(Path(sys.argv[2]))" "$1" "$VID/$g.mp4" 2>&1 | dxi tee -a "$LOG" >/dev/null || st "FALLÓ T-DEED $1 $g"
  done
  evalm "$1"
}
st "inicio"
tdeed SoccerNetBall_test
for size in big small; do
  st "puntuando AdaSpot $size"
  dx python -m scripts.benchmark.adaspot_scores --model "SoccerNetBall_$size" --videos "$VID/game_18.mp4" "$VID/game_24.mp4" "$VID/game_47.mp4" 2>&1 | dxi tee -a "$LOG" >/dev/null || st "FALLÓ AdaSpot $size"
  evalm "AdaSpot_$size"
done
n=0
until grep -qE " fin$" output/benchmark/snv2/STATUS 2>/dev/null; do sleep 120; n=$((n+1)); [ $n -eq 1 ] && st "esperando a que acabe el piloto de SN-v2 (VRAM para el 800MF)"; done
tdeed SoccerNetBall_challenge2
st "fin"
