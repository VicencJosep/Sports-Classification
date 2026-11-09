#!/usr/bin/env bash
# Benchmark E2 completo (docs/comparativa_benchmark.md §7.11): todos los spotters con pesos públicos sobre las vistas de
# cámara virtual de los partidos de EVALUACIÓN de SoccerTrack v2 (128057 + 132831, docs/particion_datasets.md), más el
# coste de cada modelo medido con la GPU libre.
#   vcam796    = cámara que sigue el balón ANOTADO del dataset (scripts/shot_benchmark/panorama_vcam.py)
#   vcam796vl1 = cámara Viterbi+L1 sobre nuestro detector de balón (scripts/shot_benchmark/vcam_path.py)
# Modelos: T-DEED SNB_pano1 / SoccerNetBall_challenge2 / SoccerNetBall_test, sn-teamspotting (baseline 2025),
# AdaSpot-b/-s y E2E-Spot de OpenSportsLab. Las cachés ya existentes se saltan (reanudable).
# Uso (host):  docker exec -w /app Sports-Classification bash scripts/benchmark/run_bas_views.sh [vcam796 vcam796vl1]
# Variables: MATCHES ("128057 132831"), SKIP_COST=1 (no medir coste), SKIP_SCAN=1 (solo evaluar).
set -u
cd /app
export PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VIEWS="${*:-vcam796 vcam796vl1}"; MATCHES="${MATCHES:-128057 132831}"
OUT=output/benchmark/bas_st2; mkdir -p "$OUT"
LOG="$OUT/run_views.log"; STATUS="$OUT/STATUS"
status() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
videos() { for m in $MATCHES; do for h in 1st 2nd; do echo "datasets/soccertrack-v2/views/$m/${h}_$1.mp4"; done; done; }
status "inicio: vistas [$VIEWS], partidos [$MATCHES]"

# 1) Coste con la GPU libre, antes de lanzar nada más (SNB_pano1 tiene la misma arquitectura que challenge2).
if [ "${SKIP_COST:-0}" != 1 ]; then
    for spec in tdeed:SoccerNetBall_test tdeed:SoccerNetBall_challenge2 adaspot:SoccerNetBall_big adaspot:SoccerNetBall_small \
                teamspot:baseline osl:snbas; do
        status "coste $spec"
        python -m scripts.benchmark.spotter_cost --model "$spec" --video output/benchmark/cost/stoke_5min.mp4 \
            >> "$LOG" 2>&1 || status "coste $spec FALLÓ"
    done
fi

# 2) Cachés de puntuaciones, dos cadenas en paralelo (la decodificación domina; cada modelo ocupa 6-9 GB de VRAM).
if [ "${SKIP_SCAN:-0}" != 1 ]; then
    chain_tdeed() {
        for v in $VIEWS; do for model in SNB_pano1 SoccerNetBall_challenge2 SoccerNetBall_test; do
            status "escaneo T-DEED $model sobre $v"
            python -c "import sys; from core.ball_spotting import TDEEDSpotter; sp = TDEEDSpotter(sys.argv[1]); [sp.ensure_scores(p) for p in sys.argv[2:]]" \
                "$model" $(videos "$v") >> "$LOG" 2>&1 || status "escaneo T-DEED $model $v FALLÓ"
        done; done
    }
    chain_ext() {
        for v in $VIEWS; do
            status "escaneo sn-teamspotting sobre $v"
            python -m scripts.benchmark.teamspot_scores --videos $(videos "$v") >> "$LOG" 2>&1 || status "escaneo teamspot $v FALLÓ"
            for m in SoccerNetBall_big SoccerNetBall_small; do
                status "escaneo AdaSpot $m sobre $v"
                python -m scripts.benchmark.adaspot_scores --model "$m" --videos $(videos "$v") >> "$LOG" 2>&1 || status "escaneo AdaSpot $m $v FALLÓ"
            done
            status "escaneo OSL sobre $v"
            python -m scripts.benchmark.osl_scores --videos $(videos "$v") >> "$LOG" 2>&1 || status "escaneo OSL $v FALLÓ"
        done
    }
    chain_tdeed & chain_ext & wait
fi

# 3) Evaluación oficial (CPU): mAP@1 y tight, resumen en $OUT/<modelo>/<vista>/summary.json
for v in $VIEWS; do
    for model in SNB_pano1 SoccerNetBall_challenge2 SoccerNetBall_test SNTeam_baseline AdaSpot_big AdaSpot_small OSL_E2E_snbas; do
        python -m scripts.benchmark.eval_bas --model "$model" --dataset st2 --matches $MATCHES --view "$v" --metrics at1 tight \
            >> "$LOG" 2>&1 || status "eval $model $v FALLÓ"
    done
done
status "fin"
