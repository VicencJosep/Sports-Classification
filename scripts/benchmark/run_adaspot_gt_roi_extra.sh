#!/usr/bin/env bash
# Segunda tanda del experimento de RoI de AdaSpot (docs/comparativa_benchmark/7.14_adaspot_roi_gt.md): dos variantes de
# sensibilidad sobre la misma anotación, para acotar la lectura de «RoI GT frente a saliencia».
#   gtfeet  FOOTPASS: RoI centrada en los pies del ejecutor (donde está el balón) en vez de en el centro de su caja
#   fix     FOOTPASS y SoccerTrack v2 (vcam796vl1): saliencia desplazada lo mínimo para que contenga al ejecutor
# Comparte cerrojo con run_adaspot_gt_roi.sh: si la primera tanda sigue en marcha, espera a que termine. Reanudable.
# Lanzar desatendido (host):  docker exec -d -w /app Sports-Classification bash scripts/benchmark/run_adaspot_gt_roi_extra.sh
# Seguir:  tail output/benchmark/adaspot_gt_roi/STATUS      Resultado:  output/benchmark/adaspot_gt_roi/tabla.md
# Parar:   docker exec Sports-Classification bash -c 'pkill -f "^bash scripts/benchmark/run_adaspot_gt_roi"; pkill -f adaspot_scores'
# Variables: SIZES ("big small"), BATCH (2), MIN_FREE_MB (5000).
set -u
cd /app
export PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
SIZES="${SIZES:-big small}"; MIN_FREE_MB="${MIN_FREE_MB:-5000}"; BATCH="${BATCH:-2}"
OUT=output/benchmark/adaspot_gt_roi; mkdir -p "$OUT"
STATUS="$OUT/STATUS"
exec >> "$OUT/run.log" 2>&1
status() { echo "$(date '+%F %T') $*" >> "$STATUS"; echo "== $(date '+%T') $*"; }
exec 9> "$OUT/lock"
flock -n 9 || { status "tanda 2 en cola: espera a que acabe la primera"; flock 9; }

PCBAS=datasets/SoccerNet/pcbas-2026/videos
st2_videos() { for m in 128057 132831; do for h in 1st 2nd; do echo "datasets/soccertrack-v2/views/$m/${h}_$1.mp4"; done; done; }

score() {  # score <descripción> <tamaño> <argumentos de adaspot_scores...>
    local what=$1 size=$2 waited=0 free; shift 2
    while free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "${free:-0}" -lt "$MIN_FREE_MB" ]; do
        [ $waited = 0 ] && status "esperando GPU ($free MB libres, hacen falta $MIN_FREE_MB)"; waited=1; sleep 120
    done
    status "puntuando AdaSpot $size: $what"
    python -m scripts.benchmark.adaspot_scores --model "SoccerNetBall_$size" --batch-size "$BATCH" "$@" || status "FALLÓ la puntuación: AdaSpot $size, $what"
}

evaluate() {  # evaluate <modelo> pcbas | st2 <vista>
    local model=$1 ds=$2 view=${3:-}
    case $ds in
        pcbas) python -m scripts.benchmark.eval_bas --model "$model" --dataset pcbas --metrics at1 at2 tight --merge-high-pass \
             && python -m scripts.benchmark.eval_prf --model "$model" --dataset pcbas --tag val_merged \
             && python -m scripts.benchmark.eval_f1_pcbas --model "$model" --dataset pcbas --tag val_merged ;;
        st2) python -m scripts.benchmark.eval_bas --model "$model" --dataset st2 --view "$view" --metrics at1 at2 tight \
             && python -m scripts.benchmark.eval_prf --model "$model" --dataset st2 --view "$view" ;;
    esac || status "FALLÓ la evaluación: $model $ds $view"
}

report() { python -m scripts.benchmark.adaspot_roi_report > /dev/null || status "FALLÓ el informe"; }

status "tanda 2, inicio: tamaños [$SIZES]"
for size in $SIZES; do
    M="AdaSpot_$size"
    score "FOOTPASS val, RoI GT en los pies + saliencia corregida" "$size" --roi gtfeet fix --videos $PCBAS/game_18.mp4 $PCBAS/game_24.mp4 $PCBAS/game_47.mp4
    evaluate "${M}_gtfeetroi" pcbas; evaluate "${M}_fixroi" pcbas; report; status "tabla actualizada: FOOTPASS $M (tanda 2)"

    score "SoccerTrack v2 vcam796vl1, saliencia corregida" "$size" --roi fix --videos $(st2_videos vcam796vl1)
    evaluate "${M}_fixroi" st2 vcam796vl1; report; status "tabla actualizada: SoccerTrack v2 vcam796vl1 $M (tanda 2)"
done
status "tanda 2, fin"
