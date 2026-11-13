#!/usr/bin/env bash
# Experimento de RoI de AdaSpot (docs/comparativa_benchmark/7.14_adaspot_roi_gt.md): puntúa AdaSpot-b y -s con la RoI
# anotada y los controles (recorte central, pseudo-GT del detector) en FOOTPASS val, SoccerTrack v2 eval y SN-BAS test,
# evalúa cada variante igual que la línea base y regenera la tabla comparativa tras cada paso.
# Reanudable (las cachés que existen se saltan). Una pasada de GPU cada vez, con lotes de BATCH clips: 2 ocupan ~4 GB de
# VRAM y 4 (el valor del config) ~7,4 GB. Como la GPU es compartida, si no hay MIN_FREE_MB libres espera en vez de fallar.
# Cambiar el lote mueve las probabilidades en el 5.º decimal: el paso de SN-BAS re-puntúa la saliencia (_salroi) para medirlo.
# Lanzar desatendido (host):  docker exec -d -w /app Sports-Classification bash scripts/benchmark/run_adaspot_gt_roi.sh
# Seguir:  tail output/benchmark/adaspot_gt_roi/STATUS      Resultado:  output/benchmark/adaspot_gt_roi/tabla.md
# Parar:   docker exec Sports-Classification bash -c 'pkill -f "^bash scripts/benchmark/run_adaspot_gt_roi"; pkill -f adaspot_scores'
# Variables: SIZES ("big small"), BATCH (2), MIN_FREE_MB (5000), SKIP_SCAN=1 (solo evaluar e informar).
set -u
cd /app
export PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
SIZES="${SIZES:-big small}"; MIN_FREE_MB="${MIN_FREE_MB:-5000}"; BATCH="${BATCH:-2}"
OUT=output/benchmark/adaspot_gt_roi; mkdir -p "$OUT"
STATUS="$OUT/STATUS"
exec >> "$OUT/run.log" 2>&1
exec 9> "$OUT/lock"; flock -n 9 || { echo "$(date '+%F %T') ya hay una ejecución en marcha" >> "$STATUS"; exit 1; }
status() { echo "$(date '+%F %T') $*" >> "$STATUS"; echo "== $(date '+%T') $*"; }

PCBAS=datasets/SoccerNet/pcbas-2026/videos
st2_videos() { for m in 128057 132831; do for h in 1st 2nd; do echo "datasets/soccertrack-v2/views/$m/${h}_$1.mp4"; done; done; }

score() {  # score <descripción> <tamaño> <argumentos de adaspot_scores...>
    local what=$1 size=$2 waited=0 free; shift 2
    [ "${SKIP_SCAN:-0}" = 1 ] && return 0
    while free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "${free:-0}" -lt "$MIN_FREE_MB" ]; do
        [ $waited = 0 ] && status "esperando GPU ($free MB libres, hacen falta $MIN_FREE_MB)"; waited=1; sleep 120
    done
    status "puntuando AdaSpot $size: $what"
    python -m scripts.benchmark.adaspot_scores --model "SoccerNetBall_$size" --batch-size "$BATCH" "$@" || status "FALLÓ la puntuación: AdaSpot $size, $what"
}

evaluate() {  # evaluate <modelo> efl | pcbas | st2 <vista>
    local model=$1 ds=$2 view=${3:-}
    case $ds in
        efl) python -m scripts.benchmark.eval_bas --model "$model" --split test --metrics at1 at2 tight \
             && python -m scripts.benchmark.eval_prf --model "$model" --dataset efl ;;
        pcbas) python -m scripts.benchmark.eval_bas --model "$model" --dataset pcbas --metrics at1 at2 tight --merge-high-pass \
             && python -m scripts.benchmark.eval_prf --model "$model" --dataset pcbas --tag val_merged \
             && python -m scripts.benchmark.eval_f1_pcbas --model "$model" --dataset pcbas --tag val_merged ;;
        st2) python -m scripts.benchmark.eval_bas --model "$model" --dataset st2 --view "$view" --metrics at1 at2 tight \
             && python -m scripts.benchmark.eval_prf --model "$model" --dataset st2 --view "$view" ;;
    esac || status "FALLÓ la evaluación: $model $ds $view"
}

report() { python -m scripts.benchmark.adaspot_roi_report > /dev/null || status "FALLÓ el informe"; }

status "inicio: tamaños [$SIZES]"
for size in $SIZES; do
    M="AdaSpot_$size"
    status "línea base $M: mismas métricas que las variantes (cachés ya puntuadas)"
    evaluate "$M" pcbas; evaluate "$M" st2 vcam796vl1; evaluate "$M" st2 vcam796; evaluate "$M" efl; report

    score "FOOTPASS val, RoI GT + recorte central" "$size" --roi saliency gt center --videos $PCBAS/game_18.mp4 $PCBAS/game_24.mp4 $PCBAS/game_47.mp4
    evaluate "${M}_gtroi" pcbas; evaluate "${M}_ctrroi" pcbas; report; status "tabla actualizada: FOOTPASS $M"

    for view in vcam796vl1 vcam796; do
        score "SoccerTrack v2 $view, RoI GT + recorte central" "$size" --roi saliency gt center --videos $(st2_videos "$view")
        evaluate "${M}_gtroi" st2 "$view"; evaluate "${M}_ctrroi" st2 "$view"; report; status "tabla actualizada: SoccerTrack v2 $view $M"
    done

    score "SN-BAS test, recorte central + pseudo-GT del detector (no hay GT) + saliencia re-puntuada" "$size" --roi saliency center det salcheck --split test
    evaluate "${M}_ctrroi" efl; evaluate "${M}_detroi" efl; evaluate "${M}_salroi" efl; report; status "tabla actualizada: SN-BAS $M"
done
status "fin"
