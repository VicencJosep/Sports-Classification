#!/usr/bin/env bash
# Variantes de dinámica de cámara (vcam_path.py) -> render 796x448 -> T-DEED -> AP de tiros, desatendido:
#   docker exec -d -w /app Sports-Classification bash output/panorama_ball/experiments/run_vcam_variants.sh [MATCH...]
# VARIANTS = "variante:tag[:parámetros] ..." (por defecto viterbi, viterbi_l1 y el control dataset_l1); la vista se llama
# vcam796<tag>; los parámetros propios de la variante van separados por comas (p. ej. viterbi_l1:vl1L200:--lead,200).
# EXTRA = parámetros comunes a todas (p. ej. "--rx 80 --w3 20"). Renders en paralelo (CPU), un solo escaneo
# de T-DEED (GPU compartida: espera a MIN_FREE_MB libres). Informe: output/panorama_views/report_vcam_<tags>.txt, con
# BASE_VIEWS (vcam796 vcam796yolo_pano) como referencia. RENDER_ARGS = parámetros extra del render (p. ej. "--enhance").
set -u
cd /app
MATCHES=("$@"); [ ${#MATCHES[@]} -eq 0 ] && MATCHES=(117093 132877)
VARIANTS="${VARIANTS:-viterbi:vit viterbi_l1:vl1 dataset_l1:dl1}"; EXTRA="${EXTRA:-}"; RENDER_ARGS="${RENDER_ARGS:-}"
MIN_FREE_MB="${MIN_FREE_MB:-7500}"; RENDER_JOBS="${RENDER_JOBS:-4}"
RUN=output/panorama_ball; EXP=$RUN/experiments; VEXP=output/panorama_views/experiments
export PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
status() { echo "$(date '+%F %T') $*" >> "$RUN/STATUS"; }
tag_of() { local x=${1#*:}; echo "${x%%:*}"; }
args_of() { local x=${1#*:}; [[ $x == *:* ]] && echo "${x#*:}" | tr ',' ' '; }
tags=(); for vt in $VARIANTS; do tags+=("$(tag_of "$vt")"); done
status "variantes de cámara (${VARIANTS}) sobre ${MATCHES[*]} ${EXTRA}: trayectorias"

jobs_file=$(mktemp)
for m in "${MATCHES[@]}"; do for h in 1st 2nd; do for vt in $VARIANTS; do
    v=${vt%%:*}; t=$(tag_of "$vt"); out="datasets/soccertrack-v2/views/$m/${h}_vcam796$t.mp4"
    python -u "$EXP/vcam_path.py" path "$m" "$h" --variant "$v" --tag "_$t" $EXTRA $(args_of "$vt") >> "$RUN/vcam_path.log" 2>&1 \
        || { status "trayectoria $m $h $v falló"; continue; }
    [ -f "$out" ] || echo "python -u $VEXP/panorama_vcam.py $m $h 0 -1 --size 796x448 --smooth-s 0 --max-speed 1e9 $RENDER_ARGS --camera $RUN/${m}_${h}_vcam_$t.npz --out $out > $RUN/render_${m}_${h}_$t.log 2>&1 || echo FALLO $out" >> "$jobs_file"
done; done; done
status "variantes de cámara: render ($(wc -l < "$jobs_file") vídeos, $RENDER_JOBS en paralelo)"
xargs -P "$RENDER_JOBS" -I{} nice -n 10 bash -c "{}" < "$jobs_file" | while read -r l; do status "render: $l"; done
rm -f "$jobs_file"

videos=()
for m in "${MATCHES[@]}"; do for h in 1st 2nd; do for t in "${tags[@]}"; do
    f="datasets/soccertrack-v2/views/$m/${h}_vcam796$t.mp4"; [ -f "$f" ] && videos+=("/app/$f")
done; done; done
status "variantes de cámara: escaneo T-DEED (${#videos[@]} vídeos)"
for attempt in 1 2 3 4 5 6; do
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$MIN_FREE_MB" ]; do sleep 60; done
    (cd models/tdeed/T-DEED && python -u "/app/$VEXP/tdeed_scan_heads.py" SoccerNetBall_challenge2 "${videos[@]}" \
        --out-tag _views --batch-size 1) >> output/panorama_views/scan.log 2>&1 && break
    status "escaneo cortado (intento $attempt/6), reintento en 10 min"; sleep 600
done
views=(${BASE_VIEWS:-vcam796 vcam796yolo_pano}); for t in "${tags[@]}"; do views+=("vcam796$t"); done
name=$(IFS=_; echo "${tags[*]}")
for m in "${MATCHES[@]}"; do
    python -u "$VEXP/panorama_views_eval.py" "$m" --views "${views[@]}" > "output/panorama_views/report_vcam_${name}_$m.txt" 2>&1 || status "informe $m falló"
done
python -u "$VEXP/panorama_views_eval.py" "${MATCHES[@]}" --views "${views[@]}" > "output/panorama_views/report_vcam_$name.txt" 2>&1 || status "informe conjunto falló"
status "FIN variantes de cámara ($name)"
