#!/usr/bin/env bash
# Vistas de la panorámica de SoccerTrack v2 para detectar tiros con T-DEED sin entrenar, sobre partidos
# COMPLETOS (ver panorama_views_eval.py): render (CPU, en paralelo) -> escaneo T-DEED (GPU, de uno en uno,
# lote 1: ~6,3 GB de pico) -> informe.
#
# Se ejecuta DENTRO del contenedor y desatendido (sigue aunque se cierre la sesión), desde una copia de los
# scripts en output/panorama_views/experiments/ para que un cambio de rama en /app no le afecte:
#   docker exec -d -w /app Sports-Classification bash output/panorama_views/experiments/run_panorama_views.sh
#   # los 10 partidos, solo la cámara virtual nativa:
#   docker exec -d -w /app -e MATCHES="117092 117093 ..." -e SIZES=796x448 -e CROPS=0 \
#       -e REPORT=report_10_partidos.txt Sports-Classification bash output/panorama_views/experiments/run_panorama_views.sh
# Reanudable: lo ya renderizado o escaneado se salta. Progreso en output/panorama_views/STATUS; informe
# final en output/panorama_views/$REPORT.
set -u
cd /app
MATCHES="${MATCHES:-117093}"
SIZES="${SIZES:-1280x720 796x448 532x300}"   # cámaras virtuales a renderizar
CROPS="${CROPS:-1}"                          # 1: también los recortes fijos de las zonas de tiro
MAX_JOBS="${MAX_JOBS:-8}"                    # renders simultáneos (cada uno decodifica la panorámica 4K)
REPORT="${REPORT:-report.txt}"
RUN=output/panorama_views
EXP=$RUN/experiments
MIN_FREE_MB="${MIN_FREE_MB:-7500}"   # VRAM libre antes de cada intento (pico medido 6,3 GB + contexto CUDA)
export PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
status() { echo "$(date '+%F %T') $*" >> "$RUN/STATUS"; }
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do wait -n; done; }

status "vistas [$MATCHES] [$SIZES] recortes=$CROPS: inicio (render)"

# 1) Render en paralelo (prioridad baja: la CPU es compartida): cámaras virtuales y recortes fijos, por mitad
for m in $MATCHES; do
    V=datasets/soccertrack-v2/views/$m
    mkdir -p "$V"
    for half in 1st 2nd; do
        for size in $SIZES; do
            out="$V/${half}_vcam${size%x*}.mp4"
            [ -f "$out" ] && continue
            throttle
            nice -n 10 python -u "$EXP/panorama_vcam.py" "$m" "$half" 0 -1 --size "$size" --out "$out" \
                > "$RUN/render_${m}_${half}_vcam${size%x*}.log" 2>&1 &
        done
        if [ "$CROPS" = "1" ]; then
            throttle
            nice -n 10 python -u "$EXP/panorama_static_crops.py" "$m" "$half" --out-dir "$V" \
                > "$RUN/render_${m}_${half}_crops.log" 2>&1 &
        fi
    done
done
wait
status "render terminado"

# 2) Escaneo T-DEED, en orden. Antes de cada intento espera a que haya VRAM libre (la GPU es compartida
#    con otros proyectos); si se corta (p. ej. OOM), reintenta y reanuda donde lo dejó.
videos=()
pats=(); for size in $SIZES; do pats+=("vcam${size%x*}"); done
[ "$CROPS" = "1" ] && pats+=(goaltile goalcrop)
for m in $MATCHES; do
    for pat in "${pats[@]}"; do
        for half in 1st 2nd; do
            for f in "datasets/soccertrack-v2/views/$m/${half}_${pat}"*.mp4; do
                case "$f" in *.tmp.mp4) continue ;; esac
                [ -f "$f" ] && videos+=("/app/$f")
            done
        done
    done
done
status "escaneo: ${#videos[@]} vídeos"
for attempt in 1 2 3 4 5 6; do
    if [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "$MIN_FREE_MB" ]; then
        status "esperando a que haya ${MIN_FREE_MB} MB de VRAM libres"
        until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$MIN_FREE_MB" ]; do sleep 60; done
    fi
    if (cd models/tdeed/T-DEED && python -u "/app/$EXP/tdeed_scan_heads.py" SoccerNetBall_challenge2 "${videos[@]}" \
            --out-tag _views --batch-size 1) >> "$RUN/scan.log" 2>&1; then
        break
    fi
    status "escaneo cortado (intento $attempt/6, ver scan.log); reintento en 10 min"
    sleep 600
done
status "escaneo terminado"

# 3) Informe
views_arg=(); for p in "${pats[@]}"; do views_arg+=("$p"); done
python -u "$EXP/panorama_views_eval.py" $MATCHES --views "${views_arg[@]}" > "$RUN/$REPORT" 2>&1 \
    || status "el informe falló, ver $REPORT"
status "FIN vistas [$MATCHES] -> $REPORT"
