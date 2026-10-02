#!/usr/bin/env bash
# Benchmark de tiros en partido completo sobre los 7 partidos EFL con
# Labels-ball.json: escaneo V-JEPA2 (reanudable) + evaluación, todo a ficheros
# en output/shot_benchmark/. Pensado para dejarlo corriendo sin sesión abierta.
#
# Desde el HOST:
#   docker exec -d -w /app Sports-Classification bash scripts/run_efl_shot_benchmark.sh
#
# - Si ya hay un scan_event_windows corriendo (lanzado a mano), espera a que
#   acabe en vez de competir por la GPU y por los mismos .tmp.
# - Si el escaneo se corta (p.ej. OOM de CUDA: la GPU es compartida con otros
#   proyectos), espera 10 min y lo reanuda desde la última ventana escrita,
#   hasta MAX_ATTEMPTS intentos.
# - Al acabar evalúa los partidos completos que haya y deja el informe en
#   vjepa2_efl.txt (legible) y vjepa2_efl.json (para comparar después).
# - Relanzarlo es seguro: los partidos ya escaneados se saltan.
#
# Progreso: output/shot_benchmark/STATUS (una línea por etapa, con hora) y
# output/shot_benchmark/scan.log (log completo del escaneo).

set -u
shopt -s nullglob
cd "$(dirname "$0")/.."

OUT=output/shot_benchmark
EFL=datasets/SoccerNet/england_efl/2019-2020
CACHE_NAME=720p.event_windows.vjepa2.w6s3.jsonl
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"
mkdir -p "$OUT"

status() { echo "$(date '+%F %T') $*" >> "$OUT/STATUS"; }

videos=("$EFL"/*/720p.mp4)
status "inicio: ${#videos[@]} partidos"

if pgrep -f "python -m scripts.scan_event_windows" >/dev/null; then
    status "hay un escaneo en curso, esperando a que termine"
    while pgrep -f "python -m scripts.scan_event_windows" >/dev/null; do sleep 60; done
fi

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    caches=("$EFL"/*/"$CACHE_NAME")
    [ "${#caches[@]}" -ge "${#videos[@]}" ] && break
    status "escaneo, intento $attempt/$MAX_ATTEMPTS (${#caches[@]}/${#videos[@]} partidos completos)"
    if ! python -m scripts.scan_event_windows --video "${videos[@]}" >> "$OUT/scan.log" 2>&1; then
        status "escaneo cortado (ver scan.log), reintento en 10 min"
        sleep 600
    fi
done

caches=("$EFL"/*/"$CACHE_NAME")
status "escaneo terminado: ${#caches[@]}/${#videos[@]} partidos completos"
if [ "${#caches[@]}" -eq 0 ]; then
    status "sin cachés completas, nada que evaluar"
    exit 1
fi

if python -m scripts.evaluate_spotting --cache "${caches[@]}" --list-missed \
        --json-out "$OUT/vjepa2_efl.json" > "$OUT/vjepa2_efl.txt" 2>&1; then
    status "evaluación escrita en $OUT/vjepa2_efl.txt"
else
    status "la evaluación falló, ver $OUT/vjepa2_efl.txt"
fi
status "FIN"
