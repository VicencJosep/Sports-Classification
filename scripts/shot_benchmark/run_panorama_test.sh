#!/usr/bin/env bash
# Viabilidad en vídeo panorámico (SoccerTrack v2, partido 117093, 1ª parte 0-630 s): T-DEED sobre la
# panorámica tal cual y sobre la cámara virtual, y el pipeline completo (tracking + T-DEED + render sin
# estelas) sobre la cámara virtual. Desde el HOST; espera a que esté generada la cámara virtual.
set -u
cd /media/dd3/ner-training/Sports-Classification
OUT=output/shot_benchmark; CL=datasets/soccertrack-v2/clips
inc() { docker exec -w /app -e PYTHONPATH=/app Sports-Classification bash -c "$1"; }
status() { inc "echo \"\$(date '+%F %T') $*\" >> $OUT/STATUS"; }
until grep -q "FIN vcam" "$OUT/STATUS"; do sleep 60; done
status "panorama: escaneo T-DEED (raw + vcam)"
inc "cd models/tdeed/T-DEED && python -u /app/$OUT/experiments/tdeed_scan.py SoccerNetBall_challenge2 /app/$CL/117093_1st_0-630.mp4 /app/$CL/117093_1st_0-630_vcam.mp4 --out-tag _pano >> /app/$OUT/tdeed/scan_pano.log 2>&1" || status "panorama: escaneo falló"
status "panorama: pipeline completo sobre la cámara virtual"
inc "python -m main --source $CL/117093_1st_0-630_vcam.mp4 --mode full-match --output-dir output/panorama_vcam --skip-classification --skip-homography --trace-length 0 > output/panorama_vcam.log 2>&1" || status "panorama: pipeline vcam falló"
status "FIN panorama"
