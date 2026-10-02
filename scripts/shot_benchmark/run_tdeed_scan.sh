#!/usr/bin/env bash
# Escaneo T-DEED de los 7 EFL (limpios primero) + modelo grande en Middlesbrough.
#   docker exec -d Sports-Classification bash /app/output/shot_benchmark/experiments/run_tdeed_scan.sh
set -u
OUT=/app/output/shot_benchmark; EFL=/app/datasets/SoccerNet/england_efl/2019-2020
status() { echo "$(date '+%F %T') $*" >> "$OUT/STATUS"; }
cd /app/models/tdeed/T-DEED; mkdir -p "$OUT/tdeed"
v() { echo "$EFL/2019-10-01 - $1/720p.mp4"; }
status "tdeed: escaneo SoccerNetBall_test"
python -u "$OUT/experiments/tdeed_scan.py" SoccerNetBall_test \
    "$(v 'Middlesbrough - Preston North End')" "$(v 'Stoke City - Huddersfield Town')" "$(v 'Reading - Fulham')" \
    "$(v 'Hull City - Sheffield Wednesday')" "$(v 'Leeds United - West Bromwich')" \
    "$(v 'Brentford - Bristol City')" "$(v 'Blackburn Rovers - Nottingham Forest')" >> "$OUT/tdeed/scan.log" 2>&1 || status "tdeed test falló"
status "tdeed: escaneo SoccerNetBall_challenge2 (Middlesbrough)"
python -u "$OUT/experiments/tdeed_scan.py" SoccerNetBall_challenge2 "$(v 'Middlesbrough - Preston North End')" >> "$OUT/tdeed/scan.log" 2>&1 || status "tdeed challenge2 falló"
status "FIN tdeed escaneo"
