# Posesión de equipo por fotograma (`core/team_possession`)

Decide por fotograma si el balón está en juego y de qué equipo es (H local, A visitante, D muerto, X sin
respuesta) a partir de posiciones en metros, con el balón ausente a menudo. Nació para la fase 1 del reto
More than a Hack 2027 de Metrica Sports y se usa igual sobre nuestro `tracks_2d.jsonl`. Estado del arte y
por qué este diseño: `docs/investigacion/reports/Estado del arte posesión de balón.md`.

## Piezas

| Módulo | Qué hace |
|---|---|
| `frames.py` | `PositionalData`: `players_xy [T,N,2]` y `ball_xy [T,2]` en metros (origen en la esquina, NaN si falta), `team [N]`, `period`, `labels`. |
| `io_metrica.py` | EPTS de Metrica (el ATD del reto, partido 3 de sample-data), CSV de sample-data, y verdad aproximada H/A/D desde eventos. |
| `io_sportec.py` | IDSSE (7 partidos DFL, CC-BY) con la verdad del proveedor por fotograma (`BallStatus`, `BallPossession`); caché `.npz`. |
| `io_pipeline.py` | Nuestro `tracks_2d.jsonl` (`field_xy` de `core/homography.py`) + `teams.json` de `core/classification.py`. |
| `features.py` | Rasgos por fotograma: distancias por equipo, cinemática del balón suavizada, duelo, actividad colectiva, huecos del balón, y `feature_matrix` (32 rasgos, con mirada atrás y adelante) para aprender. |
| `control.py` | `rule_labels` (último control válido, duelo, cambio cinemático), `emission_scores` (evidencias blandas H/A/D), `nearest_labels` (línea base del reto). |
| `decode.py` | `majority_smooth`, `min_segment_filter`, `viterbi_decode` (coste de cambio + duración mínima por expansión de estados). |
| `simulate.py` | `drop_ball` (borra el balón al 41 % en rachas, sesgado a balón alto y zonas pobladas) y `anonymize_slots` (tracklets anónimos como el ATD). |
| `spell_metrics.py` | Copia literal del scorer oficial del reto (MIT): `spell_f1_mean` (posesiones por IoU temporal) y `macro_f1`. |
| `labels.py` | Códigos, segmentos, `merge_short_spells` (verdad por toque → por control), CSV de entrega. |

## Uso

```bash
# caché de datos (una vez) y evaluación de variantes con las métricas del reto
docker exec -w /app Sports-Classification python scripts/benchmark/cache_possession_data.py --dataset idsse
docker exec -w /app Sports-Classification python scripts/benchmark/eval_possession.py --dataset idsse --merge-short 2 --drop-ball 0.41
# clasificador por fotograma (CPU, ~7 min con validación dejando un partido fuera)
docker exec -w /app Sports-Classification python scripts/benchmark/train_possession_model.py
# entrega del reto (requiere los datos del partido en datasets/posesion/mtah2027/data)
docker exec -w /app Sports-Classification python scripts/mtah/build_submission.py --variant gbm_vit_msf \
    --model output/posesion/models/hgb_possession.joblib --switch-cost 16 --min-dur 50 --out output/posesion/submissions/gbm.csv
```

Desde código, con nuestro pipeline:

```python
from core.team_possession import control, decode
from core.team_possession.io_pipeline import load_tracks_2d
from core.team_possession.features import compute_features
data = load_tracks_2d("p.tracks_2d.jsonl", "p.teams.json", fps=25)
feats = compute_features(data)
emit = control.log_softmax(control.emission_scores(feats))
codes = decode.viterbi_decode(emit, switch_cost=32, min_dur=50, runs=feats.period_runs)  # H=0 A=1 D=2
```

## Qué funciona y qué no (IDSSE con el balón borrado al 41 %, verdad fusionada a < 2 s)

- La línea base del reto da 0,255 de spellF1, igual que los 0,24 publicados: la simulación está calibrada.
- Reglas + Viterbi con coste de cambio alto y duración mínima de 1 s: 0,44. El decodificador manda: el
  mismo Viterbi pasa de 0,34 (coste 4) a 0,44 (coste 24, duración mínima 25 fotogramas).
- Clasificador por fotograma (sklearn HistGradientBoosting sobre `feature_matrix`, 89 rasgos con contexto
  temporal ±0,5/1/2 s) + Viterbi coste 16 y duración mínima 2 s: **0,66** de spellF1 y 0,91 de macroF1 (validación
  dejando un partido fuera); en los partidos de Metrica, sin reentrenar, 0,58 frente a 0,43 de la línea base. Detalle
  y análisis de errores en `docs/comparativa_benchmark/7.18_posesion_equipo_mtah.md`.
- Oráculos sobre la verdad: retrasar todas las fronteras 1 s cuesta 0,21; no detectar balón muerto cuesta
  0,35; desviar las paradas 1 s cuesta 0,09. Las transiciones y las paradas pesan más que el equipo.
- Los partidos de ejemplo de Metrica no sirven para simular el reto: en ellos el balón solo falta cuando
  está muerto (92-100 % de los D sin balón, 7-13 % de los vivos).
- `rule_labels` fragmenta (toques del rival con ruido de aceleración): sin filtro de tramos no vale.

## Trampas

- Metrica entrega coordenadas normalizadas: el reto mide el campo en 105,3 × 68,0 m (los metadatos dicen
  105 × 68). El ATD cuenta desde 1 y la entrega desde 0.
- IDSSE marca posesión por toque (mediana 3 s); la verdad del reto es por control (14 s de media): usa
  `--merge-short 2` (o 3) antes de medir.
- La ventana pública del reto tiene 17 posesiones: un ±0,15 de ruido. No elegir variantes con ella.
- Las reglas del reto prohíben subir los datos del partido a LLM alojados por terceros.
