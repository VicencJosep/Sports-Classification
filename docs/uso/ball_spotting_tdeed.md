# Uso — detección puntual de tiros y acciones de balón (`core/ball_spotting.py`, T-DEED)

> Sección sacada del README el 2026-11-09, sin cambios de contenido.

En partido completo (`--mode full-match`), los tiros/goles y el *instante* de cada acción de balón
salen de [T-DEED](https://github.com/arturxe2/T-DEED) (GPL-3.0, ganador de SoccerNet Ball Action
Spotting 2024), que da una probabilidad por frame para 12 clases (PASS, DRIVE, HEADER, HIGH PASS,
OUT, CROSS, THROW IN, SHOT, BALL PLAYER BLOCK, PLAYER SUCCESSFUL TACKLE, FREE KICK, GOAL).

### Instalación (`models/` no se versiona)

```bash
git clone https://github.com/arturxe2/T-DEED.git models/tdeed/T-DEED
pip install gdown
# Pesos (carpeta de Google Drive enlazada en el README de T-DEED):
gdown 1uCWam8EhAXt3z6QQAfhRfzHITUcml438 -O models/tdeed/SoccerNetBall_challenge2/checkpoint_best.pt
gdown 1b7REo4_Cota8SWPSmtdSEfbVNXNgziX6 -O models/tdeed/SoccerNetBall_test/checkpoint_best.pt
```

`SoccerNetBall_challenge2` (RegNetY-800MF, entrenado con 6 de los 7 partidos EFL) es el de
producción (`config.BallSpottingConfig.model_name`). `SoccerNetBall_test` (RegNetY-200MF, split
train) solo sirve para evaluar limpio en Middlesbrough, Stoke y Reading.

### Uso

```bash
# Standalone: lista los tiros/goles de un partido (escanea y cachea junto al vídeo, ~9-15 min)
python -m core.ball_spotting --source partido.mp4 --classes SHOT GOAL

# En el pipeline (valores por defecto):
python -m main --mode full-match --source partido.mp4 --output-dir output/x \
    --event-source tdeed --thresholds auto --ball-actions-trigger tdeed --ball-actions-label combined
```

* `--event-source tdeed|mixed|vjepa2`: de dónde salen los eventos de la línea temporal. `tdeed`
  (por defecto desde el 15-oct-2026) los saca de T-DEED y no carga V-JEPA2: tiro, gol y saque de
  banda de su cabeza de acciones de balón; córner y falta de su cabeza de SoccerNet-v2
  (`config.BallSpottingConfig.tdeed_event_labels`). Esa cabeza también da tarjeta, saque inicial,
  sustitución, penalti, fuera de juego y falta directa/indirecta (`core.ball_spotting.EVENT_SOURCES`),
  pero no pasan de F1 0,35 ni en partidos de SoccerNet-v2 que vio al entrenar, así que no se
  emiten salvo que se añadan a esa lista. La caché `.tdeed.<modelo>.npz`
  guarda las dos cabezas; las anteriores al 15-oct-2026 solo tienen la primera y se rehacen solas.
  `mixed` es el comportamiento anterior (V-JEPA2 + tiro/gol/saque de T-DEED) y `vjepa2`, el de
  antes de T-DEED.
* `--thresholds auto|fixed`: `auto` (por defecto) ajusta los umbrales en cada partido
  (`analytics.engine.calibrate_spot_thresholds`). Fuera de la EFL T-DEED dispara en los mismos
  sitios con menos confianza (en dos juveniles de YouTube, a 0.5 salían 4-5,6 acciones/min y 5-7
  tiros por partido): el umbral de acciones se elige para dar el ritmo de los partidos EFL de
  entrenamiento (15,8 acciones/min de juego), acotado a [0.2, 0.5], así que solo baja. En el partido
  completo de Middlesbrough sale 0.45 (F1 92%, el máximo); en los juveniles 0.24-0.27. Los eventos
  mantienen su umbral fijo: escalarlos empeoraba tiros y saques en SoccerNet-v2 (ver
  `config.BallSpottingConfig`). Los umbrales usados quedan en `telemetry.json` (`spot_thresholds`). `fixed` usa los de `config.BallSpottingConfig`, y
  `--action-min-score X` fija el de acciones.
* `scripts/respot_telemetry.py` regenera `telemetry.json` desde las cachés (`.tdeed.*.npz`,
  `.possession.jsonl`) con otros umbrales (`--thresholds`, `--action-min-score`,
  `--event-min-score shot=0.2 ...`), en segundos y sin GPU, para A/B con
  `scripts/evaluate_pipeline.py`; la etiqueta de las acciones es la de T-DEED (sin R3D-18).
* `scripts/tune_tdeed_event_thresholds.py` fija los umbrales de los eventos de la cabeza SoccerNet
  sobre partidos del split de entrenamiento de SoccerNet-v2.
* `--ball-actions-trigger tdeed|possession`: cuándo hay una acción de balón. Con `tdeed` (por
  defecto) cada acción detectada por T-DEED; el jugador sale siempre de `core/possession.py`.
* `--ball-actions-label combined|classifier|spotter`: quién pone la etiqueta (R3D-18 y T-DEED
  combinados, solo el R3D-18, o solo T-DEED).

### Qué esperar (medido, partido completo)

| Tiros | AP ±3 s | P / R |
|---|---|---|
| V-JEPA2 + SVC, etiqueta ganadora por ventana (antes) | 31,6% | 29% / 73% |
| V-JEPA2 + SVC, picos por clase (`core/spotting.py`) | 34,5% | 26% / 70% |
| T-DEED `SoccerNetBall_test`, 3 partidos EFL no vistos (69 tiros) | 75,6% | 74% / 70% |
| T-DEED `SoccerNetBall_challenge2`, 6 partidos SoccerNet-v2 de otras ligas (130 tiros) | 78,0% | 65% / 84% (umbral 0,3) |

| Acciones de balón, `ground_truth/` (367 acciones, ±1 s) | P | R | F1 |
|---|---|---|---|
| Disparo por cambio de posesión + etiqueta R3D-18 (antes) | 28% | 57% | 38% |
| Disparo T-DEED + etiqueta R3D-18 | 80% | 80% | 80% |
| Disparo T-DEED + etiqueta combinada (por defecto) | 87% | 88% | 88% |
| Disparo T-DEED + etiqueta T-DEED | 90% | 92% | 91% |

Un verificador VLM sobre los candidatos (Qwen3.5-9B con frames o vídeo, con y sin razonamiento;
VideoARM; Qwen3-Omni-30B) NO separa tiros de centros (AUC 0,49-0,63 sobre 150 candidatos
difíciles). Ese código (backends vLLM, `ShotVerifier`, `scripts/verify_shot_candidates.py` y los
benchmarks VLM) no está en `main`: se conserva en la rama `feat/T-DEED`. Los scripts del resto
de mediciones están en `scripts/shot_benchmark/` y el estudio completo en
`docs/archivo/estudio_deteccion_tiros.md`.
