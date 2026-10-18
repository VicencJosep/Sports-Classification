# Highlights: evaluación de goles y lector de marcador

Medido el 2026-10-16 en la rama `feat/highlights`. Referencia: `ground_truth/highlights_goals.json`.

```bash
# Lo ya generado
docker exec -w /app Sports-Classification python -m analytics.highlights.evaluate
# Una variante sin cortar vídeo (reutiliza señales y cachés del VLM)
docker exec -w /app Sports-Classification python -m analytics.highlights.evaluate --rerun --set vlm_max_candidates=30
docker exec -w /app Sports-Classification python -m analytics.highlights.evaluate --rerun --set scoreboard_enabled=false
# Solo el lector de marcador contra goles etiquetados (SoccerNet)
docker exec -w /app Sports-Classification python -m analytics.highlights.scoreboard --video v.mp4 --labels Labels-v2.json --half 1
```

## Problema de partida (Clásico, TV)

Los 5 goles estaban entre los 78 candidatos, pero solo 3 llegaban a un clip, ninguno se etiquetaba "gol" y
había 2 clips con goles inventados. La causa era la selección, no la detección: los candidatos de gol quedaban
en los puestos 15-34 por reglas y el VLM solo ve los 10 primeros; con `vlm_max_candidates=30` se gana un gol
y se pierde otro (el juicio del VLM sobre gol/parada es inestable); solo reglas es peor (4/7). La señal de
saque de centro no funciona con la cámara móvil de TV y el audio de TV tampoco.

## Solución: un gol es un cambio de marcador (`analytics/highlights/scoreboard.py`)

El mismo VLM lee el rótulo (recorte de la esquina donde esté; sondea 4 esquinas) cada 30 s; cada cambio se
localiza por bisección (±4 s) y se descartan lecturas sueltas, decrecientes o de +3 goles. Cada cambio se asigna
al candidato de la ventana con más tiro/gol (o crea uno), `goal_decision` lo trata como gol seguro y, con
lecturas a ambos lados sin cambio, como no-gol seguro. Título y descripción de un gol salen del marcador
(el VLM inventa jugadores y equipos). Todos los goles entran aunque superen `max_clips`. Sin rótulo legible no
hay señal y todo sigue como antes. Coste: ~1 s por lectura, ~3-4 min por partido (cacheado en `scoreboard.json`).

## Resultados

| Conjunto | Antes | Con marcador |
|---|---|---|
| Clásico + Mallorca + Espanyol (clips, 8 goles) | 5/7 en clips, 0 etiquetados gol, 5 descripciones de gol falsas | 8/8 en clips y etiquetados gol, 0 etiquetas ni descripciones falsas |
| SoccerNet-v2, 10 partidos / 20 partes / 5 ligas (solo el lector, no vistos antes) | – | 29/30 goles distintos, 0 cambios de marcador falsos |

Notas:
- La referencia de goles de Mallorca tenía 2 y el partido tiene 3 (Barcelona 1-2 en 1:34:42, 2ª parte 47:00);
  lo halló el lector y se comprobó en el vídeo. Está añadido al GT.
- El único gol no detectado de SoccerNet (PSG–Metz, 2ª parte 48:13) tiene el rótulo oculto hasta el final del
  vídeo. Un gol casi al final de la parte se confirma con lecturas vecinas (Bastia 43:31 pasó de no detectado a
  detectado). En los Barça–Bayern 0:48:06/0:48:07 son dos etiquetas del mismo gol.
- Los 7 EFL (`datasets/SoccerNet/england_efl`) no llevan rótulo: el lector devuelve 0 lecturas válidas y no
  inventa nada (prueba del caso sin marcador).
- El resto de clips (paradas, ocasiones) no se miden como acierto/fallo: son highlights legítimos; lo que se mide
  es el recall de goles y que no se afirme un gol que no existe.
- Con `/mnt` fuera del contenedor, para evaluar vídeos de SoccerNet-v2 hay que copiarlos al proyecto (ver
  `output/highlights/eval/run_sn2.sh`).

## Sin rótulo: V-JEPA2 como segundo juez (`analytics/highlights/vjepa.py`)

Medido offline (2026-10-18) sobre los 7 EFL (sin rótulo, 13 goles) con las cachés que ya existían junto a cada
vídeo: `720p.event_windows.vjepa2.w6s3.jsonl` (16 probabilidades por ventana de 6 s) y
`720p.shot_verdicts.vjepa2.w6s3.v1.jsonl` (1.149 candidatos a tiro con la P(desenlace=gol) de Qwen3.5-9B).
Scripts: `output/highlights/eval/{vjepa_goal_eval,goal_judges_efl,goal_fusion_raw}.py` (copias en scratchpad).

| Juez de gol (gol vs tiro sin gol) | AP | falsos para cubrir 10/13 goles | 12/13 |
|---|---|---|---|
| VLM solo (P(gol) del desenlace) | 0,27 | 83 | 106 |
| V-JEPA2 solo (P(goal) máx. en [-3, +9] s) | 0,19 | 79 | 128 |
| V-JEPA2 + VLM (suma de probabilidades) | 0,51 | 23 | 127 |
| + kick-off de V-JEPA2 en [+15, +120] s | 0,57 | 17 | 28 |
| + córner/banda de V-JEPA2 en [+3, +45] s (−0,5) | 0,59 | 10 | 16 |
| + tiro (0,5) — **la que usa `vjepa.goal_fusion`** | 0,65 | 7 | 29 |

- V-JEPA2 solo es bimodal: en 4 de 7 partidos el gol es la ventana con mayor P(goal) del partido (0,77-0,91), en
  los otros 3 queda enterrado (0,11-0,42, con 4-9 tiros por encima). El VLM se equivoca en otras jugadas, por
  eso sumarlos dobla la AP. Ninguno sirve solo.
- Coste: 1,2 s por ventana de 6 s en GPU (2,3 GB), ventanas en rejilla fija compartidas entre candidatos
  (`vjepa_cache.json`): golpeo para todos los candidatos, ventanas posteriores para los 30 mejores o con
  P(goal) ≥ 0,1; ~10 min por partido.

### Lo que pasó al integrarlo tal cual (fusión ≥ 1,5 → "gol probable")

Clásico sin rótulo: 12 clips, los 12 "gol probable", 8 falsos, y el gol más claro (Messi 2-3, V-JEPA2 0,91)
fuera porque el VLM dijo "parada" (fusión 1,45) y los 12 "goles" agotaron el tope. Los votos que engañan en TV:
el `is_goal` del VLM (1,0 en seis paradas repetidas, 0,0 en tres goles reales), V-JEPA2 en repeticiones de
paradas (0,69-0,90) y el `kick-off` de V-JEPA2 en juego por el círculo central (0,35-0,96). En cámara única
(Mallorca) V-JEPA2 está fuera de dominio: 0,22-0,48 en los 3 goles. Y la fusión no transfiere de EFL porque allí
el voto del VLM era la P(gol) calibrada del verificador, no el `is_goal` del juez de highlights.

Lo único que separa gol de falso en los tres partidos es la **celebración vista por el VLM en los frames
posteriores** (`_celebration_seen`, ya se calculaba para el corte): verdadera en los 8 goles, falsa en 12 de 13
falsos (tabla en `output/highlights/eval/`; el falso restante, Clásico 0:23:56, es una parada tras la que la TV
corta a la grada y luego se juega por el círculo central).

### Regla final (`pipeline.goal_decision`, `vjepa.goal_fusion`)

1. **Gol** (confirmado, título "Gol del local (1-0)" / "Gol"): cambio de marcador, o saque de centro visto
   (geometría del tracking) tras un tiro.
2. **Gol probable** (título "Gol probable", compite por el tope de clips como cualquier jugada): celebración
   vista Y sin balón parado en 45 s Y tiro, y además
   - cámara única (audio informativo): griterío sostenido ≥ 6 dB (`goal_audio_min_db`; el tercer gol del
     Mallorca tiene 6,8 y el 7,0 anterior lo perdía) → `celebration+audio`;
   - TV (audio inútil): `goal_fusion` = V-JEPA2 gol + VLM gol + V-JEPA2 kick-off − 0,5·córner/banda + 0,5·tiro
     ≥ 1,4 (`goal_fusion_min`) → `fusion`.
3. El ranking textual del VLM ya no veta a un gol (describía como "tiro fuera" un gol real), y los candidatos
   con griterío y V-JEPA2 ≥ 0,2 entran en la red de seguridad del VLM (`vlm_candidates`; 6 extra en Mallorca,
   0 en Espanyol): sin eso el tercer gol del Mallorca (puesto 26 por reglas) nunca llegaba al VLM.

| 3 partidos, 8 goles | goles en clip | etiquetados gol | "gol" falsos |
|---|---|---|---|
| Sin rótulo, cascada anterior | 5/8 | 2/8 | 0 |
| Sin rótulo, V-JEPA2 fusión ≥ 1,5 sin celebración | 6/8 | 6/8 | 9 |
| **Sin rótulo, regla final** | **8/8** | **8/8** | **1** (probable, Clásico 0:23:56) |
| Todo activado (marcador manda) | 8/8 | 8/8 | 0 |

Aviso: `goal_fusion_min`, `goal_audio_min_db` y `vjepa_loud_min_goal` se fijaron mirando estos mismos 3
partidos (8 goles). La validación limpia pendiente es correr el pipeline completo en partidos nuevos de
SoccerNet-v2 usando el marcador como verdad y midiendo la regla sin marcador contra él.

Comandos: `python -m analytics.highlights.evaluate --rerun --set scoreboard_enabled=false` (regla sin marcador),
`--set vjepa_enabled=false` (sin V-JEPA2), `--set scoreboard_enabled=false --set vjepa_enabled=false` (cascada
anterior). Tests: `tests/test_highlights_{scoreboard,vjepa}.py` (sin pytest en el contenedor: runner mínimo).

## Pendiente

- Validar la regla sin marcador en partidos nuevos (SoccerNet-v2 con el marcador como verdad): los umbrales
  están fijados con 3 partidos. Hace falta el análisis completo por partido (horas) o reutilizar las cachés de
  tracking de los 7 EFL (`720p.tracks.jsonl`, `720p.possession.jsonl`), que no tienen rótulo y tienen 13 goles.
- Cámara única sin rótulo ni audio útil: ahí solo queda el saque de centro por geometría; sin medir.
- Criterio de aceptación para integrar en `main` y decisión sobre la dependencia de Ollama (main no tiene VLM).
