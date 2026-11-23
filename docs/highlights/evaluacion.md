# Highlights: evaluación de goles y lector de marcador

Medido el 2026-10-16 en la rama `feat/highlights`. Referencia: `ground_truth/highlights_goals.json`.

```bash
# Lo ya generado
docker exec -w /app Sports-Classification python -m analytics.highlights.evaluate
# Una variante sin cortar vídeo (reutiliza señales y cachés del VLM)
docker exec -w /app Sports-Classification python -m analytics.highlights.evaluate --rerun --set vlm_max_candidates=30
docker exec -w /app Sports-Classification python -m analytics.highlights.evaluate --rerun --set scoreboard_enabled=false
# Sin modelo ni GPU: solo con los veredictos ya cacheados (cambios de corte o selección; lo no cacheado se omite)
docker exec -w /app Sports-Classification python -m analytics.highlights.evaluate --rerun --vlm-cache-only --set clip_max_goal_s=50
# Tests (sin pytest en la imagen)
docker exec -w /app -e PYTHONPATH=/app Sports-Classification python tests/test_highlights_duration.py
# Solo el lector de marcador contra goles etiquetados (SoccerNet)
docker exec -w /app Sports-Classification python -m analytics.highlights.scoreboard --video v.mp4 --labels Labels-v2.json --half 1
```

## Mecanismo de selección y corte (resumen, 2026-10-20)

Código en `analytics/highlights/`, parámetros en `config.HighlightsConfig`. Cada etapa deja caché en el directorio de salida
(`signals.npz`, `candidates.json`, `scoreboard.json`, `vjepa_cache.json`, `vlm_cache.json`; la del VLM solo se invalida con
`vlm.PROMPT_VERSION` o al cambiar de modelo).

1. **Señales** (`signals.build_signals`, rejilla de 0,5 s): curvas T-DEED de las dos cabezas sin umbral, audio sostenido
   (dB sobre la mediana móvil de 5 min, media de 15 s), poseedor/equipo/posiciones en metros, acciones y eventos.
2. **Anclas** (`candidates.find_anchors`): picos SHOT ≥ 0,08, GOAL ≥ 0,02, audio ≥ 6 dB con ≥ 3 acciones en ±30 s y eventos
   emitidos; anclas a < 15 s se funden y manda el pico de tiro.
3. **Jugada hacia atrás** (`play_start`): cadena de acciones del equipo atacante hasta toque rival, hueco > 6 s o balón
   parado; inicio = primera acción − 1,5 s, acotado a [ancla − 35, ancla − 6]; sin cadena, ancla − 10. En TV las cadenas
   casi nunca se forman (40/78 candidatos del Clásico con una sola acción) y el inicio queda en ancla − 6.
4. **Fin provisional** (`provisional_end`): ancla + 12, alargado mientras el audio sostenido supere 3 dB, acotado a
   [ancla + 8, ancla + 45]. En TV el audio nunca alarga (`audio_informative` = false).
5. **Puntuación por reglas** (`rule_score`): gol 3, tiro 2, 0,12/dB, saque de centro visto 0,8, pases, metros de
   conducción, racha de tiros, cercanía a portería. Solo ordena candidatos.
6. **Marcador** (`scoreboard.build_timeline` → `apply_scoreboard`): lectura cada 30 s, cambio localizado por bisección y
   asignado al candidato con más tiro+gol de los 20 s anteriores al cambio (`scoreboard_goal_near_s`); si no hay
   ninguno, al de más tiro+gol de los 45 s anteriores, y si tampoco, se crea uno. Lecturas iguales a ambos lados = no-gol.
7. **V-JEPA2** (`vjepa.apply_vjepa`): P(gol) en [ancla − 3, ancla + 9] y saque de centro / córner-banda posteriores para los
   30 mejores; `goal_fusion` = gol + VLM + saque − 0,5·balón parado + 0,5·tiro.
8. **Juicio del VLM** (`vlm.judge_candidate`) sobre los 10 mejores más la red de seguridad de goles: 12 frames (la mitad a
   1/s en el golpeo) → qué pasa, `is_goal`, habilidad, emoción, `worth_highlight`, `play_start`, título y descripción.
9. **Ranking relativo** (`vlm.rank_candidates`): una llamada de texto con todos los juzgados; rechaza duplicados y jugadas
   rutinarias; no veta goles.
10. **Fin del clip** (`vlm.refine_end` → `clip_end_from_kinds`): un frame cada 4 s desde ancla + 3 hasta ancla + 45,
    clasificado acción / celebración / grada / repetición / vuelta / juego; fin = último frame de reacción (celebración o
    grada) + 4 s, tope 45; las **repeticiones solo cuentan tras un gol** (no cortan una celebración partida en dos y, si
    cierran el clip, alargan ≤ `end_replay_max_s` = 8 s); sin reacción, fin del tramo de acción (≤ ancla + 18).
    `celebration_seen` sale de aquí.
11. **Decisión de gol** (`pipeline.goal_decision`): manda el marcador; si no, saque de centro visto + tiro (+ griterío si el
    audio es informativo); si no, "gol probable" solo con celebración vista y griterío ≥ 6 dB (cámara única) o fusión ≥ 1,4 (TV).
12. **Selección** (`select_highlights`): `final` = 0,4·reglas + 0,6·VLM (0,6·emoción + 0,4·habilidad; ≥ 0,9 si gol; ×0,5 si no
    vale); goles primero, luego ranking; entre 3 y 6 clips, por encima de 3 solo con `final` ≥ 0,45; los goles entran siempre.
13. **Límites finales**: inicio = `play_start` del VLM − 1 s acotado a ±10 s del de reglas (el VLM contesta "primer frame
    mostrado" en 49/64 juicios: en la práctica inicio de reglas − 4 s); fin = el del VLM acotado a [ancla + 8, ancla + 45].
    Después, **`fit_duration`** (política §"Duración de los clips"): jugada 12-30 s, gol ≤ 45 s contado desde el gol
    (`goal_time`: cambio de rótulo − 4 s si el ancla va ≥ 10 s por delante; sin rótulo, primer frame de celebración − 4 s);
    primero se recorta lo anterior al momento clave (12 s; 15 en gol), luego la cola. `_dedupe` quita solapes y
    `_merge_close` une clips a ≤ 20 s solo si el resultado no pasa de 40 s.
14. **Corte** (`cut.cut_clip`): ffmpeg a resolución original, crf 20, fundidos de 0,4 s; `highlights.json` lleva por clip
    `start_s`, `end_s`, `anchor_s`, `duration_s`, `label`, `goal_source`, `merged_from`.

## Duración de los clips (política del 2026-10-21; hoja de ruta §3.4)

**Antes** (salidas del backend del 2026-10-18): Clásico 55/45/37/41/33/61 s (mediana 43), Mallorca 25/65/25/28/55 (28),
Espanyol 37/22/37/32 (34,5), Middlesbrough 41/22/56/60 (48,5): 8/19 en rango (12-30 s; goles ≤ 45). Nada miraba la
duración total: la ponían el tope de 45 s hacia delante y las repeticiones de TV, que contaban como reacción tanto tras un
gol como tras una parada.

**Política** (`config.HighlightsConfig`): `clip_min_s` 12, `clip_max_s` 30, `clip_max_goal_s` 45, `clip_back_max_s` 12
(`clip_back_max_goal_s` 15), `end_replay_max_s` 8, `clip_max_merged_s` 40. Implementación: `pipeline.fit_duration`,
`pipeline.goal_time`, `vlm.clip_end_from_kinds(replay_as_reaction, replay_tail_max_s)`, `_merge_close(max_merged_s)`.
Tests: `tests/test_highlights_duration.py` (10).

**Después** (`evaluate --rerun --vlm-cache-only`, mismos veredictos del VLM, V-JEPA2 y marcador; sin GPU):

| Partido | clips | en rango | mediana | goles en clip / etiquetados | s antes / después de cada gol |
|---|---|---|---|---|---|
| Clásico | 6 | 6/6 | 43 s | 5/5, 0 falsos | 7/38, 10/35, 9/32, 13/20, 15/30 |
| Mallorca | 6 | 6/6 | 29 s | 3/3, 0 falsos | 15/27, 12/12, 12/33 |
| Espanyol | 4 | 4/4 | 23 s | – | – |
| Middlesbrough | 4 | 4/4 | 26 s | – | – |
| **Total** | 20 | **20/20** | 28,6 s | **8/8, 0 falsos** (antes 8/8, 0) | |

Hallazgos al medir:
- **El ancla del 1-0 del Mallorca iba 27 s antes del gol** porque `apply_scoreboard` elegía, de los 45 s anteriores
  al cambio de rótulo, el candidato con más SHOT (un tiro anterior, 0,35) y no el del gol (0,14, griterío +14 dB,
  6 s antes del rótulo). Ahora manda la cercanía al rótulo (`scoreboard_goal_near_s` = 20 s): el clip queda
  1:13:50-1:14:32 con la celebración entera, y el tiro anterior se evalúa como jugada aparte.
- **Re-anclar al primer frame de celebración es peligroso en TV**: en el 1-1 del Clásico la realización mete 7
  repeticiones (28 s) antes de la celebración; re-anclando ahí el clip se quedaba sin el gol. Por eso con rótulo manda
  el rótulo y sin rótulo solo se re-ancla con un desplazamiento claro (≥ 10 s).
- Referencia del usuario (Messi 2-3, 1:45:56-1:46:46, 50 s = 7 s de jugada + 43 s de celebración), con la política por
  defecto el clip es 1:45:48-1:46:33: inicio −8 s, fin −13 s, IoU 0,63. Variantes medidas (8/8 goles y 0 falsos en todas):

  | `clip_max_goal_s` / `clip_back_max_goal_s` | clip de Messi | Δ inicio / Δ fin | IoU |
  |---|---|---|---|
  | 45 / 15 (por defecto) | 1:45:48-1:46:33 | −8 / −13 s | 0,63 |
  | 50 / 15 | 1:45:48-1:46:38 | −8 / −8 s | 0,72 |
  | 45 / 10 | 1:45:53-1:46:38 | −3 / −8 s | 0,78 |
  | 50 / 10 | 1:45:53-1:46:43 | −3 / −3 s | 0,88 |

  Decisión pendiente del usuario (con 2-3 referencias más de corte): el tope de gol (45 o 50) y cuánto se guarda antes
  del gol (15 o 10 s).
- En modo solo caché el ranking del Mallorca no está cacheado (cambió el conjunto de candidatos juzgados) y la
  selección sale sin él (`vlm.degraded` en `highlights.json`); con el VLM en marcha se pregunta de nuevo. Los clips
  cortados del backend siguen siendo los del 18-oct hasta que se regeneren.

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
