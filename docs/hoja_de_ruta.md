# Hoja de ruta — Sports-Classification

Documento vivo. Última actualización: **2026-10-21**. Al final de cada sesión se actualizan las casillas (☐ → ☑), se
añade la entrada completa a `docs/archivo/hoja_de_ruta_bitacora.md` y una línea en la bitácora (§7). Cada cifra lleva su fuente; **[M]** = medido, **[H]** = hipótesis.
Los commits los hace el usuario.

## 0. Por dónde empezar

**Lo que más cojea** no es un modelo: es que el dominio del producto (juveniles, cámara única) no tiene ni una anotación
de eventos y T-DEED está ciego ahí (SHOT ≤ 0,16 y GOAL ≤ 0,03 en los 3 goles del Mallorca [M]; 5-7 tiros por partido
frente a 20-30 habituales). Sin verdad propia no se puede ni medir ni afinar, y las dos ideas más caras (afinar T-DEED,
VLMs grandes) se decidirían a ojo. Lo segundo que cojea es la infraestructura de medida: el único GT de acciones
(Middlesbrough, 20 min) está fuera de git y sin generador, hay cuatro arneses distintos y `pipeline_commit` sale nulo.

**Actualización 2026-11-13**: ya hay verdad propia en el dominio (Mallorca, 2.ª parte entera anotada desde el visor) y está
medida (§7.16 de la comparativa): T-DEED de producción 79 / 81 de precisión / recall en acciones al umbral automático, pero solo
el 2 % de recall en las 47 acciones que el pipeline no tenía y el 30 % en tiros; AdaSpot-b se apaga (3 %) y fusionarlo no ayuda. Lo
que toca: más verdad sin sesgo (equipo, una parte del Espanyol, un juvenil nuevo como evaluación) y afinar T-DEED con ella.
**2026-11-14**: Espanyol 1.ª parte anotada y medida (§7.17): producción 76 / 77 y 11 % en lo añadido; AdaSpot-b sí responde ahí
(60 / 48; 53 % de lo añadido) y en el Mallorca se apaga por el punto de vista de la cámara, no por ser amateur; la BatchNorm por
vídeo y AdaSpot-s lo reflotan en parte. Sigue sin haber ningún tramo anotado a ciegas: es lo primero que pedir.

**Orden recomendado** (dos semanas), en tres carriles que no se bloquean entre sí:

| Carril | Quién | Qué | Por qué primero |
|---|---|---|---|
| **A. Verdad propia** | tú (anotación), con plantilla y lector hechos por Claude | Terminar el Espanyol 25-30 (acciones) y seguir con **eventos** (tiro, gol, córner, banda, falta, ±1 s, equipo, tracker del ejecutor) en Espanyol y Mallorca (**Mallorca 2.ª parte hecha el 2026-11-12 y Espanyol 1.ª parte el 2026-11-13**, desde el visor; falta un tramo a ciegas). Fijar su papel: ambos **validación**; evaluación = 2 juveniles nuevos sin abrir. Siguiente verdad: equipo de todas las acciones en 5 min, una parte del Espanyol, un juvenil nuevo. | Desbloquea todo lo demás en el dominio real; 0 GPU. |
| **B. Código sin GPU** | sesiones Claude | (1) `build_ground_truth.py` + arnés único `evaluate_match.py` + lector del CSV de anotación; (2) duración de clips 12-30 s; (3) equipo/jugador en eventos, acción "shot" bajo un tiro, barrido de la curva GOAL; (4) GT de highlights apuntando al backend. | Todo reutiliza cachés; cada pieza se mide el mismo día. |
| **C. GPU (cabe hoy: 6,5 GB)** | lanzar de noche, desatendido | **Experimento barato**: escanear con `SNB_pano1` Middlesbrough (GT), Mallorca, Espanyol y Clásico y comparar con `challenge2` (§3.6). | 1 h de GPU decide si tu idea de los renders de SoccerTrack merece 12 h de GPU entera. |

**Primera semana, concreta**
1. ☑ (B) **Hecho 2026-10-21**: `scripts/annotation_csv.py` (lector), `scripts/evaluate_match.py` (arnés único) y `scripts/build_ground_truth.py` (reescrito: Labels-ball, Labels-v2 con desfases por mitad, CSV). Middlesbrough reproduce 87,7/89,1 (F1 88,4) y equipo 68,1 % (231/339; clústeres de pureza 76/62 %) [M]. Espanyol 25-30 contra la hoja en curso (126 filas): acciones P 81,1 / R 84,9 / F1 82,9; equipo **62,1 %** (clúster 0 con pureza 50,8 %); tracker del ejecutor igual al del pipeline 91,5 % [M]. Uso: `python -m scripts.evaluate_match --csv output/annotations/espanyol_damm_Ground_Truth.csv --backend-match f2c9315d3394c6ee --list-errors` (corre en el host, sin numpy).
2. ☑ (C) **Hecho 2026-10-21** (cadena `scripts/shot_benchmark/run_pano1_tv.sh`, 52 min de GPU a 6,2 GB; informes en `output/pano1_tv/`). Resultado mixto, tabla y veredicto en §3.6: `SNB_pano1` **sí ve los 3 goles del Mallorca como tiro** (SHOT+GOAL 0,12/0,01/0,16 → 0,47/0,55/0,41 [M]) pero **hunde la EFL** (tiros Middlesbrough AP 85,0 → 73,8; acciones F1 −14 pt, la clase header desaparece; la cabeza SoccerNet pierde "Goal" en el Clásico). Conclusión: el estilo "cámara única sin zoom" transfiere, pero afinar solo con renders especializa → `SNB_tv1` solo mezclando EFL en cada lote (o backbone congelado) y entrenando las dos cabezas.
3. ☑ (B) **Hecho 2026-10-21** (rama `feat/highlights`): política 12-30 s / gol ≤ 45 s en `pipeline.fit_duration`, repeticiones solo tras gol, tope de fusión 40 s, `evaluate.py` con duración/en rango/Δancla/s antes y después del gol/`ref_clips`, GT apuntando al backend, Middlesbrough añadido, modo `--vlm-cache-only` (sin GPU ni Ollama). Resultado: **20/20 clips en rango** (mediana 28,6 s), **8/8 goles, 0 falsos** [M]. De paso, dos arreglos: el gol se asigna al candidato de los 20 s anteriores al cambio de rótulo (el 1-0 del Mallorca iba 27 s antes) y no se re-ancla por celebración si hay rótulo (en TV las repeticiones van antes). Pendiente de ti: tope 45/50 y 15/10 s previos (tabla en `docs/highlights/evaluacion.md`).
4. ☐ (A) Espanyol 25-30 cerrado; **plantilla de eventos decidida**: la misma hoja y las mismas 4 columnas, con etiquetas `shot`, `goal`, `corner`, `foul`, `free kick`, `penalty`, `kick off`, `out` (el lector las entiende; `dorsal` = tracker del ejecutor, `equipo` = quien ejecuta); anotar la 1ª parte completa. **Hecho el 2026-11-12 en el Mallorca** (2.ª parte entera desde el visor, 721 filas; medida en §7.16 de la comparativa); Espanyol 1.ª parte hecha el 2026-11-13 (hoja fusionada con la de 25-30 min: `output/annotations/espanyol_damm_Ground_Truth_1a_parte.csv`, §7.17).
5. ☑ (B) **Hecho 2026-10-21**: `foul` sigue emitiéndose pero marcado `estimated: true` en `telemetry.json` (`EventSpan.estimated`, `BallSpottingConfig.tdeed_estimated_event_labels`) y en la API (`MatchEvent.estimated`); el panel del frontend aún no lo pinta. Para dejar de emitirlo, quitarlo de `tdeed_event_labels`.

**Segunda semana**: equipo/jugador en eventos + shot como acción + curva GOAL (B); rama nueva de panorámica con tracking por
ventanas + homografía fija + mapa de calor medido con el GSR (§3.2); prueba de `autocal`/`pancheck` de futtraffic (§3.1, media
tarde); experimento mínimo de VLM grande por API sobre los candidatos existentes (§3.5, < 3 $).

## 1. Diagnóstico (medido, 2026-10-19/20)

| Frente | Dónde está | Cuello de botella |
|---|---|---|
| main, TV | Tiros AP ±3 s 75,6 % con `SoccerNetBall_test` en los 3 EFL limpios (incluye Middlesbrough, validación) [M, estudio §2b]. Acciones F1 88,4 % en Middlesbrough [M, output/main_eval_20261015]. Gol como evento 1/5 en el Clásico a umbral 0,3 (GOAL 0,19-0,37 en los 5 goles) [M]. | Gol como evento; eventos sin equipo ni jugador (`EventSpan`); equipo de las acciones ~68 % [M, contra `team` del GT]. |
| main, juveniles | 5-7 tiros/partido; 0/3 goles en Mallorca; 13-14 córners y 17-24 faltas emitidos sin verdad [M, resumen.txt]. Espanyol es 30 fps y `STRIDE=2` fijo hace que T-DEED vea 15 fps en vez de 12,5 [M]. | T-DEED fuera de dominio (no es umbral); cero anotaciones de eventos (el CSV de `output/annotations` era un volcado del pipeline; el `Ground_Truth.csv` que estás corrigiendo es la primera verdad real). |
| highlights | Con marcador 8/8 goles, 0 falsos; lector 29/30 fuera de muestra; sin marcador 8/8 con 1 falso probable [M, docs/highlights/evaluacion.md]. Clips: Clásico mediana 43 s (33-61), Mallorca 28 s (25-65), Espanyol 34,5 s, Middlesbrough 48,5 s [M]. | Duración (nada mira la duración total; el lado posterior llega al tope de 45 s por repeticiones de TV); umbrales sin marcador fijados con esos 8 goles; decisión Ollama/VLM en main. |
| panorámica | Eval tiros 41,6 → 55,2 % con `SNB_pano1`; mAP 35 → 50 [M, estudio §10]. En 128057, 60,5 % frente a 79,5 % con la cámara de referencia: la cámara no acompaña el golpeo (cobertura SHOT 87,7 % frente a 96,5 %) [M]. 132831 (gran angular): base 34,5 %, afinado 59,3 % [M]. | Nada corre desde `main.py` ni el backend (`_CONFIG_OF` no conoce `SNB_pano1`); eval ya mirado 3 veces; sin vídeo panorámico real del usuario. |
| plataforma | Cadena subir → analizar → ver → highlights completa en 4 partidos; análisis a 1,2-2× tiempo real [M, run.json]. | GT fuera de git y sin generador; 4 arneses; `pipeline_commit` nulo; homografía calculada (12 min/partido) y sin consumir; PnLCalib calibra solo 34-71 % de los frames en TV [M]. |

GPU: otro proceso ocupa ~15 GB de 24 [M, nvidia-smi 2026-10-20]. Caben escaneos T-DEED (6,5 GB) y V-JEPA2 (2,3 GB); no caben
el afinado (18,7 GB), vLLM fp8 (10,8 GB) ni `qwen3.5:9b-q8_0` entero (10,7 GB). `/mnt/datasets` está montado en el host, no
en el contenedor (copiar al proyecto).

## 2. Hoja de ruta por frente

### 2.1 main / eventos ("lo que haya debe funcionar bien")

| | Paso | Criterio de hecho | Coste | Verdad |
|---|---|---|---|---|
| ☑ | **Arnés único** `scripts/evaluate_match.py` (acciones, eventos emparejados 1 a 1 con \|Δt\|, goles, equipo; fps real de `video.json`) + reescribir `scripts/build_ground_truth.py` + sacar `ground_truth/*.gt.json` del `.gitignore:20` | Hecho 2026-10-21: reproduce 87,7/89,1; equipo 68,1 %; gt.json y highlights_goals.json en git. El GT histórico lleva 4 anotaciones a mano (2 FOUL, 1 KICK-OFF, 1 SHOT a 9:52) que el generador no produce: se conserva tal cual | 1-2 d | Existente |
| ☑ | **Lector del CSV de anotación** (`tiempo,etiqueta,dorsal(=tracker_id),equipo`, filas en blanco y marcas `---` ignoradas) y plantilla de eventos | Hecho 2026-10-21 (`scripts/annotation_csv.py`, avisa de equipos/etiquetas ilegibles con nº de línea); Espanyol 25-30 medido | 0,5 d | A |
| ☑ | **Partición de juveniles** en `docs/particion_datasets.md` §6 (Mallorca, Espanyol = validación; 2 nuevos = evaluación) + §4 SoccerNet Tracking (el detector `yolo11s_snt_ball.pt` ya se entrenó con él) + Clásico declarado in-sample para T-DEED | Hecho 2026-10-21 | 1 h | — |
| ☑ | Decisión sobre `foul` (estimado / fuera) | Hecho 2026-10-21: marcado `estimated` (config + telemetría + API); falta pintarlo en el panel | 10 min | Existente |
| ☑ | **Tracker BoT-SORT con GMC y ReID en `main`** (sustituye a ByteTrack; `tracker="bytetrack"` sigue como opción) | Hecho 2026-11-06: test de SoccerNet-Tracking con las mismas cajas HOTA 56,4 → 59,4, IDF1 66,0 → 70,5, IDSW −14 % (la GMC sola −27 %, concentrado en paneos); reentradas 1 → 30 de 720; pipeline ~28 → ~15 fps (`docs/comparativa_benchmark.md` §7.10) | 1 d | B |
| ☐ | **Equipo y jugador en los eventos**: `EventSpan` con frame del pico, tracker y equipo del poseedor; re-etiquetar como `shot` la acción ±0,5 s bajo cada SHOT/GOAL emitido (no meter SHOT en `BALL_ACTION_LABELS`: el NMS y el umbral de acciones lo pisarían); `shot` en arnés, adaptador y vocabulario del frontend | telemetry.json de los 4 partidos con `team`/`tracker_id` en eventos; exactitud de equipo informada | 2 d | Existente |
| ☐ | **Curva GOAL**: barrido de GOAL y de "Goal" (cabeza SoccerNet) con lector Labels-ball en `tune_tdeed_event_thresholds.py` sobre `_test` (7 goles limpios EFL) y `challenge2` en los 10 SN-v2 (25 goles, in-sample) y el Clásico (5) | Tabla en estudio §12, aunque sea negativa (esperable [H]: mantener 0,3 y confiar el gol al marcador/saque de centro) | 0,5 d | Existente |
| ☐ | **Clásico como GT de TV**: alinear `Labels-v2.json` de SoccerNet (5 Goal, 20 tiros, 10 Corner, 18 Foul, 29 Throw-in) con el vídeo de YouTube (desfases ~359 s y ~3.554 s [M, preliminar]) y medir córner/falta/tiro/banda. In-sample para T-DEED: cifras optimistas, pero es la única verdad de córner y falta en TV | Informe por evento del Clásico | 0,5 d | Existente |
| ☐ | Córner, banda, tiro y falta en el juvenil anotado con `respot_telemetry.py`; prueba de remuestreo a 25 fps del Espanyol | Tabla P/R por evento en Espanyol | 1 d + 20 min GPU | A |
| ☐ | Confirmación de gol en main: por **saque de centro visto** (geometría del tracking, ya existe en highlights) y, si se decide el VLM, por marcador | Clásico 1 → 5 goles; Mallorca 0 → 3 | 1 d | Existente |
| ☑ | **Experimento barato SNB_pano1 en TV** (§3.6) | Hecho 2026-10-21: tabla en §3.6; señal en juveniles, pérdida en EFL | 1 h GPU | Existente |
| ☐ | **Etiqueta de las acciones: combinada o spotter** (hallazgo 2026-10-21 al regenerar con `respot_telemetry`, que etiqueta solo con T-DEED): Middlesbrough spotter 90,3/91,8 frente a combinada 87,7/89,1 (+2,7 F1), pero Espanyol 25-30 spotter 72,7/76,2 frente a combinada 81,1/84,9 (−8,5) [M]. El R3D-18 ayuda fuera de la EFL y estorba dentro | Decidir con el juvenil anotado entero; mientras, combinada (la de producción) | 0 | A |
| ☐ | **SNB_tv1** (§3.6 dio señal el 2026-10-21): `challenge2` + EFL train (4) **mezclado en cada lote** con renders de referencia de SoccerTrack train (+ juveniles de entrenamiento anotados cuando existan); entrenar las dos cabezas (SN con sus 10 partidos de ajuste) o `--freeze-backbone`; antes, `tdeed_finetune_data.py` tiene que aceptar EFL plano y mezclar fuentes (~0,5 d, CPU) | No bajar > 2 pt en Middlesbrough (tiros 85,0, acciones F1 ~91 con etiqueta spotter); header y cabeza SN estables; los 3 goles del Mallorca > 0,3 como tiro; precisión de tiros en el juvenil anotado | 2-3 d + 10-12 h GPU entera (hoy 8,8 GB libres: no cabe) | A (Mallorca 2.ª parte anotada: 10 tiros y 47 acciones que producción no ve, §7.16; falta el Espanyol) |
| ☐ | **Combinación T-DEED + AdaSpot en el pipeline** (§7.15; el 2026-11-13 pareció descartada porque AdaSpot-b se apaga en el Mallorca, §7.16; **reabierta el 2026-11-14** con §7.17: AdaSpot-s, o -b con BatchNorm recalculada por vídeo, responde en amateur y recoge el 19-68 % de lo que producción no ve; la media simple pierde precisión, así que tiene que ser la combinación aprendida de §7.15 §7 con rasgos de dominio): puntuar el partido también con AdaSpot-b y promediar (0,5 / 0,5) las curvas de las 10 clases comunes antes de los umbrales automáticos; FREE KICK y GOAL siguen saliendo solo de T-DEED | Acciones en Middlesbrough y en el juvenil anotado no peor que hoy; menos falsos positivos fuera de la EFL (FOOTPASS: 2.170 → 1.072 con producción); coste medido en un partido | 1-2 d; ~2× el tiempo de spotting, +4-7 GB de VRAM | Existente |
| ☐ | **Segunda etapa sobre los spotters congelados con el train de FOOTPASS** (§7.15 §6-§8; pendiente de tu visto bueno): los eventos raros se detectan (tiro, centro y bloqueo, 93-100 %) pero se etiquetan mal, y con la etiqueta perfecta el mAP pasaría de 66 a 87. Descargar el train (48 partidos, ~175 GB en Full HD; hay 803 GB libres), puntuarlo con T-DEED y AdaSpot-b (solo inferencia) y entrenar `scripts/benchmark/stack_scores.py` en CPU | Subir tiro, centro, bloqueo y entrada en FOOTPASS val y en Stoke/Reading frente a la media simple (54,3 y 64,0) | descarga + ~12 h de GPU de inferencia a 4-7 GB [NV] + CPU | Existente |
| ☐ | **Cabeza de clasificación con pesos por clase** (§7.15 §8; pendiente de tu visto bueno por la GPU): reentrenar solo la cabeza de T-DEED con SN-BAS + SoccerTrack v2 mezclados (221 entradas, 253 bloqueos y 183 tiros frente a 34, 128 y 100) y pérdida equilibrada; es `SNB_tv1` con el backbone congelado | No bajar > 1 pt en Stoke y Reading; subir bloqueo, entrada y tiro | conversor de datos (0,5 d, CPU) + GPU; VRAM con backbone congelado sin medir | Existente |
| ☐ | **Fusión por evento con filtro de acuerdo** (regla del ensemble de PAVE, `docs/estado_del_arte/pave_valoracion.md` §4.1; pendiente de tu visto bueno y **solo para TV/FOOTPASS**: en amateur AdaSpot-b no responde, §7.16): agrupar los eventos ya decodificados de T-DEED y AdaSpot-b a ±0,5 s y descartar los que apoya un solo modelo, salvo las clases que solo ve T-DEED (tiro, banda, entrada) | Falsos positivos en FOOTPASS val por debajo de 1.377 sin perder tiro ni banda; mAP no peor que 54,3 | 0,5 d, 0 GPU | Existente |
| ☐ | **Estado del juego en la segunda etapa** (`docs/estado_del_arte/pave_valoracion.md` §4.2; pendiente de tu visto bueno): posiciones y velocidades del h5 táctico de FOOTPASS val como entrada de `stack_scores.py`, 2 partidos entrenan y 1 predice; después, lo mismo con nuestro tracking | Subir banda, centro, conducción y tiro frente a §7.15 §7 (mAP 55,1); si no suben, cerrar la vía antes de descargar el train | 1 d, 0 GPU | Existente |
| ☐ | Coherencia temporal del balón en posesión si el equipo no supera el umbral acordado | Equipo > umbral | 2-3 d | Existente |
| ☐ | GT de Stoke y Reading desde Labels-ball con el generador reescrito (cachés `720p.tracks/possession` ya existen: 0 GPU) | Única cifra final limpia del pipeline en TV (`SoccerNetBall_test`) | 1 d | Existente |

### 2.2 highlights (rama `feat/highlights`)

| | Paso | Criterio de hecho | Coste |
|---|---|---|---|
| ☑ | **Duración 12-30 s con excepciones** (§3.4): `clip_min_s` 12, `clip_max_s` 30, `clip_max_goal_s` 45, `clip_back_max_s` 12 (15 en gol), `end_replay_max_s` 8, `clip_max_merged_s` 40; `fit_duration` tras fijar límites; replays solo cuentan como reacción en gol; tope de gol contado desde el gol (`goal_time`: cambio de rótulo − 4 s; sin rótulo, primer frame de celebración − 4 s solo con desplazamiento ≥ 10 s); `_merge_close` con tope | Hecho 2026-10-21: 20/20 en rango, goles con 12-38 s de celebración, 8/8 y 0 falsos, 10 tests (`tests/test_highlights_duration.py`). Los clips del backend siguen siendo los del 18-oct hasta regenerar | 0,5-1 d, 0 GPU |
| ☑ | `evaluate.py`: `duration_s`, `in_range`, `anchor_delta_s`, `start_before_goal_s`, `end_after_goal_s`, bloque `ref_clips` con límites dados por el usuario (Messi 6356-6406); GT de Mallorca/Espanyol apuntando a `backend/storage/outputs/*/full/highlights`; Middlesbrough con `goals: []`; `--vlm-cache-only` | Hecho 2026-10-21 | 0,5 d |
| ☐ | Pedir al usuario 2-3 referencias más de corte (una ocasión sin gol en TV, un gol de cámara única) antes de cerrar `clip_back_max_goal_s` y `clip_max_goal_s` (variantes ya medidas: 45/15 IoU 0,63 con Messi; 45/10 0,78; 50/10 0,88) | Referencias en `highlights_goals.json` | 15 min |
| ☑ | Propagar `goal_source`, `vlm.used`, `vjepa.used`, `scoreboard.*` y estado "degradado" a `HighlightsResponse` (`vlm_used`, `vlm_degraded`, `vjepa_used`, `scoreboard_available`, `n_scoreboard_changes`, `goal_source` por clip; test en `backend/tests/test_highlights.py`). Pendiente: panel del frontend; test "sin VLM nunca se etiqueta gol salvo por saque de centro"; pytest en la imagen (hoy se corre con `backend/.venv/bin/pytest` dentro del contenedor) | API hecha 2026-10-21; panel pendiente | 0,5 d |
| ☐ | Criterio de aceptación escrito en `docs/highlights/evaluacion.md` (propuesta: en N partidos nuevos con marcador como verdad, goles en clip ≥ 95 %, 0 "Gol" confirmados falsos, ≤ 1 "Gol probable" falso por partido sin marcador, ≤ 30 min/partido) | Sección escrita | 1 h |
| ☐ | Validación sin marcador fuera de muestra: (a) regresión barata en 7 EFL con cachés (fusión in-sample allí); (b) 3-5 SN-v2 **sin asignar** (los 10 copiados están en el entrenamiento del SVC), copiados desde `/mnt`, marcador como verdad | Tabla goles/falsos por partido | ≥ 3 h GPU por partido |
| ☐ | Experimento mínimo de VLM grande (§3.5) | AP de `is_goal`/`excitement` y Spearman con la nota del usuario, por modelo | 2-3 h, < 3 $ |
| ☐ | Decisión VLM en main (Ollama externo con degradación visible; vLLM fp8 no cabe hoy) y **merge** (un commit aquí, uno en el frontend) | Mergeado | 2 h |
| ☐ | Después: lector con rótulo oculto (PSG-Metz), `home`/`away` en títulos; publicador YouTube solo `--dry-run` con ≥ 10 GB libres y `/opt/venv-molmo` creado | — | — |

### 2.3 panorámica (rama nueva desde main; no se mergea hasta que esté completa)

| | Paso | Criterio de hecho | Coste |
|---|---|---|---|
| ☐ | Crear `feat/panorama-pipeline` desde `main`; código en módulos propios (`core/panorama/`) para que `git merge main` sea casi sin conflictos; cambios pequeños y neutros en main (registrar `SNB_pano1` en `_CONFIG_OF`, `PanoramaConfig`, valor `panorama` en `AnalysisMode`) | Rama creada | 1 h |
| ☐ | Clases emitidas: PASS, DRIVE, THROW IN, HIGH PASS, CROSS, SHOT+GOAL, OUT; fuera TACKLE (7 %) y BALL PLAYER BLOCK (6 %); FREE KICK a decisión del usuario. Umbral de tiros **0,30** desde validación (P 75 / R 64 [M, report_SNB_pano1_val.txt]), nunca el 0,13 que sale del eval | Lista y umbrales en `PanoramaConfig` | 1-2 d, 0 GPU |
| ☐ | **Tracking fuera de la cámara virtual** (tu idea, §3.2): detección por ventanas sobre la panorámica entera unificando persona+balón en un detector (evita doble pasada), ByteTrack a fps reducido, posesión en metros | Tracks de 117093/132877 medidos contra GSR `bbox_pitch` (recall, error mediano en m, fragmentación) | 3-5 d, ~0 GPU extra si se unifica |
| ☐ | **Homografía fija por mitad**: TPS píxel→metros desde los 65 keypoints (mediana 1,3-1,9 m [M, leave-one-out]; una homografía plana sobre el frame crudo falla, 49 m, por la distorsión [M]); en producto, calibración manual de 8-12 puntos en el frontend o PnLCalib una vez (no probado) | `tracks_2d.jsonl` + `heatmap.json` por equipo y jugador al final del análisis; capa `heatmap` del frontend activada | 2-3 d + 1-2 d de UI |
| ☐ | Probar `autocal.py` y `pancheck.py` de futtraffic (§3.1) sobre los 10 partidos: error de reproyección contra los 65 keypoints | Si ≤ 1-3 m: calibración sin clics para producto | 0,5 d, CPU |
| ☐ | Pipeline completo `--mode panorama` de punta a punta en 117093/132877: yolo11s_pano → Viterbi+L1 → render 796x448 → T-DEED `SNB_pano1` → tracking por ventanas → homografía fija → analytics → heatmap; pesos del detector a `models/`; polígono de campo propio en vez de `raw/*_keypoints.json` | Un `telemetry.json` panorámico con eventos, equipos y mapa | 3-5 d, ~35 min GPU/partido |
| ☐ | **Una** ronda de cámara (2-3 variantes) sobre los 214 tiros de train+val con el modelo base; go/no-go ≥ +5 pt AP±3 y cobertura SHOT ≥ 92 %; si no, cerrar por escrito | Informe | 3,5 h GPU/variante |
| ☐ | Vídeo panorámico real tuyo con lista de tiros (riesgo de dominio: un montaje no visto dio 34,5 % sin afinar) | AP±3 medido | 1 h + 35 min GPU |
| ☐ | Al final, con el frontend: `video_kind` ('broadcast' / 'panorama') elegido al subir con sugerencia automática (aspecto > 2,2 o ancho ≥ 3840; 117092 da 2,01 y 132831 2,55: el aspecto solo no basta); qué vídeo lleva overlays (4K original, render 796x448 o ambos). Condición de merge: cadena completa en val + una medida en eval + decisión de UI | Mergeado | 1-2 d |

### 2.4 plataforma

| | Paso | Coste |
|---|---|---|
| ☐ | Higiene: rotar el token que llevan los remotos de los dos repos; fps real en informes (Espanyol 136 min, no 163); `safe.directory` para `pipeline_commit`; commit y fps en `telemetry.json`; pytest en la imagen; renombrar los CSV exportados | 2-4 h |
| ☐ | Estadísticas por partido primero; por equipo cuando los eventos lleven equipo y la exactitud supere el umbral acordado; faltas "estimado" | 2-3 d |
| ☐ | Homografía en TV: consumirla (minimapa/heatmap con suavizado; solo 34-71 % de frames calibrados [M]) o `--skip-homography` por defecto (ahorra ~12 min/partido) | 3-5 d o 0 |
| ☐ | Dockerfile reproducible (T-DEED, PnLCalib, prtreid a SHA, parches de Soccer_Master, lock) | 1 d |
| ☐ | Posponer PostgreSQL, Q&A (`orquestrator/` estaba vacío y se borró el 2026-11-09) y re-identificación | — |

## 3. Valoración de las ideas del 2026-10-20

Texto completo (veredictos, diseño, comandos y tablas) en `docs/archivo/hoja_de_ruta_valoracion_ideas_2026-10-20.md`. Una
línea por idea:

| Idea | Veredicto | Cifra clave [M] | Dónde sigue |
|---|---|---|---|
| 3.1 Comparar la cámara virtual con futtraffic | No es una cámara virtual (pipeline por jugador para panorámica fija); prestar `autocal.py` (calibración sin clics) y `pancheck.py` (fijo frente a operador) | Su tracking en 128057: HOTA 0,38, por debajo del ByteTrack del leaderboard (0,42) | §2.3 |
| 3.2 Tracking fuera de la cámara virtual + homografía + mapa | Viable y la mejora de más valor de la rama panorámica: una homografía por mitad; diseño de 4 ventanas de 1024 + ByteTrack + TPS | TPS píxel→metros 1,3-1,9 m de mediana (p90 ~5 m); PnLCalib calibra el 34 % de los frames del Mallorca | §2.3 (3-5 d) |
| 3.3 No mergear la panorámica todavía | De acuerdo: rama nueva desde main y `git merge main` tras cada entrada | — | §2.3 |
| 3.4 Clips de 12-30 s con excepciones | Hecho el 2026-10-21: 12-30 s, gol ≤ 45 s contados desde el gol real, repeticiones solo tras gol | 5/19 clips en rango antes; 20/20 después, 8/8 goles | §2.2 y `docs/highlights/evaluacion.md` |
| 3.5 VLMs más grandes y «puntuar jugadas» | 25 trabajos verificados: ningún VLM puntúa jugadas mejor que las señales clásicas; el patrón que funciona es LLM sobre eventos ya extraídos. Experimento mínimo con Gemini (2-3 h, < 3 $) pendiente de tu decisión | SVHighlights: el mejor VLM+LLM acierta el primer highlight el 26,6 %; `is_goal` del 9B 4/5 goles y 8 falsos | §2.2, §4 |
| 3.6 Renders de SoccerTrack como datos para el T-DEED de TV | Hecho el 2026-10-21: afinar solo con renders especializa (Middlesbrough −11 pt en tiros, −14 en acciones, header 0/29) pero el estilo «cámara única sin zoom» transfiere → `SNB_tv1` solo mezclando EFL en cada lote o con backbone congelado | Mallorca, SHOT+GOAL en los 3 goles: 0,12 / 0,01 / 0,16 → 0,47 / 0,55 / 0,41 | §2.1 (SNB_tv1) |

## 4. Qué no hacer (vías agotadas, con cifra)

- Verificador VLM de tiros en cualquier variante: AUC tiro/centro 0,50-0,65 frente a 0,87 de T-DEED [estudio §2a-§5].
- V-JEPA2 sin recorte (31 % frente a 31-32 %) o como fuente de eventos (AP 34 %).
- Seguir ajustando la cámara L1 sola (70,2 y 70,9 frente a 70,7, ruido ±5) o recortes fijos (24-34 %).
- Mirar otra vez 128057, 132831, Stoke o Reading para ajustar (eval ya usado; la demo del 18-oct sobre 128057 ya contraviene
  la partición). Mallorca como evaluación (fijó `goal_audio_min_db` y las anclas).
- Bajar umbrales en juveniles (no hay señal); `auto_event_min_factor` < 1 (tiro 0,66 → 0,62); emitir card/kick-off/offside
  (F1 ≤ 0,35).
- Usar los 10 SN-v2 copiados para validar la fusión V-JEPA2 (están en su entrenamiento) ni como "limpios" para T-DEED
  (89 % de picos a ±0,15 s). Tratar `output/annotations/espanyol_damm_ball_actions_*.csv` como verdad (son el pipeline).
- `vlm_max_candidates` alto, `is_goal` del VLM o fusión sin celebración (8-9 falsos); `--force` en highlights (rehace todo).
- Afinados u Omni con la GPU compartida (3 OOM en octubre); publicador YouTube, PostgreSQL y Q&A ahora.
- Benchmark completo de futtraffic; comparar con cámaras de zoom variable.
- Reentrenar AdaSpot con RoI supervisada: con pesos congelados la RoI anotada queda 1-2 pt por debajo de la saliencia y no
  recupera los eventos que el selector deja fuera (§7.14). En la fusión de spotters: pesos por clase, un tercer modelo,
  un segundo T-DEED o la media geométrica (de ±0,5 a −4 pt, §7.15).
- AdaSpot-b original en el Mallorca (cámara baja tras cristal, sol): se apaga (probabilidad media de acción 0,005 frente a 0,05 en EFL;
  recall 3 %, §7.16). No generalizar a «amateur»: en el Espanyol responde (60 / 48) y la BatchNorm por vídeo o AdaSpot-s lo reflotan
  en parte (§7.17). Promediar sus curvas con T-DEED: sube el recall y hunde la precisión; combinar con pesos aprendidos.

## 5. Ramas

| Rama | Estado [M, 2026-10-20] | Decisión |
|---|---|---|
| `feat/highlights` | 1 commit propio (88e1cbd; ca27f22 ya en main por patch-id) | Mergear tras §2.2 (duración + validación + decisión VLM); a la vez el frontend (6c983af) |
| `feat/T-DEED` | 0 por delante; el código VLM se recupera con `git revert 6ceb32b` sin la rama | Tag `archive/` opcional y borrar |
| `feat/panorama-shots` | 0 por delante, mergeada | Borrar; el desarrollo nuevo va en `feat/panorama-pipeline` |
| `videoARM` | 0 por delante; vía agotada | Tag `archive/` y borrar |
| `VLLM` | 1 commit único (events_omni, 55 % en clips sesgados, 22 GB) | No mergear; tag `archive/` y borrar |
| `basketball` | 0 por delante. La carpeta `basketball/` salió de `main` el 2026-11-09 (ahora en `/media/dd3/ner-training/basketball_Sports-Classification`, con sus datasets y modelos) | Conservar la rama: es la copia versionada del código |

## 6. Riesgos transversales

- **GPU compartida**: `gpu_lock` no gobierna Ollama ni procesos ajenos; planificar qué entra cuando se libere (afinados,
  vLLM, análisis de SN-v2).
- **Evaluación contaminada**: todas las cifras de producción en TV son validación (`challenge2` entrenó con 6/7 EFL y
  preentrenó con SN-v2); solo `SoccerNetBall_test` se mide limpio; Mallorca/Espanyol ya tocados.
- **Dependencias**: Ollama ajeno (fallo silencioso → solo reglas), prtreid sin SHA, cachés T-DEED viejas (2-oct) que
  `ensure_scores` rehace en GPU bajo pasos "0 GPU", `labels_path.txt` del repo T-DEED sobrescrito hacia las etiquetas
  panorámicas.
- **GPL-3.0 de T-DEED**: `SNB_pano1` y cualquier `SNB_tv1` son derivados; decidir antes de distribuir.
- **Seguridad**: token en los remotos de los dos repos; salidas propiedad de root.
- **Confianza del usuario final**: no mostrar estadística por equipo con ~68 % de acierto ni faltas sin validar sin marcarlo.

## 7. Bitácora

Una línea por sesión. La entrada completa, con cifras y ficheros, está en `docs/archivo/hoja_de_ruta_bitacora.md`.

- **2026-10-19**: hoja de ruta inicial; hallazgos de partida (T-DEED ciego en juveniles, GT principal fuera de git).
- **2026-10-20**: segunda ronda sobre las ideas del usuario (§3); decisiones: `main` debe funcionar bien, la panorámica
  sigue en rama, clips de 12-30 s con excepciones.
- **2026-10-21**: carril B, semana 1 (lector del CSV, arnés único, GT en git, duración de clips, `foul` estimado) y
  experimento `SNB_pano1` en TV (§3.6).
- **2026-11-02**: modelo de SoccerHigh portado e integrado en highlights (`docs/highlights/soccerhigh.md`).
- **2026-11-06**: BoT-SORT + GMC + ReID OSNet sustituye a ByteTrack en `core/tracking.py` (59,4 HOTA).
- **2026-11-08**: estado del arte en tracking (`docs/estado_del_arte/tracking.md`).
- **2026-11-08**: tabla única de spotters (`docs/comparativa_benchmark.md` §7.11); el test de SN-v2 está limpio para T-DEED.
- **2026-11-09**: punto de trabajo P/R/FP y metraje nuevo PCBAS (§7.12); cabeza de 17 eventos de SN-v2 (§7.13).
- **2026-11-09**: limpieza del repo: basket fuera, docs y README partidos en resumen + detalle, `CLAUDE.md` nuevo, SoccerMaster sin lo que no se usa.
- **2026-11-09**: artículos de `docs/estado_del_arte/papers/` (T-DEED, AdaSpot, SoccerNet 2026) pasados a Markdown.
- **2026-11-09**: experimento de RoI anotada frente a saliencia en AdaSpot (§7.14 de la comparativa), rama `exp/adaspot-new-roi`; cola desatendida en marcha.
- **2026-11-11**: §7.14 cerrado (el recorte no es lo que frena a AdaSpot) y §7.15: anatomía de errores y fusión T-DEED + AdaSpot-b (SN-BAS test 62,5 → 64,0 de mAP; FOOTPASS 50,1 → 54,3 y F1 64,6 → 75,4).
- **2026-11-11**: anotación de la verdad desde el visor: el editor de clips corrige, borra y añade acciones y eventos sobre una copia revisada (`backend/…/review/analysis.json`) y exporta la hoja `tiempo,etiqueta,dorsal,equipo` de un tramo.
- **2026-11-12**: el editor atascaba el túnel SSH (38 MB de vídeo por apertura para las miniaturas): ahora las corta el backend (`/api/videos/{id}/thumbnail`, 40 KB por apertura) y la lista de acciones solo repinta la fila que cambia.
- **2026-11-12**: valoración de PAVE (ganador de PCBAS 2026) en `docs/estado_del_arte/pave_valoracion.md`: no replicable ni útil entero (su palanca es el estado del juego del dataset, no la visión); lo que se traslada es el filtro de acuerdo en la fusión y el estado del juego en la segunda etapa (2 pasos a 0 GPU propuestos en §2.1).
- **2026-11-13**: primera medida en verdad propia del dominio (Mallorca 2.ª parte, 721 filas del visor) y en Middlesbrough 0-20 min, T-DEED frente a AdaSpot (§7.16): producción 79 / 81 (umbral automático 0,27 ≈ óptimo 0,25) pero 2 % en lo añadido a mano y 30 % en tiros; AdaSpot-b se apaga (3 %) y la fusión queda descartada para amateur; `eval_gt_prf.py`; AdaSpot puntuable en CPU. Docs: `estado_del_arte/` (con `papers/`), `highlights/`, §3 archivado íntegro, baloncesto al archivo.
- **2026-11-14**: reto More than a Hack 2027 de Metrica (posesión H/A/D por fotograma con el balón ausente el 41 %, métrica spellF1 por tramos; entrega 17-11): investigación SOTA en `docs/investigacion/`, paquete `core/team_possession/` (EPTS, IDSSE, nuestro `tracks_2d.jsonl`, reglas, Viterbi, scorer oficial) y banco IDSSE degradado (§7.18): línea base 0,26, clasificador por fotograma con contexto + Viterbi **0,66**; transfiere a Metrica (0,58 frente a 0,43). Falta el partido del reto (inscripción) y probarlo en un partido nuestro. Todo en CPU; rama `exp/mtah-posesion`.
- **2026-11-14**: Espanyol 1.ª parte anotada (visor + hoja de 25-30 min fusionadas) y medida; AdaSpot probado en los dos juveniles (§7.17): -b se apaga en el Mallorca por el punto de vista de la cámara y responde en el Espanyol; BatchNorm recalculada por vídeo y AdaSpot-s lo reflotan en parte; donde responde recoge lo que producción no ve, pero la media simple pierde precisión → combinación aprendida (reabierta en §2.1). Pendiente: tramo anotado a ciegas.
- **2026-11-17**: estado del arte del procesado de cámara previo al análisis en `docs/investigacion/reports/Estado del arte procesado de cámara.md`: nadie ha medido el efecto de la cámara virtual, la restauración o la estabilización sobre el spotting amateur; lo medido favorece adaptar el modelo y garantizar la escala en píxeles, no restaurar la imagen; la caída a un tercio de Cabado et al. sale de un partido por dominio y la cámara única con operador profesional no pierde. Pendiente de decisión: qué experimento propio lanzar (entrada transformada frente a modelo afinado).
- **2026-11-26**: primera verdad anotada a ciegas (La Roca–Montmeló, 1.ª parte, 457 filas) y siete spotters medidos sobre ella (§7.19): todos empatan en F1 43-47 frente a 76-80 en los juveniles del visor, y fallan por confianza, no por instante ni por etiqueta. Resumen de todos los modelos en todos los conjuntos, con umbral óptimo y precisión / recall, en §7.20. Pendiente: revisar tres tramos sin anotar, fijar el papel de La Roca y Vandalia en la partición y decidir la prueba a 25 fps.
