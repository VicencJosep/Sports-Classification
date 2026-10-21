# Hoja de ruta — Sports-Classification

Documento vivo. Última actualización: **2026-10-21**. Al final de cada sesión se actualizan las casillas (☐ → ☑) y se
añade una entrada en la bitácora (§7). Cada cifra lleva su fuente; **[M]** = medido, **[H]** = hipótesis.
Los commits los hace el usuario.

## 0. Por dónde empezar

**Lo que más cojea** no es un modelo: es que el dominio del producto (juveniles, cámara única) no tiene ni una anotación
de eventos y T-DEED está ciego ahí (SHOT ≤ 0,16 y GOAL ≤ 0,03 en los 3 goles del Mallorca [M]; 5-7 tiros por partido
frente a 20-30 habituales). Sin verdad propia no se puede ni medir ni afinar, y las dos ideas más caras (afinar T-DEED,
VLMs grandes) se decidirían a ojo. Lo segundo que cojea es la infraestructura de medida: el único GT de acciones
(Middlesbrough, 20 min) está fuera de git y sin generador, hay cuatro arneses distintos y `pipeline_commit` sale nulo.

**Orden recomendado** (dos semanas), en tres carriles que no se bloquean entre sí:

| Carril | Quién | Qué | Por qué primero |
|---|---|---|---|
| **A. Verdad propia** | tú (anotación), con plantilla y lector hechos por Claude | Terminar el Espanyol 25-30 (acciones) y seguir con **eventos** (tiro, gol, córner, banda, falta, ±1 s, equipo, tracker del ejecutor) en Espanyol y Mallorca. Fijar su papel: ambos **validación**; evaluación = 2 juveniles nuevos sin abrir. | Desbloquea todo lo demás en el dominio real; 0 GPU. |
| **B. Código sin GPU** | sesiones Claude | (1) `build_ground_truth.py` + arnés único `evaluate_match.py` + lector del CSV de anotación; (2) duración de clips 12-30 s; (3) equipo/jugador en eventos, acción "shot" bajo un tiro, barrido de la curva GOAL; (4) GT de highlights apuntando al backend. | Todo reutiliza cachés; cada pieza se mide el mismo día. |
| **C. GPU (cabe hoy: 6,5 GB)** | lanzar de noche, desatendido | **Experimento barato**: escanear con `SNB_pano1` Middlesbrough (GT), Mallorca, Espanyol y Clásico y comparar con `challenge2` (§3.6). | 1 h de GPU decide si tu idea de los renders de SoccerTrack merece 12 h de GPU entera. |

**Primera semana, concreta**
1. ☑ (B) **Hecho 2026-10-21**: `scripts/annotation_csv.py` (lector), `scripts/evaluate_match.py` (arnés único) y `scripts/build_ground_truth.py` (reescrito: Labels-ball, Labels-v2 con desfases por mitad, CSV). Middlesbrough reproduce 87,7/89,1 (F1 88,4) y equipo 68,1 % (231/339; clústeres de pureza 76/62 %) [M]. Espanyol 25-30 contra la hoja en curso (126 filas): acciones P 81,1 / R 84,9 / F1 82,9; equipo **62,1 %** (clúster 0 con pureza 50,8 %); tracker del ejecutor igual al del pipeline 91,5 % [M]. Uso: `python -m scripts.evaluate_match --csv output/annotations/espanyol_damm_Ground_Truth.csv --backend-match f2c9315d3394c6ee --list-errors` (corre en el host, sin numpy).
2. ☐ (C) **En marcha**: cadena desatendida lanzada el 2026-10-21 08:19 dentro del contenedor (`scripts/shot_benchmark/run_pano1_tv.sh`), espera a que haya ≥ 8 GB de VRAM (ese día la GPU estaba a 23/24 GB por tres procesos ajenos), escanea los 5 vídeos y lanza `pano1_tv_report.sh`. Seguimiento: `output/pano1_tv/chain.log`; resultados en `output/pano1_tv/*.txt`. Criterio: tabla AP tiros / F1 acciones Middlesbrough y P(SHOT+GOAL) en los goles del Mallorca, base frente a afinado.
3. ☑ (B) **Hecho 2026-10-21** (rama `feat/highlights`): política 12-30 s / gol ≤ 45 s en `pipeline.fit_duration`, repeticiones solo tras gol, tope de fusión 40 s, `evaluate.py` con duración/en rango/Δancla/s antes y después del gol/`ref_clips`, GT apuntando al backend, Middlesbrough añadido, modo `--vlm-cache-only` (sin GPU ni Ollama). Resultado: **20/20 clips en rango** (mediana 28,6 s), **8/8 goles, 0 falsos** [M]. De paso, dos arreglos: el gol se asigna al candidato de los 20 s anteriores al cambio de rótulo (el 1-0 del Mallorca iba 27 s antes) y no se re-ancla por celebración si hay rótulo (en TV las repeticiones van antes). Pendiente de ti: tope 45/50 y 15/10 s previos (tabla en `docs/highlights_evaluacion.md`).
4. ☐ (A) Espanyol 25-30 cerrado; **plantilla de eventos decidida**: la misma hoja y las mismas 4 columnas, con etiquetas `shot`, `goal`, `corner`, `foul`, `free kick`, `penalty`, `kick off`, `out` (el lector las entiende; `dorsal` = tracker del ejecutor, `equipo` = quien ejecuta); anotar la 1ª parte completa.
5. ☑ (B) **Hecho 2026-10-21**: `foul` sigue emitiéndose pero marcado `estimated: true` en `telemetry.json` (`EventSpan.estimated`, `BallSpottingConfig.tdeed_estimated_event_labels`) y en la API (`MatchEvent.estimated`); el panel del frontend aún no lo pinta. Para dejar de emitirlo, quitarlo de `tdeed_event_labels`.

**Segunda semana**: equipo/jugador en eventos + shot como acción + curva GOAL (B); rama nueva de panorámica con tracking por
ventanas + homografía fija + mapa de calor medido con el GSR (§3.2); prueba de `autocal`/`pancheck` de futtraffic (§3.1, media
tarde); experimento mínimo de VLM grande por API sobre los candidatos existentes (§3.5, < 3 $).

## 1. Diagnóstico (medido, 2026-10-19/20)

| Frente | Dónde está | Cuello de botella |
|---|---|---|
| main, TV | Tiros AP ±3 s 75,6 % con `SoccerNetBall_test` en los 3 EFL limpios (incluye Middlesbrough, validación) [M, estudio §2b]. Acciones F1 88,4 % en Middlesbrough [M, output/main_eval_20261015]. Gol como evento 1/5 en el Clásico a umbral 0,3 (GOAL 0,19-0,37 en los 5 goles) [M]. | Gol como evento; eventos sin equipo ni jugador (`EventSpan`); equipo de las acciones ~68 % [M, contra `team` del GT]. |
| main, juveniles | 5-7 tiros/partido; 0/3 goles en Mallorca; 13-14 córners y 17-24 faltas emitidos sin verdad [M, resumen.txt]. Espanyol es 30 fps y `STRIDE=2` fijo hace que T-DEED vea 15 fps en vez de 12,5 [M]. | T-DEED fuera de dominio (no es umbral); cero anotaciones de eventos (el CSV de `output/annotations` era un volcado del pipeline; el `Ground_Truth.csv` que estás corrigiendo es la primera verdad real). |
| highlights | Con marcador 8/8 goles, 0 falsos; lector 29/30 fuera de muestra; sin marcador 8/8 con 1 falso probable [M, docs/highlights_evaluacion.md]. Clips: Clásico mediana 43 s (33-61), Mallorca 28 s (25-65), Espanyol 34,5 s, Middlesbrough 48,5 s [M]. | Duración (nada mira la duración total; el lado posterior llega al tope de 45 s por repeticiones de TV); umbrales sin marcador fijados con esos 8 goles; decisión Ollama/VLM en main. |
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
| ☐ | **Equipo y jugador en los eventos**: `EventSpan` con frame del pico, tracker y equipo del poseedor; re-etiquetar como `shot` la acción ±0,5 s bajo cada SHOT/GOAL emitido (no meter SHOT en `BALL_ACTION_LABELS`: el NMS y el umbral de acciones lo pisarían); `shot` en arnés, adaptador y vocabulario del frontend | telemetry.json de los 4 partidos con `team`/`tracker_id` en eventos; exactitud de equipo informada | 2 d | Existente |
| ☐ | **Curva GOAL**: barrido de GOAL y de "Goal" (cabeza SoccerNet) con lector Labels-ball en `tune_tdeed_event_thresholds.py` sobre `_test` (7 goles limpios EFL) y `challenge2` en los 10 SN-v2 (25 goles, in-sample) y el Clásico (5) | Tabla en estudio §12, aunque sea negativa (esperable [H]: mantener 0,3 y confiar el gol al marcador/saque de centro) | 0,5 d | Existente |
| ☐ | **Clásico como GT de TV**: alinear `Labels-v2.json` de SoccerNet (5 Goal, 20 tiros, 10 Corner, 18 Foul, 29 Throw-in) con el vídeo de YouTube (desfases ~359 s y ~3.554 s [M, preliminar]) y medir córner/falta/tiro/banda. In-sample para T-DEED: cifras optimistas, pero es la única verdad de córner y falta en TV | Informe por evento del Clásico | 0,5 d | Existente |
| ☐ | Córner, banda, tiro y falta en el juvenil anotado con `respot_telemetry.py`; prueba de remuestreo a 25 fps del Espanyol | Tabla P/R por evento en Espanyol | 1 d + 20 min GPU | A |
| ☐ | Confirmación de gol en main: por **saque de centro visto** (geometría del tracking, ya existe en highlights) y, si se decide el VLM, por marcador | Clásico 1 → 5 goles; Mallorca 0 → 3 | 1 d | Existente |
| ☐ | **Experimento barato SNB_pano1 en TV** (§3.6) | En marcha desde 2026-10-21 (cadena esperando VRAM; `output/pano1_tv/chain.log`) | 1 h GPU | Existente |
| ☐ | **Etiqueta de las acciones: combinada o spotter** (hallazgo 2026-10-21 al regenerar con `respot_telemetry`, que etiqueta solo con T-DEED): Middlesbrough spotter 90,3/91,8 frente a combinada 87,7/89,1 (+2,7 F1), pero Espanyol 25-30 spotter 72,7/76,2 frente a combinada 81,1/84,9 (−8,5) [M]. El R3D-18 ayuda fuera de la EFL y estorba dentro | Decidir con el juvenil anotado entero; mientras, combinada (la de producción) | 0 | A |
| ☐ | **SNB_tv1** (solo si §3.6 da señal): `challenge2` + EFL train (4) + renders de referencia de SoccerTrack train (+ juveniles de entrenamiento anotados cuando existan) | No bajar > 2 pt en Middlesbrough; cabeza SN estable; mejora visible en juveniles; una medida en eval | 2-3 d + 10-12 h GPU entera | A (≥ 2 juveniles más) |
| ☐ | Coherencia temporal del balón en posesión si el equipo no supera el umbral acordado | Equipo > umbral | 2-3 d | Existente |
| ☐ | GT de Stoke y Reading desde Labels-ball con el generador reescrito (cachés `720p.tracks/possession` ya existen: 0 GPU) | Única cifra final limpia del pipeline en TV (`SoccerNetBall_test`) | 1 d | Existente |

### 2.2 highlights (rama `feat/highlights`)

| | Paso | Criterio de hecho | Coste |
|---|---|---|---|
| ☑ | **Duración 12-30 s con excepciones** (§3.4): `clip_min_s` 12, `clip_max_s` 30, `clip_max_goal_s` 45, `clip_back_max_s` 12 (15 en gol), `end_replay_max_s` 8, `clip_max_merged_s` 40; `fit_duration` tras fijar límites; replays solo cuentan como reacción en gol; tope de gol contado desde el gol (`goal_time`: cambio de rótulo − 4 s; sin rótulo, primer frame de celebración − 4 s solo con desplazamiento ≥ 10 s); `_merge_close` con tope | Hecho 2026-10-21: 20/20 en rango, goles con 12-38 s de celebración, 8/8 y 0 falsos, 10 tests (`tests/test_highlights_duration.py`). Los clips del backend siguen siendo los del 18-oct hasta regenerar | 0,5-1 d, 0 GPU |
| ☑ | `evaluate.py`: `duration_s`, `in_range`, `anchor_delta_s`, `start_before_goal_s`, `end_after_goal_s`, bloque `ref_clips` con límites dados por el usuario (Messi 6356-6406); GT de Mallorca/Espanyol apuntando a `backend/storage/outputs/*/full/highlights`; Middlesbrough con `goals: []`; `--vlm-cache-only` | Hecho 2026-10-21 | 0,5 d |
| ☐ | Pedir al usuario 2-3 referencias más de corte (una ocasión sin gol en TV, un gol de cámara única) antes de cerrar `clip_back_max_goal_s` y `clip_max_goal_s` (variantes ya medidas: 45/15 IoU 0,63 con Messi; 45/10 0,78; 50/10 0,88) | Referencias en `highlights_goals.json` | 15 min |
| ☑ | Propagar `goal_source`, `vlm.used`, `vjepa.used`, `scoreboard.*` y estado "degradado" a `HighlightsResponse` (`vlm_used`, `vlm_degraded`, `vjepa_used`, `scoreboard_available`, `n_scoreboard_changes`, `goal_source` por clip; test en `backend/tests/test_highlights.py`). Pendiente: panel del frontend; test "sin VLM nunca se etiqueta gol salvo por saque de centro"; pytest en la imagen (hoy se corre con `backend/.venv/bin/pytest` dentro del contenedor) | API hecha 2026-10-21; panel pendiente | 0,5 d |
| ☐ | Criterio de aceptación escrito en `docs/highlights_evaluacion.md` (propuesta: en N partidos nuevos con marcador como verdad, goles en clip ≥ 95 %, 0 "Gol" confirmados falsos, ≤ 1 "Gol probable" falso por partido sin marcador, ≤ 30 min/partido) | Sección escrita | 1 h |
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
| ☐ | Posponer PostgreSQL, Q&A (`orquestrator/agent.py` está vacío) y re-identificación | — |

## 3. Valoración de las ideas del 2026-10-20

### 3.1 Comparar la cámara virtual con futtraffic

**Veredicto [M, repo leído el 2026-10-20]: no es una cámara virtual.** `danrublop/futtraffic` (Apache-2.0, 7 commits, 1 autor,
creado el 2026-10-10) es un pipeline de datos por jugador para cámara panorámica fija: calibración sin clics →
RF-DETR por tiles de 1024 → C-BIoU → equipos por color → distancias/velocidades/mapas de calor + app local. Su propio código
dice "panning, operator-driven and broadcast footage is out of scope" y no renderiza nada. Tampoco detecta eventos (su
posesión sin balón da F1 0,60). Usa el mismo dataset (SoccerTrack v2) y mide tracking en 128057 (HOTA 0,38, por debajo del
ByteTrack del leaderboard, 0,42). No existe ningún repo abierto con métricas de calidad de cámara (AutoCam-AI, reco
video-stitcher y Soccer-Panaroma-Pipeline no publican precisión): nuestra única vara sigue siendo la cámara de referencia
del dataset.

**Qué sí tomar prestado** (media tarde en total, CPU):
- `model/engine/autocal.py`: calibración de panorámica fija sin clics (modelo esférico que se ajusta a la pintura del campo
  y rechaza si no está seguro). Es justo la pieza que falta para hacer homografía y mapa en panorámica sin los keypoints del
  dataset. Prueba: correrlo sobre las 10 panorámicas y medir el error contra los 65 keypoints. Riesgo [H]: lo ajustaron en un
  solo campo; los montajes 3840x1906 y 3840x1504 pueden hacer que rechace.
- `pancheck.py`: detecta si el metraje es fijo o con operador (SIFT+RANSAC). Responde a "cómo diferenciar panorámica de
  TV" sin pedirlo al frontend.
- Dos puertas geométricas de balón que no tenemos (aspecto 0,55-1,8 y tamaño físico 0,4-3,0× el esperado para 0,22 m):
  se aplican offline sobre los candidatos ya guardados en `output/panorama_ball/*_tiles1024_pano.jsonl`. Expectativa baja:
  el hueco medido está en la cámara, no en el balón.
- Stitcher de trayectorias en metros (9 m/s, 4 m de holgura, huecos de 10 s) para los cortes de identidad en panorámica.
- Disciplina de cifras verificables: cada número del estudio con el informe que lo reproduce.

**No hacer**: su benchmark completo de balón (1 día + noche de GPU) ni comparar con AutoCam-AI (zoom variable: cambia la
escala que ve T-DEED y la comparación sería injusta).

### 3.2 Tracking dentro y fuera de la cámara virtual + homografía + mapa al acabar

**Viable y es la mejora de más valor para la rama panorámica.** La cámara fija convierte la homografía en UNA
transformación por mitad (frente a PnLCalib por frame en TV, que calibra 34 % de los frames en el Mallorca y 71 % en el
Espanyol [M]): el mapa de calor saldría completo. Hechos que condicionan el diseño [M]:
- Una homografía plana sobre el frame crudo falla (49 m de error con los 65 keypoints): hay distorsión de lente. El TPS
  píxel→metros queda en 1,3-1,9 m de mediana (p90 ~5 m): vale para mapas de 5 m y posesión en metros, no para distancias finas.
- Las cajas de imagen del GSR son sintéticas; la verdad útil es `bbox_pitch` (metros, 25 fps) + `team`/`player_id`.
- Jugadores de 46 px: ReID de equipos probablemente peor que el 68 % de TV; dorsales ilegibles [H].

Diseño: detección por 4 ventanas de 1024 sobre la panorámica entera (un solo detector persona+balón para no pagar dos
pasadas), ByteTrack a 5-12,5 fps, homografía fija (TPS de keypoints en experimentos; en producto, 8-12 clics en el frontend
o `autocal`), `tracks_2d.jsonl` + `heatmap.json` al final, capa `heatmap` del frontend. Medir en 117093/132877 contra el
GSR: recall de los 22 jugadores, error mediano en metros, fragmentación, acierto de equipo, correlación del mapa por celda.
Coste: 3-5 días + 1-2 de UI de calibración + 1 de arnés.

### 3.3 No mergear la panorámica todavía

De acuerdo. `feat/panorama-shots` ya está íntegra en main (0 commits por delante), así que el desarrollo nuevo va en una
rama nueva desde main (`feat/panorama-pipeline`), con el código en módulos propios y `git merge main` tras cada entrada en
main (highlights, umbrales, backend). Lo que habrá que decidir con el frontend al final está en §2.3 (último paso). El backend
ya guarda `width`, `height`, `fps` por vídeo; `AnalysisMode` es hoy `Literal["clip", "full-match"]`.

### 3.4 Clips de 12-30 s con excepciones

**Diagnóstico [M, 19 clips de los 4 partidos]**: 5/19 están en 12-30 s. El lado anterior no es la causa (los clips empiezan
10-16 s antes del ancla; en TV las cadenas de acciones no se forman y el VLM contesta `play_start` = primer frame mostrado en
49/64 juicios). El lado posterior sí: 23-45 s después del ancla, y 2/6 del Clásico tocan el tope `post_max_s` 45 porque
`_END_REACTION_KINDS` incluye "replay" y la realización de TV encadena repeticiones tras goles **y tras paradas** (la
ocasión hl-03 dura 37 s por 6 frames de replay). Nada en la cadena mira la duración total (máximo teórico 35 + 45 = 80 s).
Los dos clips de ocasión del Mallorca que te gustan duran 24,8 y 28,2 s: ya están en el rango. Mallorca tiene 3 goles (los 3
salen etiquetados); el partido sin goles es el Espanyol.

**Conflicto a decidir**: tu referencia del gol de Messi (1:45:56-1:46:46, 50 s) tiene 7 s de construcción y 43 s de
celebración; un tope de 30 s para goles la contradice. Propuesta: **goles son la excepción** (≤ 45 s con celebración; 50 si
quieres clavar la referencia) y el resto 12-30 s. Simulación sin GPU sobre los frames ya clasificados [M]: Clásico pasa a
45/45/21/41/33/45 s (6/6 en rango, Messi pierde 13 s de celebración con tope 45, 8 s con tope 50); Mallorca 24/45/25/28/45;
Espanyol 29/22/30/27; Middlesbrough 30/22/30/23. Qué se pierde: las repeticiones de TV (50-60 % del clip en el Clásico) y la
cola de celebraciones largas. Riesgo a cubrir: en Mallorca el ancla del 1-0 está 26 s antes del gol real; el tope de gol debe
contarse desde el gol (re-anclar al primer frame de celebración) o el clip se queda sin celebración.
Implementación y medida en §2.2 (primer paso). Alternativa barata para TV sin VLM: `signals.green_ratio_curve` (ratio de
césped, ya definida y sin uso) para detectar la vuelta al plano general [H].

### 3.5 VLMs más grandes y "puntuar jugadas"

**Veredicto tras 25 trabajos verificados [M, URLs en la bitácora de la sesión]**: la literatura confirma que los video-LLM
localizan y describen, pero nadie publica un VLM que puntúe jugadas de fútbol mejor que las señales clásicas:
- SVHighlights (KDD 2026, 320 vídeos deportivos de 2 h): el mejor método VLM+LLM acierta el primer highlight el 26,6 % de
  las veces; lo que más aporta es el **texto de la narración** (ASR), no el volumen.
- SoccerBench (ACM MM 2025): MLLM comerciales 54-57 % en tareas de vídeo de fútbol (Gemini 2.5 Pro 66 %); el salto 7B → 72B
  es de +6 pt; lo que dispara el acierto es el agente con herramientas especializadas, no el modelo grande.
- SoccerHigh (MMSports 2025): un modelo supervisado fuerte se queda en F1 0,39 porque el "highlight" editorial es subjetivo.
- DIAMOND (béisbol, 2025): el patrón que sí funciona es un LLM razonando sobre eventos y estadísticas ya extraídos (nuestro
  `telemetry.json` y nuestro ranking relativo en texto), no sobre píxeles.
- Industria (WSC Sports, Spiideo 2026, Veo, Pixellot): detectan con visión + audio + marcador/datos y usan el LLM para
  componer, no para ver el gol. Es nuestra arquitectura.

Corrección a la memoria: con el prompt actual (`PROMPT_VERSION` 3) el `is_goal` del 9B es verdadero en 4/5 goles del Clásico
y en 8 jugadas que no son gol [M, vlm_cache.json]: alto recall, baja precisión; por eso manda el marcador.

Modelos que cabrían con la GPU libre: Qwen3-VL-8B FP8 (~10 GB, vídeo nativo con timestamps), Qwen3-VL-30B-A3B AWQ (17 GB),
`qwen3.5:27b` en Ollama (17-20 GB); MiniCPM-V 4.5 int4 (~6 GB) es el único que cabe hoy. Qwen3-VL-32B AWQ (20 GB) no deja
sitio al vídeo. Cloud: Gemini es el único con vídeo + audio nativos y timestamps; coste < 1 $ por partido con 25 candidatos
de 60 s (céntimos por clip) [M, precios oficiales].

**Experimento mínimo (2-3 h, < 3 $, sin GPU)**: (0) congelar el baseline con `evaluate.py --json`; (1) mismo protocolo con
otro cerebro: `--rerun --set vlm_api=openai --set vlm_base_url=... --set vlm_model=gemini-2.5-pro --set
scoreboard_enabled=false --set vjepa_enabled=false --set vlm_max_candidates=40` y AP/AUC de `is_goal` y `excitement` frente a
los 8 goles; (2) vídeo nativo con audio: clips [inicio − 5, ancla + 45] de los 30 mejores candidatos a Gemini pidiendo
`play_start`/`play_end` en MM:SS y gol, error frente a Messi y a referencias manuales de ±2 s (criterio: mediana < 4 s); (3) tú
puntúas 1-5 los 19 clips (10 min) y se calcula Spearman de `excitement`/ranking y de `rule_score` con tu nota: si ningún
modelo supera a las reglas con ρ > 0,6, el VLM no aporta en puntuar. Para sustituir reglas + marcador, un VLM tendría que
dar 8/8 goles con ≤ 1 falso sin rótulo y repetirlo en SN-v2 no usados. Hacerlo después del cambio de duración (§3.4) para
comparar con la misma política de corte.

### 3.6 Renders de la cámara de referencia de SoccerTrack como datos "SoccerNet" para el T-DEED de TV

**Viable técnicamente, utilidad incierta; hay un experimento de 1 h que lo decide.** Hechos [M]:
- Los renders de referencia `vcam796` (796x448 nativo, cámara guiada por la pista del dataset) existen para las 20 mitades
  (`datasets/soccertrack-v2/views/<id>/`, 62 GB). Los de 132831 y 132877 están degradados (balón aparcado en (0,0)); los 5
  partidos de train de día tienen referencia sana: 10 mitades, ~10.400 eventos, 136 SHOT + 22 GOAL.
- Las etiquetas `bas/` son las 12 clases de EFL; `tdeed_finetune_data.py` ya las convierte a `Labels-ball.json`. Falta que
  acepte EFL plano y mezcle fuentes (~medio día). `tdeed_finetune.py` entrena solo la cabeza SNB pero el backbone cambia: la
  cabeza SoccerNet (córner, falta) puede desalinearse. `--freeze-backbone` existe y nunca se midió.
- `SNB_pano1` (afinado con nuestros renders) empeoró sobre la cámara de referencia (75,7 → 69,2 en val): afinar con renders
  especializa. No está medido en TV.
- Afinado: 18,7 GB de VRAM y 10-12 h con EFL + renders → necesita la GPU entera (hoy imposible).

**Experimento barato primero** (cabe hoy, ~6,5 GB, 40-70 min, solo escribe en `output/`):
```bash
docker exec -d -w /app/models/tdeed/T-DEED -e PYTHONPATH=/app Sports-Classification bash -c \
  'python -u /app/scripts/shot_benchmark/tdeed_scan_heads.py SNB_pano1 \
   "/app/datasets/SoccerNet/england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End/720p.mp4" \
   /app/ground_truth/2019-10-01_Middlesbrough-Preston_North_End-720p-trim.mp4 \
   /app/backend/storage/videos/8e759934bde2156e/source.mp4 \
   /app/backend/storage/videos/f0926c19277b28db/source.mp4 \
   /app/backend/storage/videos/573d9c1395b8f1c8/source.mp4 \
   --out-tag _tv --batch-size 1 > /app/output/shot_benchmark/tdeed/scan_pano1_tv.log 2>&1'
# Tiros en TV (Middlesbrough, 25 tiros; challenge2 dio AP 85,0 %):
cd scripts/shot_benchmark && cat common.py tdeed_eval.py | docker exec -i -w /app Sports-Classification python - SNB_pano1_heads_tv
# Acciones en el GT de 20 min: copiar el npz como ground_truth/<vídeo>.tdeed.SNB_pano1.npz, respot_telemetry --model-name SNB_pano1,
# evaluate_pipeline --compare con la telemetría de challenge2 (F1 88-91 %). Espanyol: --fps 30.
# Juveniles: máx P(SHOT+GOAL) en ±5 s de los 3 goles del Mallorca (challenge2 hoy: 0,12 / 0,01 / 0,16) y picos por 90 min.
```
Qué decide: si Middlesbrough cae > 5 pt en tiros o > 3 en acciones, los renders especializan y SNB_tv1 solo tiene sentido
mezclando EFL en cada lote o con backbone congelado; si Middlesbrough aguanta y los goles del Mallorca pasan de 0,3 en ≥ 2 de
3, el aspecto "cámara única sin zoom" transfiere y SNB_tv1 merece las 12 h; si Middlesbrough aguanta y los juveniles no
cambian, el fallo juvenil no es el estilo de cámara y hay que anotar antes que entrenar. Mi apuesta [H]: la probabilidad de
mejorar la EFL es baja y la de mejorar juveniles, desconocida; dos juveniles anotados rendirán más que SNB_tv1, pero la hora
de GPU es barata y conviene saberlo.

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

## 5. Ramas

| Rama | Estado [M, 2026-10-20] | Decisión |
|---|---|---|
| `feat/highlights` | 1 commit propio (88e1cbd; ca27f22 ya en main por patch-id) | Mergear tras §2.2 (duración + validación + decisión VLM); a la vez el frontend (6c983af) |
| `feat/T-DEED` | 0 por delante; el código VLM se recupera con `git revert 6ceb32b` sin la rama | Tag `archive/` opcional y borrar |
| `feat/panorama-shots` | 0 por delante, mergeada | Borrar; el desarrollo nuevo va en `feat/panorama-pipeline` |
| `videoARM` | 0 por delante; vía agotada | Tag `archive/` y borrar |
| `VLLM` | 1 commit único (events_omni, 55 % en clips sesgados, 22 GB) | No mergear; tag `archive/` y borrar |
| `basketball` | 0 por delante; `basketball/README.md` existe, sin mención en el README raíz | Una línea en el README raíz; borrar la rama |

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

- **2026-10-19**: hoja de ruta inicial (workflow de 12 agentes). Hallazgos: los CSV de `output/annotations` eran volcados
  del pipeline; el Clásico está en SoccerNet-v2 con `Labels-v2.json`; T-DEED ciego en juveniles (no es umbral); equipo de las
  acciones ~68 %; GT principal fuera de git y sin generador; `/mnt/datasets` montado en el host.
- **2026-10-20**: segunda ronda (5 agentes) sobre las ideas del usuario: futtraffic no es cámara virtual (sí: `autocal`,
  `pancheck`, puertas de balón); duración de clips diagnosticada (5/19 en 12-30 s; causa: replays de TV + tope 45 s); plan
  de homografía fija + tracking por ventanas medido con GSR; experimento barato `SNB_pano1` en TV definido; literatura VLM:
  sin evidencia de que un VLM mayor puntúe jugadas mejor que reglas + marcador. El usuario anota el Espanyol 25:00-30:00
  (`output/annotations/espanyol_damm_Ground_Truth.csv`, formato `tiempo,etiqueta,dorsal(=tracker_id),equipo`). Decisiones del
  usuario: main debe funcionar bien; panorámica sigue en rama; clips 12-30 s con excepciones. Documento creado; resumen del
  mecanismo de selección añadido a `docs/highlights_evaluacion.md`.
- **2026-10-21** (sesión en `feat/highlights`; el usuario anota el Espanyol en paralelo): carril B, semana 1. (1) Lector del
  CSV (`scripts/annotation_csv.py`), arnés único (`scripts/evaluate_match.py`) y generador de GT reescrito
  (`scripts/build_ground_truth.py`: Labels-ball, Labels-v2 con desfase por mitad, CSV; fps real por `video.json`/ffprobe);
  `ground_truth/*.gt.json` ya van a git. Middlesbrough reproduce 87,7/89,1; equipo 68,1 %. Espanyol 25-30 (hoja a medias,
  126 filas): F1 82,9, equipo 62,1 % (los clústeres de prtreid son impuros: 51 % y 76 %), tracker 91,5 %: el fallo de
  equipo está en el agrupamiento, no en el poseedor. (2) Política de duración de clips implementada y medida sin GPU
  (`--vlm-cache-only`): 20/20 en rango, 8/8 goles, 0 falsos; dos arreglos de paso (gol al candidato cercano al rótulo;
  nada de re-anclar por celebración en TV). (3) `foul` marcado `estimated` en telemetría y API; origen del gol y estado
  degradado del VLM en `HighlightsResponse` (backend: 22 tests OK). (4) Partición de juveniles y nota del Clásico en
  `docs/particion_datasets.md`. (5) Carril C: cadena `run_pano1_tv.sh` lanzada y esperando VRAM (GPU a 23/24 GB ajena).
  Hallazgo [M]: etiqueta spotter frente a combinada, +2,7 F1 en Middlesbrough y −8,5 en Espanyol (ver §2.1). Sin medir
  aún: córner/falta/tiro en el juvenil (la hoja no tiene eventos todavía).
