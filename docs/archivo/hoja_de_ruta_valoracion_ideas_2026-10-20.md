# Valoración de las ideas del 2026-10-20 (texto completo, archivado el 2026-11-13)

> Era el §3 de `docs/hoja_de_ruta.md`. Se archiva íntegro, sin cambios de contenido; en la hoja de ruta queda una línea por idea.

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

**Resultado (2026-10-21, `output/pano1_tv/`; base = `challenge2`, afinado = `SNB_pano1`) [M]:**

| Medida | base | afinado |
|---|---|---|
| Middlesbrough completo, tiros AP ±3 s (25 tiros; `tdeed_eval.py`) | **85,0 %** | 73,8 % (±1 s: 85,0 → 70,0) |
| Middlesbrough, AUC tiro/centro sobre candidatos | 0,98 | 0,89 |
| Middlesbrough 20 min, acciones P/R/F1 (etiqueta spotter, umbral auto) | 90,3 / 91,8 / 91,0 | 78,6 / 74,9 / 76,7; **header 0/29** (van a pass) |
| Middlesbrough 20 min, eventos tiro / banda (F1) | 66,7 / 92,9 | igual |
| Mallorca, SHOT+GOAL máx. ±5 s en los 3 goles | 0,12 / 0,01 / 0,16 | **0,47 / 0,55 / 0,41** (GOAL sigue ≤ 0,09) |
| Mallorca, picos SHOT ≥ 0,3 por 90 min | 4,5 | 14,5 (precisión sin medir: faltan tiros anotados) |
| Espanyol, picos SHOT ≥ 0,3 por 90 min | 4,6 | 8,6 |
| Clásico (in-sample), SHOT máx. en los 5 goles | 0,59-0,99 | 0,69-0,96 (igual) |
| Clásico, cabeza SoccerNet "Goal" en los 5 goles | 0,25-0,61 | 0,04-0,32; picos Goal ≥ 0,3 por 90 min 5,8 → 0 |

Veredicto: se cumplen a la vez la primera condición (Middlesbrough cae 11 pt en tiros y 14 en acciones: los renders
especializan, y además la cabeza SoccerNet se desalinea) y la segunda (3/3 goles del Mallorca pasan de 0,3 como tiro: el
estilo de cámara transfiere). Por tanto `SNB_tv1` merece la GPU entera, pero solo mezclando EFL en cada lote (o con
`--freeze-backbone`) y entrenando las dos cabezas; y su medida en juveniles necesita los tiros anotados por el usuario
(sin ellos, 14,5 picos/90 min puede ser recall o falsas alarmas). La subida de "equipo" en la tabla de acciones del
afinado (68,1 → 77,1 %) es un artefacto: cambian las parejas emparejadas, no `teams.json`.

