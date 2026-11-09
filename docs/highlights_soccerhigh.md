# Highlights con SoccerHigh (VideoMAEv2-Giant + transformer)

Hecho el 2026-11-02. Modelo del paper *SoccerHigh: A Benchmark Dataset for Automatic Soccer Video Summarization*
(Díaz-Juan, Ballester, Haro; MMSports 2025, arXiv 2509.01439; repo IPCV/SoccerHigh, CC BY-NC-SA 4.0), reimplementado
en `analytics/highlights/soccerhigh.py` y metido en el pipeline de highlights como tercera fuente de jugadas junto a
T-DEED y el audio. Contexto: `docs/estado_arte_highlights.md` §1 y §3 (paso 1), `docs/datasets.md` §2.2.

## Qué hace el modelo

Aprende, con los resúmenes oficiales de 237 partidos de SoccerNet alineados al partido completo, qué tramos de una
retransmisión entran en el resumen. Tres piezas:

1. **Rasgos**: VideoMAEv2-Giant afinado en Kinetics-710 (`models/soccerhigh/vit_g_hybrid_pt_1200e_k710_ft.pth`, 1.012 M
   parámetros, 2 GB) sobre clips de 16 fotogramas a 2 fps (8 s), 224x224 con recorte central, centrados en cada fotograma:
   un vector de 1408 por cada 0,5 s de vídeo (`extract_features`; `analytics/highlights/videomaev2.py` es el ViT
   reducido de `OpenGVLab/VideoMAEv2-giant` con atención SDPA).
2. **Resumidor** (`SoccerHighSummarizer`, 3,1 M parámetros): proyección 1408→256, codificación posicional sinusoidal,
   dos capas de transformer (8 cabezas) y cabeza anchor-free (DSNet-AF) que da, por fotograma, la confianza de estar en
   el resumen y los desplazamientos al inicio y al fin del tramo. Pesos publicados `Baseline_VideoMAEv2-Giant.ckpt`,
   extraídos a `models/soccerhigh/soccerhigh_baseline_giant.pt` sin ejecutar su pickle (Lightning + omegaconf).
3. **Inferencia**: ventanas de 60 s sin solape; cada fotograma con confianza > 0,5 propone `[i − d_ini, i + d_fin]`; NMS sin
   solape por confianza. `select_to_length` es la selección "@T" del paper (los mejores tramos hasta la duración del
   resumen real).

Coste medido (RTX 4090, fp16, lote 8): 21 clips/s en el ViT, 2,3 GB de VRAM; un partido de 95 min son ~11.400 clips,
8-9 min con la decodificación de ffmpeg solapada en un hilo. Los rasgos se cachean junto al vídeo
(`<vídeo>.soccerhigh.VideoMAEv2_Giant_K710.npy`, float16, ~32 MB por partido), así que repetir la selección no vuelve a
pasar por la GPU.

## Reproducción del baseline (sus rasgos, su test)

`scripts/soccerhigh_download.py` baja las features de Hugging Face (`imva-upf/SoccerHigh`, fichero a fichero: listar el
árbol agota el límite de la API) y `scripts/evaluate_soccerhigh.py` calcula sus métricas (P/R/F1 por fotograma a 2 fps,
agregadas y por partido; mAP; "@T"). 45 partidos de test, `--source hf`:

| Métrica | Paper | Esta reimplementación |
|---|---|---|
| Clasificación cruda por fotograma (prob > 0,5), F1 | 0,3956 (P 0,4796 / R 0,3367) | 0,3738 (P 0,6041 / R 0,2706) |
| Tramos tras NMS, F1 por fotograma | – | 0,4119 (P 0,5184 / R 0,3417; IoU 0,259) |
| F1@T (selección hasta la duración del resumen) | 0,3883 (P 0,5034 / R 0,3375) | 0,3946 (P 0,5307 / R 0,3370) |

Las cifras cuadran a 1-2 puntos: el modelo y la inferencia están bien portados. La diferencia en la clasificación cruda
viene de que el paper mezcla en "F1" el resultado de su `test.py`, cuya rama de tramos arrastraba un error de forma
(`reshape` de los desplazamientos, corregido en el repo después del paper).

## Desfase HQ → 720p

Las anotaciones van en tiempo de los `*_HQ.mkv` de SoccerNet; nuestros 720p empiezan en el saque inicial (`Kick-off` de
`Labels-v2.json` a 0:00) y `video.ini` no es público. `scripts/soccerhigh_align.py` lo resuelve alineando nuestros rasgos
con los de Hugging Face (similitud coseno media tras quitar la componente común): en el Clásico (primera mitad) el pico
está en +968,0 s con similitud 0,83 (segundo máximo 0,62, mediana 0,00), y la similitud cruda por fotograma en ese
desfase es 0,95 de media (p10 0,93): los rasgos que extraemos del 720p son los suyos. Dos comprobaciones de la
convención de clip: 16 fotogramas consecutivos a 2 fps (8 s) dan 0,83; 16 fotogramas a ritmo nativo (paso 4 de 25 fps,
2,6 s, el régimen de Kinetics) solo 0,55, así que el paper usa el clip de 8 s. Centrar o retrasar la ventana da los
mismos rasgos desplazados 3,5 s (ambigüedad que no se puede resolver con la similitud); se asume centrada.

Clásico completo (ambas mitades): +968,0 s y +277,0 s, sin deriva (el desfase es el mismo en todos los trozos de 5 min
de cada mitad, picos 0,78-0,92). El vídeo del backend (otra fuente, 1080p, las dos partes en un fichero) se mapea igual:
primera parte = HQ − 607 s; segunda parte empieza en 3276 s del vídeo y coincide con el inicio del HQ de la 2ª mitad
(`output/soccerhigh/clasico_backend/mapping.json`). Comprobación: los 5 goles del Clásico caen donde dice `Labels-v2`.

## Resultados en el Clásico (validación de SoccerHigh; 21 tramos, 126 s de resumen oficial)

| Qué se evalúa | P | R | F1 | F1@T |
|---|---|---|---|---|
| Modelo con los rasgos de HF (HQ) | 0,384 | 0,171 | 0,236 | 0,236 |
| Modelo con nuestros rasgos del 720p (+desfase) | 0,636 | 0,325 | 0,430 | 0,430 |
| Modelo sobre el vídeo del backend (1080p, YouTube): 11 tramos, 124 s | 0,319 | 0,314 | 0,316 | 0,316 |
| Clips del frontend **antes** (6 clips de 12-45 s, política de duración) | 0,189 | 0,409 | 0,259 | 0,219 |
| Los 78 candidatos de reglas **antes** (techo de la cascada) | 0,046 | 0,595 | 0,085 | 0,321 |
| Clips del frontend **después** (SoccerHigh en la cascada) | 0,198 | 0,361 | 0,256 | 0,221 |

Lectura: (1) los rasgos extraídos aquí son equivalentes a los publicados (en este partido incluso van mejor, es un
solo partido y los tramos son cortos; la comparación en 3 partidos de test está más abajo); (2) en el vídeo del
backend el modelo encuentra 11 tramos que incluyen los 5 goles (confianza 0,64-0,88) y suman 124 s, casi la duración
del resumen oficial; (3) la métrica por fotograma castiga a nuestros clips porque la política de duración
(12-30 s, goles hasta 45 s) hace clips mucho más largos que los planos de 3-10 s del resumen de la Liga: por eso la
recall de los clips (0,41) supera a la del modelo y la precisión se queda en 0,19.

## Fidelidad de nuestros rasgos en 3 partidos de test (720p + desfase por alineación)

Barcelona-Villarreal 2015-11-08, Torino-Roma 2016-09-25 y Mónaco-PSG 2016-08-28 (desfases 780/917, 315/273 y 311/581 s,
picos de similitud 0,86-0,93). Mismo modelo, mismas anotaciones:

| Rasgos | P | R | F1 por fotograma | F1@T |
|---|---|---|---|---|
| Publicados (HF, HQ) | 0,392 | 0,322 | 0,353 | 0,346 |
| Nuestros (720p, `extract_features`) | 0,468 | 0,304 | 0,368 | 0,365 |

Conclusión: la extracción propia reproduce el baseline (incluso +1,5 pt); el modelo se puede aplicar a cualquier vídeo
nuestro sin depender de las features publicadas. Rasgos y `offsets.json` quedan en `datasets/soccerhigh/videos/`.

## Qué cambia en los 4 partidos del frontend (regenerados el 2026-11-02)

`python -m analytics.highlights --backend-match <id>` para los cuatro (rasgos SoccerHigh 8-9 min por partido; el resto
cacheado). `python -m analytics.highlights.evaluate` (goles): **8/8 goles en clip, 8/8 etiquetados gol, 0 falsos,
21/21 clips en rango (mediana 28 s)**, igual que antes; lo que aporta SoccerHigh es otra cosa:

| Partido | Tramos SoccerHigh (conf ≥ 0,5) | Qué ve | Clips |
|---|---|---|---|
| Clásico (TV) | 11, 124 s | los 5 goles (0,64-0,88) + 6 ocasiones (0,51-0,81); el tramo 1:00:48 (0,81, tiro bloqueado) queda 13.º por reglas y el VLM no lo sube | 5 goles + parada (igual que antes; `soccerhigh_score` 0,64-0,88 en los goles, 0,14 en la parada) |
| Mallorca (juvenil, cámara única) | 2, 27 s | dos de los tres goles (0,69 y 0,86); el tercero (1:19:18, `score` 0,07) solo lo ve el marcador | 3 goles + 3 paradas (0,33 en la primera) |
| Espanyol (juvenil, TV) | 0 (máx. 0,41) | nada: el modelo no reconoce "resumen" sin gol en este dominio | 5 ocasiones, sin cambios |
| Middlesbrough (TV, 20 min, sin goles) | 0 (máx. 0,27) | nada | 4 ocasiones, sin cambios |

Lectura: con los pesos publicados SoccerHigh es sobre todo un **detector de goles y grandes momentos de TV** (en el
Clásico puntúa 0,8-0,9 a todos los goles sin leer el marcador), razonablemente transferible a juveniles con rótulo
(2/3 goles del Mallorca) y mudo en partidos sin gol. Como está entrenado con resúmenes de 2-6 min de ligas
profesionales, su umbral de "entra en el resumen" es alto: para ocasiones hay que usar la curva (ya es rasgo de reglas y
red de seguridad del VLM) y no los tramos. De paso se arregló un fallo que esta integración destapó: con ~30 candidatos
juzgados el ranking relativo del VLM se cortaba a 900 tokens y quedaba vacío (`rank_candidates` ahora escala el límite y
rescata las filas completas).

## Siguientes pasos

1. **Afinar el resumidor con nuestros vídeos** (3 M parámetros, entrena en minutos sobre rasgos cacheados): los 141 de
   train de SoccerHigh con rasgos propios (ya se puede: `extract_features` reproduce los suyos) y, para juveniles, el
   "resumen de referencia" por partido que pide `docs/estado_arte_highlights.md` §3.5. El ViT-g queda congelado.
2. Medir la selección completa en más partidos de test de SoccerHigh (`scripts/soccerhigh_predictions_from_highlights.py`
   + `--predictions`): el pipeline entero necesita análisis previo (T-DEED, tracking) de cada partido, ~1 h por partido.
3. Umbral y peso: `soccerhigh_threshold` 0,5 y `w_soccerhigh` 2,5 son los del paper y una primera elección; con la
   verdad de SoccerHigh en más partidos se pueden barrer sin GPU (`evaluate --rerun --vlm-cache-only`).
4. Los rasgos VideoMAEv2 ya cacheados son una representación general del vídeo: candidatos a sustituir V-JEPA2 (fuera
   de dominio en juveniles) como entrada del clasificador de gol/saque de centro.

## Integración en `analytics/highlights`

- `pipeline.run_soccerhigh` (etapa `soccerhigh`, tras las señales): rasgos cacheados junto al vídeo, resultado en
  `highlights/soccerhigh.json` + `soccerhigh_scores.npy`; `signals.attach_soccerhigh` añade la curva `soccerhigh`
  (confianza por bin de 0,5 s) y los tramos a `signals.npz`.
- `candidates.find_anchors`: cada tramo con confianza ≥ `soccerhigh_anchor_min_score` (0,5) es un ancla; el momento clave
  es el pico de tiro de T-DEED dentro del tramo si lo hay y, si no, el fotograma de más confianza. `provisional_end` se
  alarga hasta el fin del tramo que cubre el ancla. Rasgo `soccerhigh` (máximo en la jugada) con peso `w_soccerhigh`
  (2,5) en la puntuación por reglas, y red de seguridad: los candidatos con `soccerhigh` ≥ 0,75 los juzga el VLM aunque
  no entren entre los 10 primeros por reglas.
- API: `HighlightsResponse.soccerhigh_used` / `n_soccerhigh_segments` y `HighlightClip.soccerhigh_score`;
  `--no-soccerhigh` en el CLI; `HighlightsConfig.soccerhigh_*`.
- Tests sin GPU: `tests/test_highlights_soccerhigh.py` (6) y los 10 de duración siguen en verde; backend 23/23.

Licencias: SoccerHigh (modelo, datos) CC BY-NC-SA 4.0 y VideoMAEv2 CC BY-NC 4.0: valen para I+D y evaluación; antes de
un producto comercial hay que sustituir o reentrenar (y el ViT-g es el que más pesa en el tiempo de proceso).
