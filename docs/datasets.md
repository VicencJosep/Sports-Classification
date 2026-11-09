# Datasets: los que usamos y los que existen fuera

Documento de referencia, estado a 1 de noviembre de 2026 (entrada de FOOTPASS / SN-PCBAS-2026 revisada el 6 de noviembre).

Responde a dos preguntas: qué conjuntos de datos usa hoy cada rama y funcionalidad de Sports-Classification, y qué otros conjuntos públicos de fútbol (y, de forma secundaria, de baloncesto) podemos usar ya o nos pueden servir más adelante. Las particiones train/val/eval no se repiten aquí: la referencia sigue siendo `docs/particion_datasets.md`. Las licencias se citan de forma literal cuando la fuente las da; cuando un dato no se pudo comprobar se dice "no verificado". Los enlaces de cada dataset van en el cuerpo; la lista completa de fuentes está en la sección 9.

## 1. Resumen y recomendaciones

Usar ya, por orden de prioridad:

1. **SoccerHigh**: verdad de highlights (los segmentos de la retransmisión que entran en el resumen oficial) para 237 partidos que ya tenemos en `/mnt/datasets/SoccerNet`. Funcionalidad: highlights. Esfuerzo: bajo (clonar 111 MB, aplicar el desfase de `video.ini`, medir F1@T). Licencia: CC BY-NC-SA 4.0, solo evaluación e I+D.
2. **StatsBomb Open Data**: eventos con equipo, jugador, coordenadas y timestamp al milisegundo para 135 de nuestros 507 partidos locales (EPL 2015/16 completa, LaLiga 2015/16 completa, partidos del Barcelona 2014-17 incluido el Clásico). Funcionalidad: evaluación de T-DEED (atribución de equipo) y del lector de marcador en `main`. Esfuerzo: medio (descargar JSON, alinear el reloj de cada mitad). Licencia: acuerdo propio no comercial, sin redistribución, logo obligatorio.
3. **SoccerNet-Echoes**: transcripciones ASR de las 1.100 mitades de SoccerNet, CC BY 4.0 y sin NDA. Funcionalidad: gol por comentario y señal débil para highlights en TV. Esfuerzo: bajo.
4. **SN-BAS-2025 (Team Ball Action Spotting)**: equipo left/right para las 12 clases de nuestros mismos 7 partidos EFL. Funcionalidad: eventos T-DEED con equipo. Esfuerzo: medio (cuenta de Hugging Face, NDA, cabeza de equipo). Licencia: NDA de SoccerNet.
5. **Etiquetas de SoccerNet-Caption y SN-GSR-2024/2025**: comentario textual con goles, equipos y alineaciones sobre 471 de nuestros partidos; posiciones, rol, equipo y dorsal en 200 clips 1080p (GPL-3.0, públicos sin NDA). Funcionalidad: equipos, dorsales, homografía. Esfuerzo: bajo-medio.
6. **TeamTrack**: 60 min de fútbol con ojo de pez 8K fijo y dron 4K anotados por completo (Kaggle, licencia "MIT"). Rama panorámica: segundo partido para el detector de balón y la cámara virtual. Esfuerzo: medio (corregir la distorsión, generar la cámara virtual).
7. **Fuentes abiertas de goles al minuto** (Transfermarkt datasets CC0, football-data.co.uk, Understat, Kaggle Football Events): verdad de goles, tiros y recuentos para los ~380 partidos sin asignar de SoccerNet-v2. Funcionalidad: highlights sin marcador, lector de marcador. Esfuerzo: bajo. Licencia: CC0 en Transfermarkt; el resto no verificada o scraping.

Guardar para después: FOOTPASS/SN-PCBAS-2026 (jugador ejecutor por dorsal; 28,5 GB en 352x640 + datos tácticos o 236 GB con Full HD; NDA), Spiideo SynLoc (cámara fija 4K con posición métrica, licencia por pedir), SoccerSynth-Detection (Apache-2.0), MultiSports, SoccerReplay-1988, GOAL+MOMENTS, SoccerBench y otros benchmarks de VLM, SJN-210k, Kinetics "shooting goal (soccer)", ISSIA/ISSIA-3D, Alfheim, Fu et al. 2023, IDSSE/Metrica/PFF (validar métricas tácticas), SoccerTrack v1, SportsMOT.

Descartar o de poco valor: DFL Bundesliga Data Shootout, DeepSportradar (para fútbol), SkillCorner y Wyscout (sin vídeo y sin solape con nuestros partidos), Last Row, Signality, los derivados Roboflow del DFL Shootout, SoccerDB y SSET (sin descarga verificable).

Dos correcciones a las descripciones del usuario. (a) El DFL Shootout no tiene clases de tiro ni de falta: sus clases son `play`, `throwin` y `challenge` (la falta solo aparece como resultado `Fouled` de un `challenge`); ya no se puede descargar ("DATA IS NO LONGER AVAILABLE FOR THIS COMPLETED COMPETITION") y sus reglas lo limitan a "Competition-Use Only". (b) DeepSportradar es baloncesto (2022: segmentación, balón 3D, re-id, calibración) y críquet (2023); nunca ha tenido tarea de tracking ni ningún dato de fútbol.

## 2. Los cinco datasets propuestos

| Dataset | Qué es | Vídeo | Licencia | Solape con lo nuestro | Veredicto |
|---|---|---|---|---|---|
| [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) (Kaggle, 2022) | Competición de spotting de `play`, `throwin` y `challenge` en TV de Bundesliga (8 partidos de entrenamiento, 12 vídeos) | Sí, pero ya no descargable | "Competition-Use Only" (reglas Kaggle, §7.A) | Ninguno | Descartar; solo referencia metodológica |
| [SoccerHigh](https://github.com/IPCV/SoccerHigh) (UPF, MMSports 2025) | Segmentos de la retransmisión que entran en el resumen oficial de 237 partidos de SoccerNet | No (anotaciones y features; el vídeo es SoccerNet) | CC BY-NC-SA 4.0 | 237 de 237 partidos están en nuestro `/mnt` | Usar ya: evaluación de highlights |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Eventos con jugador, equipo, coordenadas y timestamp al ms; 3.961 partidos; 360 en 12 competiciones-temporada | No | StatsBomb Public Data User Agreement (no comercial, sin redistribución, logo) | 135 de 507 partidos locales | Usar ya: evaluación de eventos con equipo y jugador |
| [DeepSportradar](https://github.com/DeepSportradar) (MMSports 2022-2023) | Baloncesto: segmentación de instancias, balón 3D, re-id, calibración; críquet: bowl release | Imágenes sueltas ("instants"), no vídeo | CC BY-NC-ND 4.0 (Kaggle `deepsport-dataset`) y CC BY-NC-SA 4.0 (Kaggle `basketball-instants-dataset`) | Ninguno en fútbol; solo rama basketball | Descartar para fútbol; útil en el futuro en baloncesto (calibración, segmentación) |
| [SkillCorner Open Data](https://github.com/SkillCorner/opendata) | Tracking a 10 fps derivado de la señal de TV más Dynamic Events; hoy 20 partidos de la A-League 2024/25 | No | MIT | Ninguno (los 9 partidos de 2019/20 se borraron en 2025) | Descartar para vídeo; referencia de formato de tracking |

### 2.1 DFL - Bundesliga Data Shootout

Competición Kaggle del 29-07-2022 al 20-12-2022, organizada por la DFL y Sportec Solutions con premios cofinanciados por AWS. Las clases son `play` (atributos pass, cross, openplay, corner kick, free kick), `throwin` y `challenge` (resultados: opponent rounded, ball action carried out, fouled, opponent dispossessed, challenge during release of the ball, possession retained). No hay clase de tiro ni de gol, ni equipo, ni jugador, ni cajas.

| Dato | Valor | Fuente |
|---|---|---|
| Vídeo de entrenamiento | 8 partidos (12 vídeos de media parte); test público 1 partido + 4 medias partes; `clips/` de 10 partidos sin anotar | Kaggle data |
| Eventos (según un participante) | 3.586 `play`, 624 `challenge`, 172 `throwin` | Habr |
| Métrica | AP con tolerancias por clase: challenge 0,30-0,70 s; play y throw-in 0,15-0,35 s | Kaggle overview |
| Ganador | Team Hydrogen, 0,91 AP (2.º K_mat 0,85; 3.º Camaro 0,84) | clist.by |
| Resolución y fps nativos | no verificados (indicios de 1280x720 y 25 fps) | hilos de Kaggle |
| Disponibilidad | "DATA IS NO LONGER AVAILABLE FOR THIS COMPLETED COMPETITION" | Kaggle data |
| Licencia | §7.A: "You may access and use the Competition Data only for participating in the Competition and on Kaggle.com forums." | Kaggle rules |

Valor residual: metodológico. El ganador de SoccerNet Ball Action Spotting 2023 se declara "very inspired by the top solutions of the DFL - Bundesliga Data Shootout" (frames en escala de grises apilados de 3 en 3, codificador 2D + bloques 3D). Nuestro T-DEED ya trabaja en esa línea. Los mirrors (Hugging Face `dbal0503/Bundesliga`, `ManuD/*`) y el dataset Roboflow `football-players-detection-3zvbc` derivado de sus clips quedan fuera de los términos de Kaggle/DFL.

### 2.2 SoccerHigh

Benchmark de resumen automático (Díaz-Juan, Ballester, Haro, Universitat Pompeu Fabra; MMSports 2025, [arXiv 2509.01439](https://arxiv.org/abs/2509.01439)). Para cada partido da los segmentos de la retransmisión completa que aparecen en el resumen oficial de la liga (YouTube), como `Labels-summary.json` (extiende `Labels-v2.json`: `UrlYoutube`, `annotations` con `start`/`end` en formato `half - hh:mm:ss.ms`, `summaryLength`) y `1_intervals.srt`/`2_intervals.srt`. Sin clase de evento, sin equipo ni jugador por segmento.

| Dato | Valor |
|---|---|
| Partidos | 237 (España 116, Francia 37, Italia 84), temporadas 2014/15 a 2016/17 |
| Splits | train 141 / validation 51 / test 45 (siguen los de SoccerNet) |
| Duración media del resumen | 3:19 |
| Solape con nuestro `/mnt` | 237 de 237 (141/141, 51/51, 45/45; nombres de carpeta idénticos) |
| Descarga | anotaciones y código en GitHub (111 MB); features VideoMAEv2 en [Hugging Face](https://huggingface.co/datasets/imva-upf/SoccerHigh) (22 GB); vídeo aparte (SoccerNet, NDA) |
| Baseline | F1@T 0,3883 (P 0,5034, R 0,3375); F1 por fotograma 0,3956 |
| Licencia | "This dataset is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license." |
| Ejemplo (nuestro Clásico 2017-04-23) | 21 segmentos, `summaryLength` 01:55, resumen en YouTube `aFCO6WidGVM` |

Los timestamps se refieren a los `*_HQ.mkv` de SoccerNet; con nuestros 720p hay que aplicar el desfase de `video.ini` antes de comparar. **Estado 2026-11-02**: anotaciones y listas en `datasets/soccerhigh/`, features Giant de los 45 de test y del Clásico bajadas con `scripts/soccerhigh_download.py`, baseline reproducido y modelo integrado en el pipeline (`docs/highlights_soccerhigh.md`); el desfase se estima alineando rasgos (`scripts/soccerhigh_align.py`) porque `video.ini` exige la contraseña de SoccerNet. Es la verdad de highlights que faltaba (`docs/highlights_evaluacion.md` solo tiene goles y un clip de referencia); los 45 de test dan un benchmark comparable con el baseline. Como es NC, debe quedar como conjunto de evaluación, no de entrenamiento del producto.

### 2.3 StatsBomb Open Data

Repositorio activo (commits hasta 2026-05) con 80 competiciones-temporada y 3.961 partidos en JSON (`competitions.json`, `matches/`, `events/`, `lineups/`, `three-sixty/`). Cada evento lleva `period`, `timestamp` al milisegundo, `team`, `player`, `location`, `play_pattern` y detalles por tipo (en `Shot`: `outcome`, `statsbomb_xg`, `body_part`, `freeze_frame`). Herramientas: `statsbombpy`, `kloppy`, `socceraction`, `mplsoccer`.

| Cruce con nuestros partidos locales | Locales | Con eventos StatsBomb |
|---|---|---|
| england_epl 2015-16 | 49 | 49 |
| spain_laliga 2015-16 | 36 | 36 |
| spain_laliga 2016-17 | 63 | 28 (solo Barcelona) |
| spain_laliga 2014-15 | 18 | 8 (solo Barcelona) |
| italy_serie-a 2015-16 | 9 | 9 |
| france_ligue-1 2015-16 | 3 | 3 |
| germany_bundesliga 2015-16 | 18 | 2 (solo Leverkusen) |
| UCL 2014-17, EPL 2014-15 y 2016-17, EFL, resto de temporadas | 311 | 0 |
| Total | 507 | 135 (26,6 %) |

Hechos verificados sobre las creencias del usuario: las temporadas 2015/16 completas existen para Premier League (380), La Liga (380) y Serie A (380); Ligue 1 tiene 377; Bundesliga 2015/16 solo tiene los 34 partidos del Bayer Leverkusen. Del Barcelona hay 524 partidos de liga entre 2004/05 y 2020/21, no todos los de cada temporada. El Clásico 2017-04-23 es el `match_id` 267569: 3.667 eventos, 38 `Shot`, 5 goles con timestamp (Casemiro P1 27:46.929, Messi P1 32:07.943, Rakitić P2 27:10.314, James P2 39:59.424, Messi P2 46:47.260), 11 córners, 6 tarjetas incluida la roja a Ramos. Las finales de Champions 2015, 2016 y 2017 están, pero no las tenemos en vídeo. Ninguna de nuestras temporadas tiene 360. Un partido pesa unos 3,2 MB.

Licencia ([LICENSE.pdf](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf), "last updated 8 September 2023"): 1.2.1 "edit, distort, distribute, reproduce, sell or in any way provide the data to any external or third party" prohibido; 1.2.2 prohibido "commercially exploit the data or any analysis derived from the use of the Service"; 1.4 "The User is required to accredit any publication of analysis formed from StatsBomb Data with the StatsBomb brand logo."; 2.2 registro con nombre y correo. Por eso: herramienta de evaluación y análisis, no de entrenamiento del producto. Hay que alinear relojes: `timestamp` cuenta desde el saque de cada mitad y el vídeo de SoccerNet incluye previa y descanso (anclar con el `Kick-off` de Labels-v2 o con los goles).

### 2.4 DeepSportradar

Suite de retos de visión de Sportradar en MMSports 2022 y 2023 ([arXiv 2208.08190](https://arxiv.org/abs/2208.08190)): "DeepSportradar currently supports four challenging tasks related to basketball: ball 3D localization, camera calibration, player instance segmentation and player re-identification"; en 2023 se añadió críquet (bowl release). No hubo edición en 2024 y el reto de MMSports 2025 fue el SoccerTrack Challenge 2025 de la Universidad de Nagoya, sin relación con Sportradar. Imágenes de la cámara fija lateral Keemotion/Synergy (2-5 Mpx, 80-150 px/m) de partidos de la LNB Pro A.

| Tarea | Datos |
|---|---|
| Segmentación de instancias | train 223 imágenes / 1.674 instancias, val 37 / 344, test 64 / 477, challenge 84 |
| Calibración | 728 vistas (480 / 164 / 84) + 84 de challenge; vistas de test a 960x540 según el README (920x540 según el paper) |
| Balón 3D | recortes con posición-oráculo; 364 "instants" en 15 pabellones |
| Re-identificación | 18.703 recortes; train 436 IDs / 8.569 imágenes |
| Licencia | Kaggle `deepsport-dataset`: "Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)"; Kaggle `basketball-instants-dataset`: "CC BY-NC-SA 4.0" (el README del reto de segmentación dice CC BY-NC-ND 4.0) |

Para fútbol no aporta nada. Para la rama basketball es el único conjunto público con máscaras, calibración y balón sobre una cámara fija de pabellón, pero sin eventos y con licencia no comercial (ver sección 5).

### 2.5 SkillCorner Open Data

Repositorio MIT creado el 2020-10-08. La versión original tenía 9 partidos 2019/20 (Clásicos 2019-12-18 y 2020-03-01, Liverpool-City, Bayern-Dortmund...) con tracking derivado de la señal de TV a 10 fps; el commit del 2025-10-06 ("Uploading new data and deleting old data") los borró y los sustituyó por partidos de la A-League australiana 2024/25.

| Dato | Valor |
|---|---|
| Partidos hoy | 20 carpetas en `data/matches` y `matches.json` (el README sigue diciendo 10) |
| Por partido | `_match.json`, `_tracking_extrapolated.jsonl` (10 fps, ~97 MB en git-LFS), `_dynamic_events.csv`, `_phases_of_play.csv` |
| Dynamic Events | `off_ball_run`, `passing_option`, `on_ball_engagement`, `player_possession`, con EPV y presión; 5.222 filas y 322 columnas en el partido 1925299 |
| Body pose | 29 articulaciones 3D a 25 fps en 2 partidos (`SkillCorner/opendata-bodypose`, ~3,3 GB cada uno, MIT) |
| Precisión declarada | "Around 97% of the player identity we provide are accurate." |
| Vídeo | No; los partidos de la A-League no son obtenibles legalmente por nosotros |

No hay solape con ninguno de nuestros partidos y los 9 de 2019/20 solo quedan en el historial git (recuperables con `git checkout 1b99d401 -- data`, contra la decisión del proveedor). Valor real: modelo de datos de referencia para nuestra salida de tracking desde una cámara (`is_detected` frente a extrapolado, polígono de campo visible, posesión por frame) y taxonomía de eventos dinámicos. No se encontró ninguna colaboración SkillCorner-PFF; PFF FC publica aparte el Mundial 2022 (sección 4B).

## 3. Lo que usamos hoy

Inventario hecho el 2026-11-01 leyendo repo y disco (`main`, HEAD 9234c43, `feat/highlights` ya fusionada), sin ejecutar nada del proyecto. Rutas del host `/media/dd3/ner-training/Sports-Classification/...`; dentro del contenedor `Sports-Classification` la misma carpeta es `/app/...`. `datasets/`, `models/` y `output/` están fuera de git; lo único versionado con datos son `ground_truth/*.gt.json`, `ground_truth/highlights_goals.json` y las listas de `docs/particiones/*.txt`. Los papeles (entrenamiento / validación / evaluación) son los de `docs/particion_datasets.md`.

| Carpeta | Tamaño (`du -sh`) | Contenido |
|---|---|---|
| `datasets/soccertrack-v2` | 195G | videos 78G, gsr 53G, views 62G (derivado nuestro), raw 2,5G, clips 1,3G (derivado), mot 58M, bas 4,5M, ball 2,6M |
| `datasets/SoccerNet` | 66G | england_efl 18G (7 partidos extraídos + cachés), spotting-ball-2024 14G (3 zip), tracking 35G (2 zip + `extracted/`), etiquetas de las 6 ligas ~52M |
| `datasets/dataset_vjepa_backup` | 23G | 11.401 clips de 6 s |
| `datasets/sn_tuning_videos` | 19G | 10 partidos de SoccerNet-v2 (2 mitades `.mkv`) |
| `datasets/ball_panorama` | 7,3G | 17.507 recortes JPG de 1024 px con etiquetas YOLO |
| `datasets/clasico_2017` | 3,3G | un `.mp4` de 1h48m |
| `datasets/partidos_tv` | 2,8G | 2 juveniles |
| `datasets/dataset_ball_actions` | 1,4G | 11.143 clips de 32 frames |
| `datasets/clips_analisis` | 294M | 212 clips cortos |
| `datasets/soccernet_tracking_ball` | 131M | etiquetas + enlaces a los JPG de `SoccerNet/tracking/extracted` |
| `datasets/tdeed_pano` | 94M | etiquetas, splits e índices (`frames/` no está en disco) |
| `basketball/datasets` | ~3,2G | dataset_fusionado 2,5G, basketball_final 378M, player-detection 139M, court-detection 122M |
| `/mnt/datasets/SoccerNet` (solo host, NFS `192.168.1.130:/export/datasets`) | 928 GB | 500 carpetas de partido con `1_720p.mkv` y `2_720p.mkv` (1.000 mitades) |

### 3.1 Rama `main`: pipeline de TV

| Dataset (ruta) | Origen | Tamaño | Papel (`docs/particion_datasets.md`) | Licencia según el repo | Modelos derivados y consumo |
|---|---|---|---|---|---|
| SoccerNet-v2: etiquetas `datasets/SoccerNet/<6 ligas>/.../Labels-v2.json` (500) y `Labels-cameras.json` (200); vídeos en `/mnt` (500 partidos: EPL 95, UCL 101, Ligue 1 38, Bundesliga 53, Serie A 96, LaLiga 117; 1280x720, 25 fps) | SDK `pip install SoccerNet` (`scripts/download_ball_action_data.py`); el script de descarga de los vídeos no consta | 52M + NFS | train 100 (SVC) / eval 6 + 57 (62 distintos) / 13 contaminados / ~380 sin asignar; in-sample para T-DEED | etiquetas públicas sin NDA; vídeos con NDA; texto de la NDA no consta | benchmark de tiros (6 partidos), umbrales de la cabeza SoccerNet de T-DEED, GT del Clásico, lector de marcador medido en 10 partidos (29/30 goles) |
| SoccerNet Ball Action Spotting 2024: `datasets/SoccerNet/england_efl/2019-2020/` (7 partidos, `720p.mp4` + `224p.mp4` + `Labels-ball.json`) y `spotting-ball-2024/{train,valid,test}.zip` | SDK con contraseña NDA | 18G + 14G | train Blackburn, Brentford, Hull, Leeds / val Middlesbrough / eval Stoke, Reading | NDA SoccerNet | T-DEED `SoccerNetBall_test` (limpio, AP tiros 75,6 % en Stoke+Reading) y `SoccerNetBall_challenge2` (producción, sin eval limpia); R3D-18 v1/v2 vía `dataset_ball_actions`; `ground_truth/`; fusión V-JEPA2 de highlights medida en los 7 (13 goles) |
| 12 clases de `Labels-ball.json`, 12.433 anotaciones | PASS 4.985, DRIVE 4.300, HIGH PASS 761, HEADER 713, OUT 551, THROW IN 362, CROSS 261, BALL PLAYER BLOCK 223, SHOT 169, TACKLE 74, FREE KICK 21, GOAL 13 | | | | sin rótulo de marcador en estos vídeos |
| SoccerNet Tracking: `datasets/SoccerNet/tracking/{train.zip 9,0G, test.zip 8,2G}` (57 + 49 secuencias de 30 s, 1920x1080) → `datasets/soccernet_tracking_ball` (YOLO, 1 clase) | SDK (log en `output/`); script de descarga no consta; conversión `scripts/ball_detector/prepare_soccernet_tracking_ball.py` | 35G / 131M | train oficial / test oficial (= `val` del YOLO; se usó para elegir época, así que no queda test intacto) | no consta | `models/ball/yolo11s_snt_ball.pt` (detector de balón TV, opcional en `config.py:45-47`); train 14.250 imágenes, val 2.450; balón mediano 14x13 px |
| `datasets/dataset_vjepa_backup` (11.401 clips de 6 s, 17 clases, 100 partidos) | recortes de `/mnt` según Labels-v2; generador no consta | 23G | entrenamiento del SVC | derivado de vídeo SoccerNet (NDA) | `models/svc_classifier_more_penalties.joblib` (V-JEPA2 ViT-L + SVC, 16 clases); desde 2026-10-15 no pone eventos, sigue como segundo juez de gol en highlights |
| `datasets/clips_analisis` (212 clips; 184 con partido en el nombre = 70 partidos; 28 clips de tiro sin partido) | recortes de SoccerNet-v2; generador no consta | 294M | evaluación (57 partidos) + 13 contaminados para el SVC | derivado de SoccerNet (NDA) | comparación de modelos y VLM en `main`; también lo usan VLLM y videoARM |
| `datasets/dataset_ball_actions` (11.143 clips 224x224 de 32 frames; 6 clases: pass 4.652, drive 3.913, background 1.046, high_pass 678, header 586, throw_in 268) | `scripts/extract_ball_action_clips.py` desde los 7 EFL | 1,4G | R3D-18 v2 entrenó con 6 partidos incluidos Stoke y Reading: no limpio; el próximo reentreno debe usar solo los 4 de entrenamiento | derivado de BAS (NDA) | `models/r3d18_ball_actions-v2.pt` (producción, val_acc 0,782), `r3d18_ball_actions.pt` |
| `datasets/sn_tuning_videos` (10 partidos SN-v2) | copias de `/mnt` | 19G | los 10 están en la lista de entrenamiento del SVC; in-sample para T-DEED | NDA | umbrales de la cabeza SoccerNet de T-DEED (`scripts/tune_tdeed_event_thresholds.py` → `config.py`) |
| `datasets/clasico_2017/real_madrid_barcelona_2017.mp4` (1920x1080, 25 fps, 6.515 s) | YouTube `cXBfVPp1DJU`; etiquetas = Labels-v2 del mismo partido (desfases ~359 s y ~3.554 s por mitad) | 3,3G | validación de highlights; única verdad de córner, falta y banda en TV; in-sample para T-DEED | no consta (YouTube) | `ground_truth/highlights_goals.json` (5 goles, clip de referencia de Messi) |
| `datasets/partidos_tv`: Mallorca-Barcelona juvenil (1280x720, 25 fps, 5.975 s, 3 goles) y Espanyol-Damm juvenil (1280x720, 30 fps, 8.175 s, 0 goles) | YouTube | 2,8G | validación (fijaron `goal_audio_min_db` y anclas); evaluación = 2 juveniles nuevos pendientes de elegir; entrenamiento ninguno | no consta | backend, `scripts/evaluate_match.py`, highlights; `output/annotations/espanyol_damm_Ground_Truth.csv` (181 líneas: 63 drive, 56 pass, 3 high pass, 3 header, 1 throw in; sin eventos todavía) |
| `ground_truth/2019-10-01_Middlesbrough-Preston_North_End.gt.json` (405 anotaciones, 20 min, 4 retoques a mano) | `scripts/build_ground_truth.py` | 0,5G con vídeo y cachés | validación | derivado de BAS (NDA) | arnés `scripts/evaluate_match.py` / `evaluate_pipeline.py` |
| Externos sin dataset local: PRTReId SoccerNet (`models/reid/`), PnLCalib (`models/homography/PnLCalib/`), YOLO11 COCO, Qwen3.5-9B (Ollama `qwen3.5:9b-q8_0`), MolmoWeb-4B | pesos de terceros | | | PRTReId y PnLCalib se entrenaron con SoccerNet GSR y Calibration | equipos (`core/classification.py`), homografía (`core/homography.py`), detector por defecto, VLM de highlights |

### 3.2 Rama panorámica (`feat/panorama-shots`, fusionada en `main` en 170cdea; `feat/panorama-pipeline` prevista)

| Dataset (ruta) | Origen | Tamaño | Papel | Licencia | Modelos derivados |
|---|---|---|---|---|---|
| SoccerTrack v2: `datasets/soccertrack-v2/{videos,gsr,bas,ball,mot,raw}` (10 partidos universitarios, 934 min, panorámica fija 4K BePro, 25 fps; 8 partidos 4096x1080, 117092 3840x1906, 132831 3840x1504) | Hugging Face `atomscott/soccertrack-v2` (acceso restringido, `HF_TOKEN`), release v1.2, `scripts/download_soccertrack_v2.sh`; `mot/` del Drive del SoccerTrack Challenge 2025 (2026-08-29) | ~132G descargados | partición oficial del Challenge 2025: train 117092, 118575, 118576, 118577, 118578, 128058 / val 117093, 132877 / eval 128057, 132831 | **CC BY 4.0** (`README.md` del dataset; la única licencia explícita en disco) | `bas/` 21.428 eventos de 12 clases con `team` y `player_id`; `gsr/` 1.398.025 frames y 30,7 M objetos; `ball/` pista reconstruida (solo referencia); `raw/` 65 keypoints por partido |
| Derivados nuestros en la misma carpeta: `views/` (cámara virtual 796x448: `vcam796` para 10 partidos, `vcam796vl1` para 9) y `clips/` | `scripts/shot_benchmark/panorama_vcam.py`, `vcam_path.py`, `run_panorama_*.sh` | 62G + 1,3G | | CC BY 4.0 (derivado) | entrada de T-DEED y del detector; los renders de 132831 y 132877 con la pista del dataset están degradados (balón aparcado en (0,0)) |
| `datasets/ball_panorama` (train 12.103 ficheros = 9.366 positivos + 2.737 negativos; val 5.404) | `scripts/ball_detector/make_pseudo_labels.py` (pista `ball/*.npz` proyectada + confirmación YOLO11x COCO; sin revisión manual) | 7,3G | train 6 / val 117093, 132877 | CC BY 4.0 (derivado) | `output/ball_panorama/runs/yolo11s_pano/weights/best.pt` (mAP50 0,990, mAP50-95 0,626 sobre pseudoetiquetas; pendiente copiar a `models/`) |
| `datasets/tdeed_pano` (train 10 mitades = 10.441 eventos; val 4 mitades = 4.529 eventos; `frames/` ausente) | `scripts/shot_benchmark/tdeed_finetune_data.py` (sobrescribe `labels_path.txt` de T-DEED) | 94M | train 5 partidos (117092 nocturno se dejó fuera) / val 117093, 132877 | CC BY 4.0 (derivado) | `models/tdeed/SNB_pano1/checkpoint_best.pt` (mAP val 0,301 → 0,438; eval tiros 41,6 → 55,2 %); `SNB_smoke` (humo) |

Nada de esta rama corre desde `main.py` ni el backend (`core/ball_spotting.py` no conoce `SNB_pano1`). El eval 128057 + 132831 ya se ha mirado 3 veces y la demo del 18-oct sobre 128057 contraviene la partición (`docs/hoja_de_ruta.md`).

### 3.3 Rama `VLLM` (no fusionada; 1 commit propio 64ce7af, 2026-09-07)

Añade `core/events_omni.py` y `scripts/serve_events_omni.sh`: Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit servido por vLLM en el puerto 8002, clip entero (vídeo + audio) a 224p. Único dataset: `datasets/clips_analisis/` (resultado `output/events_omni_eval_full.jsonl`, 213 líneas, 55 % de acierto de clase). Decisión vigente: no fusionar, etiquetar `archive/` y borrar.

### 3.4 Rama `videoARM` (0 commits por delante de `main`; último propio f274aff, 2026-09-04)

Historia de los backends VLM (`core/events_vllm.py`: VideoARM + Qwen3.5-9B/vLLM fp8, puerto 8010) medida sobre `clips_analisis/` (12 clips y los 61 de gol + tiro: SVC 27/61, VideoARM 30/61, híbrido 42/61). Ese código se quitó de `main` en 6ceb32b (2026-10-06) y se recupera con `git revert 6ceb32b`. Vía agotada; no consume ningún otro dataset. Los veredictos del verificador sobre los 7 EFL quedaron como caché `720p.shot_verdicts.vjepa2.w6s3.v1.jsonl`.

### 3.5 Carpeta y rama `basketball` (0 commits por delante de `main`)

| Dataset (ruta) | Origen | Tamaño | Papel | Licencia | Modelos |
|---|---|---|---|---|---|
| Roboflow `basketball-player-detection-3-ycjdo` v4 (654 imágenes 464/96/94, 10 clases: ball, ball-in-basket, number, player, player-in-possession, player-jump-shot, player-layup-dunk, player-shot-block, referee, rim; 1280x1280) | `basketball/scripts/download_datasets.py` | 139M | train/valid/test de Roboflow | CC BY 4.0 (`README.dataset.txt`) | modelo hospedado en Roboflow usado por el pipeline (`inference.get_model`) |
| Roboflow `basketball-court-detection-2` v14 (1.460 imágenes, keypoints COCO, 640x640) | ídem | 122M | ídem | CC BY 4.0 | modelo hospedado de keypoints de cancha |
| `dataset_fusionado` (Roboflow `ugiat-basketball_merged_dataset` v1: 5.593 imágenes 4.583/486/524, 17 clases → 6) | propio en Roboflow | 2,5G | | marcado "Public Domain" en Roboflow, pero el README del derivado avisa de que incluye fuentes CC BY 4.0 (mal marcado) | ninguno |
| `basketball_final` (1.305/294/289 imágenes, 10 clases) y `basketball_final_5cls` | `build_final_dataset.py` (2026-10-16), partición por clip de origen NBA | 378M + 15M | entrenamiento sugerido, no ejecutado | hereda las anteriores | ningún modelo propio en disco; `fashion-clip` para equipos |

### 3.6 Modelos en disco y de qué dataset salen

| Modelo | Fichero | Entrenado con | ¿Evaluación limpia? |
|---|---|---|---|
| T-DEED `SoccerNetBall_test` (RegNetY-200MF) | `models/tdeed/SoccerNetBall_test/checkpoint_best.pt` (49,6 MB) | 4 EFL de entrenamiento + SoccerNet-v2; val Middlesbrough | Sí: Stoke y Reading (AP tiros 75,6 %) |
| T-DEED `SoccerNetBall_challenge2` (RegNetY-800MF, producción) | `models/tdeed/SoccerNetBall_challenge2/checkpoint_best.pt` (185 MB) | 6 EFL (todo menos Middlesbrough) + SoccerNet-v2 | No: ningún partido EFL ni de v2 queda fuera |
| T-DEED `SNB_pano1` | `models/tdeed/SNB_pano1/` | `challenge2` + `tdeed_pano` (5 partidos SoccerTrack) | Eval 128057 + 132831 ya usado 3 veces |
| R3D-18 acciones de balón v1 / v2 | `models/r3d18_ball_actions*.pt` | `dataset_ball_actions` (6 EFL) | No: entrenó con Stoke y Reading |
| V-JEPA2 ViT-L + SVC | `models/svc_classifier_more_penalties.joblib` (300 MB) | `dataset_vjepa_backup` (100 partidos SN-v2) | Sí en los 7 EFL y en los 62 de evaluación |
| YOLO11s balón TV | `models/ball/yolo11s_snt_ball.pt` (19,2 MB) | `soccernet_tracking_ball` | El test oficial sirvió para elegir época |
| YOLO11s balón panorámica | `output/ball_panorama/runs/yolo11s_pano/weights/best.pt` | `ball_panorama` | Solo medido en pseudoetiquetas de validación |
| PRTReId + HRNet, PnLCalib, YOLO11 COCO, Qwen3.5-9B, MolmoWeb-4B, Soccer_Master `SoccerNetGSR_Detection` | `models/reid/`, `models/homography/`, `models/hub/` | externos (SoccerNet GSR / Calibration, COCO) | no aplica |

Licencias de código que afectan a los pesos: T-DEED es GPL-3.0 y `SNB_pano1`/`SNB_tv1` son derivados suyos (`docs/estudio_deteccion_tiros.md`, `docs/hoja_de_ruta.md`).

### 3.7 Hallazgos del inventario

- `/mnt/datasets/SoccerNet` está montado hoy en el host con las 1.000 mitades de los 500 partidos; `docs/particion_datasets.md:20` sigue diciendo que a 8-oct-2026 el montaje no estaba disponible. El contenedor no lo ve: usar esos vídeos exige copiarlos al proyecto. Los `Labels-v2.json` no están junto a los vídeos en `/mnt`, sino en `datasets/SoccerNet/<liga>/`.
- Los 10 partidos de `sn_tuning_videos` están todos en la lista de entrenamiento del SVC: no valen para validar la fusión V-JEPA2 ni como "limpios" para T-DEED. La validación de highlights sin marcador pide 3-5 partidos sin asignar copiados desde `/mnt`.
- Derivados sin generador en el repo: `dataset_vjepa_backup/`, `clips_analisis/`, el entrenamiento del SVC, la descarga de SoccerNet Tracking y de los vídeos de `/mnt`, y el origen de los 28 clips de tiro sin partido en el nombre. `tdeed_pano/frames/` no está en disco (se regenera con `tdeed_finetune_data.py`).
- La única licencia explícita guardada con un dataset es la de SoccerTrack v2 (CC BY 4.0) y las de los dos Roboflow de baloncesto (CC BY 4.0). El repo solo anota que los vídeos de SoccerNet, Ball Action Spotting y Tracking exigen la NDA; el texto de la NDA no está guardado y todos los derivados de esos vídeos (`dataset_vjepa_backup`, `clips_analisis`, `dataset_ball_actions`, `sn_tuning_videos`, `ground_truth/*.mp4`) heredan esa condición.
- Cero anotaciones de eventos en el dominio del producto (juveniles): el CSV del Espanyol solo tiene acciones; faltan los dos juveniles de evaluación; faltan tiros con equipo y jugador para medir `SNB_pano1`/`SNB_tv1`.
- Sin cifra final limpia del pipeline en TV: `challenge2` y R3D-18 v2 entrenaron con 6 de 7 EFL; el detector de balón de TV no tiene test intacto ni test del dominio real (faltan 300-500 frames de YouTube etiquetados).
- Basketball: ningún modelo propio entrenado y un dataset fusionado con la licencia mal marcada.

## 4. Catálogo externo por utilidad

Convención de cada ficha: nombre (año, autores/organización, enlace); qué contiene (vídeo, cámara, resolución, horas o partidos; anotaciones y granularidad; equipo y jugador sí/no); licencia (literal cuando la fuente la da); solape con lo nuestro; veredicto y rama/funcionalidad; qué habría que hacer para usarlo. "Gated" quiere decir acceso en Hugging Face previa solicitud.

Nota sobre el acceso a SoccerNet en 2026: la distribución se ha migrado a Hugging Face (organización `SoccerNet`, 27 repositorios). El paquete `SoccerNet` 0.2.0 (MIT, 2026-09-07) descarga desde HF por defecto y la antigua contraseña del NDA ya no se usa ("password= is accepted but ignored"). Los vídeos (`SoccerNet_raw_HQ`, `SoccerNet-Tracking-RAW-Video`, `SN-BAS-2023/2024/2025`, `SN-PCBAS-2026`, `ActionAnticipation`, `SN-VQA-2027`) son gated con aprobación manual y NDA nominal; las etiquetas `SN-Labels`, `SN-GSR-2024/2025`, `SN-MVFouls-2024/2025`, `SN-echoes`, `SN-Depth-2025`, `SN-Tracking-2023`, `SN-Jersey-2023`, `SN-ReID-2023`, `SN-Calibration(-2023)` y `SN-VQA-2026` son públicas ([soccer-net.org/data](https://www.soccer-net.org/data), [API HF](https://huggingface.co/api/datasets?author=SoccerNet)).

### 4A. Usar ya

**SoccerHigh (2025, UPF)**, [github.com/IPCV/SoccerHigh](https://github.com/IPCV/SoccerHigh). Ficha completa en 2.2.
- Veredicto: usar ya como verdad de highlights de TV. Funcionalidad: highlights (`analytics/highlights/evaluate.py`, política 12-30 s).
- Para usarlo: clonar el repo, emparejar los 237 `Labels-summary.json` con nuestras carpetas (nombres idénticos), convertir los tiempos `_HQ.mkv` → 720p con `video.ini`, medir P/R/F1@T de nuestros candidatos; empezar por el Clásico (21 segmentos) y seguir con los 45 de test.

**StatsBomb Open Data (StatsBomb Services Ltd)**, [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data). Ficha completa en 2.3.
- Veredicto: usar ya como verdad de tiros, goles, córners, faltas y tarjetas con equipo y jugador en 135 partidos locales. Funcionalidad: evaluación de eventos T-DEED y lector de marcador en `main`; la atribución de equipo que el pipeline aún no emite.
- Para usarlo: descargar `events/<match_id>.json` y `lineups/` de los 135 partidos (~3,2 MB por partido), guardar la tabla local ⇄ `match_id`, anclar el reloj de cada mitad al vídeo (etiqueta `Kick-off` de Labels-v2 o goles), registrarse en el resource centre de StatsBomb y usar el logo en cualquier publicación.

**SoccerNet-Echoes (2024, SimulaMet / SoccerNet)**, [huggingface.co/datasets/SoccerNet/SN-echoes](https://huggingface.co/datasets/SoccerNet/SN-echoes), [arXiv 2405.07354](https://arxiv.org/html/2405.07354).
- Qué contiene: transcripciones ASR (Whisper large v1, v2 y v3, más traducción al inglés) del comentario de "all 1,100 game halves from SoccerNet" (550 partidos); 70 mitades sin comentario; 10 idiomas de locución. JSON por mitad con `segments` → `start_time`, `end_time`, `text`. Sin equipo ni jugador estructurados.
- Calidad: WER 0,443 (v1), 0,458 (v2), 0,551 (v3) frente a 40 mitades de GOAL; repeticiones alucinadas de Whisper con silencio o música.
- Licencia: CC BY 4.0 (etiqueta de HF). Público, sin NDA.
- Solape: cubre nuestros 500 partidos de `/mnt` y los 50 de challenge.
- Veredicto: usar ya. Funcionalidad: gol por comentario (palabras clave multilingües) y señal débil para highlights en TV; validación del lector de marcador.
- Para usarlo: descargar desde HF, emparejar por ruta de partido, comprobar el desfase audio-vídeo en 2-3 partidos con goles conocidos.

**SoccerNet-Caption, etiquetas (2023-2024, SoccerNet)**, [github.com/SoccerNet/sn-caption](https://github.com/SoccerNet/sn-caption), [arXiv 2304.04565](https://arxiv.org/abs/2304.04565).
- Qué contiene: 36.894 comentarios con timestamp sobre 471 partidos de TV (715,9 h) + 42 de challenge; versiones anonimizadas e identificadas (jugadores, equipos, árbitros, entrenadores) y alineaciones; features a 2 fps. Mismos ficheros de vídeo que ya tenemos.
- Licencia: no declarada para las etiquetas (no verificado); los vídeos siguen bajo NDA.
- Solape: 471 de nuestros 500 partidos (por comprobar la correspondencia de carpetas).
- Veredicto: usar ya (solo etiquetas, sin bajar vídeo). Funcionalidad: goles y equipos en texto, enlace nombre de jugador ↔ dorsal, contexto para el VLM de highlights.
- Para usarlo: `downloadDataTask(task="caption-2024")` sin vídeo; cruzar con Labels-v2.

**SN-BAS-2025 Team Ball Action Spotting (2025, SoccerNet)**, [huggingface.co/datasets/SoccerNet/SN-BAS-2025](https://huggingface.co/datasets/SoccerNet/SN-BAS-2025), devkit [sn-teamspotting](https://github.com/SoccerNet/sn-teamspotting) (GPL-3.0).
- Qué contiene: los mismos 7 partidos EFL de BAS-2024 + 2 de challenge, 720p, 25 fps; 12 clases con el equipo codificado como left/right respecto a la cámara. 19,3 GB (incluye vídeo).
- Métrica: Team-mAP@1 (AP por par acción-equipo, tolerancia 1 s). Baseline T-DEED con cabeza de equipo: 47,18 Team-mAP@1 / 53,59 mAP@1 en test; ganador 2025 (dude.k, T-DEED reimplementado con cabeza conjunta equipo×acción) 60,03.
- Licencia: gated, `license: other`, NDA de SoccerNet.
- Solape: total con `datasets/SoccerNet/england_efl/2019-2020/`; solo necesitamos las etiquetas con equipo.
- Veredicto: usar ya (prioridad 1 para T-DEED). Funcionalidad: eventos y acciones con equipo en `main`, misma partición train 4 / val Middlesbrough / eval Stoke y Reading.
- Para usarlo: cuenta HF, solicitud y NDA; añadir cabeza de equipo (sigmoid + BCE) o la cabeza conjunta; medir Team-mAP@1 con `scripts/evaluate_match.py`.

**SoccerNet Game State Reconstruction, SN-GSR-2024/2025 (2024-2025, SoccerNet)**, [github.com/SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate), [arXiv 2404.11335](https://arxiv.org/html/2404.11335).
- Qué contiene: 200 clips de 30 s (1920x1080, 25 fps, ~150.000 frames) de cámara principal construidos sobre SoccerNet-Tracking; cajas de jugadores, árbitros y balón; >9,37 M puntos de líneas (26 clases); calibración de cámara por frame; >2,36 M posiciones en el campo (`bbox_pitch`); rol, equipo left/right, dorsal y track ID. Versión 1.3.
- Métrica: GS-HOTA. Baseline TrackLab (YOLO + PRTReId + TVCalib/PnLCalib + MMOCR) 22,26; mejor 2025 63,90 (KIST-GSR).
- Licencia: público en HF, etiqueta GPL-3.0 (licencia de software aplicada a datos; alcance sobre modelos ambiguo).
- Solape: las secuencias son las de SoccerNet-Tracking, de las que tenemos 106.
- Veredicto: usar ya. Funcionalidad: equipos, dorsales, homografía y posiciones en `main` (nuestro stack YOLO11 + prtreid + PnLCalib coincide con el baseline).
- Para usarlo: `snapshot_download("SoccerNet/SN-GSR-2025")`; medir GS-HOTA de nuestro pipeline; afinar prtreid/PnLCalib con los splits oficiales.

**OSL-loc-SPL-highlight (OpenSportsLab)**, [huggingface.co/datasets/OpenSportsLab/OSL-loc-SPL-highlight](https://huggingface.co/datasets/OpenSportsLab/OSL-loc-SPL-highlight).
- Qué contiene: 171 vídeos con etiquetas Start/End/Goal de segmentos de highlight de gol; train 102 vídeos / 922 eventos, valid 17 / 158, test 52 / 498; formato OpenSportsLib (Parquet + WebDataset). Liga, resolución y origen: no verificados (ficha de HF sin descripción).
- Licencia: gated; no declarada (no verificado).
- Solape: desconocido.
- Veredicto: usar ya si la composición resulta razonable; es la única verdad externa de "dónde empieza y acaba el clip de un gol". Funcionalidad: política de corte de highlights (12-30 s, gol ≤ 45 s).
- Para usarlo: pedir acceso, inspeccionar liga y resolución, comparar nuestras duraciones con las suyas.

**TeamTrack (2024, Scott et al., Nagoya / Tsukuba / Tokai / AIST)**, [kaggle.com/datasets/atomscott/teamtrack](https://www.kaggle.com/datasets/atomscott/teamtrack), [arXiv 2404.13868](https://arxiv.org/html/2404.13868v1).
- Qué contiene: fútbol universitario con ojo de pez 8K fijo (Z CAM E2-F8, 30 min) y dron 4K cenital (DJI Mavic 3, 30 min), 30 fps; 2.484.000 cajas de fútbol; también baloncesto y balonmano (total 155,5 min, 279.900 frames, 4.374.900 cajas). IDs persistentes durante todo el partido y afiliación de equipo (formato SportsLabKit y MOTChallenge); sin dorsales, sin eventos, sin coordenadas de campo. Es con toda probabilidad la captura de Tsukuba de SoccerTrack v1 anotada por completo (mismos dispositivos y mismo recuento de cajas).
- Resultados: ByteTrack HOTA 59,3 (ojo de pez) y 53,7 (dron).
- Licencia: Kaggle declara "MIT" (16,7 GB, 2024-11-26); la web del proyecto no la indica. Confirmar por escrito con los autores antes de entrenar modelos de producto.
- Solape: ninguno; complementa SoccerTrack v2 (otro campo, otra óptica).
- Veredicto: usar ya. Rama panorámica: segundo partido para el detector `yolo11s_pano` y la cámara virtual; tracking de jugadores con verdad de IDs.
- Para usarlo: descargar, comprobar si el balón conserva cajas (no verificado), corregir la distorsión del ojo de pez (método de Zhang) y generar `vcam796` como con SoccerTrack v2.

**Transfermarkt datasets (dcaribou)**, [github.com/dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets).
- Qué contiene: tablas `games` (88.000+), `game_events` (1.270.000+ filas: goles, tarjetas, sustituciones, con `minute`, `type`, `club_id`, `player_id`), `appearances`, `players`, `clubs`, `competitions` (65, incluida la UCL); datos hasta el 6 de julio de 2026 (actualizaciones en pausa). También en Kaggle `davidcariboo/player-scores` y como DuckDB/CSV. Sin vídeo.
- Licencia: "Creative Commons Zero v1.0 Universal" (CC0-1.0).
- Solape: probable para los 500 partidos de SoccerNet-v2 incluida la UCL; la EFL Championship 2019-20 no está confirmada.
- Veredicto: usar ya. Funcionalidad: verdad de goles y marcador al minuto para highlights sin marcador y para el lector de marcador en los ~380 partidos sin asignar.
- Para usarlo: mapear nombres de club y fecha a nuestra lista; ventana de ±60 s más descuento (`extra_minute`).

**Understat (understat.com)**, [understat.com](https://understat.com/league/EPL/2014).
- Qué contiene: tiros con xG, `minute`, `result` (Goal, SavedShot, MissedShots, BlockedShot), `situation`, `x`/`y`, jugador y asistente para EPL, La Liga, Bundesliga, Serie A, Ligue 1 (y RFPL) desde 2014/15. Sin vídeo. Lectores: `soccerdata`, `understatapi`, `penaltyblog` (scraping).
- Licencia: sin términos de uso publicados (scraping; no verificado).
- Solape: nuestras 5 ligas 2014-17 (no UCL, no EFL).
- Veredicto: usar ya con cautela, solo internamente. Funcionalidad: recuento y resultado de tiros por partido para contrastar T-DEED y el benchmark de tiros.
- Para usarlo: `soccerdata.Understat.read_shot_events`; tolerancia al minuto.

**Kaggle "Football Events" (secareanualin)**, [kaggle.com/datasets/secareanualin/football-events](https://www.kaggle.com/datasets/secareanualin/football-events).
- Qué contiene: 941.009 eventos de 9.074 partidos de las 5 grandes ligas, temporadas 2011-12 a 2016-17; `time` en minuto; `event_type` (Attempt, Corner, Foul, Yellow card, Second yellow, Red card, Substitution, Free kick won, Offside, Hand ball, Penalty conceded), `shot_place`, `shot_outcome`, `is_goal`, `location`, `situation`, `fast_break`, equipo y jugador; texto de comentario de BBC/ESPN/onefootball; cuotas en `ginf.csv`. Algunos partidos sin eventos detallados (`adv_stats`). Sin vídeo.
- Licencia: no verificable (página solo JavaScript); el texto de comentario tiene derechos propios.
- Solape: por competición y temporada cubre nuestras 2014-15, 2015-16 y 2016-17 de EPL, LaLiga, Serie A, Bundesliga y Ligue 1 (no UCL, no EFL); la 2016-17 probablemente incompleta (corte a principios de 2017, no confirmado).
- Veredicto: usar ya para evaluación interna. Funcionalidad: tiros, goles, córners, faltas y tarjetas con equipo y jugador al minuto; "momentos obligatorios" para medir el recall de highlights.
- Para usarlo: descargar los CSV, cruzar fecha + equipos, filtrar `adv_stats`.

**European Soccer Database (Hugo Mathien)**, [kaggle.com/datasets/hugomathien/soccer](https://www.kaggle.com/datasets/hugomathien/soccer).
- Qué contiene: SQLite (~299 MB) con más de 25.000 partidos de 11 ligas, temporadas 2008 a 2016; eventos en XML (goles con `elapsed` en minutos, tiros a puerta/fuera, faltas, tarjetas, centros, córners, posesión) para más de 10.000 partidos; cuotas de hasta 10 casas; atributos FIFA. Derivado en CSV: Kaggle `jiezi2004/soccer`.
- Licencia: ODbL según espejos (no verificado en la página).
- Solape: nuestras temporadas 2014-15 y 2015-16 de liga (no 2016-17, no UCL, no EFL).
- Veredicto: usar ya como segunda fuente para cruzar con Kaggle Football Events. Funcionalidad: verdad de goles/tiros/faltas/tarjetas al minuto; prior de "partido desequilibrado" por cuotas.

**football-data.co.uk**, [football-data.co.uk](https://www.football-data.co.uk/notes.txt).
- Qué contiene: una fila por partido con marcador final y al descanso, tiros (`HS`, `AS`), a puerta (`HST`, `AST`), córners, faltas, fueras de juego, tarjetas, árbitro y cuotas; LaLiga desde 1993/94 con estadísticas desde 2005/06; ligas inglesas incluida la Championship. Sin minutos de gol, sin vídeo.
- Licencia: "My data is free"; sin texto de licencia (no verificado).
- Solape: todas las ligas de SoccerNet-v2 y la EFL 2019-20 (no la UCL).
- Veredicto: usar ya. Funcionalidad: validar los recuentos por partido de nuestro detector (tiros, córners, faltas, tarjetas) y reconstruir el marcador esperado (`HTHG/HTAG`) para el lector de marcador.

### 4B. Útil en el futuro

**FOOTPASS / SN-PCBAS-2026, Player-Centric Ball Action Spotting (2025-2026, Footovision / Mines Paris)**, [github.com/JeremieOchin/FOOTPASS](https://github.com/JeremieOchin/FOOTPASS), [arXiv 2511.16183](https://arxiv.org/html/2511.16183), [CVIU 269 (2026) 104790](https://doi.org/10.1016/j.cviu.2026.104790), HF [`SoccerNet/SN-PCBAS-2026`](https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026) (236 GB, gated). Entrada revisada el 2026-11-06.
- Son el mismo dataset con dos nombres. FOOTPASS ("Footovision Play-by-Play Action Spotting in Soccer") es el nombre del dataset, del paper (preprint HAL hal-05373478 de septiembre de 2025, arXiv de noviembre de 2025, CVIU vol. 269, junio de 2026) y del devkit de GitHub (baselines, `evaluation.py`, etiquetas públicas). SN-PCBAS-2026 es el nombre del repositorio de Hugging Face de la organización SoccerNet que distribuye vídeos y datos tácticos (creado el 2025-10-02) y de la tarea "Player-Centric Ball Action Spotting" del reto SoccerNet 2026 (patrocinada por Footovision con 1.000 USD, abierta hasta el 2026-04-24). El README de FOOTPASS remite a ese repositorio para descargar y la página del reto remite a FOOTPASS como devkit; no hay otra distribución.
- Qué contiene: 54 partidos completos 2023/24 (Ligue 1, Bundesliga, Serie A, La Liga, UCL; 50 equipos), broadcast nativo 1920x1080 a 25 fps; 102.992 eventos en 8 clases (Pass 49,9 %, Drive 39,0 %, Header 4 %, Cross 2,3 %, Throw-in 1,9 %, Block 1,4 %, Shot 1,2 %, Tackle 0,3 %) como tuplas (frame, team 0 izquierda / 1 derecha, dorsal, clase), anclados a un solo frame, anotados también en repeticiones y primeros planos; el 81,5 % tiene caja visible. Datos tácticos por jugador y frame en HDF5 con 14 columnas (`FRAME, PLAYER_ID, LEFT_TO_RIGHT, SHIRT_NUMBER, ROLE_ID, X_POS, Y_POS, X_SPEED, Y_SPEED, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT, CLS`): tracklet (caja en coordenadas 1920x1080, NaN cuando el jugador no se ve), posición y velocidad en el campo, equipo, dorsal y uno de 13 roles tácticos (Goalkeeper; Left, Left Central, Mid Central y Right Central Back; Right Back; Left y Right Midfielder; Defensive y Attacking Midfielder; Left y Right Winger; Central Forward). Los ganadores del reto trabajan con 26 huecos de rol (13 por equipo). Los eventos los anotó un equipo profesional con software propio; el tracking, los dorsales y las trayectorias de los jugadores no visibles salen de un pipeline automático con imputación (no consta verificación humana); los roles, de trayectorias + clustering + anotación experta.
- Diferencias entre el paper y la distribución del reto:
  1. Particiones. El paper divide 48 / 3 / 3 partidos (train 91.327, validación 6.070 y test 5.595 eventos; el test son los partidos 0, 36 y 50). Hugging Face publica `TRAIN`, `VAL` y `CHALLENGE`, sin `TEST`: el test del paper es el conjunto *challenge* (el README lo llama "Challenge/Test"), con verdad oculta y evaluación en Codabench; las etiquetas públicas del devkit (`playbyplay_train.json`, `playbyplay_val.json`) cubren 51 partidos. Codabench tiene dos servidores, "test" (11249) y "challenge" (11232); qué partidos evalúa el primero no se pudo verificar (un participante calibra umbrales con etiquetas de su "test", así que no es el conjunto oculto; probablemente la validación).
  2. Métrica. El paper da F1 global (micro) a τ = 0,15 y ±12 frames más F1 y AP de 11 puntos por clase: TAAD 35,9, TAAD+GNN 52,1, TAAD+DST 67,5 en su test. El reto puntúa Macro-F1 (media de las 8 clases, con clase, equipo y dorsal correctos): el mismo TAAD+DST sobre los mismos partidos da 46,41 (la media de sus F1 por clase del paper es 46,4; tackle 0,0 y block 18,0 hunden la media). Ganador 2026 (PAVE, TAHAKOM) 58,94; 2.º 56,40; 3.º 55,69; 6 equipos y 124 envíos.
  3. Resoluciones. El paper describe Full HD y entrena a 352x640. Hugging Face trae las dos: `videos_fullHD_*` (207,2 GB en 7 zips) y `videos_352x640_*` (25,7 GB), más `tactical_data_*` (2,8 GB). Para reproducir los baselines bastan 28,5 GB.
  4. Unidad. Las anotaciones y los datos tácticos van por mitad (`game_18_H1`, `game_18_H2`); por eso PAVE habla de 96 "partidos" de entrenamiento y 6 de validación.
  5. Índices de clase. El código usa 9 (0 fondo + drive, pass, cross, throw-in, shot, header, tackle, block); el "0 a 9" del paper es una errata.
- Licencia: paper CVIU en acceso abierto CC BY 4.0 (Crossref); README "Dataset annotations & baselines released under CC BY-NC 4.0." y "Redistribution of SoccerNet broadcast videos is prohibited under the NDA."; fichero LICENSE del repo Apache-2.0 (discrepancia sin resolver); HF `license: other` enlazando el NDA de SoccerNet (formulario con nombre, afiliación y planes de uso, aprobación manual, NDA personal por correo).
- Solape: ninguno (temporada 2023/24; nuestros EFL son 2019/20 y SoccerNet-v2 2014-17).
- Veredicto: útil en el futuro, cuando se ataque la atribución de jugador. Funcionalidad: eventos con equipo y jugador en `main`; tracklets para supervisar el tracking. Es el único dataset de SoccerNet con jugador ejecutor sobre partidos completos. Nuestra cadena actual (T-DEED para la clase + sn-teamspotting para el equipo + dorsal por tracklet) se podría medir con su `evaluation.py` sobre los 3 partidos de validación (`docs/comparativa_benchmark.md` §7.7).
- Para usarlo: cuenta HF + NDA, 28,5 GB (352x640 + tácticos) o 236 GB (Full HD), baselines TAAD / TAAD+GNN / TAAD+DST públicos sin pesos; Codabench 11232 para el challenge.

**Spiideo SoccerNet SynLoc (2025, Spiideo + SoccerNet; VISAPP 2025 y reto 2026)**, [github.com/Spiideo/sskit](https://github.com/Spiideo/sskit), resultados en [arXiv 2607.07320](https://arxiv.org/html/2607.07320v1).
- Qué contiene: imágenes 4K (y versión fullhd) de 17 instalaciones reales con 2 cámaras fijas por campo (cada una cubre medio campo), con atletas 3D sintéticos (BEDLAM, 683 HDRI) renderizados sobre fondos reales; train 42.504, validación 6.777, test 9.309, challenge 11.352 imágenes; 1.107.009 humanos anotados. COCO con caja, keypoints 2D/3D de pelvis y punto de apoyo, `position_on_pitch` en metros, articulaciones SMPL-X y calibración por imagen (`camera_matrix`, `dist_poly`). Sin balón, sin eventos, sin vídeo continuo.
- Métrica: mAP-LocSim (similitud por distancia real) y frame accuracy; baseline 77,30 / 33,74; ganador 2026 (SELab, YOLO26-L con tiling + RTMPose-X) 97,67 / 81,91. El paper mide 79,3 mAP-LocSim con punto de apoyo frente a 52,4 con el centro inferior de la caja.
- Licencia: no consta en sskit, en el paper ni en la API de GitHub; descarga previo registro en research.spiideo.com o `task="SpiideoSynLoc"` del paquete SoccerNet.
- Solape: ninguno, pero es el dominio de la rama panorámica (cámara fija 4K, jugadores lejanos).
- Veredicto: útil en el futuro, alto potencial. Rama panorámica: preentrenar el detector de jugadores pequeños, cabeza de "punto de apoyo" para la proyección a campo, validar la homografía en metros.
- Para usarlo: pedir las condiciones a Spiideo por escrito antes de descargar.

**SoccerSynth-Detection y SoccerSynth-Field (2025, Qin, Yeung, Umemoto, Fujii, Nagoya)**, [github.com/open-starlab/SoccerSynth-Detection](https://github.com/open-starlab/SoccerSynth-Detection), [arXiv 2501.09281](https://arxiv.org/html/2501.09281v1), [arXiv 2503.13969](https://arxiv.org/html/2503.13969v1).
- Qué contiene: Detection: 37.621 imágenes nítidas 1920x1080 y 17.052 con desenfoque 800x600 generadas en Unreal Engine con una sola cámara central tipo TV; clases player y soccer ball en formato YOLO; generador (solo Windows) y datos en Google Drive. Field: 20.000 imágenes 1920x1080 con líneas, landmarks y keypoints; sin enlace de descarga a 2026-11-01.
- Resultados: YOLOv8n entrenado solo con SoccerSynth da AP50 89,4 en SoccerNet-Tracking (82,8 entrenando con SportsMOT); preentrenar con sintético mejora con 1/8 de datos reales.
- Licencia: Detection "Apache License Version 2.0" (LICENSE.md); Field no consta.
- Veredicto: útil en el futuro para la rama TV (detector de balón y jugadores) y como semilla de un sintético de cámara fija si se reconfigura el generador. Funcionalidad: detector de balón TV (`yolo11s_snt_ball`).

**MultiSports (2021, MCG-NJU, ICCV 2021)**, [github.com/MCG-NJU/MultiSports](https://github.com/MCG-NJU/MultiSports), HF `MCG-NJU/MultiSports` (65,1 GB, gated).
- Qué contiene: 800 clips de fútbol de YouTube (competiciones profesionales, re-escalados a 720p, anotación a 25 fps) con tubos espacio-temporales de 15 clases: Shoot, Long Ball, Pass, Through Ball, Cross, Dribble, Trap, Throw, Save, Interception, Tackle, Clearance, Block, Defence, Aerial duels; 12.254 instancias y ~225k cajas; 0,7 s por acción, 22,6 s por clip; anotaciones de test retenidas. Sin equipo ni identidad.
- Licencia: "MultiSports Dataset License: CC BY_NC 4.0".
- Veredicto: útil en el futuro. Funcionalidad: tiros localizados en caja para afinar el detector de tiros y para enseñar al VLM "quién chuta"; solo I+D.

**SoccerReplay-1988 / UniSoccer / MatchVision (2024-2025, SJTU, CVPR 2025)**, [huggingface.co/datasets/Homie0609/SoccerReplay-1988](https://huggingface.co/datasets/Homie0609/SoccerReplay-1988), [arXiv 2412.01820](https://arxiv.org/html/2412.01820).
- Qué contiene: 1.988 partidos completos (Premier League, LaLiga, Bundesliga, Serie A, Ligue 1, UCL; 2014-15 a 2023-24), 3.323 h, vídeo "collected from the Internet" a 360p-720p y 25-30 fps; ~150K eventos en 24 clases (corner, goal, own goal, penalty, red card, yellow card, substitution, throw in, free kick, saved by goal-keeper, shot off target, clearance, off-side, var, foul...) con marca al segundo derivada de comentarios de flashscore alineados por modelo; ~150K comentarios; nombres anonimizados ([PLAYER], [TEAM]). Splits 1.488 / 250 / 250.
- Resultados: MatchVision top-1 84,0 % en SoccerNet (SigLIP 50,2 %) y 68,03 % en SoccerReplay-test.
- Licencia: cc-by-nc-sa-4.0 + formulario NDA (revisión de 2-3 semanas); "these data WON'T BE USED FOR COMMERCIAL!!".
- Solape: posible con nuestras temporadas 2014-17 (no verificado).
- Veredicto: útil en el futuro para I+D (preentrenar y evaluar el spotter y el VLM de eventos a escala); nulo para producto sin negociar licencia. Precisión de segundos, no de frame; sin equipo ni jugador.

**GOAL (2022, Suglia et al., Heriot-Watt) y MOMENTS (2026, Universiteit van Amsterdam)**, [arXiv 2211.04534](https://arxiv.org/abs/2211.04534), [github.com/akskuchi/MOMENTS](https://github.com/akskuchi/MOMENTS), [arXiv 2601.16333](https://arxiv.org/html/2601.16333v3).
- Qué contiene: GOAL: 1.107 vídeos de resúmenes de YouTube (2018-2020, 88 ligas; Serie A 638, Premier League 167), 73 h, 3,96 min de media, comentario en inglés transcrito por profesionales, base de conocimiento de flashscore; acceso por NDA, sin licencia ("fair-use"). MOMENTS: 3.954 fragmentos (1.977 "important" por aparecer en esos resúmenes / 1.977 no importantes) de 100 pares resumen-partido, con frames, audio del comentario (`*_v2.wav`) y transcripción; los vídeos exigen acceso a SoccerReplay-1988 y escribir a los autores; sin licencia declarada.
- Resultados: Gemini-2.5-Flash audio + vídeo MCC 0,598 / acc. 0,795; mejor abierto Qwen2.5-Omni-7B MCC 0,502.
- Veredicto: útil en el futuro. Funcionalidad: verdad externa de "qué entra en el resumen" con audio y comentario; validar gol por audio+vídeo en TV. Acceso con dos NDA; el procedimiento (alinear resúmenes de YouTube con partidos completos) es reproducible por nosotros.

**SoccerBench / SoccerAgent (2025, SJTU, ACM MM 2025) y otros benchmarks de VLM**, [huggingface.co/datasets/Homie0609/SoccerBench](https://huggingface.co/datasets/Homie0609/SoccerBench), [arXiv 2505.03735](https://arxiv.org/html/2505.03735v1).
- SoccerBench: 13 tareas y ~9.500 preguntas de texto, imagen y vídeo (clasificación de acción 1.000, replay grounding 400, marcador y tiempo 600, dorsal 200, color de camiseta 700, faltas multivista 300...); materiales incluidos (15,1 GB); cc-by-nc-sa-4.0. Resultados en vídeo: SoccerAgent 60,9 %, GPT-4o 57,5 %, Qwen2.5-VL-7B 46,8 %, Gemini 2.5 Pro 66,2 %. Base del reto SoccerNet VQA 2026.
- Sports-QA: 1.030 vídeos de fútbol (de MultiSports) y 10.486 QA; HF `HopLeeTop/Sports-QA` etiqueta "mit" pero deriva de MultiSports (CC BY-NC): licencia efectiva dudosa. SPORTU (ICLR 2025): 1.701 clips a cámara lenta en 7 deportes, 12.048 QA, Google Drive sin licencia. SportR (ICLR 2026): 2.052 vídeos, 50 tipos de infracción. RefereeBench (2026): 925 vídeos y 6.475 QA en 11 deportes, vídeos CC BY-NC de YouTube, mejor modelo 61,1 %; sin enlace de descarga.
- Veredicto: útil en el futuro. Funcionalidad: medir el VLM Qwen3.5 de highlights (acción, marcador, replay) y calibrar faltas por VLM; solo investigación.

**SJN-210k (2018, Li, Xu, Liu, Li, Wang; CVPRW 2018)**, [zenodo.org/records/14247819](https://zenodo.org/records/14247819).
- Qué contiene: "nearly 210,000 jersey number images for soccer games"; train 8.952 positivos con eje + 6.004 sin eje + 130.890 negativos; test 3.794 + 2.660 + 55.795; 442,9 MB. Resolución y origen no constan.
- Licencia: "Creative Commons Attribution Share Alike 4.0 International".
- Veredicto: útil en el futuro. Funcionalidad: lector de dorsales en `main` (TV); en panorámica los dorsales no son legibles a esa distancia. ShareAlike obliga a compartir igual las obras derivadas; los pesos son zona gris.

**Kinetics-700 (clases de fútbol) y AudioSet**, [label_map_k700.txt](https://raw.githubusercontent.com/open-mmlab/mmaction2/main/tools/data/kinetics/label_map_k700.txt), [AudioSet Cheering](https://research.google.com/audioset/dataset/cheering.html).
- Qué contiene: Kinetics-700 tiene cuatro clases de fútbol ("juggling soccer ball", "kicking soccer ball", "passing soccer ball", "shooting goal (soccer)"; Kinetics-400 sin "passing") con al menos 700 clips de ~10 s por clase, mezcla de TV, entrenamientos y vídeo amateur de YouTube; sin marca del golpeo ni equipo. AudioSet: "Cheering" 4.380 vídeos (12,2 h), "Crowd" 10.403 vídeos (28,8 h).
- Licencia: no declarada en el paper de Kinetics (vídeos de YouTube); AudioSet CC BY 4.0 (etiquetas; no verificado aquí).
- Veredicto: útil en el futuro. Funcionalidad: "shooting goal (soccer)" como única fuente grande de tiros en vídeo no televisivo para un clasificador de clip en juveniles; AudioSet para preentrenar un detector de "explosión de público" para gol sin marcador. UCF101/THUMOS (SoccerPenalty, 114 instancias), Sports-1M y YouTube-8M solo valen para preentrenar backbones.

**ISSIA-CNR Soccer (2009, D'Orazio et al., CNR) e ISSIA-3D (2025)**, descarga vía README de [FootAndBall](https://github.com/jac99/FootAndBall), [arXiv 2504.10106](https://arxiv.org/html/2504.10106v1).
- Qué contiene: seis cámaras fijas Full HD (1920x1080, 25 fps, tres por banda) sincronizadas en un partido de Serie A, unos 2 minutos (~20.000 frames); cajas de jugadores (63-144 px de alto) y centro del balón. ISSIA-3D añade líneas en formato SoccerNet-v3, calibración PnLCalib de las seis cámaras, balón 3D por triangulación y 10.544 frames (train cámaras 3-6: 8.686; test cámaras 1-2: 1.858).
- Resultados: AP de balón en ISSIA: FootAndBall 0,909, DeepBall 0,877, Faster R-CNN 0,866; YOLO afinado en ISSIA-3D AP@0.5 0,65 y MAE 4,2 m.
- Licencia: no consta (ni en el original ni en ISSIA-3D).
- Veredicto: útil en el futuro. Rama panorámica: único benchmark clásico de AP de balón desde cámara fija Full HD; con ISSIA-3D permite evaluar nuestro PnLCalib y el detector de balón en cámara fija profesional.

**Alfheim dataset (2014, Simula / UiT, Tromsø IL)**, [datasets.simula.no/alfheim](https://datasets.simula.no/alfheim/).
- Qué contiene: 3 partidos de noviembre de 2013 (Strømsgodset, Anzhi, Tottenham ~40 min) con posiciones de jugadores a 20 Hz por sensor ZXY y vídeo de dos arrays de cámaras fijas en el centro del campo (tres cámaras individuales o panorámica cosida; cinco cámaras 2K en el tercer partido); posición del balón en píxeles solo en el tercero; vídeo H.264 en segmentos de 3 s; "slightly above 100 GB". Las matrices de transformación se perdieron salvo un ejemplo. Sigue en línea (modificado 2026-10-29).
- Licencia (literal): "The data may only be used for non-commercial research purposes." y prohibición de reidentificar jugadores o crear perfiles de rendimiento.
- Veredicto: útil en el futuro. Rama panorámica: banco de pruebas de homografía y tracking con verdad de posición por sensor sobre panorámica cosida; sin uso comercial; habría que calibrar a mano.

**Fu et al. 2023, dataset multicámara fijo (Zhejiang / PolyU, Applied Sciences 13(9):5361)**, [PDF](https://ira.lib.polyu.edu.hk/bitstream/10397/105399/1/applsci-13-05361.pdf).
- Qué contiene: 17 clips de 120 s a 20 fps más 1.100+ min de vídeo multicámara en cuatro campos de fútbol 5, 7, 8 y 11 ("real competitions", 34 equipos); 5-7 cámaras fijas por campo, con cámara principal cosida de hasta 5950x1450 (fútbol 11) y 5120x1400 (fútbol 8); 316 identidades; 727.179 anotaciones a 20 fps + 137.846 a 1 fps; calibración conjunta por bundle adjustment sobre marcas del campo. 13 clips train / 4 test.
- Resultados: fútbol 11 HOTA 64,89 (MVFlow); fútbol 7 HOTA 77,52.
- Licencia: artículo CC BY; dataset "will be available at https://www.true-think.com" (dominio no verificado); licencia del dataset no consta.
- Veredicto: útil en el futuro si sigue disponible. Rama panorámica: la cámara cosida se parece a nuestra 4096x1080 y cubre fútbol amateur de 5/7/8/11 con calibración métrica.

**IDSSE, Sportec Solutions / DFL (2025, Bassek, Rein, Weber, Memmert; Scientific Data)**, [PMC11787359](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/), figshare 10.6084/m9.figshare.28196177.
- Qué contiene: 7 partidos de Bundesliga y 2. Bundesliga 2022/23 (Köln-Bayern, Bochum-Leverkusen y 5 del Fortuna Düsseldorf) con tracking oficial TRACAB a 25 Hz (1.002.644 frames; X/Y en metros, velocidad, aceleración; balón con altura, posesión y estado) y 11.137 eventos oficiales de la DFL sincronizados (Needleman-Wunsch, desfase medio −0,37 ± 1,82 s), con xG en tiros. Tres XML por partido, 2,63 GB. Sin vídeo "due to licensing restrictions".
- Licencia: "CC-BY 4.0" con autorización de la DFL.
- Herramientas: floodlight (`IDSSEDataset`), kloppy (`sportec.load_open_*`), databallpy (`get_open_game()`).
- Veredicto: útil en el futuro como referencia "gold". Funcionalidad: rangos de sanidad para posesión, velocidades, distancias y mapas de calor; proyecciones sintéticas de cámara única para probar homografía y tracking; pseudoetiquetas evento-trayectoria.

**Metrica Sports sample data (2020-2021)**, [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data).
- Qué contiene: 3 partidos anonimizados con tracking (25 fps según databallpy) y eventos sincronizados; dos en CSV propio y el tercero en formato EPTS FIFA + JSON; campo 105x68 m, coordenadas 0-1. Sin vídeo.
- Licencia: sin licencia formal: "Please be responsible with the use of this data." / "If you use it for anything public, please aknowledge the source."
- Veredicto: útil en el futuro. Funcionalidad: banco de pruebas estándar (kloppy, databallpy) para fórmulas de posesión y sincronización evento-trayectoria; adoptar el formato EPTS FIFA como exportación de nuestro tracking facilitaría comparaciones.

**PFF FC / Gradient Sports Open Data, Mundial 2022 (y espejos OpenSportsLab/PFF y SoccerNet-GAR)**, [PFF FC blog](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data), [OpenSportsLab/PFF](https://huggingface.co/datasets/OpenSportsLab/PFF), [SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR).
- Qué contiene: tracking derivado de la señal de TV (~30 fps) + eventos + grades de los 64 partidos del Mundial 2022, en JSONL/JSON previo formulario; sin vídeo y sin texto de licencia publicado. Espejo 1: `OpenSportsLab/PFF`, vídeo 720p en formato OSL Action Spotting, 326 GB, etiqueta GPL-3.0, acceso condicionado. Espejo 2: SoccerNet-GAR, clips 720p de ~4,8 s sincronizados con tracking, 87.939 actividades de grupo en 10 clases (nombres no publicados), splits 45/9/10 partidos, gated con NDA; base del SoccerNet Challenge 2027 (baseline 78,64 % con posiciones).
- Veredicto: útil en el futuro. Funcionalidad: único tracking abierto derivado de TV a escala para estudiar cobertura y ruido frente a nuestro pipeline; con los espejos, vídeo + tracking + eventos para validar homografía y tracking. StatsBomb Open Data tiene los mismos 64 partidos (con 360): dos anotadores sobre el mismo partido.

**SoccerTrack v1 (2022, Scott et al., CVPRW 2022)**, [kaggle.com/datasets/atomscott/soccertrack](https://www.kaggle.com/datasets/atomscott/soccertrack).
- Qué contiene: ojo de pez 8K (7680x4320) y dron 4K a 30 fps de partidos universitarios de Tsukuba; 20 clips de 30 s anotados (10 min; 15 train / 5 test) más "extra footage with complete annotations" (2023-02-08); cajas de jugadores y balón, IDs, GNSS a 10 Hz, 65 keypoints de campo; 6,8 GB.
- Licencia: Kaggle "GNU Lesser General Public License 3.0" (atípica para datos).
- Veredicto: útil en el futuro; superado por TeamTrack (misma captura, anotación completa). Rama panorámica.

**SportsMOT (2023, MCG-NJU, ICCV 2023)**, [github.com/MCG-NJU/SportsMOT](https://github.com/MCG-NJU/SportsMOT).
- Qué contiene: 240 vídeos (45 / 45 / 150) de baloncesto, voleibol y fútbol (≈80 clips de fútbol de Olympics, NCAA y Premier League), 720p, 25 fps, 150.379 frames, 1.629.490 cajas; solo jugadores en el campo (sin balón ni árbitros); formato MOT17; espejo de 35,1 GB.
- Resultados: MixSort-OC HOTA 74,1; fútbol HOTA 66,4.
- Licencia: "Creative Commons Attribution-NonCommercial 4.0 International License" y "You agree not to distribute the SportsMOT dataset without prior written permission".
- Veredicto: útil en el futuro para evaluar o afinar el tracking de jugadores de la rama TV, solo I+D.

**SoccerNet-v3 (2022, Scientific Data)**, [PMC9210334](https://pmc.ncbi.nlm.nih.gov/articles/PMC9210334/), repo [SoccerNet-v3](https://github.com/SoccerNet/SoccerNet-v3).
- Qué contiene: 33.986 frames (12.764 de acción en directo + 21.222 de repetición) de 400 partidos de v2 con 1.324.732 anotaciones: 344.660 cajas de personas (7 clases), 26.939 cajas de objetos (balón ~7,1 % de todas las cajas, bandera, tarjeta), 167.589 líneas, 53.577 partes de portería, 172.622 correspondencias de re-identificación y 106.592 dorsales. Sin vídeo (PNG).
- Licencia (literal): "will be downloadable freely, with no restriction whatsoever (open licence), as per the fair use principle".
- Veredicto: útil en el futuro. Funcionalidad: cajas de balón y dorsales a 720p de nuestros propios partidos; líneas para calibración TV.

**Otras capas de SoccerNet no descargadas**: 12 vídeos completos de Tracking (~18 h de cámara principal 1080p, gated RAW) y 58 + 37 clips no publicados; SN-Jersey-2023 (2.853 tracklets + 1.211 de challenge, clases 1-99 y −1, público; mejor 92,85 % de acierto); SN-ReID-2023 (340.993 miniaturas, público); SN-MVFouls-2024/2025 (3.901 acciones con 2-4 vistas a 224p/720p, 8 tipos de falta y 4 severidades; GPL-3.0 público en HF aunque el README hable de NDA); SoccerNet-XFoul (22k tripletas vídeo-pregunta-respuesta). Veredicto: útil en el futuro para dorsales (TV), re-id y faltas; ninguna anota equipo ni jugador de la falta.

**EIGD-S (2021, L3S/TIB Hannover y DSHS Köln; MMSports 2021)**, [github.com/MM4SPA/eigd](https://github.com/MM4SPA/eigd).
- Qué contiene: 5 partidos × 5 secuencias × 5 min (≈125 min) de vídeo de la FIFA en YouTube (1280x720, 25 fps; solo se distribuyen las URL); anotación frame a frame de posesión, Ball Reception/Release → Pass (Successful / Unsuccessful) y Shot (Successful / Unsuccessful: Goal Frame, Blocked, Off Target), eventos a balón parado y decisiones arbitrales; equipo solo en la posesión, sin jugador.
- Licencia: "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International".
- Veredicto: útil en el futuro como validación externa pequeña del spotter de pases y tiros con resultado.

**TACDEC (2024, SimulaMet / Eliteserien; MMSys 2024)**, [zenodo.org/records/10611979](https://zenodo.org/records/10611979).
- Qué contiene: 425 vídeos de la Eliteserien noruega con 836 anotaciones en 4 clases (tackle live / replay, completas e incompletas); `videos.zip` 6,9 GB.
- Licencia: "Creative Commons Attribution Non Commercial No Derivatives 4.0 International"; Simula: "Use of the dataset for competitions or commercial purposes requires prior written permission".
- Veredicto: útil en el futuro para una clase de entrada/falta y para detectar repeticiones en TV.

**Imágenes de eventos: "A comprehensive dataset of soccer event images" (2025, Data in Brief) e IAUFD (2022)**, [PMC11999477](https://pmc.ncbi.nlm.nih.gov/articles/PMC11999477/).
- Qué contiene: 38.728 imágenes JPG 640x360 de 45-50 vídeos de YouTube en 10 clases (Goal 10.922, Player celebration 10.219, Plenty stock 3.307, Free kick 3.235, Goal attempt 3.055, Corner 2.768, Red card 2.000, Offside 1.491, Yellow card 890, Spectator 841) más una parte de tipos de plano; en Kaggle "Soccer View and Event Score" (dsv/7700429); VGG16 acc. 0,921. IAUFD: ~100.000 imágenes en 10 categorías (goal, celebration, red card, yellow card, penalty-kick...), sin enlace de descarga.
- Licencia: artículo CC BY-NC; licencia del dataset no declarada (no verificado).
- Veredicto: útil en el futuro. Funcionalidad: clasificador de frame "celebración / gol / tarjeta" como apoyo a la detección de gol sin marcador (hoy el ratio de césped separa plano general de celebración).

**xGHub (2026, Gutiérrez-Pérez, Yeung, Fujii, Agudo; CVIU 268:104732)**.
- Qué anuncia: "an open-source dataset of football shots enriched with player pose estimation, body orientation" y contexto, extraído de vídeo de TV; artículo CC BY 4.0.
- Estado: repositorio, número de tiros y licencia del dataset no localizados (ScienceDirect, UPF e IRI devolvieron 403). 3DSP (2024, Apache-2.0) es el precedente pequeño: 210 tiros de SoccerNet con pose y etiqueta on/off target.
- Veredicto: útil en el futuro si aparece el repositorio; escribir a los autores (IRI CSIC-UPC y Nagoya).

**MatchTime, SoccerChat y SoccerLens (2024-2026, sobre vídeos de SoccerNet)**, [HF MatchTime](https://huggingface.co/datasets/Homie0609/MatchTime), [HF SoccerChat](https://huggingface.co/datasets/SimulaMet/SoccerChat), [SoccerLensDataset](https://github.com/IsmaelElsharkawi/SoccerLensDataset).
- Qué contienen: MatchTime: comentarios de los 471 partidos de Caption realineados (`event_aligned_gameTime`, `contrastive_aligned_gameTime`; 49 partidos alineados a mano), 78,1 MB, cc-by-sa-4.0. SoccerChat: 85.220 ejemplos train / 4.625 val de clips ≤10 s con pregunta, respuesta y las 17 clases de v2; anotaciones MIT, vídeo bajo NDA. SoccerLens: 200 clips de 30 s a 1 fps con 4.687 cajas de "pistas" de 13 eventos; CC BY-NC 4.0; ningún VLM supera el 50 % de grounding.
- Veredicto: útil en el futuro como protocolos de evaluación del VLM sobre vídeo que ya tenemos.

**WC14, TS-WorldCup y WorldPose (calibración TV)**, [nhoma.github.io](https://nhoma.github.io/), [KpSFR](https://github.com/ericsujw/KpSFR), [PnLCalib](https://arxiv.org/html/2404.08401v3).
- Qué contienen: WC14: 209 imágenes train / 186 test con homografías manuales; TS-WorldCup: 3.812 imágenes de 43 vídeos de los Mundiales 2014 y 2018 con marcas de campo; WorldPose: 88 secuencias del Mundial 2022 con >2,5 M poses 3D y parámetros de cámara, vídeo de FIFA solo académico bajo solicitud. Licencias de WC14 y TS-WorldCup no constan.
- Referencia: PnLCalib (el que usamos) da error de proyección 0,60 m en WC14 y 0,23 m en TSWC.
- Veredicto: útil en el futuro solo para la rama TV, para comparar nuestro PnLCalib con los números de referencia.

**Impect open data (Bundesliga 2023/24)**, [github.com/ImpectAPI/open-data](https://github.com/ImpectAPI/open-data). Eventos con KPIs por evento y jugador, alineaciones; `kloppy.impect.load_open_data`; licencia en `LICENSE.pdf` ("By using this data you agree to our terms and conditions", logo obligatorio); número de partidos y precisión temporal no leídos. Veredicto: útil en el futuro solo como prior de élite y para medir discrepancias entre proveedores con StatsBomb 2023/24.

| Otros de valor menor o condicionado | Qué es | Licencia | Nota |
|---|---|---|---|
| SoccerNet Ball Action Anticipation (SN-BAA 2026) | predecir las acciones de los 5 s siguientes sobre los 9 partidos de BAS; 10 clases | gated NDA; código FAANTRA GPL-3.0 | mismos vídeos; relevancia baja |
| SoccerNet Camera Calibration 2022-2023 | 25.506 imágenes con 226.305 polilíneas | público, sin etiqueta | PnLCalib ya se entrenó con él |
| SN-VQA-2026 / 2027 | 500 + 500 preguntas de 4 opciones; 2027 en streaming (gated) | cc-by-sa-4.0 (2026) | relevancia baja |
| SoccerNet-Depth, NVS, BannerReplacement | sintéticos de videojuego, síntesis de vistas, publicidad | GPL-3.0 / sin etiqueta | sin uso para nosotros |
| Soccer Player Detection (Lu et al., BMVC 2017) | 2.019 imágenes, 22.586 jugadores, 3 cámaras, 1280x720 | no consta | sin balón |
| Huang et al. 2022 (Applied Sciences 12:7473) | 17 vídeos de 2 min de partidos no profesionales con cámaras fijas, solo tracking (736.800 cajas según el buscador) | artículo CC BY; descarga no localizada | lo más parecido a cámara fija amateur |
| Wyscout 2017/18 (Pappalardo et al.) | 1.941 partidos, 3.251.294 eventos al segundo | CC BY 4.0 | sin solape; priors (ver 4C) |
| `infactory-ai/soccer-events` (HF, 2026) | 211 clips de Serie A 2024-25 (goal 100, yellow_card 100, red_card 11), 1280x720 a 15 fps | "proprietary… non-commercial research purposes" | pequeño |

### 4C. Descartar o de poco valor para nosotros

**DFL - Bundesliga Data Shootout (Kaggle, 2022)**. Ficha en 2.1. Razones: no descargable ("DATA IS NO LONGER AVAILABLE FOR THIS COMPLETED COMPETITION"); licencia "Competition-Use Only"; sin tiros, goles, córners, equipo ni jugador; TV con realización. Cualquier copia en circulación (mirrors de HF, Roboflow) está fuera de los términos. Conservar solo la metodología (tolerancias 0,15-0,7 s, frames en gris apilados, decodificar frame a frame).

**DeepSportradar para fútbol**. Ficha en 2.4. Es baloncesto y críquet; sin tracking, sin fútbol. Para la rama basketball ver sección 5.

**SkillCorner Open Data**. Ficha en 2.5. Sin vídeo, sin solape (A-League 2024/25; los 9 partidos 2019/20 borrados) y vídeo de la A-League no obtenible legalmente. Guardar solo como referencia de formato (`is_detected`, polígono visible, Dynamic Events).

**Wyscout, "Soccer match event dataset" (Pappalardo et al., Scientific Data 2019)**, [figshare 4415000](https://figshare.com/collections/Soccer_match_event_dataset/4415000), [PMC6817871](https://pmc.ncbi.nlm.nih.gov/articles/PMC6817871).
- Qué contiene: 1.941 partidos y 3.251.294 eventos de las 5 grandes ligas 2017/18, Mundial 2018 y Euro 2016; 7 tipos y subtipos con tags; `eventSec` al segundo; coordenadas 0-100; `events.zip` 77 MB. Sin vídeo.
- Licencia: CC BY 4.0.
- Por qué descartar para un proyecto de vídeo: ningún partido coincide con nuestros vídeos (terminan en mayo de 2017). Valor residual: priors de highlights (qué precede a un tiro, tasas base por partido) y, con StatsBomb, dos anotadores sobre el Mundial 2018 para fijar tolerancias.

**Last Row (Friends of Tracking, 2020)**, [github.com/Friends-of-Tracking-Data-FoTD/Last-Row](https://github.com/Friends-of-Tracking-Data-FoTD/Last-Row). 19 goles del Liverpool 2019 a 20 fps, hechos para animaciones; el README dice que "lacks the accuracy needed for research"; sin eventos, sin vídeo, sin licencia formal ("Feel free to use the dataset for whatever purpose, but please credit the source"). Descartar.

**Signality (Hammarby IF)**. 3 partidos con tracking y vídeo no sincronizado, sin eventos, usados en el curso Soccermatics; ninguna URL pública de descarga ni licencia; Signality fue adquirida por Spiideo. Descartar mientras no haya acceso.

**Derivados Roboflow del DFL Shootout y familia `roboflow/sports`**, [github.com/roboflow/sports](https://github.com/roboflow/sports), [Roboflow blog](https://roboflow.com/blog/track-football-players). `football-players-detection-3zvbc` (frames de "a few dozen" clips del Shootout, 10 por vídeo, clases player/goalkeeper/referee/ball), `football-ball-detection-rejhg`, `football-field-detection-f07vi`; Kaggle `football-soccer-ball-detection-dfl` (76 MB, licencia "Unknown"). Las páginas de Universe devuelven HTTP 403, así que la licencia declarada no se pudo leer; aunque digan CC BY 4.0, el permiso real depende de las condiciones del Shootout. Descartar para producto; a lo sumo pruebas internas.

**SoccerDB (2020), Comprehensive Soccer (2018) y SSET (2020)**, [github.com/newsdata/SoccerDB](https://github.com/newsdata/SoccerDB), [arXiv 1912.04465](https://ar5iv.labs.arxiv.org/html/1912.04465). SoccerDB: 346 partidos (270 de SoccerNet), 668,6 h, 10 clases (Shot 14.358, Goal 2.559, Corner 3.275, Foul 5.276, Saves 5.467...), 702.096 cajas sin IDs, 17.115 highlights "playback"; los vídeos no SoccerNet bajo una NDA "similar to the SoccerNet protocol"; el repo no muestra enlaces de descarga. SSET: 350 vídeos, 282 h, 11 eventos; la página de HUST devolvía 404. Cubrirían nuestras clases, pero a 2026-11-01 no hay vía de descarga comprobable ni licencia. Descartar salvo contacto con los autores.

**Otros de poco valor**: Footovision STAD (~20.000 clips de 997 partidos con jugador ejecutor; propietario de facto); datasets de penaltis de ULPGC (1.010 clips) y Ranasinghe (755), sin descarga pública; Kaggle "football match actions video dataset" y "FOOTBALL-SOCCER-VIDEOS-DATASET" (licencia desconocida, páginas no legibles); Sports-1M y YouTube-8M (etiquetas a nivel de vídeo completo; YouTube-8M solo features); SVW (vídeos de usuarios de Coach's Eye, cobertura de fútbol no verificada); Gerats et al. (TNO, 4 partidos de panorámica fija con 3.717 eventos, no publicado); KTH Multiview Football (pose), Open Soccer Ball Dataset (RoboCup), dataset térmico de AAU, hockey jersey (bajo petición), SportsHHI (sin fútbol); el reto MMSys'22 de la Eliteserien (datos no públicos); VisAudSoccer (460 partidos, no disponible).

### 4D. Lo que no existe (a 2026-11-01)

| Carencia del proyecto | Resultado de la búsqueda | Lo más cercano |
|---|---|---|
| Vídeo de fútbol amateur o juvenil con una cámara sin realización y eventos anotados | Nada público. Búsquedas: "amateur grassroots youth soccer video dataset single camera events annotated", "youth football video dataset annotated computer vision", "amateur football video dataset action spotting non-broadcast lower league semi-professional", "Veo OR Pixellot OR Spiideo OR Trace football video dataset research public release" | SoccerTrack v2 (universitario, panorámica fija, ya en disco); Fu et al. 2023 (fútbol 5/7/8/11, solo tracking); Huang et al. 2022 (17 × 2 min, solo tracking); Kinetics "shooting goal (soccer)" como fuente débil de tiros no televisivos |
| Audio de estadio o de comentario etiquetado por gol | Nada fuera de SoccerNet-Echoes. Búsquedas: "soccer broadcast audio dataset goal detection crowd cheering commentary", "football stadium sound event dataset whistle crowd cheer goal audio classification" | MOMENTS (audio + transcripción, gated), GOAL Heriot-Watt (NDA), AudioSet Cheering/Crowd (preentrenamiento) |
| Calibración o keypoints de cámara fija amateur | Nada en Roboflow, HF ni arXiv | 65 keypoints por partido de SoccerTrack v2, SynLoc (34 cámaras reales 4K, licencia por pedir), ISSIA-3D, Fu et al. 2023 |
| Vídeo de celebraciones de gol | Ningún dataset dedicado. Búsquedas: "goal celebration recognition soccer video dataset clips", "kaggle dataset football highlights video clips goal celebration classification" | clases de imagen "celebration" (IAUFD, Data in Brief 2025); clase "celebrate" de VisAudSoccer (no público) |
| Tiros con resultado (gol / parado / fuera) a escala fuera de SoccerNet | Nada descargable. Búsquedas: "expected goals from broadcast video dataset shots annotated", "xGHub football dataset" | EIGD-S (125 min), 3DSP (210 tiros), xGHub (anunciado, repositorio no localizado), Understat/Kaggle events (al minuto, sin vídeo) |
| Alineación publicada SoccerNet × StatsBomb/Wyscout | Ningún dataset ni paper | Hay que hacerla en casa (receta de SoccerNet-v1: reloj en pantalla + mapeo lineal + refinamiento) |
| Datasets públicos de Veo, Pixellot, Hudl Focus, Trace | Ninguno | Spiideo solo vía SynLoc |
| Dron de fútbol posterior a 2024 | Ninguno (ni en Zenodo) | SoccerTrack v1 / TeamTrack (DJI Mavic 3, 30 min) |
| Nombres buscados que no existen como dataset | "SoccerSynth-Tracking", "SynthSoccer", "SoccerNet-Synthetic", "SoccerNet-v2 MLV" (fue el nombre de un equipo de MVFoul 2024), "SN-Calib 2025", "FootballDB", "Soccer-Caption", "Basketball-51", "HoopTrack", "BasketballGAR" | SoccerSynth-Detection/-Field, SoccerNet-Caption, SoccerDB, NBA dataset de Yan et al. |
| Re-identificación o "SVHN de dorsales" de fútbol fuera de SoccerNet | Ninguno | SJN-210k (dorsales), SN-ReID-2023 |

## 5. Baloncesto

DeepSportradar (ficha en 2.4) es el único conjunto público con máscaras de instancia, calibración de cancha y balón sobre la misma cámara fija lateral de pabellón (Keemotion, vista desde el centro de la pista), un encuadre parecido al de una cámara amateur fija. Su utilidad para `basketball/` es alta en calibración y segmentación de jugadores, media en balón (solo recortes con posición-oráculo) y nula en eventos (tiro, canasta) o marcador. Las licencias CC BY-NC-ND 4.0 y CC BY-NC-SA 4.0 impiden el uso comercial y la redistribución de derivados. Sus datos siguen descargables (Kaggle + GitHub); la organización no tiene actividad desde 2023.

| Dataset (año, enlace) | Qué contiene | Licencia | Para qué nos sirve |
|---|---|---|---|
| [TrackID3x3](https://github.com/open-starlab/TrackID3x3) (2025, Nagoya / open-starlab; MMSports 2025) | Baloncesto 3x3 a cancha completa con vídeo: subconjuntos Indoor y Outdoor (cámaras fijas) y Drone (127.119 frames a 3840x2160, no verificado en el PDF); cajas de los 6 jugadores en todos los frames, 10 keypoints de pose en algunos, identidades; Google Drive | "Creative Commons Attribution 4.0 International (CC BY 4.0), unless otherwise stated" (el módulo de dorsales es solo no comercial) | Alta: cámara fija interior y exterior con licencia permisiva; salvedad: 3x3 (6 jugadores) |
| [TeamTrack](https://atomscott.github.io/TeamTrack/), parte de baloncesto (2024) | 65,5 min y 1.296.900 cajas en tres vistas: ojo de pez 6K (Z CAM E2-F6, 17,5 min), dron lateral 4K (24 min) y dron cenital 4K (24 min); IDs persistentes; Kaggle + Google Drive | Kaggle "MIT"; la web del proyecto no la indica | Alta-media: cámara fija gran angular de partidos universitarios |
| [DeepSportradar / basketball-instants](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset) (2019-2023) | ~364 instants en 15 pabellones (LNB Pro A); 408 imágenes con máscaras; 728 vistas calibradas; 18.703 recortes de re-id; balón 3D con posición-oráculo | CC BY-NC-ND 4.0 / CC BY-NC-SA 4.0 | Alta en calibración y segmentación, solo investigación |
| [SportsMOT](https://github.com/MCG-NJU/SportsMOT), parte de baloncesto (2023) | Clips de retransmisión (Olympics, NCAA, NBA) a 720p y 25 fps; 845,4 frames y 9,1 jugadores por clip de media; solo jugadores | CC BY-NC 4.0, sin redistribución | Media: tracking en TV, cámara móvil |
| Roboflow Universe: `basketball-player-detection-3-ycjdo` (654 imágenes NBA, 10 clases), `basketball-court-detection-2` (keypoints), `persistence/basketball-hoop-player` (216 imágenes), `tomas-34ray/basketball-detection-cjjxp` | Pequeños, de origen NBA; los dos primeros ya los usa el pipeline como modelos hospedados | CC BY 4.0 (los cuatro, según `README.dataset.txt` y las fichas de Universe; páginas de Universe no legibles hoy: 403) | Media: aro y balón; licencias heterogéneas |
| Eventos en retransmisión NBA: [NBA dataset GAR](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530205.pdf) (181 partidos, 9.172 clips, 9 clases con resultado, "available upon request"), [FineSports](https://github.com/PKU-ICST-MIPL/FineSports_CVPR2024) (~10.000 vídeos, 16.000 instancias, 52 subtipos, Release Agreement por correo), [NSVA](https://github.com/jackwu502/NSVA) (>32K clips con subtítulos; CC BY-NC), [BARD](https://bodai.unibs.it/?p=1198) (60 partidos NBA, 14.676 clips a 720p; licencia sin confirmar) | Tiro, canasta, pase, rebote etiquetados a nivel de clip | bajo petición o no comercial | Media-baja: única fuente de eventos de tiro/canasta, pero TV y acceso restringido |
| [SpaceJam](https://github.com/simonefrancia/SpaceJam) (2018-2020, tesis Univ. Bolonia) | ~32.560 clips de 16 frames de un solo jugador + articulaciones 2D; 10 clases (Block, Pass, Run, Dribble, Shoot, Ball in Hand, Defence, Pick, No Action, Walk) | MIT (enlace en el repo) | Baja-media: clasificar "tiro" por pose del jugador recortado |

Lo que no existe en baloncesto: ningún dataset público con jugadores + balón + aro + cancha + eventos desde una cámara única amateur; los eventos de tiro y canasta solo están etiquetados sobre retransmisiones NBA/NCAA. Para la rama basketball la vía realista es TrackID3x3 + los sets de aro y balón de Roboflow para detección, DeepSportradar solo para investigación en calibración, y anotación propia o reglas (trayectoria balón-aro) para eventos. "Basketball-51", "HoopTrack" y "BasketballGAR" no existen como datasets; "NBA SportVU 2015-16" es tracking posicional sin vídeo.

## 6. Licencias y uso comercial

| Dataset | Licencia | ¿Permite producto comercial? | Nota |
|---|---|---|---|
| SoccerNet: vídeos (500 + 50 partidos, Tracking RAW, BAS 2023-2025, PCBAS-2026, GAR) | NDA con KAUST: "Use only for non-commercial research purposes and do not distribute to tiers" | No | Cláusula 8: no crea licencia alguna; vigencia 2 años; ley saudí |
| SoccerNet `Labels-v2.json` / `Labels-ball.json` (`SN-Labels`) | sin licencia declarada en HF | No verificado | sin licencia explícita no hay permiso de reutilización más allá del uso de investigación |
| SN-echoes | CC BY 4.0 | Sí (las transcripciones) | su valor depende de vídeos bajo NDA |
| SN-GSR-2024/2025, SN-Depth-2025, SN-MVFouls-2024/2025 | GPL-3.0 (etiqueta HF) | Ambiguo | licencia de software aplicada a datos; alcance sobre modelos sin resolver |
| SN-VQA-2026 | CC BY-SA 4.0 | Sí, con share-alike | |
| SoccerNet-v3 | "open licence", "no restriction whatsoever" | Sí (las anotaciones) | frames de vídeo bajo NDA |
| FOOTPASS / SN-PCBAS-2026 (anotaciones) | CC BY-NC 4.0 (README); LICENSE del repo Apache-2.0; paper CVIU CC BY 4.0 | No | vídeo bajo NDA, en HF a 352x640 y Full HD |
| SoccerTrack v2 | CC BY 4.0 | Sí, con atribución y cita | la única licencia explícita en disco; acceso HF restringido |
| TeamTrack | Kaggle "MIT" | En principio sí | confirmar por escrito con los autores |
| SoccerTrack v1 | Kaggle "GNU Lesser General Public License 3.0" | Atípica | licencia de software sobre datos |
| TrackID3x3 | CC BY 4.0 (salvo el módulo de dorsales) | Sí | baloncesto 3x3 |
| SoccerHigh | CC BY-NC-SA 4.0 | No | solo evaluación e I+D |
| StatsBomb Open Data | StatsBomb Public Data User Agreement | No ("commercially exploit the data or any analysis derived" prohibido) | sin redistribución; logo obligatorio; registro |
| SkillCorner Open Data | MIT | Sí | sin vídeo; inútil para visión |
| DFL - Bundesliga Data Shootout | "Competition-Use Only" (Kaggle §7.A) | No | ya no se distribuye |
| DeepSportradar | CC BY-NC-ND 4.0 / CC BY-NC-SA 4.0 | No | fuentes discrepan entre ND y SA |
| MultiSports, SportsMOT, NSVA | CC BY-NC 4.0 | No | SportsMOT además prohíbe redistribuir |
| SoccerReplay-1988, SoccerBench | CC BY-NC-SA 4.0 (+ NDA en SoccerReplay) | No | |
| MatchTime | CC BY-SA 4.0 | Sí (anotaciones) | sobre vídeo SoccerNet |
| SoccerChat (anotaciones) | MIT | Sí (anotaciones) | vídeo bajo NDA |
| SJN-210k | CC BY-SA 4.0 | Sí, con share-alike | pesos: zona gris |
| SoccerSynth-Detection | Apache-2.0 | Sí | sintético, cámara TV |
| Spiideo SynLoc | no consta | Desconocido | pedir condiciones a Spiideo |
| Alfheim | "The data may only be used for non-commercial research purposes." | No | prohibida la reidentificación |
| ISSIA-CNR, Fu et al. 2023, WC14, TS-WorldCup, Metrica | no consta / sin licencia formal | Desconocido | Metrica: "please aknowledge the source" |
| IDSSE (Sportec/DFL) | CC BY 4.0 con autorización de la DFL | Sí | sin vídeo |
| Wyscout 2017/18 | CC BY 4.0 | Sí | sin solape con nuestro vídeo |
| Transfermarkt datasets | CC0-1.0 | Sí | |
| openfootball | dominio público | Sí | solo resultados |
| Kaggle Football Events, European Soccer Database | no verificable (JS); ESD ODbL según espejos | Desconocido | texto de comentario con derechos de BBC/ESPN/onefootball |
| Understat, FBref, WhoScored | sin términos / propietario | No (scraping) | zona gris legal |
| PFF FC Open Data | términos no públicos (formulario) | Desconocido | espejo OpenSportsLab/PFF GPL-3.0 |
| EIGD-S | CC BY-NC-SA 4.0 | No | |
| TACDEC, SoccerSum | CC BY-NC-ND 4.0 | No | permiso escrito para competiciones o uso comercial |
| Roboflow baloncesto (los dos del pipeline) | CC BY 4.0 | Sí, con atribución | `dataset_fusionado` marcado "Public Domain" por error |
| Roboflow derivados del DFL Shootout | no verificable (403) | No recomendable | procedencia "Competition-Use Only" |
| T-DEED (código) y pesos `SNB_pano1`/`SNB_tv1` | GPL-3.0; pesos entrenados con vídeo bajo NDA | Decidir antes de distribuir | ver párrafo siguiente |

El NDA de SoccerNet es un acuerdo bilateral entre el solicitante ("Collaborator", con nombre y afiliación) y King Abdullah University of Science and Technology, vinculado desde todos los repositorios gated ([texto](https://docs.google.com/document/d/1Efz8yP-baa8CtcMB7SYIID7Bn5sH7sGTB4-u1YdVRf4/export?format=txt)). Su Exhibit A fija el uso permitido en una frase: "Use only for non-commercial research purposes and do not distribute to tiers" (sic, por third parties). La cláusula 3.c prohíbe revelar la información "to any third party except authorized representatives, consultants, or subcontractors engaged to assist in pursuing the Purposes"; la 6 obliga a devolver las copias a petición de KAUST; la 8 dice: "No license is created under this Agreement, nor shall any be implied therefrom under any patent, trademark, application, copyright, trade secret, or other intellectual property right of either Party, other than the use of Confidential Information for the Purposes"; la 10 fija la vigencia en dos años y la 11 la ley de Arabia Saudí. El formulario de Hugging Face añade tres casillas: "I understand that the use of this dataset is for research and non-commercial use only", "I understand that this dataset and the videos are protected by copyright" y "I agree not to share the dataset with anyone else". Los 7 partidos EFL de Ball Action Spotting tienen el mismo gating, el mismo `license: other` y el mismo NDA que el resto.

Zona gris: ni el NDA ni las reglas de los retos mencionan los pesos de modelos entrenados con los datos. Afecta a los T-DEED que usamos en producción (preentrenados por sus autores con SoccerNet-v2 y los 7 EFL), a `SNB_pano1` (derivado de `challenge2`, además GPL-3.0), al SVC (clips de vídeo SoccerNet), al R3D-18 (`dataset_ball_actions`) y a PRTReId/PnLCalib (SoccerNet GSR y Calibration). No es posible verificar desde el repositorio si las copias locales de BAS-2024 y Tracking están cubiertas por un NDA vigente a nombre de alguien del equipo. Este documento no saca conclusiones jurídicas: antes de distribuir un producto que incluya cualquiera de esos pesos conviene consultar asesoría legal, y mientras tanto tratar los datos con NDA y las licencias NC como material de evaluación e I+D, y preferir para entrenar lo que es CC BY 4.0 (SoccerTrack v2, IDSSE, SN-echoes), CC0 (Transfermarkt), MIT (TeamTrack, por confirmar) o Apache-2.0 (SoccerSynth-Detection) y datos propios.

## 7. Pasos concretos sugeridos

Plan para el usuario; aquí no se ha ejecutado nada. El esfuerzo es una estimación orientativa de trabajo de una persona.

1. **Evaluar los highlights contra SoccerHigh.** Clonar `IPCV/SoccerHigh`, emparejar los 237 `Labels-summary.json` con nuestras carpetas de `/mnt`, convertir los tiempos `_HQ.mkv` → 720p con `video.ini`, correr el pipeline de highlights sobre el Clásico (21 segmentos) y después sobre los 45 partidos de test, y medir P/R/F1@T frente al baseline 0,3883. Hueco que cierra: verdad de highlights más allá de los goles. Esfuerzo: 2-3 días (los partidos hay que copiarlos del host al contenedor).
2. **Alinear StatsBomb con los 135 partidos.** Descargar `events/` y `lineups/` de los 135 `match_id`, guardar el fichero de mapeo carpeta local ⇄ `match_id`, anclar el reloj de cada mitad al vídeo (etiqueta `Kick-off` de Labels-v2 o los goles con marcador conocido) y medir con `scripts/evaluate_match.py` la atribución de equipo de T-DEED y el lector de marcador (eventos `Shot` con `outcome` Goal). Registrarse en el resource centre de StatsBomb. Hueco: eventos con equipo y jugador en TV. Esfuerzo: 3-4 días.
3. **Descargar SoccerNet-Echoes (CC BY 4.0, sin NDA)** y prototipar la señal "gol por comentario" (palabras clave en 10 idiomas + picos de energía) sobre 3-5 partidos sin asignar de SoccerNet-v2; nunca sobre `sn_tuning_videos` (entrenamiento del SVC). Hueco: gol sin marcador en TV y validación fuera de muestra de los umbrales de highlights. Esfuerzo: 2 días.
4. **Construir la verdad de goles al minuto de los ~380 partidos sin asignar** con Transfermarkt datasets (CC0) y football-data.co.uk (marcador al descanso, recuentos), y elegir con ella los 3-5 partidos sin asignar que pide `docs/hoja_de_ruta.md` para validar highlights sin marcador. Hueco: validación con datos no vistos. Esfuerzo: 1 día.
5. **Pedir acceso a `SoccerNet/SN-BAS-2025`** (cuenta HF, afiliación, NDA nominal), añadir a T-DEED la cabeza de equipo (sigmoid + BCE del devkit o la cabeza conjunta equipo×acción del ganador 2025) y medir Team-mAP@1 en Stoke y Reading con la partición vigente. Hueco: equipo en acciones y eventos. Esfuerzo: 1-2 semanas.
6. **Descargar TeamTrack (16,7 GB)**, comprobar si el balón conserva cajas, corregir la distorsión del ojo de pez y generar `vcam796` como con SoccerTrack v2; usarlo como segundo partido para `yolo11s_pano`, la cámara Viterbi+L1 y el tracking con verdad de IDs. Hueco: la rama panorámica solo tiene un dataset. Esfuerzo: 3-4 días.
7. **Descargar `SoccerNet/SN-GSR-2025` (público)** y medir GS-HOTA de nuestro stack YOLO11 + prtreid + PnLCalib frente al baseline TrackLab (22,26) y al estado del arte (63,90); decidir si afinar prtreid y PnLCalib con sus splits. Hueco: equipo, dorsal y posición medidos con una métrica estándar. Esfuerzo: 1 semana.
8. **Escribir a Spiideo (research.spiideo.com) para conocer las condiciones de SynLoc** y, si permiten I+D, preentrenar el detector de jugadores pequeños y una cabeza de "punto de apoyo" para la proyección a campo en panorámica. Hueco: calibración y localización métrica en cámara fija. Esfuerzo: correo + 1 semana si llega.
9. **Documentar licencias y generadores.** Guardar en el repo el texto del NDA y del formulario de HF, anotar la licencia de cada dataset de `datasets/` (hoy solo SoccerTrack v2 la tiene), corregir `docs/particion_datasets.md:20` (el montaje de `/mnt` sí está disponible en el host), y reescribir o documentar los generadores que faltan (`dataset_vjepa_backup`, `clips_analisis`, descarga de Tracking y de `/mnt`, origen de los 28 clips de tiro). Esfuerzo: medio día.
10. **Consultar asesoría legal** sobre los pesos entrenados con vídeo bajo NDA (T-DEED, `SNB_pano1`, SVC, R3D-18) y sobre la GPL-3.0 de T-DEED antes de distribuir el producto. Esfuerzo: externo.
11. **Anotar lo que ningún dataset externo cubre**: elegir y anotar los dos juveniles de evaluación (eventos con equipo y jugador, plantilla de eventos pendiente) y 300-500 frames de balón de YouTube como test real del detector de TV. Hueco: dominio del producto. Esfuerzo: 1-2 semanas de anotación.
12. **Cuando se ataque la atribución de jugador**, pedir `SoccerNet/SN-PCBAS-2026` (FOOTPASS; 28,5 GB en 352x640 + datos tácticos o 236 GB con Full HD; NDA + CC BY-NC 4.0) y reutilizar sus baselines TAAD+DST; y como laboratorio de métricas tácticas, cargar IDSSE y Metrica con kloppy para fijar rangos de sanidad de posesión, velocidades y mapas de calor. Esfuerzo: 1 semana cada uno.

## 8. Huecos de esta investigación

Lo que los investigadores no pudieron verificar a 2026-11-01:

- Licencias declaradas en Roboflow Universe (`football-players-detection-3zvbc`, `football-ball-detection-rejhg`, `football-field-detection-f07vi`, los dos de baloncesto que usa el pipeline y sus homónimos): las páginas devolvieron HTTP 403; los recuentos proceden de tutoriales y fichas de workspace.
- Repositorio, tamaño y licencia del dataset xGHub (CVIU 2026): ScienceDirect, UPF e IRI devolvieron 403; solo consta el registro Crossref/Semantic Scholar y un post del autor.
- Dominio true-think.com (Fu et al. 2023): no se comprobó si sigue sirviendo el dataset ni sus condiciones.
- Resolución, fps y tamaño total de los vídeos del DFL Shootout (indicios de 1280x720 y 25 fps); contenido exacto del write-up del 1.º puesto (CAPTCHA de Kaggle).
- Licencia de Spiideo SynLoc (no consta en sskit, en el paper ni en research.spiideo.com) y número de imágenes reales frente a sintéticas.
- Licencias de los datasets de Kaggle leídos solo por metadatos o fuentes secundarias: "Football Events", European Soccer Database (ODbL según espejos), "Soccer View and Event Score", "football match actions video dataset", "FOOTBALL-SOCCER-VIDEOS-DATASET"; fecha de corte de "Football Events" en 2016-17.
- Enlaces de descarga de SoccerDB (repo sin enlaces) y de Comprehensive Soccer/SSET (HUST devolvía 404); tamaño, clases y enlace del dataset de vídeo "SoccerEvent" (Karimi et al.); URL de descarga del dataset de Huang et al. 2022.
- Composición de OSL-loc-SPL-highlight (liga, resolución, licencia) y de OpenSportsLab/PFF (326 GB; si trae partidos completos); términos reales de PFF FC (requieren registro); texto del NDA de SoccerReplay-1988.
- Si las copias locales de BAS-2024 y SoccerNet Tracking están cubiertas por un NDA vigente a nombre de alguien del equipo; licencia de `Labels-v2.json`/`Labels-ball.json` (ningún documento de SoccerNet la fija); cláusulas sobre pesos de modelos (no existen en el NDA).
- Reparto exacto train/valid/test/challenge de los 200 clips de GSR y liga de origen de los 12 partidos de Tracking ("Swiss Super League" solo aparece en notas internas del proyecto); nombres de las 10 clases de SoccerNet-GAR; tamaños de la mayoría de repos de HF (la API devolvió listas vacías).
- TeamTrack: si el balón conserva cajas y la licencia del dataset en la web del proyecto; frames y resolución de los subconjuntos indoor/outdoor de TrackID3x3.
- Transfermarkt datasets: si incluye la EFL Championship 2019-20 y cuántos partidos de UCL 2014-17 tienen `game_events` completos.
- StatsBomb: fecha de publicación original (2023) de las temporadas 2015/16 completas (los `match_updated` son de 2024-2026); tamaño total del repositorio; número de ficheros de 360.
- Alfheim: resolución y fps de las cámaras (el PDF de MMSys 2014 no se obtuvo); Signality: ninguna fuente primaria de acceso; Impect: número de partidos, licencia (`LICENSE.pdf`) y precisión temporal; MMSys'22 Eliteserien: datos no accesibles; SVW: lista de deportes no legible.
- Páginas bloqueadas o caídas: deeperaction.github.io/multisports (404), media.hust.edu.cn/dataset (404), datasets.simula.no (certificado), mdpi.com (403), link.springer.com (login), kaggle.com (solo JavaScript), sciencedirect.com (403), repositori.upf.edu (acceso denegado), cvlab.cse.msu.edu (conexión rechazada), mmsys2022.ie (403), atomscott.github.io/SoccerTrack (404), research.spiideo.com (sin contenido estático), bundesliga.com/en/aws-bundesliga-data-shootout (404), fbref.com (403).

## 9. Fuentes

Notas de investigación de las que procede este documento (2026-11-01): `output/research_notes/Datasets de fútbol para el proyecto/` (`inventario_local.md`, `propuestos_por_el_usuario.md`, `soccernet_ecosistema.md`, `otros_videos_tv_eventos_highlights.md`, `deteccion_tracking_panoramica_sinteticos.md`, `datos_posicionales_y_eventos_sin_video.md`, `deepsportradar_y_baloncesto.md`; lista de partidos locales en `soccernet_v2_partidos_locales.md`).

### Sección 2 (los cinco propuestos) y 4C

1. DFL Shootout, overview: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/overview
2. DFL Shootout, data: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data
3. DFL Shootout, rules: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/rules
4. DFL Shootout, event descriptions: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/overview/event-descriptions
5. Bundesliga Group, nota del 17-05-2023: https://www.bundesliga-group.com/innovation/bundesliga-data-shootout-machine-learning-competition-on-football-data/
6. Leaderboard (clist.by): https://clist.by/standings/dfl-bundesliga-data-shootout-video-data-sports-football-38475019/
7. Participante (Habr): https://habr.com/ru/articles/848468/
8. 6.º puesto: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/360872
9. 2.º puesto: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/writeups/k-mat-2nd-place-solution
10. 3.º puesto: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/writeups/camaro-public-3rd-place-solution
11. Ganador BAS 2023 inspirado en el Shootout: https://github.com/lRomul/ball-action-spotting
12. Roboflow, origen DFL del dataset de jugadores: https://roboflow.com/blog/track-football-players
13. Mirror HF del Shootout: https://huggingface.co/datasets/dbal0503/Bundesliga
14. SoccerHigh, arXiv: https://arxiv.org/abs/2509.01439 y https://arxiv.org/html/2509.01439v1
15. SoccerHigh, proyecto: https://ipcv.github.io/SoccerHigh/
16. SoccerHigh, código y anotaciones: https://github.com/IPCV/SoccerHigh y https://github.com/IPCV/SoccerHigh/tree/main/dataset
17. SoccerHigh, features (HF): https://huggingface.co/datasets/imva-upf/SoccerHigh
18. SoccerHigh, Labels-summary.json del Clásico: https://huggingface.co/datasets/imva-upf/SoccerHigh/resolve/main/spain_laliga/2016-2017/2017-04-23%20-%2021-45%20Real%20Madrid%202%20-%203%20Barcelona/Labels-summary.json
19. StatsBomb Open Data: https://github.com/statsbomb/open-data
20. StatsBomb, LICENSE.pdf: https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf
21. StatsBomb, competitions.json: https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json
22. StatsBomb, eventos del Clásico (267569): https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/267569.json
23. StatsBomb, especificaciones: https://github.com/statsbomb/open-data/tree/master/doc
24. statsbombpy: https://github.com/statsbomb/statsbombpy
25. kloppy: https://github.com/PySport/kloppy y https://kloppy.pysport.org/
26. socceraction (StatsBomb): https://socceraction.readthedocs.io/en/latest/documentation/data/statsbomb.html
27. DeepSportradar-v1 (arXiv 2208.08190): https://arxiv.org/abs/2208.08190
28. DeepSportradar, organización GitHub: https://github.com/DeepSportradar
29. DeepSportradar, retos 2023: https://deepsportradar.github.io/challenge.html
30. DeepSportradar-v2 (EPFL Infoscience): https://infoscience.epfl.ch/record/308980
31. Kaggle deepsport-dataset: https://www.kaggle.com/datasets/gabrielvanzandycke/deepsport-dataset
32. Kaggle basketball-instants-dataset: https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset
33. Reto de segmentación (licencia ND): https://github.com/DeepSportradar/instance-segmentation-challenge
34. MMSports 2025 (SoccerTrack Challenge): http://mmsports.multimedia-computing.de/mmsports2025/challenge.html
35. MMSports 2024 (sin reto): http://mmsports.multimedia-computing.de/mmsports2024/index.html
36. SkillCorner opendata: https://github.com/SkillCorner/opendata
37. SkillCorner, commits (borrado 2025-10-06): https://github.com/SkillCorner/opendata/commits/master
38. SkillCorner, README 2020: https://github.com/SkillCorner/opendata/blob/1b99d401/README.md
39. SkillCorner, matches.json: https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches.json
40. SkillCorner body pose (HF): https://huggingface.co/datasets/SkillCorner/opendata-bodypose
41. SkillCorner, Dynamic Events CSV Specifications: https://26560301.fs1.hubspotusercontent-eu1.net/hubfs/26560301/Guides/Dynamic%20Events/20250216%20-%20Dynamic%20Events%20CSV%20Specifications.pdf
42. SkillCorner Open Data #5: https://skillcorner.com/articles/skillcorner-open-data-5-visualising-football-tracking-data
43. Wyscout, figshare: https://figshare.com/collections/Soccer_match_event_dataset/4415000
44. Wyscout, artículo (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC6817871
45. Last Row: https://github.com/Friends-of-Tracking-Data-FoTD/Last-Row
46. Signality (mención): https://soccermatics.readthedocs.io/en/latest/ y https://www.svgeurope.org/blog/headlines/spiideo-acquires-optical-tracking-and-sports-data-specialist-signality/
47. roboflow/sports: https://github.com/roboflow/sports
48. Kaggle football-soccer-ball-detection-dfl: https://www.kaggle.com/datasets/michaelmortenson/football-soccer-ball-detection-dfl
49. SoccerDB: https://github.com/newsdata/SoccerDB y https://ar5iv.labs.arxiv.org/html/1912.04465
50. SSET (Springer): https://link.springer.com/article/10.1007/s11042-020-09414-3
51. Survey 2023 de datasets de fútbol: https://ar5iv.labs.arxiv.org/html/2309.12067
52. Survey 2025 de detección de eventos deportivos: https://arxiv.org/html/2505.03991
53. Footovision STAD: https://arxiv.org/html/2505.09455
54. Penaltis ULPGC: https://arxiv.org/abs/2507.12617
55. Gerats et al. (TNO): https://www.scitepress.org/PublishedPapers/2021/103035/pdf/index.html
56. MMSys'22 Grand Challenge: https://arxiv.org/html/2202.01031

### Sección 3 (inventario local)

57. Ficheros del repo citados: `docs/particion_datasets.md`, `docs/hoja_de_ruta.md`, `docs/estudio_deteccion_tiros.md`, `docs/highlights_evaluacion.md`, `README.md`, `config.py`, `scripts/download_ball_action_data.py`, `scripts/download_soccertrack_v2.sh`, `scripts/ball_detector/*.py`, `scripts/shot_benchmark/*.py`, `scripts/extract_ball_action_clips.py`, `scripts/build_ground_truth.py`, `training/train_ball_actions.py`, `basketball/README.md`, `basketball/datasets/basketball_final/README.md`, `output/ball_experiments/CONCLUSIONES.md`, `datasets/soccertrack-v2/README.md`, `CHANGES_v1_1.md`, `CHANGES_v1_2.md`, `mot/README.md`.
58. SoccerNet, página de datos y acceso: https://www.soccer-net.org/data
59. SoccerTrack v2 (HF): https://huggingface.co/datasets/atomscott/soccertrack-v2
60. SoccerTrack v2, código: https://github.com/AtomScott/SoccerTrack-v2
61. SoccerTrack v2, informe técnico: https://arxiv.org/html/2508.01802v2
62. SoccerTrack Challenge 2025, resultados: https://arxiv.org/html/2602.00484v1

### Sección 4A (usar ya)

63. SN-echoes (HF): https://huggingface.co/datasets/SoccerNet/SN-echoes
64. SoccerNet-Echoes, paper: https://arxiv.org/html/2405.07354 y repo https://github.com/SoccerNet/sn-echoes
65. sn-caption: https://github.com/SoccerNet/sn-caption
66. SoccerNet-Caption, paper: https://arxiv.org/abs/2304.04565
67. SN-BAS-2025 (HF): https://huggingface.co/datasets/SoccerNet/SN-BAS-2025
68. sn-teamspotting (devkit y reglas): https://github.com/SoccerNet/sn-teamspotting y https://raw.githubusercontent.com/SoccerNet/sn-teamspotting/main/ChallengeRules.md
69. SoccerNet 2025, resultados: https://arxiv.org/html/2508.19182v1
70. sn-gamestate: https://github.com/SoccerNet/sn-gamestate
71. SoccerNet-GSR, paper: https://arxiv.org/html/2404.11335
72. SN-GSR-2025 (HF): https://huggingface.co/datasets/SoccerNet/SN-GSR-2025
73. SoccerNet 2024, resultados: https://arxiv.org/html/2409.10587
74. OSL-loc-SPL-highlight: https://huggingface.co/datasets/OpenSportsLab/OSL-loc-SPL-highlight
75. OpenSportsLab, API HF: https://huggingface.co/api/datasets?author=OpenSportsLab
76. TeamTrack (Kaggle): https://www.kaggle.com/datasets/atomscott/teamtrack
77. TeamTrack, paper: https://arxiv.org/html/2404.13868v1 y proyecto https://atomscott.github.io/TeamTrack/
78. Transfermarkt datasets: https://github.com/dcaribou/transfermarkt-datasets
79. Transfermarkt, modelo `game_events`: https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/master/dbt/models/curated/models.yml
80. Understat: https://understat.com/league/EPL/2014
81. soccerdata, Understat: https://soccerdata.readthedocs.io/en/latest/datasources/Understat.html
82. Kaggle Football Events: https://www.kaggle.com/datasets/secareanualin/football-events
83. Columnas de Football Events (ModStatR): https://search.r-project.org/CRAN/refmans/ModStatR/html/data_event.html
84. Descripción de Football Events (README de terceros): https://raw.githubusercontent.com/JingxianLin/DSND-Write-A-Data-Science-Blog-Post/master/README.md
85. European Soccer Database: https://www.kaggle.com/datasets/hugomathien/soccer
86. ESD en OpenML (licencia): https://www.openml.org/api/v1/json/data/43762
87. football-data.co.uk, notas: https://www.football-data.co.uk/notes.txt y https://www.football-data.co.uk/spainm.php
88. SoccerNet, API HF (gating y licencias): https://huggingface.co/api/datasets?author=SoccerNet
89. Paquete pip SoccerNet: https://github.com/SoccerNet/SoccerNet y https://pypi.org/pypi/SoccerNet/json

### Sección 4B (útil en el futuro)

90. FOOTPASS: https://github.com/JeremieOchin/FOOTPASS, LICENSE https://raw.githubusercontent.com/JeremieOchin/FOOTPASS/main/LICENSE, árbol https://api.github.com/repos/JeremieOchin/FOOTPASS/git/trees/main?recursive=1, `evaluation.py` https://raw.githubusercontent.com/JeremieOchin/FOOTPASS/main/evaluation.py y `utils/TAAD_Dataset.py` https://raw.githubusercontent.com/JeremieOchin/FOOTPASS/main/utils/TAAD_Dataset.py
91. FOOTPASS, paper: https://arxiv.org/html/2511.16183; CVIU https://doi.org/10.1016/j.cviu.2026.104790 (licencia vía Crossref https://api.crossref.org/works/10.1016/j.cviu.2026.104790); preprint HAL https://hal.science/hal-05373478v1
92. SN-PCBAS-2026 (HF): https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026; API https://huggingface.co/api/datasets/SoccerNet/SN-PCBAS-2026; árbol con tamaños https://huggingface.co/api/datasets/SoccerNet/SN-PCBAS-2026/tree/main
93. SoccerNet 2026, resultados: https://arxiv.org/html/2607.07320v1; página del reto https://www.soccer-net.org/challenges/2026; Codabench https://www.codabench.org/competitions/11232/ (challenge) y https://www.codabench.org/competitions/11249/ (test); PAVE https://arxiv.org/abs/2606.28389; extensiones de los baselines https://arxiv.org/abs/2606.09679; ME-DST https://arxiv.org/abs/2608.01696
94. Spiideo sskit: https://github.com/Spiideo/sskit y https://api.github.com/repos/Spiideo/sskit
95. SynLoc, VISAPP 2025: https://scitepress.org/publishedPapers/2025/131082/pdf/index.html
96. SoccerSynth-Detection: https://github.com/open-starlab/SoccerSynth-Detection y https://arxiv.org/html/2501.09281v1
97. SoccerSynth-Field: https://arxiv.org/html/2503.13969v1
98. MultiSports: https://github.com/MCG-NJU/MultiSports, https://huggingface.co/datasets/MCG-NJU/MultiSports y https://ar5iv.labs.arxiv.org/html/2105.07404
99. SoccerReplay-1988 (HF): https://huggingface.co/datasets/Homie0609/SoccerReplay-1988
100. UniSoccer / MatchVision: https://arxiv.org/html/2412.01820 y https://github.com/jyrao/UniSoccer
101. GOAL (Heriot-Watt): https://ar5iv.labs.arxiv.org/html/2211.04534 y https://gitlab.com/grounded-sport-convai/goal-baselines
102. MOMENTS: https://github.com/akskuchi/MOMENTS y https://arxiv.org/html/2601.16333v3
103. SoccerBench (HF): https://huggingface.co/datasets/Homie0609/SoccerBench
104. SoccerAgent: https://arxiv.org/html/2505.03735v1 y https://jyrao.github.io/SoccerAgent/
105. Sports-QA: https://github.com/HopLee6/Sports-QA y https://huggingface.co/datasets/HopLeeTop/Sports-QA
106. SPORTU: https://arxiv.org/html/2410.08474v4
107. SportR: https://arxiv.org/abs/2511.06499v1
108. RefereeBench: https://arxiv.org/html/2604.15736
109. SJN-210k (Zenodo): https://zenodo.org/records/14247819
110. Kinetics-700, label map: https://raw.githubusercontent.com/open-mmlab/mmaction2/main/tools/data/kinetics/label_map_k700.txt
111. Kinetics-700-2020: https://arxiv.org/abs/2010.10864
112. AudioSet Cheering y Crowd: https://research.google.com/audioset/dataset/cheering.html y https://research.google.com/audioset/dataset/crowd.html
113. FootAndBall (enlace a ISSIA-CNR): https://github.com/jac99/FootAndBall y https://ar5iv.labs.arxiv.org/html/1912.05445
114. ISSIA-3D / SoccerNet-v3D: https://arxiv.org/html/2504.10106v1
115. Alfheim: https://datasets.simula.no/alfheim/
116. Fu et al. 2023 (PDF): https://ira.lib.polyu.edu.hk/bitstream/10397/105399/1/applsci-13-05361.pdf
117. IDSSE (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/ y figshare https://api.figshare.com/v2/articles/28196177
118. floodlight datasets: https://floodlight.readthedocs.io/en/latest/modules/io/datasets.html
119. Metrica sample data: https://github.com/metrica-sports/sample-data
120. databallpy (25 fps de Metrica): https://databallpy.readthedocs.io/en/v0.5.0/getting_started/loading_in_a_match_page.html
121. PFF FC Open Data: https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data y https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset
122. OpenSportsLab/PFF (HF): https://huggingface.co/datasets/OpenSportsLab/PFF
123. SoccerNet-GAR (HF): https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR
124. GAR, paper y reto 2027: https://arxiv.org/abs/2511.12606 y https://github.com/SoccerNet/sn-gar-2027
125. SoccerTrack v1 (Kaggle): https://www.kaggle.com/datasets/atomscott/soccertrack
126. SoccerTrack v1, paper: https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Scott_SoccerTrack_A_Dataset_and_Tracking_Algorithm_for_Soccer_With_Fish-Eye_CVPRW_2022_paper.pdf
127. SportsMOT: https://github.com/MCG-NJU/SportsMOT y https://arxiv.org/html/2304.05170
128. SoccerNet-v3 (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC9210334/
129. sn-tracking, sn-jersey, sn-mvfoul: https://github.com/SoccerNet/sn-tracking, https://github.com/SoccerNet/sn-jersey, https://github.com/SoccerNet/sn-mvfoul
130. X-VARS / XFoul: https://arxiv.org/abs/2404.06332
131. EIGD-S: https://github.com/MM4SPA/eigd y https://data.uni-hannover.de/dataset/eigd
132. TACDEC: https://zenodo.org/records/10611979 y https://www.simula.no/research/tacdec-dataset-tackle-events-soccer-game-videos
133. Imágenes de eventos (Data in Brief 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC11999477/
134. IAUFD: https://doaj.org/article/fc5f6c212de841aaa120126d68006ce4
135. xGHub (registro): https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.cviu.2026.104732?fields=title,abstract,year,authors,externalIds,openAccessPdf
136. 3DSP: https://github.com/calvinyeungck/3D-Shot-Posture-Dataset
137. MatchTime (HF): https://huggingface.co/datasets/Homie0609/MatchTime
138. SoccerChat (HF): https://huggingface.co/datasets/SimulaMet/SoccerChat y https://arxiv.org/pdf/2505.16630
139. SoccerLens: https://github.com/IsmaelElsharkawi/SoccerLensDataset y https://arxiv.org/html/2605.09598v2
140. WC14: https://nhoma.github.io/
141. TS-WorldCup / KpSFR: https://github.com/ericsujw/KpSFR
142. PnLCalib: https://arxiv.org/html/2404.08401v3
143. WorldPose: https://eth-ait.github.io/WorldPoseDataset/
144. Impect open data: https://github.com/ImpectAPI/open-data
145. Huang et al. 2022 (registro): https://api.semanticscholar.org/graph/v1/paper/DOI:10.3390/app12157473?fields=title,abstract,year,authors,externalIds,openAccessPdf
146. infactory-ai/soccer-events: https://huggingface.co/datasets/infactory-ai/soccer-events
147. FAANTRA (Ball Action Anticipation): https://github.com/MohamadDalal/FAANTRA
148. sn-depth: https://github.com/SoccerNet/sn-depth
149. SoccerNet, retos 2027: https://www.soccer-net.org/challenges/2027

### Sección 5 (baloncesto)

150. TrackID3x3: https://github.com/open-starlab/TrackID3x3 y https://arxiv.org/abs/2503.18282
151. NBA dataset (GAR, ECCV 2020): https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530205.pdf
152. FineSports: https://github.com/PKU-ICST-MIPL/FineSports_CVPR2024
153. NSVA: https://github.com/jackwu502/NSVA
154. BARD: https://bodai.unibs.it/?p=1198
155. SpaceJam: https://github.com/simonefrancia/SpaceJam
156. Roboflow, tutorial YOLO26 (dataset de jugadores NBA): https://blog.roboflow.com/how-to-train-yolo26-custom-data/
157. Roboflow Universe, aro y balón: https://universe.roboflow.com/persistence/basketball-hoop-player y https://universe.roboflow.com/tomas-34ray/basketball-detection-cjjxp
158. Repo DeepSport (licencia del código): https://github.com/gabriel-vanzandycke/deepsport

### Sección 6 (licencias)

159. NDA de SoccerNet (texto): https://docs.google.com/document/d/1Efz8yP-baa8CtcMB7SYIID7Bn5sH7sGTB4-u1YdVRf4/export?format=txt
160. Formulario gated y licencias por repo (API HF): https://huggingface.co/api/datasets?author=SoccerNet
161. FALCON-Bench (ejemplo de condiciones heredadas de SoccerNet): https://huggingface.co/datasets/cplou99/FALCON-Bench/blob/main/README.md
