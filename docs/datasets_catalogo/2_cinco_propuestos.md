> Parte de [docs/datasets.md](../datasets.md): catálogo a 2026-11-01, con la numeración original del documento.

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
