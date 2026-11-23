# Datasets públicos con tracking + verdad de posesión / estado del balón, y herramientas de posesión (estado: 13-11-2026)

Contexto verificado del reto More than a Hack 2027 (Metrica Sports × FC Barcelona × Mobile World Capital):
- Repo oficial: [metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027). Un partido (FC Barcelona vs Athletic Club, LaLiga 2024-25, jornada 2), vídeo táctico 199.319 fotogramas a 25 fps 1920×1080, tracking "Automatic Tracking Data (ATD)" automático y sin corregir, 46 slots sin identidad, balón presente en el 56,9 % de los fotogramas (59,3 % de los puntuados), etiquetas H/A/D/X, verdad pública de 5 minutos (17 posesiones, 14,2 s de media), métrica spellF1 (baseline 0,2433) con macroF1 de desempate — [OVERVIEW.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/OVERVIEW.md), [README.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/README.md).
- Formato: exportación FIFA/EPTS de Metrica (`BAR-ATH_tactical_atd.xml` + `.txt`), se carga con `kloppy.metrica.load_tracking_epts`; columnas `frame_id, ball_x, ball_y, 1_x, 1_y, 1_s …`; coordenadas normalizadas 0–1; `NaN` cuando falta un objeto; no hay bandera de balón muerto/vivo (solo las fases del Smart Tagging y los saques como pista) — [DATA.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md).
- Reglas: se permite cualquier dato externo de otros partidos, incluidos etiquetados a mano "en cualquier cantidad"; se prohíbe etiquetar a mano el partido del reto o usar cualquier dato humano sobre él (eventos, tracking corregido); código MIT, datos del partido licenciados solo para la competición — [OVERVIEW.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/OVERVIEW.md), [README.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/README.md).
- Fase 2 (Innovation Challenge, 26-11-2026 a 14-01-2027): los 20 mejores reciben "10 partidos profesionales con tracking y eventos"; formato no especificado — [blog de Metrica](https://www.metrica-sports.com/blog/more-than-a-hack-2027-build-something-real-with-fc-barcelona-match-data).

Nota de método: connectedpapers.com no se pudo usar (requiere JavaScript); en su lugar se consultó la API de Semantic Scholar para los 35 artículos que citan el dataset IDSSE (ver Gaps de la pregunta 1).

## Pregunta 1: ¿Qué datasets traen ball status (alive/dead) y equipo poseedor por fotograma, que es exactamente la verdad del reto?

### Takeaway
Solo dos fuentes públicas traen las dos cosas por fotograma desde el proveedor: IDSSE/DFL (7 partidos, `BallStatus` 0/1 + `BallPossession` 1/2, 25 fps, CC-BY 4.0, descarga libre) y PFF FC/Gradient WC 2022 (64 partidos, `game_event` por fotograma con `home_ball` y tipos OUT/END/KICKOFF/OTB, 30 fps, acceso por formulario). SkillCorner da poseedor por fotograma (jugador + equipo) pero no estado del balón; Metrica solo lo da a nivel de evento; SoccerTrack v2 trae estado dentro/fuera del balón por fotograma pero no poseedor.

### Cited Findings

#### Catálogo (resumen por dataset)

| Dataset | Partidos / duración | fps | Balón | Ball status/frame | Poseedor/frame | Vídeo | Tracking | Licencia | Loader |
|---|---|---|---|---|---|---|---|---|---|
| Metrica sample-data | 3 (~96 min el G3) | 25 | sí (x,y) | no (solo eventos BALL OUT / SET PIECE) | no (solo eventos) | no | óptico, corrección no indicada | sin licencia formal | kloppy (`metrica`), codeball |
| SkillCorner opendata (2024/25 A-League) | 10 | 10 | sí (puede faltar) | no | sí (`possession.player_id/group`) | no | automático de broadcast, sin corregir | MIT | kloppy (`skillcorner.load_open_data`), floodlight |
| IDSSE (DFL/Sportec) | 7 (1.002.644 frames ≈ 668 min) | 25 | sí | sí (`BallStatus`) | sí (`BallPossession`, equipo) | no | TRACAB Gen5 con operadores y QC | CC-BY 4.0 | kloppy (`sportec.load_open_*`), floodlight, databallpy |
| PFF FC / Gradient WC 2022 | 64 | 30 (29,97 nativo) | sí (3D) | derivable por frame (`game_event`) | sí (`home_ball` por frame) | no (sí en SoccerNet-GAR, con NDA) | automático de broadcast + refinado manual | formulario; términos no publicados | kloppy (`pff.load_tracking`) |
| StatsBomb open data | 80 competición-temporada; 12 con 360 | — | no | por evento | por evento (`possession_team`) | no | — | User Agreement (atribución) | kloppy, socceraction, statsbombpy |
| SoccerTrack v2 | 10 (932 min) | 25 | reconstruido entre anclas | sí (dentro/fuera) | no | sí (4K panorámico) | sistema BePro, solo comprobaciones visuales | CC BY 4.0 (gated) | OpenSTARLab (BePro) |
| SoccerNet-Tracking | 200 clips 30 s + 1 mitad de 45 min + 12 partidos completos | 25 | sí (bbox) | no | no | sí (broadcast) | anotación manual (GT) | NDA SoccerNet | pip SoccerNet |
| SoccerNet-GSR | 200 clips 30 s | 25 | no (ignorado) | no | no | sí (broadcast) | semiautomático (GT) | NDA SoccerNet | sn-gamestate / TrackLab |
| SoccerTrack v1 | 30 min fisheye + 30 min dron | n/v | n/v | no | no | sí | bbox anotadas + GNSS | n/v | SportsLabKit |
| TeamTrack | 30 + 30 min fútbol (+ basket, balonmano) | n/v | n/v | no | no | sí | bbox anotadas | n/v (código MIT) | — |
| FOOTPASS | 54 partidos 2023/24 | 25 | no indicado | no | no (eventos con equipo) | sí (vía NDA SoccerNet) | pipeline propio "FIFA EPTS certified" | pendiente | — |
| Wyscout (Pappalardo 2019) | 7 competiciones, 1 temporada | — | no | por evento | por evento | no | — | CC BY (conflicto: "non-commercial" en D4Science) | kloppy (`wyscout`), socceraction |
| Last Row / Liverpool | 19 goles | 20 | sí (con z) | no | no | no | aproximado, "sin precisión para investigación" | sin licencia | — |
| DFL Bundesliga Data Shootout (Kaggle 2022) | vídeo de 8 partidos (train) | n/v | no | no | no | sí | sin tracking | reglas Kaggle (n/v) | — |
| Impect open data | Bundesliga 2023/24 | — | no | por evento | por evento | no | — | LICENSE.pdf | kloppy (`impect`) |

n/v = no verificado en esta investigación.

#### Metrica Sports open data
- 3 partidos; los partidos 1 y 2 en CSV (tracking y eventos) y el partido 3 (añadido el 15-04-2021) en formato EPTS/FIFA para tracking y metadatos con eventos en JSON; datos anonimizados; campo 105×68 m, coordenadas 0–1; sin licencia (solo "sé responsable" y reconoce la fuente) — [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)
- Metadatos EPTS del partido 3: `FrameRate` 25, 1.ª parte frames 1–69.661, 2.ª 69.662–143.761; tipo de tracking "Optical"; solo canales de posición x,y para 35 jugadores y el balón; no hay canal de ball status ni de posesión — [Sample_Game_3_metadata.xml](https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_3/Sample_Game_3_metadata.xml)
- Eventos CSV (partido 1): columnas `Team, Type, Subtype, Period, Start Frame, Start Time [s], End Frame, End Time [s], From, To, Start X, Start Y, End X, End Y`; tipos `SET PIECE, PASS, BALL LOST, RECOVERY, CHALLENGE, SHOT, BALL OUT, FAULT RECEIVED, CARD`; subtipos de SET PIECE `KICK OFF, THROW IN, CORNER KICK, FREE KICK`; BALL LOST incluye `END HALF` — [Sample_Game_1_RawEventsData.csv](https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/Sample_Game_1_RawEventsData.csv)
- El deserializador EPTS de kloppy fija `ball_owning_team=None` y `ball_state=None` para Metrica — [kloppy metrica_epts/deserializer.py](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/metrica_epts/deserializer.py); la documentación muestra `ball_state` y `ball_owning_team_id` a `None` en tracking y rellenados en eventos (p. ej. "alive", "TEAM_A"); `metrica.load_open_data(match_id=1)` — [kloppy: Metrica](https://kloppy.pysport.org/user-guide/loading-data/metrica/)

#### SkillCorner open data
- Versión actual: 10 partidos de la A-League australiana 2024/25; tracking a 10 fps (pose corporal a 25 fps); por fotograma `frame, timestamp, period, ball_data, possession {player_id, group}, image_corners_projection, player_data` (con `is_detected`); eventos dinámicos CSV, fases de juego CSV "definidas solo mientras el balón está en juego"; licencia MIT; datos obtenidos de broadcast por visión por computador, "algunos puntos son erróneos", ~97 % de identidades correctas — [SkillCorner/opendata](https://github.com/SkillCorner/opendata)
- Ramas: `master` (actualizada 14-09-2026) y `data/2026-09-add-10-matches-v3-refresh` (08-09-2026); no aparece rama con la versión antigua — [branches](https://github.com/SkillCorner/opendata/branches)
- La versión original tenía 9 partidos 2019/20 (campeón vs subcampeón de Premier, Ligue 1, LaLiga, Serie A y Bundesliga), "recogidos automáticamente del broadcast sin corrección manual", MIT — resumen de búsqueda a partir del texto del repo: [skillcorner.github.io/opendata](https://skillcorner.github.io/opendata/), [znstrider/Skillcorner_Opendata_Match_Analysis](https://uithub.com/znstrider/Skillcorner_Opendata_Match_Analysis)
- kloppy: `ball_owning_team` sale de `possession.group` ("home team"/"away team"); `ball_state = ALIVE if ball_owning_team is not None else DEAD` (no depende de la posición del balón); `frame_rate = 10`; `only_alive` descarta frames sin posesión; `include_empty_frames=False` descarta frames sin `player_data` — [kloppy skillcorner.py](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/skillcorner.py); `load_open_data(match_id="1925299")` lee `data/matches/{id}/{id}_match.json` y `_tracking_extrapolated.jsonl` de la rama master — [kloppy _providers/skillcorner.py](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/_providers/skillcorner.py)

#### IDSSE – "An integrated dataset of spatiotemporal and event data in elite soccer" (Bassek, Rein, Weber, Memmert; Scientific Data, 01-02-2025)
- 7 partidos de la Bundesliga 2022/23 (2 de 1.ª, 5 de 2.ª división; 26-08-2022 a 27-05-2023); 207 jugadores, 10 equipos; 11.137 eventos; 1.002.644 fotogramas de x/y de jugadores y balón; tres XML por partido (match information, events, positions); tracking óptico TRACAB Gen5 (16–20 cámaras a 25 Hz), identidades asignadas manualmente antes del partido, operadores en directo y control de calidad final; `BallStatus` 0 inactivo / 1 activo; `BallPossession` 1 local / 2 visitante ("el jugador que controla el balón"); no todos los eventos están sincronizados (6.251 pases y tiros alineados con DataBallPy v0.5.3, desfase medio −0,37 ± 1,82 s); CC-BY 4.0; sin vídeo por licencias; figshare DOI 10.6084/m9.figshare.28196177 — [PMC11787359](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/)
- Espejo en Hugging Face: 25 fps, ids `J03WPY, J03WMX, J03WN1, J03WOH, J03WOY, J03WPF, J03WQF`; `sportec.load_open_tracking_data(match_id=...)` / `load_open_event_data`; propietario DFL, CC-BY 4.0 — [pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md); repo de los autores [spoho-datascience/idsse-data](https://uithub.com/spoho-datascience/idsse-data)
- kloppy lee `BallPossession` (1 → local, si no → visitante; ausente → 0 → visitante) y `BallStatus` (1 → ALIVE, si no → DEAD); con `only_alive=True` descarta los frames muertos — [kloppy sportec/deserializer.py](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/sportec/deserializer.py)
- floodlight construye `Code("possession", definitions={1:"Home", 2:"Away"})` y `Code("ballstatus", definitions={0:"Dead", 1:"Alive"})` desde los mismos atributos; estima el framerate del intervalo entre frames — [floodlight.io.dfl (fuente)](https://floodlight.readthedocs.io/en/latest/_modules/floodlight/io/dfl.html), [doc del módulo](https://floodlight.readthedocs.io/en/latest/modules/io/dfl.html)

#### PFF FC (ahora Gradient Sports) – FIFA World Cup 2022
- 64 partidos; tracking de broadcast; eventos con "grades"; acceso rellenando un formulario; versión "enhanced" con un fichero de eventos por partido y carpetas Metadata/Rosters — [blog PFF FC](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data), [enhanced dataset](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset); el formulario no se pudo abrir y los términos de uso no están publicados en el blog.
- Según la ficha de SoccerNet-GAR (derivado): posiciones 2D de jugadores y 3D del balón a 30 fps (≈29,97 nativo), "extraídas automáticamente del broadcast y refinadas manualmente por anotadores" — [OpenSportsLab/SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR)
- kloppy: `pff.load_tracking(meta_data, roster_meta_data, raw_data)`; `ball_state` se deriva de `game_event.game_event_type` (OUT, END → DEAD; FIRSTKICKOFF…FOURTHKICKOFF y OTB → ALIVE), `ball_owning_team` de `game_event.home_ball`; ambos persisten entre frames; `frame_rate = metadata["fps"]` — [kloppy pff.py](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/pff.py)

#### Hudl StatsBomb open data
- 80 entradas competición-temporada, 24 competiciones, 12 con datos 360 (Bundesliga 23/24, AFCON 2023, WC 2022, LaLiga 20/21, Ligue 1 21/22 y 22/23, MLS 2023, Euro 2020 y 2024, Women's Euro 2022 y 2025, Women's WC 2023) — [competitions.json](https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json)
- Se exige citar StatsBomb y usar su logo; la licencia está en `LICENSE.pdf` ("StatsBomb Open Data User Agreement"); 360 "para partidos seleccionados" — [statsbomb/open-data](https://github.com/statsbomb/open-data); el anuncio de 2018 habla de "uso no comercial" — [blog 2018](https://blogarchive.statsbomb.com/news/statsbomb-release-free-fifa-world-cup-data/)
- kloppy: `ball_owning_team` = `possession_team` del evento; `ball_state` ALIVE por defecto y DEAD solo mediante eventos sintéticos "out" (pases/tiros/duelos/porteros fuera); Half End y Foul no ponen el balón muerto; freeze frames 360 con `visible_area` — [kloppy statsbomb/specification.py](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/event/statsbomb/specification.py), [deserializer.py](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/event/statsbomb/deserializer.py)

#### SoccerTrack v2 (Scott, Uchida, Kuroda, Kim, Fujii; arXiv 2508.01802, v1 03-08-2025, v2 05-10-2026)
- 10 partidos universitarios, 932 min a 25 fps, vídeo 4K panorámico fijo; coordenadas métricas por fotograma, dorsales, identidades persistentes, roles y equipo; ~1,4 M frames y 30,7 M registros jugador-frame; 21.428 eventos de balón en 12 clases; pista del balón "reconstruida" (rectas a velocidad constante entre anclas, casi todas eventos anotados); un estado del balón dentro/fuera por fotograma "presente en los datos" pero excluido como entrada del modelo porque es "casi una etiqueta directa de la clase Out"; sin bandera de posesión; cajas solo en un clip de 4 min por partido (59.842 frames, ~1,3 M cajas); posiciones e identidades del sistema BePro, inspección visual sin auditoría cuantitativa; eventos corregidos a mano contra vídeo — [arXiv 2508.01802v2](https://arxiv.org/html/2508.01802v2), [abstract](https://arxiv.org/abs/2508.01802)
- Hugging Face (v1.2): carpetas `videos/` (20 mitades), `gsr/` (JSON por mitad), `bas/`, `ball/` (una fila por frame), `mot/`, `raw/` (calibración y exportaciones BePro); 145 GB; CC BY 4.0; acceso gated (aceptar condiciones); split 6/2/2 — [atomscott/soccertrack-v2](https://huggingface.co/datasets/atomscott/soccertrack-v2); 2 partidos con BePro Cerberus y 8 con stitching de 3 cámaras — [búsqueda](https://arxiv.org/html/2508.01802v1)

#### SoccerNet-Tracking y SoccerNet-GSR
- SN-Tracking: 200 clips de 30 s, una mitad completa de 45 min, 12 partidos completos de cámara principal; clases jugador, portero, árbitro, balón; CSV estilo MOT; descarga con `SoccerNetDownloader` — [SoccerNet/sn-tracking](https://github.com/SoccerNet/sn-tracking), [arXiv 2204.06918](https://arxiv.org/abs/2204.06918)
- SN-GSR: 200 clips de 30 s (750 frames a 25 fps, 1920×1080), ~2,36 M posiciones de atletas en el campo (105×68, origen en el centro), balón ignorado "porque pasa mucho tiempo en el aire", anotación semiautomática, hereda splits de SN-Tracking — [arXiv 2404.11335](https://arxiv.org/html/2404.11335v1); código GPL-3.0, dataset v1.3, `downloadDataTask(task="gamestate-2024")`, métrica GS-HOTA — [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate)
- Acceso: los datasets con vídeo protegido están "gated" y requieren cuenta de Hugging Face y aceptar el NDA de SoccerNet; vídeos broadcast a 25 fps en 720p/224p — [soccer-net.org/data](https://www.soccer-net.org/data)

#### SoccerTrack v1 y TeamTrack (Scott et al.)
- SoccerTrack (CVPRW 2022): GNSS y cajas anotadas sobre vídeo fisheye 8K (Z CAM E2-F8) y dron 4K (DJI Mavic 3); 82.800 frames y 2.484.000 cajas; código en SportsLabKit (GPL-3.0); datos en Kaggle — [paper CVF](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Scott_SoccerTrack_A_Dataset_and_Tracking_Algorithm_for_Soccer_With_Fish-Eye_CVPRW_2022_paper.html), [AtomScott/SoccerTrack](https://github.com/AtomScott/SoccerTrack)
- TeamTrack (arXiv 2404.13868, publicado 22-04-2024): fútbol lateral fisheye 8K 30 min y cenital dron 4K 30 min, 1.242.000 cajas cada uno; 279.900 frames y 4.374.900 cajas en total (fútbol, baloncesto, balonmano); Kaggle y Google Drive; código MIT; balón y licencia del dataset no indicados — [atomscott.github.io/TeamTrack](https://atomscott.github.io/TeamTrack), [AtomScott/TeamTrack](https://github.com/AtomScott/TeamTrack)

#### FOOTPASS (Ochin et al.; arXiv 2511.16183, 20-11-2025; CVIU)
- 54 partidos completos 2023/24 (Ligue 1, Bundesliga, Serie A, LaLiga, UCL), broadcast 1920×1080 a 25 fps; tracking del pipeline de los autores ("FIFA EPTS certified": posiciones, velocidades, equipo, dorsal, rol y tracklets); 102.992 eventos en 8 clases (drive, pass, cross, shot, header, throw-in, tackle, block) como (frame, clase, equipo, dorsal); sin etiquetas explícitas de posesión o estado del balón; vídeos vía Hugging Face de SoccerNet con NDA; licencia "se especificará"; baselines en [JeremieOchin/FOOTPASS](https://github.com/JeremieOchin/FOOTPASS) — [arXiv 2511.16183](https://arxiv.org/html/2511.16183)

#### SoccerNet-GAR (OpenSportsLab, arXiv 2511.12606, nov. 2025)
- Vídeo broadcast 720p sincronizado + tracking Gradient/PFF (2D jugadores, 3D balón, 30 fps) en ventanas de 4,5 s alrededor de 87.939 eventos de los 64 partidos del WC 2022; 10 clases (PASS, TACKLE, OUT, HEADER, THROW IN, CROSS, FREE KICK, SHOT, GOAL, HIGH PASS); splits por partido 45/9/10; licencia "other", acceso con NDA de SoccerNet; ligado al SoccerNet Challenge 2027 — [OpenSportsLab/SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR)

#### Otros
- Wyscout open data (Pappalardo et al., Scientific Data 2019, DOI 10.1038/s41597-019-0247-7): "la mayor colección abierta de soccer-logs", todos los eventos de una temporada de 7 competiciones (5 grandes ligas 2017/18, WC 2018, Euro 2016), cada evento con posición, tiempo, resultado y jugador; licencia CC BY según metadatos; 233 citas — [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s41597-019-0247-7?fields=title,abstract,year,citationCount,openAccessPdf); colección figshare 4415000 (v5 28-01-2020) y catálogo D4Science que la lista como "Non-commercial only" — [búsqueda](https://catalogue.d4science.org/dataset/public_data_set_of_spatio-temporal_match_events_in_soccer_competitions)
- Last Row / Liverpool: `liverpool_2019.csv` con 19 goles, 20 fps, balón con `z`, "no todos los jugadores aparecen en cada jugada", recogido "principalmente para animaciones" y "sin la precisión necesaria para investigación"; sin licencia — [Last-Row](https://github.com/Friends-of-Tracking-Data-FoTD/Last-Row)
- DFL Bundesliga Data Shootout (Kaggle, 30-07 a 13-10-2022, DFL + AWS + Sportec): vídeos de 8 partidos en train, etiquetas Play / Throw-In / Challenge con marcadores start/end; sin tracking — fuentes de terceros: [notebook de EDA](https://xxxxyyyy80008.github.io/3_machine_learning/1_data_processing/video%20data/dfl-exploratory-data-analysis-and-video-processing.html), [habr](https://habr.com/en/articles/848468); la página oficial de Kaggle no se pudo leer (solo JavaScript).
- Impect open data: eventos JSON de la Bundesliga 2023/24, sin tracking, `LICENSE.pdf` con atribución — [ImpectAPI/open-data](https://github.com/ImpectAPI/open-data)
- BePro: no tiene open data propio; su sistema es el origen de SoccerTrack v2 y OpenSTARLab tiene loaders "BePro (SoccerTrackv2)" — [OpenSTARLab](https://openstarlab.readthedocs.io/en/latest/Pre_Processing/Sports/Phase_data/Data_Provider/Football/Bepro/contents.html)
- Second Spectrum: no se encontró muestra abierta; kloppy y floodlight tienen parsers — [kloppy](https://github.com/PySport/kloppy), [floodlight](https://github.com/floodlight-sports/floodlight)
- Signality/Hammarby: el README de Metrica menciona "3 partidos del Hammarby por Signality, tracking y vídeo (no sincronizados), sin eventos" para el curso de David Sumpter — citado vía [búsqueda sobre sample-data](https://github.com/metrica-sports/sample-data); enlace de descarga no verificado.

#### Herramientas de código abierto y su lógica de posesión
- kloppy 3.19.1 (BSD-3): `BallState` = {`alive` "Ball is in play", `dead` "Ball is not in play"}; `DataRecord.ball_owning_team` "el equipo que tenía la posesión" y `ball_state`; flags `BALL_OWNING_TEAM`/`BALL_STATE` — [kloppy common.py](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/domain/models/common.py); tabla de proveedores con datos públicos: Impect, Metrica, PFF, SkillCorner, Sportec, StatsBomb (con 360), Wyscout — [PySport/kloppy](https://github.com/PySport/kloppy), [kloppy.pysport.org](https://kloppy.pysport.org/). kloppy no calcula posesión: la copia del proveedor (Sportec, SkillCorner, PFF, StatsBomb) o la deja a `None` (Metrica).
- floodlight (MIT): objetos `Code` "p. ej. información de posesión"; parsers Tracab, DFL/STS, Kinexon, Second Spectrum, SkillCorner, StatsPerform (tracking) y DFL/STS, Opta, Second Spectrum, Sportradar, StatsPerform, StatsBomb (eventos); datasets públicos EIGD-H, IDSSE, StatsBomb; los parsers de Tracab, DFL/STS y Second Spectrum devuelven Codes; no hay función que calcule posesión desde posiciones — [floodlight-sports/floodlight](https://github.com/floodlight-sports/floodlight)
- databallpy (JOSS 2026): posesión de equipo desde eventos sincronizados: "el equipo con posesión es el del último evento con balón" (pass, dribble, shot) y solo cambia con un evento del rival con `outcome == 1`; se escribe `team_possession` por frame; validado contra el `team_possession` de Tracab en frames `alive`; posesión individual según Vidal-Codina et al. (2022): candidato = jugador más cercano dentro de `pz_radius` 1,5 m, validado si el balón cambia de dirección > `ba_threshold` 10° o de velocidad > `bv_threshold` 5 m/s, columna `player_possession`; depende de la calidad del balón — [databallpy: Team and Player Possession](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html); sincronización con Needleman-Wunsch y `get_open_game()` para IDSSE; "para Metrica la posesión se pone a None" — [alek050/databallpy](https://github.com/alek050/databallpy)
- socceraction: SPADL/Atomic-SPADL, VAEP, xT y loaders StatsBomb/Wyscout/Opta; sin posesión por fotograma ni tracking — [socceraction docs](https://socceraction.readthedocs.io/en/latest/)
- codeball (Metrica, MIT, "very much WIP"): solo datasets Metrica Elite vía kloppy; "patterns" sobre eventos (p. ej. pases al área); sin cálculo de posesión — [metrica-sports/codeball](https://github.com/metrica-sports/codeball)
- unravelsports `KloppyPolarsDataset`: si el proveedor no trae posesión la infiere por distancia al balón con `ball_carrier_threshold` 25 m por defecto, avisando de resultados inesperados con balón disputado o en el aire; columna `ball_state`; recomienda `only_alive=True` o `include_empty_frames=False` para proveedores distintos de Sportec — [unravelsports docs](https://unravelsports.readthedocs.io/en/latest/api/soccer/dataset.html)
- PathCRF (KDD 2026, MPL-2.0): infiere la "ruta de posesión" (jugador en posesión o balón viajando entre dos) a partir de trayectorias de jugadores sin usar el balón, con Set Attention + CRF; usa solo el Sportec Open DFL con eventos sincronizados por ELASTIC; `datatools/preprocess.py` produce "tracking con estados de posesión verdad por fotograma" — [hyunsungkim-ds/pathcrf](https://github.com/hyunsungkim-ds/pathcrf), [arXiv 2602.12080](https://arxiv.org/abs/2602.12080)

### Inferences
- La verdad del reto (H/A/D) se reproduce de forma literal solo en IDSSE (`BallStatus`=0 → D; si no, `BallPossession` 1 → H, 2 → A) y, con una definición distinta (eventos "on the ball" que persisten), en PFF.
- SkillCorner solo cubre H/A: su `ball_state` en kloppy es un artefacto ("vivo si hay poseedor"), no una medición; D habría que derivarlo de las fases de juego (solo existen con balón en juego) o de los eventos dinámicos.
- SoccerTrack v2 es el único dataset amateur con vídeo panorámico, posiciones por fotograma y estado dentro/fuera del balón; el poseedor habría que derivarlo de los 21.428 eventos con equipo/actor, y su balón no es una medición.
- Metrica es el dataset más parecido en formato al reto (mismo exportador EPTS, 25 fps, 105×68, 0–1), pero su verdad hay que construirla desde los eventos (BALL OUT/SET PIECE/END HALF → D; pases/recuperaciones → H/A).

### Gaps
- No se pudo abrir el formulario ni los términos de uso del dataset de PFF FC/Gradient; tampoco si incluye `BallStatus` explícito fuera de `game_event`.
- El 2.º fichero del `competitions.json` no dice cuántos partidos hay por competición; el PDF de licencia de StatsBomb no se pudo leer.
- La ficha de SoccerTrack v2 no indica el formato (CSV/JSON) de `ball/` ni de `bas/`, ni si el estado dentro/fuera está en los ficheros publicados o solo en `raw/`.
- No se encontró ningún dataset 2025-2026 nuevo con posesión por fotograma más allá de IDSSE y PFF; de los 35 artículos que citan IDSSE en Semantic Scholar, los relevantes son SoccerTrack v2, FOOTPASS, PathCRF, ELASTIC (2508.09238 y 2608.30227), VERSA (2601.21981) y DataBallPy (JOSS) — [citas de IDSSE](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s41597-025-04505-y/citations?fields=title,year,externalIds,venue&limit=100). Un trabajo con etiquetas por fotograma pass/possession/uncontrolled/out of play (arXiv 2512.19528) usa datos propietarios.

## Pregunta 2: ¿Cuáles son de tracking automático sin corregir (como la entrada del reto) frente a corregido a mano (como la verdad)?

### Takeaway
Automático y sin corregir: SkillCorner (broadcast), el propio ATD del reto, FOOTPASS y, en la práctica, SoccerTrack v2 (BePro sin auditoría). Corregido o curado: IDSSE (TRACAB con operadores), PFF (refinado manual), SoccerNet-Tracking/GSR, SoccerTrack v1 y TeamTrack (cajas anotadas). Ninguno de los públicos combina tracking automático bruto y verdad corregida del mismo partido, que es lo que haría falta para simular el reto.

### Cited Findings
- SkillCorner: "collected from broadcast video using computer vision and machine learning", "some data points are erroneous", ~97 % de identidades correctas, `is_detected` distingue detectado de extrapolado — [SkillCorner/opendata](https://github.com/SkillCorner/opendata); la versión 2019/20 "no ha recibido corrección manual" — [skillcorner.github.io/opendata](https://skillcorner.github.io/opendata/)
- Reto: "automatic and uncorrected, with gaps, fragmented tracks and identity switches", 46 slots que "aparecen, desaparecen e intercambian identidad", balón ausente sobre todo en ocultaciones, áreas pobladas y balones aéreos — [DATA.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md)
- IDSSE: TRACAB Gen5 óptico, identidades manuales previas, operadores en directo y control de calidad final; el artículo no describe corrección manual de coordenadas — [PMC11787359](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/)
- PFF/Gradient: "automatically extracted from broadcast footage and manually refined by annotators" — [SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR)
- SoccerTrack v2: posiciones del sistema BePro, "inspeccionadas visualmente y con comprobaciones puntuales, sin auditoría cuantitativa"; eventos corregidos a mano — [arXiv 2508.01802v2](https://arxiv.org/html/2508.01802v2)
- SN-GSR: líneas del campo anotadas a mano en keyframes e interpoladas; tracking heredado de SN-Tracking (verdad anotada) — [arXiv 2404.11335](https://arxiv.org/html/2404.11335v1)
- FOOTPASS: tracking de un pipeline propio; no se describe corrección — [arXiv 2511.16183](https://arxiv.org/html/2511.16183)
- Metrica: metadatos indican tracking "Optical"; el README no dice si está corregido — [metadata.xml](https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_3/Sample_Game_3_metadata.xml), [sample-data](https://github.com/metrica-sports/sample-data)

### Inferences
- Para simular la entrada del reto a partir de datos corregidos (IDSSE, PFF) habría que degradarlos: borrar el balón en ~41 % de los frames (sesgado hacia áreas pobladas y balones aéreos), fragmentar las pistas en slots anónimos y añadir intercambios de identidad.
- SkillCorner es el más parecido a la entrada (automático, balón ausente, identidades imperfectas) pero a 10 fps y con poseedor sin estado del balón; es el candidato natural para validar un modelo sin verdad corregida.
- SoccerTrack v2 se parece más a nuestro caso de vídeo amateur: cámara panorámica fija, partidos universitarios, 25 fps.

### Gaps
- No se verificó cuánta corrección manual lleva el tracking de Metrica sample-data ni los 10 partidos de la fase 2 del reto.

## Pregunta 3: ¿Qué datasets permiten derivar la verdad del reto de forma mecánica para medir spellF1 fuera del partido entregado?

### Takeaway
De forma directa: IDSSE (7 partidos, ~668 min, dos códigos por fotograma) y PFF WC 2022 (64 partidos vía `game_event` de kloppy). De forma derivada: SkillCorner (H/A por fotograma, D por fases de juego), Metrica (eventos BALL OUT/SET PIECE/END HALF y pases) y SoccerTrack v2 (estado dentro/fuera + eventos con equipo). El scorer del reto (`evaluation/possession_metrics.py`) es MIT y se puede aplicar a cualquiera de ellos.

### Cited Findings
- Scorer y métricas del reto: `evaluation/possession_metrics.py`, `run_evaluation.py`, `validate_submission.py`, código MIT; spellF1 = emparejamiento 1-a-1 de posesiones por solape temporal promediado sobre cinco umbrales; macroF1 por fotograma sobre H, A y D — [README](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/README.md), [OVERVIEW](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/OVERVIEW.md)
- IDSSE: `BallStatus` 0/1 y `BallPossession` 1/2 por fotograma — [PMC11787359](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/); floodlight los expone como `Code` con definiciones {0 Dead, 1 Alive} y {1 Home, 2 Away} — [floodlight.io.dfl](https://floodlight.readthedocs.io/en/latest/_modules/floodlight/io/dfl.html)
- PFF: `ball_state` y `ball_owning_team` por fotograma derivados de `game_event` y persistentes — [kloppy pff.py](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/pff.py)
- SkillCorner: `possession {player_id, group}` por fotograma; fases de juego "definidas solo mientras el balón está en juego" — [SkillCorner/opendata](https://github.com/SkillCorner/opendata)
- Metrica: eventos con `Start Frame`/`End Frame` y tipos `BALL OUT`, `SET PIECE` (KICK OFF, THROW IN, CORNER KICK, FREE KICK), `BALL LOST` con subtipo `END HALF` — [RawEventsData.csv](https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/Sample_Game_1_RawEventsData.csv); databallpy aplica la regla "último evento con balón exitoso" para generar `team_possession` por fotograma — [databallpy](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html)
- SoccerTrack v2: estado dentro/fuera por fotograma y eventos con equipo y actor — [arXiv 2508.01802v2](https://arxiv.org/html/2508.01802v2), [HF](https://huggingface.co/datasets/atomscott/soccertrack-v2)
- PathCRF ya construye "estados de posesión verdad por fotograma" a partir del Sportec Open DFL — [pathcrf](https://github.com/hyunsungkim-ds/pathcrf)

### Inferences
- Con IDSSE se puede construir un banco de pruebas completo (7 partidos) con el mismo scorer; la definición DFL ("jugador que controla el balón") puede diferir de la de Metrica, así que conviene calibrar la duración media de posesión contra los 5 minutos públicos (14,2 s).
- Las posesiones derivadas de eventos (Metrica, PFF, databallpy) cambian solo en eventos exitosos; las de IDSSE cambian en el fotograma en que el operador marca el control, lo que afecta directamente a spellF1 en los bordes.
- Las etiquetas de SkillCorner y PFF se obtienen de proveedores distintos a Metrica, por lo que el sesgo de definición es un riesgo a medir, no a asumir.

### Gaps
- No se comprobó si las fases de juego de SkillCorner cubren todo el tiempo de balón en juego (si hay huecos, D quedaría sobreestimado).
- No se verificó qué define exactamente "control" en el `BallPossession` de DFL (el artículo no da la definición operativa).

## Pregunta 4: ¿Alguno trae vídeo sincronizado con el tracking?

### Takeaway
Sí, pero casi siempre con NDA o sin verdad de posesión: SoccerTrack v2 (vídeo panorámico 4K + posiciones + eventos + estado del balón; CC BY 4.0 gated), SoccerNet-Tracking/GSR (broadcast + cajas, NDA), SoccerNet-GAR (broadcast + tracking PFF en ventanas de 4,5 s, NDA), FOOTPASS (broadcast + tracking propio, NDA), SoccerTrack v1 y TeamTrack (fisheye/dron + cajas). IDSSE, PFF, SkillCorner y Metrica no traen vídeo.

### Cited Findings
- SoccerTrack v2: 20 vídeos panorámicos de mitades, `gsr/` por mitad, `ball/`, `bas/`, `mot/` con clips de 4 min — [HF](https://huggingface.co/datasets/atomscott/soccertrack-v2); 25 fps — [arXiv](https://arxiv.org/html/2508.01802v2)
- SN-Tracking: 12 partidos completos de cámara principal + 200 clips con cajas de jugadores, árbitros y balón — [sn-tracking](https://github.com/SoccerNet/sn-tracking); SN-GSR: 750 frames/clip a 25 fps 1920×1080 con posiciones en el campo — [arXiv 2404.11335](https://arxiv.org/html/2404.11335v1); vídeo protegido por NDA — [soccer-net.org/data](https://www.soccer-net.org/data)
- SoccerNet-GAR: "vídeo broadcast 720p sincronizado y tracking" en ventanas de 4,5 s, NDA — [HF](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR)
- FOOTPASS: vídeo 1080p 25 fps vía SoccerNet con NDA — [arXiv 2511.16183](https://arxiv.org/html/2511.16183)
- IDSSE: "el vídeo no está disponible por restricciones de licencia" — [PMC11787359](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/); Metrica, SkillCorner y PFF no mencionan vídeo — [sample-data](https://github.com/metrica-sports/sample-data), [opendata](https://github.com/SkillCorner/opendata), [PFF blog](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset)
- El reto entrega dos vídeos (táctico y broadcast) con homografía por fotograma — [DATA.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md)

### Inferences
- Para nuestro pipeline de vídeo amateur, SoccerTrack v2 es el único con vídeo abierto (CC BY) + posiciones + estado del balón; serviría para probar un detector de balón muerto y una posesión derivada de eventos sobre vídeo panorámico.
- Los datasets SoccerNet exigen NDA de investigación, incompatible con uso en producto sin negociación.

### Gaps
- No se verificó si el vídeo de SoccerTrack v2 incluye audio ni la resolución exacta de cada mitad.

## Pregunta 5: ¿Qué licencias permiten uso en competición y en investigación/producto?

### Takeaway
Sin trabas: IDSSE (CC-BY 4.0), SkillCorner (MIT) y SoccerTrack v2 (CC BY 4.0, previa aceptación). Con condiciones: StatsBomb (User Agreement con atribución; anuncio de 2018 habla de uso no comercial), Wyscout (CC BY según el paper, "non-commercial only" según D4Science), SoccerNet/FOOTPASS/SoccerNet-GAR (NDA), PFF (formulario, términos no publicados), Metrica y Last Row (sin licencia formal). Los datos del reto se licencian solo para la competición.

### Cited Findings
- IDSSE: CC-BY 4.0 con autorización de la DFL — [PMC11787359](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/); hay que nombrar a la DFL y citar el artículo — [HF](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md)
- SkillCorner: MIT — [opendata](https://github.com/SkillCorner/opendata)
- SoccerTrack v2: CC BY 4.0, acceso gated — [HF](https://huggingface.co/datasets/atomscott/soccertrack-v2)
- StatsBomb: "al usar el repositorio aceptas el User Agreement", citar y usar el logo — [open-data](https://github.com/statsbomb/open-data); "non-commercial use" en el anuncio de 2018 — [blog](https://blogarchive.statsbomb.com/news/statsbomb-release-free-fifa-world-cup-data/)
- SoccerNet: datos con vídeo gated tras aceptar el NDA — [soccer-net.org/data](https://www.soccer-net.org/data); SoccerNet-GAR licencia "other" + NDA — [HF](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR)
- Metrica: sin licencia, "sé responsable" y reconoce la fuente — [sample-data](https://github.com/metrica-sports/sample-data); Last Row: sin licencia, "cita la fuente" — [Last-Row](https://github.com/Friends-of-Tracking-Data-FoTD/Last-Row)
- Reto: código MIT, datos del partido "licenciados por separado solo para esta competición" — [README](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/README.md)
- Wyscout: CC BY según Semantic Scholar — [S2](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/s41597-019-0247-7?fields=title,abstract,year,citationCount,openAccessPdf); "Non-commercial only" en D4Science — [catálogo](https://catalogue.d4science.org/dataset/public_data_set_of_spatio-temporal_match_events_in_soccer_competitions)
- Impect: `LICENSE.pdf`, atribución con logo — [ImpectAPI/open-data](https://github.com/ImpectAPI/open-data)

### Inferences
- Para entrenar y documentar en el reto, IDSSE + SkillCorner + SoccerTrack v2 cubren tracking corregido, tracking automático y vídeo amateur sin riesgo de licencia; PFF conviene solicitarlo y leer los términos antes de usarlo en producto.

### Gaps
- No se leyó el texto de `LICENSE.pdf` de StatsBomb ni de Impect, ni los términos de PFF/Gradient, ni las reglas de datos del Kaggle DFL (página no accesible sin JavaScript); la licencia de los datasets SoccerTrack v1 y TeamTrack (Kaggle) tampoco se pudo verificar.
