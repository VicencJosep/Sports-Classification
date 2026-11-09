> Parte de [docs/datasets.md](../datasets.md): catálogo a 2026-11-01, con la numeración original del documento.

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
