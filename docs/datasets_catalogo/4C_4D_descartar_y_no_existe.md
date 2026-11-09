> Parte de [docs/datasets.md](../datasets.md): catálogo a 2026-11-01, con la numeración original del documento.

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
