> Parte de [docs/datasets.md](../datasets.md): catálogo a 2026-11-01, con la numeración original del documento.

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
