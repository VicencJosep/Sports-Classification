# Datasets: los que usamos y los que existen fuera

Documento de referencia, estado a 1 de noviembre de 2026 (entrada de FOOTPASS / SN-PCBAS-2026 revisada el 6 de noviembre).

Responde a dos preguntas: qué conjuntos de datos usa hoy cada rama y funcionalidad de Sports-Classification, y qué otros conjuntos públicos de fútbol (y, de forma secundaria, de baloncesto) podemos usar ya o nos pueden servir más adelante. Las particiones train/val/eval no se repiten aquí: la referencia sigue siendo `docs/particion_datasets.md`. Las licencias se citan de forma literal cuando la fuente las da; cuando un dato no se pudo comprobar se dice "no verificado". Los enlaces de cada dataset van en el cuerpo; la lista completa de fuentes está en la sección 9.

**Estructura (2026-11-09).** Aquí quedan el resumen (§1), lo que usamos hoy (§3) y los pasos sugeridos (§7). El resto
está, con la misma numeración, en `docs/datasets_catalogo/` (abrir solo el fichero que haga falta):

| § | Fichero | Contenido |
|---|---|---|
| 2 | `docs/datasets_catalogo/2_cinco_propuestos.md` | DFL Shootout, SoccerHigh, StatsBomb, DeepSportradar y SkillCorner: ficha y veredicto |
| 4A | `docs/datasets_catalogo/4A_usar_ya.md` | Catálogo externo: lo que sirve ya |
| 4B | `docs/datasets_catalogo/4B_util_en_el_futuro.md` | Catálogo externo: útil más adelante (el más largo, 30 KB) |
| 4C, 4D | `docs/datasets_catalogo/4C_4D_descartar_y_no_existe.md` | Descartados y nombres que no existen como dataset |
| 5 | `docs/datasets_catalogo/5_baloncesto.md` | Baloncesto, con el inventario local del antiguo §3.5 |
| 6 | `docs/datasets_catalogo/6_licencias.md` | Licencias y uso comercial |
| 8 | `docs/datasets_catalogo/8_huecos.md` | Huecos de la investigación |
| 9 | `docs/datasets_catalogo/9_fuentes.md` | Fuentes por sección |

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

### 3.5 Baloncesto

Fuera del repo desde el 2026-11-09 (`/media/dd3/ner-training/basketball_Sports-Classification`). Inventario en `docs/datasets_catalogo/5_baloncesto.md`.

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

Licencias de código que afectan a los pesos: T-DEED es GPL-3.0 y `SNB_pano1`/`SNB_tv1` son derivados suyos (`docs/archivo/estudio_deteccion_tiros.md`, `docs/hoja_de_ruta.md`).

### 3.7 Hallazgos del inventario

- `/mnt/datasets/SoccerNet` está montado hoy en el host con las 1.000 mitades de los 500 partidos; `docs/particion_datasets.md:20` sigue diciendo que a 8-oct-2026 el montaje no estaba disponible. El contenedor no lo ve: usar esos vídeos exige copiarlos al proyecto. Los `Labels-v2.json` no están junto a los vídeos en `/mnt`, sino en `datasets/SoccerNet/<liga>/`.
- Los 10 partidos de `sn_tuning_videos` están todos en la lista de entrenamiento del SVC: no valen para validar la fusión V-JEPA2 ni como "limpios" para T-DEED. La validación de highlights sin marcador pide 3-5 partidos sin asignar copiados desde `/mnt`.
- Derivados sin generador en el repo: `dataset_vjepa_backup/`, `clips_analisis/`, el entrenamiento del SVC, la descarga de SoccerNet Tracking y de los vídeos de `/mnt`, y el origen de los 28 clips de tiro sin partido en el nombre. `tdeed_pano/frames/` no está en disco (se regenera con `tdeed_finetune_data.py`).
- La única licencia explícita guardada con un dataset es la de SoccerTrack v2 (CC BY 4.0) y las de los dos Roboflow de baloncesto (CC BY 4.0). El repo solo anota que los vídeos de SoccerNet, Ball Action Spotting y Tracking exigen la NDA; el texto de la NDA no está guardado y todos los derivados de esos vídeos (`dataset_vjepa_backup`, `clips_analisis`, `dataset_ball_actions`, `sn_tuning_videos`, `ground_truth/*.mp4`) heredan esa condición.
- Cero anotaciones de eventos en el dominio del producto (juveniles): el CSV del Espanyol solo tiene acciones; faltan los dos juveniles de evaluación; faltan tiros con equipo y jugador para medir `SNB_pano1`/`SNB_tv1`.
- Sin cifra final limpia del pipeline en TV: `challenge2` y R3D-18 v2 entrenaron con 6 de 7 EFL; el detector de balón de TV no tiene test intacto ni test del dominio real (faltan 300-500 frames de YouTube etiquetados).

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
