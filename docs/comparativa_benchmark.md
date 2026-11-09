# Comparativa técnica con el estado del arte y plan de benchmarking

Fecha: 2026-11-01. Rama analizada: `main` (HEAD 9234c43). No se ha tocado código.

Estructura (2026-11-09): este fichero es el resumen, el plan y el índice; los resultados de §7 (medidos del
2026-11-02 al 2026-11-09) están uno por fichero en `docs/comparativa_benchmark/`.

Convenciones: **[M]** medido por nosotros (con fichero de origen), **[P]** publicado por terceros (con URL),
**[NV]** no verificado. Ninguna cifra [P] es comparable con otra salvo que se diga explícitamente: cada una
sale de un dataset, split, detector o tolerancia distintos.

---

## 0. Resumen

- **Tracking, medido después de escribir este resumen** (§7.2, §7.4, §7.10): con nuestro detector híbrido, ByteTrack
  daba 56,4 HOTA / 66,0 IDF1; `main` usa desde el 2026-11-06 BoT-SORT + GMC + ReID OSNet (59,4 / 70,5). Deep-EIoU
  (62,4 / 76,8) y Deep-EIoU + GTA-Link (64,3 / 81,3) siguen por encima. El punto siguiente es el estado a 2026-11-01.
- **Tracking**: nuestro tracker es ByteTrack puro (`supervision`), sin ReID, sin compensación de cámara y sin
  tracking de balón. **No existe ninguna medida de tracking de nuestro pipeline** (ni HOTA, ni MOTA, ni IDF1, ni
  GS-HOTA). Es el hueco principal: sin ella no se puede comparar con nada de la lista.
- **Eventos**: T-DEED, que ya usamos, es la familia ganadora de Ball Action Spotting 2024 y la base del baseline de 2025.
  CALF y NetVLAD++ son baselines de 2020-21 a 2 fps con tolerancias de segundos. FOOTPASS es un dataset sin pesos.
  Ninguno se ha evaluado en panorámica ni en fisheye.
- **Merece la pena ejecutar** contra lo nuestro:
  - tracking: Deep-EIoU (+GTA-Link), BoT-SORT, sn-gamestate/TrackLab, FutTraffic y el detector YOLOv8x6 de
    SoccerMaster;
  - eventos: el baseline de equipo de sn-teamspotting (tras verificar sus pesos), y nuestro T-DEED limpio medido con
    la métrica oficial.
- **Spotters con pesos públicos, tabla única (§7.11, 2026-11-08)**: en TV (test oficial de SN-BAS, 2 partidos) T-DEED
  `SoccerNetBall_test` da 61,2 mAP@1 frente a 55,5 de sn-teamspotting, 54,9 de AdaSpot-b (10 clases) y 34,5 del E2E-Spot a
  2 fps. En panorámica (eval de SoccerTrack v2) la cámara virtual "ideal" que sigue el balón anotado da lo mismo que la nuestra
  (±2 puntos; `SNB_pano1` 38,4 frente a 38,3): el hueco con TV es de dominio, salvo en tiros (+15 AP en 128057 con la cámara
  ideal). Coste con GPU libre: todos los modelos a 796x448 van a 12-14× tiempo real de punta a punta (manda la
  decodificación). Ningún otro proyecto con pesos hace spotting con instante (SoccerMaster y UniSoccer clasifican clips;
  sn-gamestate no tiene eventos; los ganadores de 2025-26 no publican pesos).
- **Punto de trabajo y metraje nuevo (§7.12, 2026-11-09)**: con `eval_prf.py` (P/R/FP al umbral óptimo de cada modelo, ±0,5 s)
  en TV, panorámica y **SN-PCBAS-2026 val** (3 partidos de 2023/24, nuevos para todos los modelos). En PCBAS T-DEED 200MF
  mantiene el mejor mAP (50,0 frente a 41,1 de AdaSpot-b, por tiro/banda/centro) pero emite 2,3× más falsos positivos
  (2.600 frente a 1.121) y su F1 cae de 80,5 a 64,0 (AdaSpot-b 78,2 → 71,5): ordena mejor, calibra peor fuera de la EFL.
- **Cabeza de 17 eventos de SoccerNet-v2 (§7.13, 2026-11-09)**: en 20 partidos limpios del test oficial, el 200MF da 21,4 de
  Average-mAP tight y el 800MF de producción 25,3 (dedicados: 62-73). Córner (P 81 / R 74), banda, balón fuera y libre
  directo son utilizables; tarjeta, cambio, fuera de juego y saque inicial no (AP < 11). En PCBAS el modelo de producción
  da 53,1 de mAP@1 y F1 65,2 con 2.136 falsos positivos: mismo problema de calibración que el 200MF fuera de la EFL.
- **No merece la pena**: CALF, NetVLAD++, FOOTPASS como modelo, el tracker MOTIP de SoccerMaster y el detector de
  GTATrack. Los tres últimos no publican pesos.

---

## 1. Qué usa nuestro pipeline (`main`)

| Módulo | Modelo / algoritmo | Dónde |
|---|---|---|
| Detección de personas | SoccerMaster `SoccerNetGSR_Detection`: Deformable-DETR sobre SigLIP2, clips de 30 frames a 512x512, umbral 0,1. Da rol y dorsal. | `core/tracking.py:317-490` |
| Detección de balón | YOLO11x COCO (clase 32) a 1280 px, conf 0,1. El `yolo11s_snt_ball.pt` afinado existe pero no está por defecto. | `core/tracking.py:242-314`, `config.py:46` |
| Tracking | **Desde el 6-nov-2026**: BoT-SORT de ultralytics (Kalman + GMC sparseOptFlow + ReID OSNet x1.0 "sports") con NMS previo 0,7, entrada conf ≥ 0,1 y área ≥ 300 px²; emite la caja detectada. Antes (y aún como opción `tracker="bytetrack"`): `supervision.ByteTrack`, activación 0,25, buffer 30, matching 0,8, prefiltro conf ≥ 0,3, sin ReID ni GMC. Medido en §7.10. | `core/tracking.py` (`BotSortTracker`, `PlayerTracker`), `core/osnet.py` |
| Balón | Sin tracking: mejor detección por frame (≥ 0,15) más `StaticBallFilter` contra logos | `core/tracking.py:531-587,798-826` |
| Equipos | PRTReId (BPBReID + HRNet-W32, 256-d) por tracklet y KMeans k=2. Es un port de sn-gamestate. | `core/classification.py` |
| Dorsal | Solo la cabeza de SoccerMaster, por detección, sin votación por tracklet | `core/tracking.py:179,693` |
| Homografía | PnLCalib (HRNet kp+líneas) cada 5 frames. Calibra 34-71 % de los frames de TV [M] y aguas abajo no se usa. | `core/homography.py`, `docs/hoja_de_ruta.md:42` |
| Eventos | T-DEED `SoccerNetBall_challenge2` (RegNetY-800MF, 796x448, 12,5 fps). Cabeza BAS de 12 clases más cabeza SN-v2 de 17 clases. Umbrales automáticos por partido. | `core/ball_spotting.py`, `config.py:111-209` |
| Etiqueta de acciones | R3D-18 combinado con T-DEED | `core/ball_actions.py` |
| Panorámica | Solo scripts experimentales: cámara virtual Viterbi+L1, `SNB_pano1`, `yolo11s_pano` (balón). Nada desde `main.py`. | `scripts/shot_benchmark/`, `docs/hoja_de_ruta.md:41` |
| Fisheye | No hay código | — |

Herramientas de evaluación actuales:

- `scripts/evaluate_match.py`: acciones ±1 s, eventos ±3 s, equipo y jugador.
- `scripts/evaluate_spotting.py` y `scripts/shot_benchmark/`: AP de detección clásico a ±1/3/5 s. **No** es la mAP
  oficial de SoccerNet.
- `Soccer_Master/tracklab` incluye una copia de TrackEval, pero no se usa.

Cifras internas de referencia [M]:

- Tiros, T-DEED `SoccerNetBall_test` en 3 partidos EFL limpios: AP ±3 s 75,6 % y AP ±1 s 70,0 %
  (`docs/archivo/estudio_deteccion_tiros.md:48,58`).
- Acciones en Middlesbrough (20 min): F1 88,4, equipo 68,1 % (`docs/hoja_de_ruta.md:24`).
- Panorámica, eval de SoccerTrack v2: tiros AP ±3 s 41,6 → 55,2 % con `SNB_pano1` (`docs/archivo/estudio_deteccion_tiros.md:218-245`).

**Ojo con la contaminación**: el checkpoint de producción `challenge2` se entrenó con todos los splits de SN-BAS.
Las medidas limpias en EFL deben hacerse con `SoccerNetBall_test`. T-DEED preentrenó con los splits `train` + `val` de
SoccerNet-v2 (400 partidos); el `test` oficial (100 partidos) está limpio para su cabeza de eventos (§7.11, corregido el 2026-11-08).

---

## 2. Proyectos de tracking

| Proyecto | Tipo | ¿Fútbol? | Pesos verificados | Licencia | Cámara | Cifra publicada [P] |
|---|---|---|---|---|---|---|
| **FutTraffic** ([repo](https://github.com/danrublop/futtraffic)) | Sistema completo (motor + API + UI) | Sí, fútbol base | N/A: usa RF-DETR COCO de stock y no publica pesos propios | Apache-2.0 | Panorámica fija | HOTA 0,3836 / IDF1 0,4022 en ST2 128057, GT cada 2 frames, medida propia ([benchmarks.md](https://github.com/danrublop/futtraffic/blob/main/docs/benchmarks.md)) |
| **SoccerMaster** ([repo](https://github.com/haolinyang-hlyang/SoccerMaster), [arXiv 2512.11016](https://arxiv.org/html/2512.11016)) | Modelo base con cabezas por tarea, más el pipeline de datos SoccerFactory sobre TrackLab | Sí | Parcial, en [HF xleprime/SoccerMaster](https://huggingface.co/xleprime/SoccerMaster): backbone, detección GSR, keypoints, líneas y YOLOv8x6 (ya en disco). **Sin pesos de la cabeza MOT (MOTIP).** | Ambigua: el repo no tiene licencia, HF dice apache-2.0 e incluye GPL | TV | MOT en SoccerNet-Tracking: HOTA 59,1 (split no indicado [NV]). SoccerFactory: GS-HOTA 64,1 en el test de SN-GSR. |
| **GTATrack** ([repo](https://github.com/ron941/GTATrack-STC2025), [arXiv 2602.00484](https://arxiv.org/html/2602.00484)) | Pipeline de concurso: YOLOv11x afinado + Deep-EIoU + GTA-Link (offline) | Sí | Parcial: el OSNet deportivo está en el repo. **El YOLOv11x afinado no está publicado.** | Sin licencia (todos los derechos reservados) | Panorámica 4096x1080 (ST2; el paper la llama «fisheye») | HOTA 0,60, 1.º del SoccerTrack Challenge 2025 sobre 128057+132831 |
| **sn-gamestate / TrackLab** ([sn-gamestate](https://github.com/SoccerNet/sn-gamestate), [tracklab](https://github.com/TrackingLaboratory/tracklab)) | Sistema GSR completo más framework modular | Sí / genérico | **Sí**, en Zenodo con descarga automática y comprobado con 200 OK (PRTReId, NBJW, PnLCalib, HRNet) | GPL-3.0 / MIT | TV | Baseline GS-HOTA 29,01, ganador 2025 63,90, ambos en el set *challenge* ([arXiv 2508.19182](https://arxiv.org/html/2508.19182v1)). Nuestro `docs/datasets.md:248` da 22,26 para el baseline, quizá de otra edición [NV]. |
| **Deep-EIoU** ([repo](https://github.com/hsiangwei0903/Deep-EIoU), [arXiv 2306.13074](https://arxiv.org/html/2306.13074)) | Algoritmo de tracking para deporte: IoU expandida iterativa + OSNet, sin Kalman | Deportes | **Sí**, [Drive](https://drive.google.com/drive/folders/1wItcb0yeGaxOS08_G9yRWBTnpVf0vZ2w) con tamaños listados: YOLOX-X de 756 MB y OSNet `sports_model.pth.tar-60` de 29 MB | Sin licencia | TV | HOTA 77,2 en el test de SportsMOT. 85,4 en SoccerNet-Tracking **con detecciones oráculo**. |
| **BoT-SORT** ([repo](https://github.com/NirAharon/BoT-SORT)) | Tracker genérico: Kalman + GMC + ReID opcional | No | Sí (MOT17/20, no de deporte). Viene integrado en ultralytics. | MIT | Peatones | HOTA 65,0 en el test de MOT17. 76,999 en SoccerNet con oráculo, según el paper de Deep-EIoU. |
| **ByteTrack** ([repo](https://github.com/ifzhang/ByteTrack)) | Asociación genérica en dos pasadas | No | Sí (YOLOX MOT17). Lo usamos vía `supervision`. | MIT | Peatones | HOTA 63,1 en el test de MOT17. 62,8 en el test de SportsMOT. 71,5 en SoccerNet con oráculo. |

### Notas importantes

- **Corrección a `docs/hoja_de_ruta.md:115`.** El «ByteTrack 0,42» no es una línea base medida en 128057. Es
  ByteTrack sobre el detector afinado de GTATrack, en *su* validación (tabla 3 de GTATrack). Por eso
  «FutTraffic 0,38 frente a ByteTrack 0,42» compara conjuntos distintos.
  - El propio FutTraffic mide en 128057, con su arnés: su sistema 0,3836, su tracker sobre detecciones del ganador
    0,5149 y el pipeline del ganador 0,5583.
  - Su conclusión es que alrededor del 75 % del hueco es el detector.
- **Las medidas con oráculo** (Deep-EIoU 85,4, BoT-SORT 77,0, ByteTrack 71,5 en SoccerNet) aíslan la asociación.
  Son las únicas cifras de tracking de TV algo comparables entre sí, porque salen de la misma tabla y el mismo split.
- **GTATrack tiene incoherencias internas**: el texto da 0,52→0,55 y la tabla 0,42→0,54. Usar solo la tabla del
  leaderboard (0,60).
- **El OSNet deportivo** `sports_model.pth.tar-60` está en Deep-EIoU y en GTATrack, ambos sin licencia. Vale para
  benchmark interno, no para producto.

---

## 3. Proyectos de detección de eventos

### 3.1 Qué métricas son comparables

| Familia | Clases | Métrica | Tolerancia |
|---|---|---|---|
| SoccerNet-v2 Action Spotting | 17 | Average-mAP *loose* (5-60 s) o *tight* (1-5 s) | segundos |
| Ball Action Spotting 2024 | 12 | mAP@1 | ventana total de 1 s (±0,5 s) |
| Team BAS 2025 | 12 × equipo | Team-mAP@1 | ±0,5 s |
| FOOTPASS / PCBAS 2026 | 8 + equipo + dorsal | Macro-F1 @ conf 0,15 | ±12 frames (±0,48 s) |

Nuestro AP de tiros «±3» está en **segundos** (`scripts/shot_benchmark/common.py:15`) y es un AP de detección
agrupado, no la mAP oficial. Lo más cercano a la Shot AP 70,07 publicada de T-DEED en el test de SN-BAS (±0,5 s) es
nuestro AP ±1 s = 70,0 % [M], que sigue sin ser comparable: otra tolerancia y otros partidos.

### 3.2 Tabla

| Proyecto | Tipo | Entrada | Pesos verificados | Licencia | Cifra publicada [P] |
|---|---|---|---|---|---|
| **T-DEED** ([repo](https://github.com/arturxe2/T-DEED), [arXiv 2404.05392](https://arxiv.org/abs/2404.05392)) | Modelo E2E especializado | Frames crudos; BAS a 796x448 y 12,5 fps | Sí, [Drive](https://drive.google.com/drive/folders/1sxZalU_hCwL8ITZCU9VqSWE8dB94lJty) (subcarpetas listadas). Ya en disco. | GPL-3.0 | BAS 2024 challenge mAP@1 73,39 (1.º, ensemble). Test de SN-BAS mAP 58,91 con Shot 70,07 y Goal 11,94. SN-AS test tight 63,42 ([T-DEED Revisited](https://arturxe2.github.io/files/TDEED_Revisited.pdf)). |
| **sn-teamspotting** ([repo](https://github.com/SoccerNet/sn-teamspotting)) | Baseline T-DEED con cabeza de equipo | Igual que T-DEED | URL en el README ([Drive](https://drive.google.com/drive/folders/16IqSkctIGp76ZYKKvJvMB_ggHcQsessM)) **[NV]** | GPL-3.0 | Test 47,18 Team-mAP@1 / 53,59 mAP@1. Ganador de TBAS 2025 (dude.k): 60,03, sin pesos publicados ([arXiv 2508.19182](https://arxiv.org/html/2508.19182v1)). |
| **FOOTPASS** ([repo](https://github.com/JeremieOchin/FOOTPASS), [arXiv 2511.16183](https://arxiv.org/abs/2511.16183)) | Dataset + 3 baselines | Vídeo 352x640 **más tracklets e identidades** | **No**: solo código. Datos en HF con NDA, 236 GB (28,5 GB con los vídeos a 352x640). | Anotaciones y baselines CC BY-NC 4.0 (README); fichero LICENSE Apache-2.0; paper CVIU CC BY 4.0 | F1 micro 67,5 en su test (TAAD+DST), que en Macro-F1 son 46,4: el 46,41 del baseline del reto, mismo conjunto. Ganador de PCBAS 2026: 58,94. |
| **SoccerNet Action Spotting** ([sn-spotting](https://github.com/SoccerNet/sn-spotting)) | Benchmark | Features o frames | — | MIT | 2023 tight 71,31 (MEDet). Test tight 73,10 (COMEDIAN, con features de Baidu). |
| **CALF** ([arXiv 1912.01326](https://arxiv.org/abs/1912.01326)) | Baseline | Features ResNet-152 con PCA512 a 2 fps. Extractor disponible (TF 2.3 / CUDA 10.1). | Sí, dentro de sn-spotting, con script de inferencia sobre vídeo | MIT / Apache-2.0 | SN-v2 test: 40,7 loose / 14,10 tight |
| **NetVLAD++** ([arXiv 2104.06779](https://arxiv.org/abs/2104.06779)) | Baseline | Igual que CALF. Sin script para vídeo arbitrario. | Sí, dentro de sn-spotting | MIT | SN-v2 test: 53,4 loose / 11,51 tight |

Ninguno se ha evaluado en panorámica ni en fisheye. Para panorámica, nuestra cifra con `SNB_pano1` es la única que conocemos.

---

## 4. Qué merece la pena ejecutar

| Prioridad | Candidato | Contra qué | Por qué | Coste estimado |
|---|---|---|---|---|
| 0 | **Arnés de tracking (TrackEval)** | — | Requisito previo: hoy no podemos medir tracking | 1 d |
| 1 | **Deep-EIoU** y **BoT-SORT** (con GMC y ReID) vs nuestro ByteTrack, con las **mismas** detecciones | `core/tracking.py` | Mejora barata de asociación; con oráculo publicado van +5,5 / +14 HOTA sobre ByteTrack | 1-2 d |
| 2 | **GTA-Link** como post-proceso offline sobre nuestros tracklets | ídem | En ST2 aporta +0,06 HOTA [P]. Encaja con el modo partido completo. | 1 d |
| 3 | **sn-gamestate** (baseline de TrackLab) vs nuestra cadena completa en SN-GSR | tracking, equipo, dorsal y homografía | Es el único benchmark que mide **toda** nuestra cadena (GS-HOTA). Pesos verificados. | 2-3 d |
| 4 | **Detector YOLOv8x6 de SoccerMaster** (ya en disco) vs nuestro híbrido | detección de personas | Mismo autor y sin coste de descarga; aísla DetA | 0,5 d |
| 5 | **FutTraffic** completo en ST2 eval | rama panorámica (que hoy no tiene tracking de personas) | Único sistema abierto para panorámica fija, Apache-2.0 | 1-2 d |
| 6 | **T-DEED limpio con mAP@1 oficial** en Stoke+Reading | nuestros scripts de spotting | Situar nuestro modelo en la escala publicada | 0,5 d |
| 7 | **sn-teamspotting** (si los pesos se verifican) | atribución de equipo de las acciones (68 % [M]) | Mide equipo por acción frente a nuestro PRTReId+KMeans | 1 d |
| — | CALF, NetVLAD++ | — | **No ejecutar.** Granularidad de segundos (tight 11-15) y SN-v2 contaminado para T-DEED. | — |
| — | FOOTPASS | — | **No ejecutar.** No hay pesos y necesita tracklets e identidades. Interesa como dataset futuro. | — |
| — | SoccerMaster MOTIP, detector de GTATrack | — | Sin pesos publicados. Sus cifras solo sirven de techo de referencia. | — |

---

## 5. Plan de benchmarking

### 5.1 Reglas comunes (para que sea justo)

1. **Mismos vídeos y mismos frames.**
   - Cada vídeo se decodifica una sola vez a JPG a la resolución nativa (formato MOT `img1/`).
   - Todos los métodos leen esos frames y aplican su propio preprocesado; se anota cuál es.
   - Mismo fps: 25 en origen. Si un método submuestrea, el GT se evalúa al mismo stride para todos.
2. **Mismo GT y mismo evaluador.**
   - Tracking: TrackEval (copia en `Soccer_Master/tracklab`, versión fijada) con HOTA, DetA, AssA, MOTA e IDF1.
   - GSR: GS-HOTA de tracklab ≥ 1.3.24, que corrige la evaluación.
   - Eventos: mAP@1 oficial (paquete `SoccerNet`) más nuestras AP ±1/±3 s.
3. **Dos regímenes de tracking:**
   - (a) **detecciones comunes**: el mismo fichero `det.txt` para todos los trackers, para aislar la asociación;
   - (b) **sistema completo**: cada uno con su detector.

   Sin esta separación, el detector se come la comparación; en panorámica es el 75 % del hueco según FutTraffic.
4. **Particiones fijas** (`docs/particion_datasets.md`). Umbrales e hiperparámetros solo en val; una única pasada en eval.
   - Nada de afinar sobre 128057/132831.
   - Nada de usar `challenge2` en EFL, porque está contaminado.
5. **Coste**: se registran fps, VRAM pico y tiempo por minuto de vídeo en la misma GPU, con la GPU libre (sin vLLM ni Ollama cargados).
6. **Clases**: el HOTA de jugadores se calcula **sin balón**; el balón se evalúa aparte (recall y precisión a ≤ N px).
7. **Resultados** en un JSON por método y escenario bajo `output/benchmark/`, más una tabla en este documento.

### 5.2 Escenarios

| Escenario | Cámara | Datos (en disco) | Métodos | Métrica |
|---|---|---|---|---|
| **T1. Tracking TV** | Broadcast | SoccerNet-Tracking test, 49 secuencias de 30 s (`datasets/SoccerNet/tracking/`) | Régimen (a): ByteTrack-nuestro, BoT-SORT, Deep-EIoU, Deep-EIoU+GTA-Link, más el oráculo para cuadrar con las cifras [P]. Régimen (b): nuestro híbrido, YOLOv8x6 de SoccerMaster + ByteTrack, y YOLOX de Deep-EIoU. | HOTA, DetA, AssA, MOTA, IDF1 |
| **T2. Game state TV** | Broadcast | SN-GSR-2025, etiquetas públicas en HF (`snapshot_download`). Split valid para ajustar y test para medir. | Nuestro pipeline exportado a formato SNGS vs baseline de sn-gamestate | GS-HOTA más sus componentes |
| **T3. Tracking panorámica** | Panorámica fija 4K | SoccerTrack v2: val 117093/132877, eval 128057/132831 (`datasets/soccertrack-v2/mot`) | Régimen (a): detecciones RF-DETR por tiles (las de FutTraffic) para ByteTrack, Deep-EIoU+GTA-Link y C-BIoU. Régimen (b): FutTraffic completo frente a yolo11x COCO por tiles con nuestro tracker. GTATrack 0,60 solo como referencia [P]. | HOTA, DetA, AssA, IDF1. GT cada 2 frames, como FutTraffic. |
| **T4. Fisheye** (opcional) | Ojo de pez 8K | TeamTrack (no está en disco; descarga de Kaggle) | ByteTrack frente al mejor de T1/T3 | HOTA. ByteTrack publicado: 59,3 [P, `docs/datasets.md:263`]. |
| **E1. Eventos TV** | Broadcast | EFL Stoke + Reading (eval limpio) | T-DEED `SoccerNetBall_test` (limpio) frente a sn-teamspotting | mAP@1 oficial de 12 clases, Team-mAP@1, AP de tiros ±1/±3 s |
| **E2. Eventos panorámica** | Panorámica (cámara virtual) | ST2 eval, `bas/` | T-DEED base frente a `SNB_pano1` (ya medido [M]); añadir la mAP@1 oficial | mAP@1, AP de tiros |
| **E3. Dominio propio** | TV juvenil | Espanyol 25-30 (CSV del usuario), Mallorca | Mismos que E1 | `evaluate_match.py`. Solo indicativo: la anotación es parcial. |

### 5.3 Orden de ejecución

1. **Arnés.**
   - Script nuevo `scripts/benchmark/export_mot.py` que vuelque nuestros tracks a MOT txt y SNGS.
   - Wrapper de TrackEval.
   - Validación: con el GT como predicción, HOTA = 1.
   - Esto **no cambia el pipeline**: solo lee sus salidas.
2. **T1, régimen (a) con oráculo.** Reproducir ByteTrack 71,5, BoT-SORT 77,0 y Deep-EIoU 85,4 dentro de unos
   puntos. Si cuadra, el arnés es fiable. Después, nuestro ByteTrack con sus parámetros.
3. **T1, régimen (b)** y **T2**: aquí sale la primera cifra real de nuestro pipeline.
4. **T3** (requiere decidir el detector de personas para panorámica, que hoy no existe en `main`).
5. **E1/E2**: añadir la mAP@1 oficial a `evaluate_spotting.py` o usar el paquete `SoccerNet`.
6. T4 y E3 si queda tiempo.

Cada paso tiene un criterio de parada: si un candidato no supera a lo nuestro en val por ≥ 2 HOTA o ≥ 2 mAP, no
pasa a eval.

### 5.4 Riesgos

- **Licencias:**
  - Deep-EIoU y GTATrack no tienen licencia; sn-gamestate y T-DEED son GPL-3.0. Vale para benchmark interno.
  - Para producto hay que revisar. Ya usamos T-DEED (GPL) en producción.
- **Entornos antiguos:** Deep-EIoU (torch 1.13, Python 3.7) y CALF (TF 2.3). Hay que ejecutarlos en contenedores
  aparte y no tocar el de `Sports-Classification`.
- **GPU compartida:** hubo OOM en octubre. Las pasadas se lanzan con la GPU libre.
- **Test contaminado:** el test de SoccerNet-Tracking ya se usó para elegir la época del detector de *balón*. No
  afecta al HOTA de jugadores, pero sí a cualquier métrica de balón en T1.

---

## 6. No verificado

- Que los ficheros `.pt` de las carpetas de Drive de T-DEED, sn-teamspotting, BoT-SORT y ByteTrack descarguen de
  verdad. Solo se comprobó que la página responde o que la carpeta lista los ficheros.
- El split de la tabla MOT de SoccerMaster (HOTA 59,1) y su consumo de VRAM.
- ~~La licencia real de FOOTPASS y su publicación en CVIU.~~ Resuelto el 2026-11-06: CVIU vol. 269, art. 104790,
  acceso abierto CC BY 4.0; anotaciones y baselines CC BY-NC 4.0; detalle en `docs/datasets.md` §4B.
- Si los pesos del ganador de TBAS 2025 (dude.k) son públicos.
- El valor por defecto del tile de FutTraffic (1024 sale de un docstring) y la hoja del leaderboard que cita.
- El baseline GS-HOTA: 22,26 (nuestro `docs/datasets.md`) frente a 29,01 (informe del challenge 2025).
  Probablemente sean ediciones distintas.
- ~~Si Stoke y Reading pertenecen al split *test* oficial de SN-BAS, condición para comparar con el 58,91 publicado.~~ Resuelto en §7.3: son exactamente el `test` oficial.
- Con qué splits de SN-BAS-2025 se entrenó el E2E-Spot de OpenSportsLab (su model card solo dice "SN-BAS-2025"; el dataset HF tiene train/valid/test). Se supone train (+ valid) y por tanto test limpio.

---

## 7. Resultados

Cada resultado está en su fichero de `docs/comparativa_benchmark/` (abrir solo el que haga falta). Un resultado nuevo
es un fichero `7.NN_*.md` nuevo más una fila aquí; las citas «§7.N» del código y de otros documentos apuntan a esta tabla.

| § | Fecha | Qué mide | Titular [M] | Fichero |
|---|---|---|---|---|
| 7.1 | 2026-11-02 | Tracking TV con detecciones oráculo | ByteTrack nuestro 82,3 HOTA; BoT-SORT 83,8 y −67 % de cambios de identidad. Arnés calibrado. | `docs/comparativa_benchmark/7.01_tracking_tv_oraculo.md` |
| 7.2 | 2026-11-02 | Tracking TV con nuestro detector híbrido | ByteTrack 56,4 HOTA / 66,0 IDF1; BoT-SORT sin ReID 57,7 / 67,1. | `docs/comparativa_benchmark/7.02_tracking_tv_detector_hibrido.md` |
| 7.3 | 2026-11-03 | Eventos TV: test oficial de SN-BAS (Stoke + Reading), mAP@1 | T-DEED `SoccerNetBall_test` 61,2; sn-teamspotting 55,5; AdaSpot-b 54,9 (10 clases); E2E-Spot de OSL 34,5. `challenge2` está contaminado ahí. | `docs/comparativa_benchmark/7.03_eventos_tv_bas_test.md` |
| 7.4 | 2026-11-03 | Tracking TV: Deep-EIoU y GTA-Link | Deep-EIoU `rgb` 62,4 HOTA / 76,8 IDF1; con GTA-Link (offline) 64,3 / 81,3: +7,9 HOTA sobre nuestro ByteTrack. Repos sin licencia. | `docs/comparativa_benchmark/7.04_tracking_deep_eiou_gta_link.md` |
| 7.5 | 2026-11-03 | Eventos TV con Average-mAP *tight* | T-DEED 65,9; el orden entre modelos no cambia. | `docs/comparativa_benchmark/7.05_eventos_tv_map_tight.md` |
| 7.6 | 2026-11-04 | Eventos en panorámica (eval de SoccerTrack v2) | `SNB_pano1` 38,3 mAP@1; `challenge2` 25,4; sn-teamspotting 19,5; AdaSpot-b 4,4 (colapsa). | `docs/comparativa_benchmark/7.06_eventos_panoramica_soccertrack.md` |
| 7.7 | 2026-11-06 | AdaSpot frente a PAVE (ganador de PCBAS 2026) | PAVE no se puede ejecutar: sin código ni pesos, y su entrada son tracklets, no vídeo. | `docs/comparativa_benchmark/7.07_adaspot_vs_pave.md` |
| 7.8 | 2026-11-06 | T-DEED frente a AdaSpot: precisión y coste | En las 10 clases comunes, T-DEED 200MF 62,5 mAP@1 frente a 54,9 de AdaSpot-b; coste en el fichero. | `docs/comparativa_benchmark/7.08_tdeed_vs_adaspot_precision_coste.md` |
| 7.9 | 2026-11-06 | Servicios comerciales: Scoutiqo y «STREET» | Qué ofrecen y cómo probarlos con SoccerTrack v2 (único vídeo que se puede subir). Pruebas pendientes del usuario. | `docs/comparativa_benchmark/7.09_servicios_comerciales_scoutiqo_street.md` |
| 7.10 | 2026-11-06 | BoT-SORT (GMC + ReID) en `core/tracking.py` | Valor por defecto actual: 59,4 HOTA / 70,5 IDF1 / 4.199 IDSW (ByteTrack 56,4 / 66,0 / 4.892). Las reentradas siguen sin resolverse. | `docs/comparativa_benchmark/7.10_tracking_botsort_gmc_reid.md` |
| 7.11 | 2026-11-08 | Tabla única de spotters con pesos públicos | TV: T-DEED 61,2 > sn-teamspotting 55,5 > AdaSpot-b 54,9 > OSL 34,5. Panorámica: la cámara ideal da lo mismo que la nuestra (±2 pt). Coste 12-14× tiempo real. | `docs/comparativa_benchmark/7.11_tabla_unica_spotters.md` |
| 7.12 | 2026-11-09 | Punto de trabajo (P/R/FP) y metraje nuevo (SN-PCBAS-2026 val) | En PCBAS, T-DEED 200MF mAP 50,0 / F1 64,0 / 2.600 FP frente a AdaSpot-b 41,1 / 71,5 / 1.121; `challenge2` 53,1 / 65,2 / 2.136. | `docs/comparativa_benchmark/7.12_punto_de_trabajo_prf_pcbas.md` |
| 7.13 | 2026-11-09 | Cabeza de 17 eventos de SoccerNet-v2 (20 partidos limpios) | Average-mAP tight: 200MF 21,4, producción 25,3 (dedicados 62-73). Córner utilizable (P 81 / R 74); tarjeta, cambio, fuera de juego y saque inicial no. | `docs/comparativa_benchmark/7.13_cabeza_17_eventos_snv2.md` |
