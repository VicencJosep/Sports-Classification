# Comparativa técnica con el estado del arte y plan de benchmarking

Fecha: 2026-11-01. Rama analizada: `main` (HEAD 9234c43). No se ha tocado código.

Convenciones: **[M]** medido por nosotros (con fichero de origen), **[P]** publicado por terceros (con URL),
**[NV]** no verificado. Ninguna cifra [P] es comparable con otra salvo que se diga explícitamente: cada una
sale de un dataset, split, detector o tolerancia distintos.

---

## 0. Resumen

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
  (`docs/estudio_deteccion_tiros.md:48,58`).
- Acciones en Middlesbrough (20 min): F1 88,4, equipo 68,1 % (`docs/hoja_de_ruta.md:24`).
- Panorámica, eval de SoccerTrack v2: tiros AP ±3 s 41,6 → 55,2 % con `SNB_pano1` (`docs/estudio_deteccion_tiros.md:218-245`).

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

### 7.1 T1 tracking TV, régimen (a) con detecciones oráculo [M, 2026-11-02]

Test de SoccerNet-Tracking (49 clips de 30 s), modo `persons` (sin balón). Las detecciones son el GT sin
identidades. Arnés en `scripts/benchmark/`; el evaluador da 100 con el GT contra sí mismo.
Ficheros: `output/benchmark/snmot_test/results/oracle__*.persons.json`.

| Tracker | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|---|---|
| Nuestro (`supervision.ByteTrack`, `PlayerTracker` sin cambios) | 82,3 | 97,5 | 69,5 | 99,9 | 97,0 | 77,0 | 5.074 |
| ByteTrack de ultralytics (YAML por defecto) | 73,4 | 85,9 | 62,8 | 90,3 | 96,5 | 76,8 | 5.133 |
| BoT-SORT de ultralytics (GMC sparseOptFlow, sin ReID) | **83,8** | 95,4 | **73,7** | 95,8 | **99,0** | **81,6** | **1.695** |
| Publicado (Deep-EIoU, tabla con oráculo, balón incluido) [P] | ByteTrack 71,5 · BoT-SORT 77,0 · Deep-EIoU 85,4 | | | | | | |

Lectura:

- **El arnés está calibrado.** El ByteTrack de ultralytics (73,4) queda cerca del ByteTrack publicado (71,5).
- **Nuestro HOTA alto con oráculo es sobre todo localización.** `supervision` devuelve la caja detectada (LocA 99,9)
  y no la caja suavizada por Kalman (90,3). Con cajas perfectas eso favorece; con un detector real puede no hacerlo.
- **BoT-SORT gana en asociación**: −67 % de cambios de identidad (1.695 frente a 5.074), +4,2 AssA y +4,6 IDF1.
  Es la compensación de movimiento de cámara, que a nosotros nos falta.
- **Pendiente:**
  - el régimen (b) con nuestro detector híbrido real (espera GPU libre; `run_snmot.sh`);
  - Deep-EIoU en un entorno aparte.

### 7.2 T1 tracking TV, régimen (b) con nuestro detector híbrido [M, 2026-11-02]

Mismas 49 secuencias, modo `persons`. Las detecciones son de SoccerMaster (personas) + YOLO11x (balón): 28 fps
y 5,7 GB de VRAM pico. Ficheros: `output/benchmark/snmot_test/results/hybrid__*.persons.json`.

| Tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|---|
| Nuestro (`supervision.ByteTrack`) | 56,4 | 69,2 | 46,2 | 87,1 | 66,0 | 4.892 |
| ByteTrack de ultralytics | 54,6 | 67,0 | 44,7 | 83,8 | 64,0 | 4.830 |
| **BoT-SORT** de ultralytics (GMC, sin ReID) | **57,7** | 69,3 | **48,3** | 86,7 | **67,1** | **3.401** |

Referencia [P, split no indicado]: SoccerMaster+MOTIP 59,1 y YOLOv8+PRTReID 59,8 (arXiv 2512.11016).

### 7.3 E1 eventos TV: Ball Action Spotting, test oficial (Stoke + Reading) [M, 2026-11-03]

Stoke y Reading son exactamente el split `test` oficial de SN-BAS (`models/tdeed/T-DEED/data/soccernetball/test.json`).

- Métrica: mAP@1 oficial (`evaluate_SNB` del repo de T-DEED).
- Postproceso: el mismo para todos, NMS suave con ventana de 1 s y candidatos ≥ 0,01.
- Partidos completos.
- Scripts: `scripts/benchmark/{eval_bas,teamspot_scores,adaspot_scores,osl_scores}.py`.
- Resultados: `output/benchmark/bas/*/test/summary.json`.

| Modelo | Clases | mAP@1 (12) | mAP@1 (10 comunes) | Tiros | Goles | Publicado [P] |
|---|---|---|---|---|---|---|
| **T-DEED `SoccerNetBall_test`** (nuestro, limpio, RegNetY-200MF, BAS + SN-v2) | 12 | **61,2** | **62,5** | **73,9** | **24,2** | 58,91 (800MF, Revisited, tabla 7) |
| sn-teamspotting (T-DEED + cabeza de equipo, SN 2025) | 12 | 55,5 | 56,4 | 59,7 | 17,4 | 53,59 |
| AdaSpot-b (arXiv 2602.22073, HF `arturxe/AdaSpot`) | 10 | — | 54,9 | 52,8 | — | **56,24 ± 0,3** mAP@1 (10 clases; su columna «δ=0,5 s»); 59,82 es mAP@2 |
| AdaSpot-s | 10 | — | 54,2 | 51,0 | — | 53,12 ± 1,4 mAP@1; 56,82 es mAP@2 |
| E2E-Spot de OpenSportsLab (SN-BAS-2025, ≈2 fps) | 12 | 34,5 | 31,1 | 15,4 | 18,2 | «tight» 47,98 (otra métrica) |
| T-DEED `challenge2` (producción) | 12 | *83,2* | — | *89,9* | *80,8* | **contaminado**: entrenado con Stoke y Reading |

Lectura:

- **El arnés reproduce lo publicado.** T-DEED (61,2 frente a 58,91) y sn-teamspotting (55,5 frente a 53,59) quedan a ~2 puntos.
- **AdaSpot reproduce su cifra publicada** (corregido el 2026-11-06). En el código oficial de SoccerNet, `at1`
  empareja a ±0,5 s y `at2` a ±1 s; el `main.py` de AdaSpot imprime las dos, y su tabla 2 las llama «δ=0,5 s» y
  «δ=1 s». La cifra comparable con nuestro mAP@1 es por tanto **56,24 ± 0,3** (AdaSpot-b) y 53,12 ± 1,4 (AdaSpot-s),
  no 59,82 / 56,82 (que son mAP@2). Nuestra medida (54,9 / 54,2 con un solo seed publicado) queda a 1,3 / −1,1
  puntos: dentro de lo esperable.
  - Su `inference.py` pasa `(448, 796)` a `cv2.resize`, que espera (ancho, alto), y deforma la imagen. Aquí se usa
    796x448.
  - **Su comparación con T-DEED no es la receta real de T-DEED**: según su apéndice B.2, E2E-Spot y T-DEED se
    reentrenaron «en nuestro pipeline», solo con BAS y «a 398x224». T-DEED para BAS se entrena a 796x448 con
    preentrenamiento conjunto en SoccerNet-v2 (vale +7,5: 51,43 → 58,91 en Revisited). De ahí que su T-DEED-200MF
    dé 45,43 y el nuestro 62,5 en las mismas 10 clases.
- **E2E-Spot de OpenSportsLab no sirve a ±0,5 s**: muestrea a 2 fps.
- **Equipo de cada acción con sn-teamspotting** (`scripts/benchmark/eval_team.py`): acierto **85,5 %** en 3.980 acciones y Team-mAP@1 aproximado 50,9
  (publicado 47,18).
  - El mapeo home/away → left/right se elige a posteriori, por eso es optimista: una orientación y un punto de
    cambio por partido.
  - El método sitúa el cambio de campo de Reading en el minuto 46,3, justo en el descanso.
  - El Team-mAP exacto necesita las etiquetas de SN-BAS-2025 (HF gated con NDA, sin acceso).
  - **Mismo tramo que nuestro pipeline** (Middlesbrough, minutos 0-20, una sola orientación como en
    `evaluate_match.py`):

    | Método | Acierto de equipo | Acciones |
    |---|---|---|
    | sn-teamspotting | **82,8 %** | 401 (todas las del GT) |
    | Nuestro pipeline (PRTReId + KMeans) | 68,1 % | 339 (las que nuestro pipeline empareja) |

    - Middlesbrough es la validación de sn-teamspotting (early stopping): sesgo leve a su favor.
    - El Team-mAP de este tramo sale `nan`: el recorte a 20 minutos deja clases sin GT. No se usa.

**Criba**:

- **Pasa a la comparación extensa:** **sn-teamspotting**, por la cabeza de equipo. En acciones no supera a T-DEED,
  pero en equipo apunta a mucho más que nuestro PRTReId + KMeans.
- **Descartados como sustitutos de T-DEED en TV:** AdaSpot y E2E-Spot.
  - AdaSpot se puede reconsiderar si se le añade el pre-entrenamiento SN-v2, o en panorámica: comparte código con
    `tdeed_finetune`.
- **Coste**: ~2 min de GPU por partido y modelo; el resto es decodificar el vídeo (5-7 min por partido en total).

### 7.4 T1 tracking TV: Deep-EIoU y GTA-Link [M, 2026-11-03]

Mismas 49 secuencias y mismas detecciones (filtro de `track_mot.py`), modo `persons`.

- Scripts: `scripts/benchmark/track_deep_eiou.py` y `gta_link.py`. Usan el código original tal cual, con calzos
  para no instalar nada.
- ReID: OSNet x1_0 "sports" del Drive de Deep-EIoU.
- Parámetros: los de `tools/sport_track.py`.
- La variante `rgb` corrige el demo original, que pasa recortes BGR a un ReID entrenado en RGB.
- `dti` = su interpolación.
- `gta` = GTA-Link encima, con los parámetros de su README.

| Tracker | HOTA oráculo | HOTA híbrido | AssA híbrido | IDF1 híbrido | IDSW híbrido |
|---|---|---|---|---|---|
| Nuestro (`supervision.ByteTrack`) | 82,3 | 56,4 | 46,2 | 66,0 | 4.892 |
| BoT-SORT (ultralytics) | 83,8 | 57,7 | 48,3 | 67,1 | 3.401 |
| Deep-EIoU (BGR, como el original) | 87,4 | 62,1 | 55,2 | 76,3 | 2.760 |
| **Deep-EIoU `rgb`** | 87,9 | **62,4** | **55,7** | **76,8** | **2.655** |
| Deep-EIoU `rgb` + interpolación | 87,5 | 62,3 | 55,6 | 76,7 | 2.615 |
| **Deep-EIoU `rgb` + GTA-Link** | **89,9** | **64,3** | **59,2** | **81,3** | 3.496 |
| Publicado (Deep-EIoU, oráculo, balón incluido) [P] | 85,4 | | | | |

- **Deep-EIoU es el mejor tracker medido:** +6,0 HOTA y +10,8 IDF1 sobre el nuestro con las mismas detecciones,
  y −46 % de cambios de identidad.
- Con oráculo supera la cifra publicada (87,9 frente a 85,4; aquí sin balón).
- La interpolación no aporta.
- GTA-Link añade +2,0 HOTA con oráculo y +1,9 con nuestro detector (IDF1 +4,5).
  - Con GTA-Link sube el IDSW (3.496 frente a 2.655): al partir tracklets (split) cuenta cambios nuevos, pero el
    IDF1 y la AssA mejoran.
  - Es offline (necesita el tracklet completo): encaja con el modo partido completo, no con tiempo real.
- **Total frente a nuestro tracker: +7,9 HOTA, +15,3 IDF1.**
- **Licencia:** ninguno de los dos repos tiene licencia; ver §5.4.

### 7.5 E1 con otra métrica: Average-mAP *tight* (1-5 s) en EFL test [M]

| Modelo | mAP@1 | tight | tight (10 clases) | Tiros tight |
|---|---|---|---|---|
| **T-DEED `SoccerNetBall_test`** | **61,2** | **65,9** | **67,8** | **77,4** |
| sn-teamspotting | 55,5 | 60,1 | 61,9 | 62,9 |
| AdaSpot-b | — | — | 59,8 | 55,3 |
| E2E-Spot de OpenSportsLab | 34,5 | 48,5 | 46,3 | 32,1 |

El orden no cambia. El E2E-Spot de OpenSportsLab es el que más gana con la tolerancia amplia, coherente con sus 2 fps.

### 7.6 E2 eventos en otros partidos: SoccerTrack v2, panorámica [M, 2026-11-04]

- Partidos: eval 128057 + 132831 (partición oficial), 4 partes.
- Vídeo: cámara virtual `vcam796vl1` (796x448, nuestra trayectoria Viterbi+L1).
- 4.372 acciones del GT. HEADER solo tiene 5 y GOAL 10, por eso se da también la media sobre las clases con
  ≥ 10 eventos.
- Etiquetas: `datasets/soccertrack-v2/bas/<m>/<m>_12_class_events.json`, convertidas por `eval_bas.py --dataset st2`.
- Ningún modelo vio estos partidos.

| Modelo | mAP@1 | mAP@1 (clases ≥ 10 GT) | tight | Tiros @1 |
|---|---|---|---|---|
| **T-DEED `SNB_pano1`** (nuestro afinado en cámara virtual) | **38,3** | **41,5** | **42,3** | **39,0** |
| T-DEED `challenge2` (producción, TV) | 25,4 | 27,7 | 30,6 | 27,2 |
| sn-teamspotting | 19,5 | 21,2 | 23,6 | 16,0 |
| AdaSpot-b (10 clases) | 4,4 | 4,8 | 5,1 | 0,0 |

- **El afinado `SNB_pano1` es el único que funciona razonablemente en panorámica:** +13 puntos sobre el modelo de TV.
- **Todos caen mucho respecto a TV** (T-DEED de TV: 61 → 25).
- **AdaSpot colapsa:** predice fondo casi siempre (P media de fondo 0,998 frente a 0,93 en TV).
  - Se comprobó que no es un desfase: su curva de PASS correlaciona con la de T-DEED en desfase 0.
  - Su recorte de alta resolución no generaliza a la cámara virtual.
- **Equipo con sn-teamspotting en panorámica**: Team-mAP@1 **exacto** 15,6, porque aquí el GT ya trae left/right.
  - Acierto de equipo por acción: 66,5 %, el mismo directo que con mejor mapeo, así que el left/right del GT
    coincide con el lado en pantalla.
  - En TV daba 85,5 %: la cabeza de equipo también sufre el cambio de dominio.

### 7.7 AdaSpot frente a PAVE (ganador de SoccerNet 2026 PCBAS) [2026-11-06]

**PAVE no se puede ejecutar.** Verificado el 2026-11-06:

- Es un informe técnico de 2 páginas ([arXiv 2606.28389](https://arxiv.org/abs/2606.28389), Altawijri y Mathkour,
  TAHAKOM). No hay código, pesos ni promesa de publicarlos: ni en el paper, ni en GitHub (búsqueda por PAVE,
  PCBAS, TAHAKOM), ni en Hugging Face.
- Tampoco publican código ni pesos el 2.º clasificado (PC-SSAS) ni las extensiones de los baselines
  (arXiv 2606.09679, 2608.01696). El único repo de terceros (`Fousekhs/footpass-pcbas`) es código de entrenamiento
  sin pesos.
- **Su entrada no es vídeo**: parte de los tracklets por jugador y del "game state" (posiciones, velocidades, 26
  huecos de rol, 13 roles por equipo) que distribuye el dataset FOOTPASS / SN-PCBAS-2026, y de features X3D sobre esos tracklets. Para
  correrlo sobre nuestros vídeos habría que generar antes esas entradas con nuestro propio tracking, con otra
  calidad.
- Los datos están en HF `SoccerNet/SN-PCBAS-2026` (236 GB, gated con NDA, sin acceso desde esta cuenta).
- **Tarea y métrica distintas a BAS**: 8 clases (drive, pass, cross, shot, header, throw-in, tackle, block) y cada
  acción debe acertar además **equipo y dorsal**; Macro-F1 a umbral 0,15 y ±12 frames (±0,48 s). BAS son 12 clases
  sin jugador y mAP@1.

**Cifras publicadas de PAVE** [P, arXiv 2606.28389 y 2607.07320]: Macro-F1 **58,94** en el set *challenge*
(baseline TAAD+DST 46,41; sus propios pasos: 48,6 → 55,1 → 58,9). En validación (6 mitades, 3 partidos), F1 por clase del
ensemble: drive 83,9, pass 84,0, cross 76,4, throw-in 82,8, shot 67,7, header 45,0, tackle 12,3, block 35,4
(media 60,9). Todo con equipo y dorsal exigidos.

**Lo que sí se puede poner en la misma escala**: nuestras predicciones (TV, test de BAS, y panorámica) con la
métrica de PCBAS, solo clase (`scripts/benchmark/eval_f1_pcbas.py`). Es una **cota superior** (sin equipo ni
dorsal) y sobre **otros partidos**, así que PAVE no es comparable de forma directa; se incluye solo para situar
los órdenes de magnitud.

| Modelo | Datos | Macro-F1@0,15 ±12f (8 clases) | drive | pass | cross | shot | header | throw-in | tackle | block |
|---|---|---|---|---|---|---|---|---|---|---|
| PAVE [P] (clase + equipo + dorsal) | FOOTPASS val | 60,9 | 83,9 | 84,0 | 76,4 | 67,7 | 45,0 | 82,8 | 12,3 | 35,4 |
| PAVE [P] (clase + equipo + dorsal) | FOOTPASS challenge | 58,9 | — | — | — | — | — | — | — | — |
| **T-DEED `SoccerNetBall_test`** (solo clase) | BAS test (TV) | **55,9** | 68,6 | 77,5 | 58,6 | **65,5** | 65,4 | 68,6 | 9,2 | **33,5** |
| AdaSpot-s (solo clase) | BAS test (TV) | 54,3 | 75,9 | 83,4 | **66,7** | 56,5 | 64,1 | 65,3 | 0,0 | 22,2 |
| AdaSpot-b (solo clase) | BAS test (TV) | 53,8 | **75,2** | **85,2** | 59,8 | 53,0 | **66,8** | 67,3 | 0,0 | 23,0 |
| sn-teamspotting (solo clase) | BAS test (TV) | 51,9 | 68,8 | 77,7 | 52,8 | 49,6 | 57,9 | 68,4 | 13,3 | 26,4 |
| E2E-Spot OSL (solo clase) | BAS test (TV) | 32,5 | 51,9 | 59,9 | 41,1 | 26,3 | 39,9 | 30,7 | 0,0 | 10,1 |
| `SNB_pano1` (solo clase) | ST2 eval (panorámica) | 36,6 | 58,0 | 69,7 | 37,6 | 44,7 | 0,0 | 66,7 | 5,3 | 10,9 |
| AdaSpot-b (solo clase) | ST2 eval (panorámica) | 1,3 | 3,4 | 4,4 | 0,0 | 0,0 | 0,0 | 2,4 | 0,0 | 0,0 |

Lectura:

- **AdaSpot frente a T-DEED cambia según la métrica.** Con mAP@1 (ranking por confianza) T-DEED gana por 7 puntos
  en TV; con F1 a umbral fijo 0,15 la distancia se reduce a 1,6-2,1 puntos y AdaSpot gana en drive, pass, cross y
  header, mientras T-DEED gana en shot, block y tackle. AdaSpot calibra mejor sus confianzas; T-DEED ordena mejor.
- **AdaSpot frente a PAVE**: aunque la cota de AdaSpot (solo clase, 53,8-54,3) ya queda por debajo del 58,9-60,9
  de PAVE con equipo y dorsal, los partidos y la dificultad de los vídeos son distintos (FOOTPASS: 5 ligas 2023/24
  en 1080p; BAS: EFL 2019 en 720p), así que no se puede concluir cuál reconoce mejor las acciones. Lo que sí se
  puede afirmar: PAVE resuelve una tarea que ninguno de nuestros modelos resuelve (quién ejecuta la acción).
- En panorámica, AdaSpot sigue en colapso (1,3) también con esta métrica.
- Tackle es la peor clase para todos, incluido PAVE (12,3).

**Cómo haríamos una comparación real con PAVE**, si se quiere:
1. Pedir acceso a `SoccerNet/SN-PCBAS-2026` (cuenta HF, NDA nominal, aprobación manual).
2. Reentrenar los baselines públicos de FOOTPASS (TAAD + DST, código en `JeremieOchin/FOOTPASS`, sin pesos):
   46,41 en challenge. PAVE en sí no es reproducible sin su código.
3. Evaluar nuestra cadena (T-DEED para la clase + sn-teamspotting para el equipo + dorsal de SoccerMaster por
   tracklet) con el `evaluation.py` oficial de FOOTPASS, sobre sus vídeos de validación (3 partidos). Esa sí sería
   la misma tarea, los mismos vídeos y la misma métrica. Coste: 2-3 días más la espera del acceso.

### 7.8 Verificación T-DEED frente a AdaSpot: precisión y coste [M, 2026-11-06]

AdaSpot (Xarles et al., 2026) es el sucesor de T-DEED por los mismos autores. Se comparan los **pesos publicados**
de ambos (un solo seed) sobre los mismos vídeos, con el evaluador oficial y el mismo postproceso (NMS suave 1 s).

**Conjunto elegido**: el test oficial de SN-BAS (Stoke + Reading, partidos completos, ~4.000 acciones) porque es el
único conjunto de TV etiquetado que ninguno de los dos checkpoints vio al entrenar (ambos: train = 4 partidos,
val = Middlesbrough) y es el test de los dos papers, así que cada cifra nuestra se contrasta con la publicada.
Middlesbrough se añade como tercer partido, marcado: ambos lo usaron para el *early stopping*.

**Precisión, mAP sobre las 10 clases comunes (sin FREE KICK ni GOAL)**

| Modelo | Receta | Test mAP@1 | Test mAP@2 | Val mAP@1 | Val mAP@2 | Publicado mAP@1 / mAP@2 [P] |
|---|---|---|---|---|---|---|
| **T-DEED 200MF** `SoccerNetBall_test` | BAS + SN-v2 conjunto, 796x448 | **62,5** | **66,7** | **62,8** | **68,1** | — |
| AdaSpot-b | solo BAS, 398x224 + recortes 112² | 54,9 | 58,7 | 55,3 | 59,2 | 56,24 ± 0,3 / 59,82 ± 0,9 |
| AdaSpot-s | ídem | 54,2 | 59,1 | 52,6 | 56,3 | 53,12 ± 1,4 / 56,82 ± 1,9 |
| T-DEED 200MF según el paper de AdaSpot [P] | solo BAS, **398x224**, su pipeline | 45,43 | 48,41 | — | — | |

- AdaSpot reproduce lo publicado a ~1 punto en las dos tolerancias y en los dos modelos: la ejecución es correcta.
- **T-DEED gana por 7-8 puntos** en los tres partidos y en las dos tolerancias. Con F1 a umbral fijo (§7.7) la
  distancia baja a ~2 puntos: AdaSpot calibra mejor, T-DEED ordena mejor.
- La ventaja viene de la **receta** de T-DEED (preentrenamiento en SN-v2 y 796x448), que AdaSpot no replicó al
  comparar. No está medido si la arquitectura de AdaSpot sería mejor con la misma receta; hacerlo exige entrenar
  (T-DEED solo-BAS a 796x448 o AdaSpot con SN-v2, que su código no contempla), 1-2 jornadas de GPU cada uno.
- **Lectura del paper y del código de AdaSpot [M, 2026-11-08]** (arXiv 2602.22073, CVPR 2026, mismos autores que T-DEED;
  `models/adaspot/repo/config/SoccerNetBall/*.json`, `model/modules.py:68`, `data/soccernetball/*.json`):
  - **Protocolo idéntico al de `SoccerNetBall_test`**: entrena con los 4 de `train`, para en Middlesbrough (`criterion: map`,
    validación desde la época 10 de 50) y evalúa en Stoke + Reading. Mismo sesgo de validación (~1 punto) y test igual de
    limpio: la comparación 61,2 / 62,5 frente a 54,9 es con los mismos datos de acciones.
  - **Entrena solo con SN-BAS** (10 clases: quita FREE KICK y GOAL "por su frecuencia extremadamente baja", 6 y 2 ejemplos
    de test), backbone RegNetY-200MF (-s) / 400MF (-b) preentrenado en ImageNet (`timm`, `pretrained=True`), AdamW 8e-4,
    coseno, mixup, 3 semillas. El paper no menciona SoccerNet-v2 salvo como origen del dataset: no hace el
    preentrenamiento conjunto de T-DEED (400 partidos más de etiquetas).
  - **Resolución**: el vídeo se fija a 796x448; la rama global va a la mitad (398x224) y los recortes de 112x112 se toman
    del frame a 796x448. Sus baselines, en cambio, "se entrenan a 398x224" con los 100 frames enteros ("all methods are
    trained on sequences of L=100 frames with a spatial resolution of 398×224"), sin SN-v2.
  - **En qué gana, según su propio paper**: en su marco controlado (todo a 398x224, solo BAS, 3 semillas) AdaSpot-b 56,24
    frente a T-DEED 800MF 49,39 y E2E-Spot 800MF 54,49, con 10,6 M parámetros frente a 46,2 y 90 frente a 151 GFLOPs; su
    ablación en SN-BAS da 47,98 solo con la rama de baja resolución y 53,02 con los recortes (+5); en los benchmarks de
    precisión por frame gana con holgura (Tennis 74,0 frente a 58,4; FineDiving 27,1 frente a 19,6-21,3; FineGym empate).
    El RoI adaptativo no ayuda en SN-BAS ("a fixed RoI size performs best").
  - **En qué no gana**: frente a la receta completa de T-DEED (SN-v2 + 796x448 enteros) pierde 7-8 puntos con los mismos
    datos de acciones; y en panorámica colapsa. Su aportación es eficiencia (resolución donde hace falta), no precisión
    absoluta en TV.

**Coste** (`scripts/benchmark/spotter_cost.py`; misma entrada para todos: clips de 100 frames muestreados a 796x448,
uint8, lote 4; cada modelo con su propio autocast; GPU RTX de 24 GB **compartida** con otros procesos de 9-13 GB, así
que los tiempos llevan ruido). Punta a punta = decodificar + redimensionar + inferir 5 min de Stoke (7.502 frames).

| Modelo | Parámetros | GFLOPs/clip (thop, ≈) | Solape de inferencia | Clips/s | s de vídeo por s de GPU | VRAM pico | Punta a punta 5 min | × tiempo real |
|---|---|---|---|---|---|---|---|---|
| T-DEED 200MF `SoccerNetBall_test` | 12,3 M | 288 | 75 % | 7,6 | 15,2 | 8,6 GB | 29,6 s | 10,2× |
| T-DEED 800MF `challenge2` (producción) | 46,2 M | 1.149 | 75 % | 5,0 (lote 2, OOM) | 10,0 | 7,1 GB (lote 2) | 33,5 s | 9,0× |
| AdaSpot-s | 7,6 M | 87 | 50 % | 13,2 | 52,7 | 5,7 GB | 26,4 s | 11,4× |
| AdaSpot-b | 10,6 M | 171 | 50 % | 11,0 | **44,0** | 6,3 GB | **24,6 s** | **12,2×** |

- Los GFLOPs de thop salen ≈ 2× los del paper de AdaSpot (que da 90 para -b y 39,6 para T-DEED-200MF a 398x224):
  convención MACs/FLOPs y resolución distintas. Las **proporciones** sí coinciden: AdaSpot-b cuesta ~0,6× T-DEED-200MF
  por clip y, con solape del 50 % en vez del 75 %, ~0,3× por segundo de vídeo.
- **En inferencia pura AdaSpot-b es ~2,9× más rápido que T-DEED-200MF** (44 frente a 15 s de vídeo por s de GPU) y
  ~4,4× más que el 800MF de producción, con 2-3 GB menos de VRAM.
- **De punta a punta la diferencia casi desaparece** (24,6 frente a 29,6 s por 5 min): la decodificación y el resize
  del vídeo dominan (unos 300 fps). Un partido de 95 min cuesta ~8 min con AdaSpot-b y ~9,5 min con T-DEED-200MF.
- Nuestro pipeline usa el 800MF de producción: 33,5 s por 5 min, 9× tiempo real.

**Conclusión**: AdaSpot es más barato (0,3× cómputo de inferencia) pero 7-8 puntos peor con los pesos publicados;
como la decodificación domina el coste real, el ahorro en pared es ~15 %. No compensa cambiar en TV. Sí podría
interesar reentrenarlo con la receta de T-DEED (SN-v2 + 796x448) si el cómputo pasara a ser limitante, o para la
panorámica, donde ambos modelos de TV fallan y hay que afinar de todos modos (§7.6).

### 7.9 Servicios comerciales con muestra gratuita: Scoutiqo y «STREET» [2026-11-06]

**Restricción previa**: los vídeos de SoccerNet (EFL, los 500 partidos, Tracking) están bajo NDA y no pueden subirse a un
tercero. El único vídeo etiquetado que sí se puede usar es **SoccerTrack v2 (CC BY 4.0)**, panorámica fija 4K con
las 12 clases de BAS, equipo y jugador. Es además el dominio de estos servicios (cámaras tipo Veo).

**Scoutiqo** ([scoutiqo.com](https://scoutiqo.com); Scoutiqo Labs, Proud Investments MB, Vilna, Lituania; ley lituana/UE).
Verificado en su web el 2026-11-06:

- Qué es: plataforma para fútbol base (clubes y academias): subida de vídeo (archivo, enlace Veo o Google Drive;
  YouTube no), seguimiento de jugadores y balón, separación de equipos, posesión, «momentos clave», telestración,
  exportación de clips MP4 y gestión del club. Sin API documentada ni exportación de datos (CSV/JSON).
- Eventos que muestra su demo (10 min de un partido de academia): *Shots*, *Passes*, *Turnovers* y *Set pieces*,
  etiquetados «AI-checked, not human-verified»; solo *Set pieces* tenía recuento (4). Su página de producto habla de
  «timestamped event estimates» y advierte de que no garantiza «every pass or foul».
- Entrada: «footage from a phone, Veo export or another camera»; el onboarding pregunta cómo se graba (Veo o
  cámara automática / teléfono en trípode / cámara operada / TV o montaje) y avisa si el montaje no sirve. Sin
  límites de tamaño o duración publicados («upload and processing limits still apply»).
- Precisión que declaran (su demo, sus cifras): jugadores encontrados 78 % de 499 revisados a mano; balón 72 % de
  25 frames; equipo correcto 99,9 %. Nada sobre eventos.
- Gratis: «your first 10 minutes free», sin tarjeta (Starter: 1 equipo, 25 jugadores, vídeo 3 meses). Club 39 €/mes
  y Academy 99 €/mes con prueba de 14 días (tarjeta y un partido completo).
- Condiciones relevantes: hay que tener derechos sobre el contenido («original content or proper attribution
  required»: CC BY 4.0 con atribución cumple); el vídeo del *preview* de 10 min «is not used to train models»;
  almacenamiento en Backblaze (UE), procesado propio y «AI analysis providers, including Google»; se borra con la
  cuenta. Nada prohíbe evaluar o publicar comparativas.

**STREET** ([streetplay.ai](https://streetplay.ai); STREETPLAY Inc., Bengaluru, India; app «Streetplay Football
Highlights» en App Store y Google Play). Verificado el 2026-11-06 en su web, FAQ, boletines de mayo a noviembre de
2026, política de privacidad y ficha de la App Store:

- Qué es: **producción de highlights con IA para fútbol base**, no una plataforma de análisis. Se sube el partido
  completo (archivo desde la app, o enlace público de YouTube, Veo, Hudl o Google Drive) y en menos de 24 h devuelve
  un vídeo de highlights de 2-5 min, reels verticales de cada gol, reels por jugador y una «página del partido» con
  marcador, goleadores y línea de tiempo. «Camera agnostic» (teléfono, GoPro, dron, Veo, XBotGo, Hudl). Con cámara
  fija genera una vista con zoom que sigue el juego, rótulos de marcador, seguimiento del balón y repeticiones.
- Lo que detecta, según su propia descripción: «every goal, skill, assist and save». No hay lista de eventos
  exportable, ni API pública, ni ninguna cifra de precisión. «Player and Game Stats» se anuncian para noviembre de
  2026 (boletín de noviembre: «Match Timeline and Momentum», valoración por jugador, resumen de acciones, alineación)
  y por ahora solo en su API de socios, con precio a medida.
- Gratis: plan «Basic: FREE, any game» subiendo desde la app STREET (iOS 17+ o Android). Sin límites publicados.
  «League Package» y «Community» con precio a medida. Tamaño de la empresa: 447 partidos procesados en total a
  noviembre de 2026 y más de 1.000 descargas de la app (su boletín).
- Condiciones relevantes (política de privacidad): el usuario conserva la propiedad pero concede «an unrestricted,
  royalty-free license to distribute, publish, and share this content»; STREET «post[s] the full game videos on our
  official YouTube channel» y usa los datos «to improve and train our AI models (e.g., ball and player tracking,
  highlight detection)»; alojamiento en Google Cloud; el contenido con menores de 18 años no se distribuye
  públicamente. Con SoccerTrack v2 (CC BY 4.0, seudonimizado) es legal con atribución, pero el partido puede acabar
  publicado en su canal y en su conjunto de entrenamiento: decisión del usuario.

Qué se compara con qué: **Scoutiqo** da eventos con tiempo (tiros, pases, pérdidas, balón parado) y se mide contra
nuestro spotting de 12 clases agrupado; **STREET** da un vídeo de highlights y los goles del partido, y se mide
contra nuestros candidatos a highlight y goles (`analytics/highlights`, `analytics/goals.py`).

**Paquete de prueba para Scoutiqo** (`output/benchmark/vendor/`):

- `st2_128057_2nd_35m_10min.mp4`: 10 min (704 MB, 4096x1080 H.264 copiado sin recodificar) de SoccerTrack v2,
  partido **128057** (eval), 2.ª parte desde el minuto 35:30. Es la ventana de 10 min con más eventos «vistosos» de
  los dos partidos de eval: 195 acciones, con 2 goles, 3 tiros, 5 centros, 8 saques de banda, 1 falta.
- `st2_128057_2nd_35m_10min.gt.json`: verdad de terreno del clip (tiempo relativo, clase, equipo, jugador).
- `scripts/benchmark/eval_vendor.py`: evalúa la línea de tiempo del servicio (CSV `mm:ss,etiqueta[,equipo]`
  transcrito a mano) y nuestro T-DEED sobre la **misma ventana** con la misma regla: grupos *shot* (SHOT+GOAL),
  *pass* (PASS+HIGH PASS+CROSS), *set_piece* (FREE KICK+THROW IN), *turnover* (TACKLE+BLOCK+OUT); P/R/F1 a ±1 s
  y ±3 s, emparejamiento 1 a 1.
- Nuestra referencia en esa ventana, `SNB_pano1` sobre la cámara virtual (umbral 0,3, NMS 1 s), F1 a ±1 s / ±3 s:
  shot 62,5 / 62,5 (5 GT), pass 86,3 / 87,2 (95), set_piece 94,1 / 94,1 (9), turnover 43,9 / 53,7 (20).

**Pasos con Scoutiqo (los hace una persona, en el navegador)**:
1. Cuenta gratuita en scoutiqo.com/start con la opción «Veo or a similar automatic camera»; subir el MP4 (o un
   enlace de Drive «anyone with the link»). Indicar la atribución SoccerTrack v2 (CC BY 4.0) si se pide.
2. Cuando termine, transcribir la línea de tiempo del Studio a `output/benchmark/vendor/scoutiqo_timeline.csv`
   (`mm:ss,shot|pass|set_piece|turnover[,equipo]`), y anotar también lo que no sea evento (posesión, tiempo de
   proceso) y capturas.
3. `python -m scripts.benchmark.eval_vendor --clip output/benchmark/vendor/st2_128057_2nd_35m_10min
   --vendor-csv output/benchmark/vendor/scoutiqo_timeline.csv --vendor-name scoutiqo --ours SNB_pano1`.

Límites con Scoutiqo: 10 minutos de un partido; sus eventos son 4 grupos gruesos frente a nuestras 12 clases;
no da tiempos exactos exportables, así que la transcripción manual añade ±1 s de ruido (por eso se mide también a
±3 s); su seguimiento de jugadores se podría medir contra el GSR del dataset si el Studio mostrara posiciones.

**Clip propio para Scoutiqo con cámara que sigue el balón** [2026-11-06]. El usuario cierra por ahora la
panorámica: su caso de estudio es la cámara única o TV que sigue el balón. Vídeos propios disponibles (`docs/
particion_datasets.md` §6): Mallorca–Barcelona juvenil J28 (cámara única con rótulo propio, 1280x720, 25 fps,
3 goles) y Espanyol–Damm juvenil J26 (TV con repeticiones, 0 goles). El Espanyol tiene anotación manual de acciones
de balón solo en 25:13-30:00 y sin tiros; el Mallorca no tiene anotación de acciones, pero sí los goles confirmados
por el marcador. Los partidos de SoccerNet y SoccerHigh no se pueden subir (NDA / derechos de retransmisión).

- Elegido: **Mallorca–Barcelona, 72:00-82:00 de vídeo** (2.ª parte, minutos 24-34 de la parte), que contiene los
  dos primeros goles (1-0 en 2:05.2 y 1-1 en 7:10.8 del clip) y, según nuestro pipeline, 1 córner, 1 falta y
  7 saques de banda. Fichero `output/benchmark/vendor/mallorca_barcelona_j28_72m_10min.mp4` (600 s, 225 MB,
  recodificado con corte exacto en 72:00, H.264 CRF 20, audio AAC). En el onboarding de Scoutiqo encaja con
  «camera operated» (no es Veo ni TV con montaje).
- Derechos: el vídeo original procede de YouTube; las condiciones de Scoutiqo piden contenido propio o
  atribución. Lo decide el usuario (indicar el canal de origen como atribución si se pide).
- GT: no existe anotación de eventos de este partido. Plantilla
  `mallorca_barcelona_j28_72m_10min.annotation.csv` (formato de `scripts/annotation_csv.py`, **tiempos del
  clip**) con los dos goles ya puestos; hay que anotar al menos tiros, saques de banda, faltas/córneres y
  pérdidas (out, entradas, bloqueos), que son los grupos de Scoutiqo; los pases son opcionales (unos 100 en
  10 min). `mallorca_barcelona_j28_72m_10min.pipeline_candidates.csv` lista los 122 eventos y acciones que emitió
  nuestro pipeline en esa ventana (por si ayuda a situar cosas), pero anotar a partir de él sesga el GT hacia
  nuestro modelo: mejor anotar viendo el clip y usarlo solo para comprobar.
- Evaluación: `eval_vendor.py --gt-csv <hoja> --vendor-csv <línea de tiempo de Scoutiqo> --ours-npz
  <clip>.tdeed.SoccerNetBall_challenge2.npz` (caché de T-DEED calculada sobre el propio clip con el modelo
  de producción). Los grupos sin GT en la hoja no puntúan. Nuestra referencia con la hoja completa se añade
  cuando exista la anotación.
- Primer dato nuestro en el clip [M, 2026-11-06], con solo los dos goles como GT: T-DEED `challenge2` emite
  83 eventos a umbral 0,3 (62 de pase, 7 de balón parado, 14 de pérdida) y **ningún tiro**: la probabilidad
  máxima de SHOT a ±8 s de los goles es 0,09 (1-0) y 0,19 (1-1), y la de GOAL 0,02. Solo a umbral 0,05 cubre
  los dos goles (5 picos). En producción esos dos goles los confirma el marcador (`telemetry.json`: 4445,25 s y
  4750,75 s, confianza 1,0), no T-DEED. Frente a los *Shots* de Scoutiqo, pues, nuestro tiro en este dominio es
  el punto débil conocido (`shot-detection-findings`).

**Clip etiquetado con cámara que sigue el balón para Scoutiqo** [2026-11-08]. El usuario pide un tramo de 10 min
de un partido **ya etiquetado** para no anotar a mano. Los únicos partidos con eventos etiquetados que se pueden
subir a un tercero son los de SoccerTrack v2 (SoccerNet y SoccerHigh, no). Para que siga el balón se usa la
**cámara virtual** de `scripts/shot_benchmark/panorama_vcam.py` (ventana 16:9 sin reescalar que sigue la pista
del balón del dataset, suavizada y con velocidad limitada): es el equivalente al vídeo «follow» que exporta una
Veo, que es justo la entrada para la que está hecho Scoutiqo.

- Ventana elegida: **132831, 1.ª parte, 31:00-41:00** (partido de evaluación, partido diurno). Entre las 20 partes
  de los 10 partidos es la ventana de 10 min con más tiros: 238 acciones, con 1 gol (36:05), 8 tiros, 3 centros,
  4 faltas, 2 saques de banda, 10 fueras, 4 entradas y 3 bloqueos. Alternativa con más goles: 128058 2.ª 5:30
  (2 goles, 4 tiros) pero es partido de entrenamiento de `SNB_pano1`.
- Ficheros (`output/benchmark/vendor/`): `st2_132831_1st_31m_10min_vcam1280.mp4` (1280x720, 25 fps, para subir;
  audio del partido remuxado), `st2_132831_1st_31m_10min_vcam796.mp4` (misma cámara a 796x448, el formato
  nativo de `SNB_pano1`, para nuestra referencia) y `st2_132831_1st_31m_10min_vcam1280.gt.json` (GT relativo
  al clip, 12 clases, equipo, jugador; generado con `eval_vendor.py --gt-st2 132831 1st 1860`).
- Aviso: la pista del balón del dataset está reconstruida entre los eventos anotados, así que el movimiento de la
  cámara lleva implícito cuándo ocurren; vale para probar un sistema ya entrenado (Scoutiqo) pero no es una cámara
  independiente del GT. Onboarding de Scoutiqo: «Veo or a similar automatic camera».
- Evaluación: `eval_vendor.py --clip output/benchmark/vendor/st2_132831_1st_31m_10min_vcam1280 --vendor-csv
  <línea de tiempo> --vendor-name scoutiqo --ours-npz output/benchmark/vendor/st2_132831_1st_31m_10min_vcam796.tdeed.SNB_pano1.npz`.
- Nuestra referencia en el clip [M, 2026-11-08], umbral 0,3 y NMS 1 s, F1 a ±1 s / ±3 s
  (`st2_132831_1st_31m_10min_vcam1280.eval.json`):

| Modelo (vista) | shot (9 GT) | pass (119) | set_piece (6) | turnover (17) |
|---|---|---|---|---|
| `SNB_pano1` (cámara 796x448 nativa) | 16,7 / 16,7 (P 33, R 11) | 78,7 / 84,3 | 28,6 / 28,6 | 48,0 / 56,0 |
| `SoccerNetBall_challenge2`, modelo TV de producción (cámara 1280x720) | 0 / 0 (sin picos) | 69,8 / 77,2 | 25,0 / 25,0 | 0 / 0 |

  El tiro vuelve a ser el punto débil: `SNB_pano1` solo emite 3 tiros en 10 min con 8 tiros y 1 gol, y el modelo
  de TV ninguno. Para saques y faltas los dos ven poco (1-2 de 6). Con esta ventana la comparación con Scoutiqo
  es exigente para nosotros en *shots*, que es lo que interesa medir.

**Paquete de prueba para STREET** (partido completo, que es lo que procesa su plan gratuito):

- Vídeo: `datasets/soccertrack-v2/videos/128057/128057_panorama_{1st,2nd}_half.mp4` (4096x1080, 25 fps, 3,3 y
  3,5 GB; sin recodificar), el mismo partido que el clip de Scoutiqo. 5 goles, todos en la 2.ª parte (1:31 right,
  24:57 right, 31:14 left, 35:54 left, 45:11 right) y 22 tiros; la 1.ª parte, sin goles, mide los goles falsos.
  Alternativa si el plan admite más partidos: 132831 (3840x1504, 6,8 GB por parte, 5 goles repartidos 2+3,
  35 tiros). GT de los dos en `output/benchmark/vendor/st2_<partido>_full.gt.json` (12 clases, equipo, jugador).
- `scripts/benchmark/eval_vendor_highlights.py`: goles de su página del partido (CSV `parte,mm:ss[,left|right]`)
  → P/R/F1 a ±10 s y ±30 s y marcador por lado; segmentos de su reel (CSV `parte,inicio,fin[,etiqueta]`, tiempos
  del vídeo original) → goles y tiros del GT cubiertos, segmentos con gol o tiro (precisión), número y duración.
  Nuestra referencia con la misma regla: picos de T-DEED `SNB_pano1` (SHOT+GOAL, NMS 1 s) como candidatos de 15 s
  ([t−10, t+5]), que es el generador de candidatos de tiro de `analytics/highlights` en panorámica.

Nuestra referencia [M, 2026-11-06], `output/benchmark/vendor/st2_<partido>_full.eval.json`:

| Partido | Umbral | Candidatos (duración) | Goles cubiertos | Tiros cubiertos | Candidatos con tiro o gol |
|---|---|---|---|---|---|
| 128057 | 0,3 | 44 (660 s) | 4/5 | 16/22 | 25 (56,8 %) |
| 128057 | 0,15 | 59 (885 s) | 5/5 | 16/22 | 28 (47,5 %) |
| 132831 | 0,3 | 23 (345 s) | 2/5 | 15/35 | 16 (69,6 %) |
| 132831 | 0,15 | 47 (705 s) | 4/5 | 25/35 | 30 (63,8 %) |

La clase GOAL de T-DEED no da ningún pico a 0,3 en panorámica (0/5 en los dos partidos; a 0,15, 2 falsos y
0 aciertos en 128057). En `main` los goles se confirman por marcador o por saque de centro (`analytics/goals.py`):
en SoccerTrack v2 no hay marcador, y el saque de centro exige correr la rama panorámica completa con tracking sobre
el partido, cosa que no se ha hecho. Por tanto, frente a los goles de STREET tenemos hoy «lo que marcaríamos como
tiro»; la cifra de gol real del pipeline en este partido queda pendiente.

**Pasos con STREET (los hace una persona, con la app)**:
1. Instalar la app STREET (iOS o Android) y crear la cuenta gratuita. Subir las dos partes con el mismo nombre de
   partido el mismo día (es lo que su FAQ pide para que salga el marcador), o enviar enlaces de Drive «anyone with
   the link». Poner «SoccerTrack v2 (CC BY 4.0)» en el nombre o la descripción del partido como atribución.
2. Al recibirlo (menos de 24 h según ellos): de la página del partido, transcribir marcador, goleadores y línea de
   tiempo a `output/benchmark/vendor/street_128057_goals.csv` (`parte,mm:ss[,left|right]`); del vídeo de highlights,
   anotar cada segmento con los tiempos del vídeo original en `street_128057_segments.csv`
   (`parte,inicio,fin,etiqueta`); si el reel no deja situar un segmento, anotar solo el instante central (se usa
   ±7,5 s). Guardar el reel y capturas de la página del partido en `output/benchmark/vendor/`.
3. `python -m scripts.benchmark.eval_vendor_highlights --match 128057 --vendor-name street
   --goals output/benchmark/vendor/street_128057_goals.csv --segments output/benchmark/vendor/street_128057_segments.csv
   --ours SNB_pano1`.

Límites con STREET: su salida es un vídeo editado (zoom, repeticiones, rótulos), así que situar cada segmento en el
tiempo original es manual (±2-3 s); el reel es una selección editorial de 2-5 min, no una detección exhaustiva, por
lo que la cobertura de tiros mide criterio además de detección; de precisión solo responden los goles de la página
del partido. Si las «stats» de noviembre llegan a la app con línea de tiempo de eventos, se transcriben al mismo
CSV de goles (y, si dan pases o tiros, al formato de `eval_vendor.py` por parte).

### 7.10 T1 tracking TV: BoT-SORT (GMC + ReID) integrado en `core/tracking.py` [M, 2026-11-06]

**Qué cambió en el código.** `PlayerTracker` usa por defecto `tracker="botsort"`: BoT-SORT de ultralytics (Kalman XYWH,
compensación global de movimiento GMC por flujo óptico disperso y ReID opcional) envuelto en `core.tracking.BotSortTracker`.
El ReID es un `core.classification.Embedder` inyectado: `reid="osnet"` (OSNet x1.0 "sports", `core/osnet.py`, pesos en
`models/reid/osnet_x1_0_sports.pth.tar-60`) o `reid="prtreid"` (el mismo PRTReId de la asignación de equipos). El tracker
emite la caja DETECTADA, no la del Kalman (con el oráculo de §7.1 el Kalman perdía 9 puntos de LocA), así que rol, dorsal y
confianza siguen siendo los de la detección del Deformable-DETR. `tracker="bytetrack"` conserva el `supervision.ByteTrack`
anterior y reproduce al decimal la fila de §7.2 (HOTA 56,4 / IDF1 66,0 / 4.892 IDSW). Tests: `tests/test_tracking_botsort.py`.
Variantes en `scripts/benchmark/track_mot.py` (`ours_*`); análisis de identidades en `scripts/benchmark/id_analysis.py`.

**Test de SoccerNet-Tracking, 49 secuencias, mismas detecciones híbridas, modo `persons`.** Los fps incluyen la lectura del
JPG 1080p y van con la máquina compartida (carga 10-48); el coste aislado está más abajo.

| Tracker | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW | FP | FN | fps |
|---|---|---|---|---|---|---|---|---|---|---|
| ByteTrack de `supervision` (baseline, `tracker="bytetrack"`) | 56,4 | 69,2 | 46,2 | 81,6 | 87,1 | 66,0 | 4.892 | 22.783 | 40.687 | 751 |
| BoT-SORT + GMC, sin ReID (`reid="none"`) | 57,5 | 69,0 | 48,1 | 81,4 | 86,4 | 67,0 | 3.565 | 40.466 | 28.028 | 28 |
| BoT-SORT + GMC + ReID OSNet (`reid="osnet"`) | 59,1 | 67,2 | 52,1 | 81,4 | 83,4 | 69,6 | 4.209 | 57.524 | 26.325 | 17 |
| BoT-SORT + GMC + NMS 0,7 + ReID OSNet (candidata 1 de train) | 59,1 | 68,8 | 51,1 | 81,5 | 85,9 | 69,8 | 4.055 | 41.147 | 29.355 | 29 |
| **BoT-SORT + GMC + NMS 0,7 + OSNet + buffer 60 + perdidos por apariencia 0,95** (candidata 2, mejor de train; **valor por defecto desde el 6-nov-2026**) | **59,4** | 68,9 | 51,3 | 81,5 | 86,1 | **70,5** | 4.199 | 39.395 | 29.943 | 33 |
| BoT-SORT + GMC + ReID PRTReId (`reid="prtreid"`) | 58,7 | 67,3 | 51,4 | 81,4 | 83,4 | 68,9 | 4.505 | 57.213 | 26.348 | 10 |
| Deep-EIoU `rgb` (§7.4, referencia) | 62,4 | 70,1 | 55,7 | 81,5 | 88,3 | 76,8 | 2.655 | 26.864 | 32.612 | — |
| Deep-EIoU `rgb` + GTA-Link (§7.4, offline) | 64,3 | 70,1 | 59,2 | — | — | 81,3 | 3.496 | — | — | — |

Oráculo (cajas del GT sin identidad; aísla la asociación):

| Tracker | HOTA | AssA | IDF1 | IDSW |
|---|---|---|---|---|
| ByteTrack de `supervision` (§7.1) | 82,3 | 69,5 | 77,0 | 5.074 |
| BoT-SORT de ultralytics tal cual (§7.1, cajas del Kalman) | 83,8 | 73,7 | 81,6 | 1.695 |
| BoT-SORT + GMC, sin ReID (`core/tracking.py`, cajas detectadas) | 86,6 | 75,6 | 81,5 | 1.695 |
| BoT-SORT + GMC + ReID OSNet | 86,4 | 75,4 | 81,3 | 1.632 |
| Deep-EIoU `rgb` (§7.4) | 87,9 | 77,6 | 84,0 | 1.865 |

Con cajas perfectas el ReID no aporta nada (86,4 frente a 86,6): su ganancia con el detector real viene de resolver
asociaciones que las cajas imprecisas dejan ambiguas, no de recuperar identidades. Emitir la caja detectada en vez de la
del Kalman vale +2,8 HOTA con oráculo (86,6 frente a 83,8), por localización.

**De dónde vienen los cambios de identidad** (`id_analysis.py`, test, 49 secuencias):

- *Paneo de cámara*: mediana por frame del desplazamiento de las cajas del GT (un paneo mueve a todos a la vez). Terciles:
  quieta 3,9 px/frame (16 seq), media 5,4 (16), mucho paneo 7,8 (17).

  | Tracker | IDSW quieta | IDSW media | IDSW mucho paneo | total |
  |---|---|---|---|---|
  | ByteTrack | 1.177 | 1.317 | 2.398 | 4.892 |
  | BoT-SORT + GMC | 1.062 (−10 %) | 877 (−33 %) | 1.626 (−32 %) | 3.565 (−27 %) |
  | BoT-SORT + GMC + OSNet | 1.202 | 1.010 | 1.997 | 4.209 |
  | BoT-SORT + GMC + NMS + OSNet | 1.232 | 938 | 1.885 | 4.055 |
  | ídem + buffer 60 + perdidos por apariencia 0,95 (por defecto) | 1.244 | 1.028 | 1.927 | 4.199 |
  | BoT-SORT + GMC + PRTReId | 1.321 | 1.124 | 2.060 | 4.505 |
  | Deep-EIoU `rgb` | 670 | 664 | 1.321 | 2.655 |

  El baseline dobla los IDSW en el tercil con más paneo; la GMC recorta justo ahí (un tercio) y casi nada en cámara quieta.
- *Reentradas* (identidad del GT que desaparece > 5 frames y vuelve; id del tracker en los 10 frames antes y después):
  720 en el test, hueco mediano **109 frames (4,4 s)**, p25/p75 57/199; el **97 % salen por el borde del plano**. Solo el
  10 % de los huecos cabe en el buffer de perdidos de 1 s que usan ByteTrack y BoT-SORT (`lost_track_buffer=30`).

  | Tracker | recuperada | cambiada | sin emparejar | ids del tracker por identidad del GT |
  |---|---|---|---|---|
  | ByteTrack | 1 (0,1 %) | 552 | 167 | 4,78 |
  | BoT-SORT + GMC | 10 (1,4 %) | 689 | 21 | 3,94 |
  | BoT-SORT + GMC + OSNet | 9 (1,2 %) | 692 | 19 | 4,57 |
  | BoT-SORT + GMC + NMS + OSNet | 9 (1,2 %) | 692 | 19 | 3,96 |
  | ídem + buffer 60 + perdidos por apariencia 0,95 (por defecto) | 30 (4,2 %) | 671 | 19 | 3,94 |
  | BoT-SORT + GMC + PRTReId | 9 (1,2 %) | 692 | 19 | 4,63 |
  | Deep-EIoU `rgb` | 107 (14,9 %) | 585 | 28 | 2,96 |
  | Deep-EIoU `rgb` + GTA-Link (offline) | **317 (44,0 %)** | 375 | 28 | 2,82 |

  **El ReID de BoT-SORT no recupera reentradas por diseño**: solo interviene cuando la caja predicha y la detección ya
  tienen IoU ≥ `proximity_thresh` (0,5), y el track perdido se borra al segundo. Lo que sí hace es asociar mejor dentro
  del plano (AssA +4, IDF1 +2,6).
- *Duplicados*: a conf ≥ 0,1 el detector híbrido emite un **12,2 %** de cajas con otra a IoU > 0,7 (2,3 % a conf ≥ 0,3).
  En la salida: ByteTrack 1,4 %, BoT-SORT 3,0 %, BoT-SORT + OSNet **4,5 %**, Deep-EIoU 1,4 %. El ReID "resucita" un track
  perdido sobre el duplicado del mismo jugador: de ahí los +17 k FP y los IDSW de más con OSNet. Arreglo: NMS previo
  (`nms_iou=0.7`, sin clases) solo en la ruta BoT-SORT: en test quita 16 k FP (57,5 k → 41,1 k), devuelve la
  fragmentación al nivel sin ReID (3,96 ids por identidad) y deja el HOTA igual (59,1) con DetA +1,6 y AssA −1,0.

**Ajuste en el split de train** (12 secuencias SNMOT-060..071, detecciones híbridas nuevas en `output/benchmark/snmot_train/`;
`scripts/benchmark/run_snmot_train_tuning.sh`). El test se usa una sola vez con la variante elegida aquí.

| Variante (train, 12 seq) | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN | reentradas recuperadas | fps |
|---|---|---|---|---|---|---|---|---|---|---|
| BoT-SORT + GMC | 55,3 | 68,0 | 45,1 | 85,2 | 64,4 | 1.257 | 13.319 | 8.281 | 1/186 | 43 |
| + NMS 0,7 | 54,6 | 69,4 | 43,1 | 87,2 | 63,4 | 1.231 | 9.326 | 9.304 | 1/186 | 44 |
| + OSNet | 55,6 | 66,0 | 47,0 | 81,4 | 64,6 | 1.595 | 19.512 | 7.709 | 1/186 | 23 |
| **+ NMS 0,7 + OSNet** | **56,0** | 68,1 | 46,2 | 85,1 | 65,5 | 1.509 | 12.803 | 8.795 | 2/186 | 23 |
| + NMS + OSNet, buffer 60 frames | 56,0 | 68,1 | 46,2 | 85,0 | 65,2 | 1.535 | 12.848 | 8.797 | 2/186 | 24 |
| + NMS + OSNet, perdidos solo por apariencia (0,8) | 54,0 | 67,5 | 43,3 | 82,0 | 65,6 | 4.640 | 14.210 | 9.022 | 19 (10,2 %) | 24 |
| ídem, buffer 60 | 54,5 | 67,4 | 44,1 | 81,6 | 67,2 | 4.943 | 14.611 | 8.841 | 24 (12,9 %) | 24 |
| ídem, buffer 125 | 54,8 | 67,3 | 44,7 | 81,5 | 68,5 | 5.087 | 14.647 | 8.813 | 18 (9,7 %) | 23 |
| ídem, buffer 60, apariencia 0,9 | 55,6 | 67,6 | 45,9 | 83,0 | 67,7 | 3.317 | 14.017 | 8.901 | 31 (16,7 %) | 24 |
| **ídem, buffer 60, apariencia 0,95** | **57,0** | 68,2 | 47,7 | 85,0 | 67,7 | 1.767 | 12.470 | 8.941 | 5 (2,7 %) | 23 |

Lectura del ajuste: el ReID sin la puerta de IoU para los tracks perdidos (`reid_lost_without_gate=True`) con el umbral
de apariencia por defecto (0,8) sí recupera el 10-13 % de las reentradas, pero cuadruplica los IDSW y cada id del tracker
acaba mezclando 3,6-4,5 identidades del GT: los compañeros con la misma equipación pasan el umbral. A 0,9 sigue mezclando
(2,7 identidades por id). A **0,95** deja de mezclar (1,57, como sin ReID), recupera solo el 2,7 % de las reentradas pero
da el mejor HOTA de train (57,0, +1,0 sobre NMS + OSNet) con un 17 % más de IDSW que esa variante. Dos candidatas pasan
a test: NMS + OSNet (elegida antes de probar los umbrales estrictos) y NMS + OSNet + buffer 60 + perdidos por apariencia
a 0,95 (la mejor de train).

**Coste por frame del tracker, aislado** (SNMOT-117, 150 frames en memoria, sin detector ni lectura de vídeo; GPU compartida):

| Tracker | ms/frame |
|---|---|
| ByteTrack de `supervision` | 0,8 |
| BoT-SORT, GMC apagada | 0,9 |
| BoT-SORT + GMC sparseOptFlow | 13,9 |
| BoT-SORT + GMC + NMS | 14,5 |
| BoT-SORT + GMC + NMS + OSNet | 32,0 |
| BoT-SORT + GMC + NMS + PRTReId | 97,8 |

El detector híbrido tarda 36 ms/frame (28 fps, §7.2): con GMC el tracking completo pasa de ~28 a ~20 fps y con OSNet a
~15 fps (un partido de 90 min: 80 → 110 → 150 min). PRTReId como ReID del tracker no compensa (98 ms y, ver tabla, sin
ventaja de precisión). La normalización de los crops de OSNet se hace en la GPU: en numpy costaba 220 ms/frame con la CPU
compartida.

**Conclusión y valor por defecto.** El test confirma el orden de train (59,4 ≥ 59,1), así que `TrackingConfig` queda en
BoT-SORT + GMC sparseOptFlow + NMS 0,7 + ReID OSNet, `lost_track_buffer=72` (60 frames a 25 fps),
`reid_lost_without_gate=True` y `appearance_thresh=0.95`. Frente al ByteTrack anterior, con las mismas cajas:

| | ByteTrack (antes) | BoT-SORT por defecto (ahora) | Δ |
|---|---|---|---|
| HOTA | 56,4 | 59,4 | +3,0 |
| IDF1 | 66,0 | 70,5 | +4,5 |
| IDSW | 4.892 | 4.199 | −14 % |
| ids del tracker por identidad del GT | 4,78 | 3,94 | −18 % |
| reentradas recuperadas (de 720) | 1 | 30 | — |
| tracker, ms/frame (aislado) | 0,8 | ~32 | pipeline 28 → ~15 fps |

- **La GMC es la pieza que ataca el paneo**: −27 % de IDSW con las mismas cajas, concentrado en el tercil con más
  movimiento de cámara (−32 %) y casi nada en cámara quieta; +1,1 HOTA, +1,0 IDF1, 14 ms/frame y sin pesos nuevos. Si el
  tiempo de proceso manda, `reid="none"` se queda con esto (57,5 / 67,0 / 3.565) a ~20 fps.
- **El ReID aporta menos de lo que promete**: +1,6 HOTA y +2,6 IDF1 sobre la GMC, pero sube los IDSW (resucita tracks sobre
  duplicados del detector: hace falta el NMS) y, tal como lo integra BoT-SORT, **no recupera reentradas** (1,2 %): la puerta
  de IoU y el buffer de 1 s lo impiden, y las reentradas duran 4,4 s de mediana. Dejarle recuperar perdidos por apariencia
  solo funciona con un umbral muy estricto (0,95 → 4,2 %, sin mezclar compañeros); a 0,8-0,9 mezcla identidades.
- **El problema "sale del plano y vuelve" sigue abierto**: Deep-EIoU (15 %) y sobre todo un enlace offline como GTA-Link
  (44 %, IDF1 81,3) siguen siendo la palanca, y encajan con el modo partido completo. Deep-EIoU (62,4 HOTA) marca el techo
  con estas cajas; el cuello mayor sigue siendo el detector (DetA 69 frente a 97,5 con oráculo; 12 % de duplicados).
- **Avisos**: los pesos "sports" de OSNet no tienen licencia (benchmark interno; para producto, OSNet de torchreid (MIT) o
  PRTReId, que da 58,7 pero a 98 ms/frame); todo esto es TV profesional 1080p: en juveniles no hay GT de tracking, solo
  el efecto indirecto en `evaluate_match.py` (tracker del ejecutor, equipo), y la caché `*.tracks.jsonl` de los partidos ya
  analizados hay que borrarla para que el nuevo tracker se note.

### 7.11 Tabla única de spotters con pesos públicos: TV (test oficial de SN-BAS) y panorámica (cámara virtual) [M, 2026-11-08]

Pregunta del usuario: qué modelos del estado del arte detectan eventos, qué métricas dan en nuestros conjuntos de
evaluación y cuánto tardan. Aquí se consolida lo de §7.3-7.8 con lo nuevo: coste de **todos** los modelos con la GPU
libre y la panorámica con la cámara virtual que sigue el **balón anotado** del dataset (`vcam796`), además de la
nuestra (`vcam796vl1`). Script: `scripts/benchmark/run_bas_views.sh` (cachés, evaluador oficial y coste; reanudable).

#### Conjuntos de evaluación y contaminación (splits leídos en cada repo, no de memoria)

| Conjunto | Partidos | Acciones GT | Limpio para | No limpio para |
|---|---|---|---|---|
| **TV: test oficial de SN-BAS** (`models/tdeed/T-DEED/data/soccernetball/test.json`) | **2**: Stoke City–Huddersfield Town y Reading–Fulham (EFL Championship, 2019-10-01, 720p, 25 fps, partidos completos) | 3.980 (12 clases; FREE KICK 2, GOAL 6, TACKLE 28) | T-DEED `SoccerNetBall_test` (train = Blackburn, Brentford, Hull, Leeds; val = Middlesbrough), sn-teamspotting baseline (su devkit entrena en `train`, para en `valid` y evalúa en `test`: mismos ficheros), AdaSpot-s/-b (train 4, val Middlesbrough: paper §4 y configs), E2E-Spot de OSL (SN-BAS-2025; el split no consta en su card, se supone train [NV]) | T-DEED `challenge2` (`train_challenge.json` = los 6 incluidos Stoke y Reading) y `SNB_pano1` (deriva de challenge2) |
| **Panorámica: eval oficial de SoccerTrack v2** | **2** partidos, 4 partes: 128057 (M7, 4096x1080) y 132831 (M9, 3840x1504) | 4.372 (HEADER 5, GOAL 10, FREE KICK 31) | Todos: ningún modelo de TV vio panorámica; `SNB_pano1` afinó solo con los 6 de entrenamiento (117092 fuera) y validó en 117093 + 132877 | — |

Las dos cámaras virtuales son 796x448 a 25 fps, sin reescalar a los jugadores (~46 px de alto):

- `vcam796`: sigue la pista de balón del dataset (reconstruida entre eventos anotados), media móvil 1 s y tope de
  40 px/frame (`scripts/shot_benchmark/panorama_vcam.py`). Es la cámara "ideal": sabe dónde está el balón. En 132831
  la pista está parada en el centro del campo (0,0) el 17-18 % de los frames, pero el 98,2-98,5 % de los eventos
  tienen balón válido en su frame (128057: 100 %) [M, `ball/*.npz` frente a `bas/*_12_class_events.json`].
- `vcam796vl1`: pista Viterbi sobre nuestro detector `yolo11s_pano` y cámara L1 (`scripts/shot_benchmark/vcam_path.py`).
  Es la cámara que tendría el producto. Resultados anteriores en §7.6.

**Hallazgo colateral sobre SoccerNet-v2 (17 eventos)**: `train_tdeed.py` → `dataset/datasets.py:79-90` carga para el
preentrenamiento solo `data/soccernet/train.json` (300 partidos) y `val.json` (100). El **`test` oficial de SoccerNet-v2
(100 partidos) no lo vio ningún checkpoint de T-DEED ni el baseline de sn-teamspotting** (mismos ficheros de split). Eso
corrige lo dicho en §1 y en `docs/particion_datasets.md` §1 ("ningún partido de v2 sirve para evaluar T-DEED"): la
cabeza de eventos de producción (Goal, Corner, Foul, tarjetas…) se puede medir limpia sobre esos 100 partidos con la
métrica oficial (Average-mAP tight/loose), y ahí sí son comparables las cifras publicadas (T-DEED Revisited tight 63,42,
E2E-Spot 61,8 tight / 74,1 loose, COMEDIAN 73,1 tight). Además, **los 6 partidos del benchmark de tiros** (Bayern–Augsburg,
Levante–R. Madrid, Arsenal–PSG, PSG–Angers, Lazio–Roma, Liverpool–Swansea) **están todos en ese `test`**: las AP de tiros
de `docs/estudio_deteccion_tiros.md` sobre ellos eran limpias para T-DEED. El Clásico está en `val` (in-sample, como se
decía). Los 100 partidos (1,6 GB cada uno, `1_720p.mkv` + `2_720p.mkv`) están en `/mnt/datasets/SoccerNet/` (solo host,
el contenedor no lo ve; lista en `output/benchmark/sn_v2_test_games.txt`). No se ha ejecutado: copiar 100 partidos son
~160 GB y ~16 h de GPU por modelo (800MF); un piloto de 20 partidos serían ~32 GB y ~3 h.

#### Modelos del estado del arte buscados (2026-11-08) y por qué están o no en la tabla

| Modelo | ¿Detecta eventos con instante? | ¿Pesos públicos? | Decisión |
|---|---|---|---|
| T-DEED (`SoccerNetBall_test` 200MF limpio, `challenge2` 800MF de producción, `SNB_pano1` afinado) | Sí, 12 BAS + 17 SN-v2 | Sí (Drive del repo; GPL-3.0) | **En la tabla** |
| sn-teamspotting baseline 2025 (T-DEED 200MF + cabeza de equipo) | Sí, 12 × equipo + 17 SN-v2 | Sí (Drive del devkit; GPL-3.0) | **En la tabla** |
| AdaSpot-b / -s (Xarles et al. 2026) | Sí, 10 BAS (sin FREE KICK ni GOAL) | Sí (HF `arturxe/AdaSpot`) | **En la tabla** |
| E2E-Spot de OpenSportsLab (SN-BAS-2025, 224p, ≈2 fps) | Sí, 12 BAS | Sí (HF, AGPL-3.0) | **En la tabla** (sirve de suelo: 2 fps) |
| **SoccerMaster** (arXiv 2512.11016, SJTU) | **No**: su cabeza de eventos es clasificación de clip (24 clases de SoccerReplay-1988, exactitud 73,8 %), sin instante. Cabezas: detección, keypoints, líneas, clasificación de eventos, alineación visión-texto; MOT y comentarios como adaptaciones | Parcial (HF `xleprime/SoccerMaster`: backbone 1,44 GB, detección GSR, keypoints, líneas, YOLOv8x6, `CaptionClassification.pt`; nada de spotting) | No comparable; no se ejecuta |
| **sn-gamestate** (TrackLab) | No: tracking + equipo + dorsal + localización en el campo (GSR) | Sí | No tiene eventos |
| UniSoccer / MatchVision (CVPR 2025, mismos autores que SoccerMaster) | No: clasificación de clip (24 clases) y comentarios | Sí (HF `Homie0609/UniSoccer`) | No comparable |
| Ganadores de Team BAS 2025 (dudek 60,03, Intellindust 56,78, KIST 56,08 Team-mAP@1) | Sí | **No** (dudek publica el framework `dude.k`, sin pesos) | Solo cifra [P] |
| PCBAS 2026 (PAVE 58,94 Macro-F1; baselines FOOTPASS) | Sí, con jugador | No / sin pesos; necesita tracklets e identidades | Ver §7.7 |
| lRomul, 1.º de BAS **2023** (EfficientNetV2-B0 + bloques 3D, gris 1280x736, 15 frames) | Sí, pero solo **PASS y DRIVE** | Sí (Drive, MIT), por fold de validación cruzada (7 folds, uno por partido) | No se ejecuta: 2 de 12 clases, entorno Docker propio con VideoProcessingFramework y sin script para vídeo suelto. Publicado: 87,03 mAP@1 en test (2 clases) [P]. Se podría añadir (~1 día) si interesara PASS/DRIVE |
| CALF, NetVLAD++, COMEDIAN, ASTRA, Baidu+Transformer (SN-v2) | Sí, 17 clases a segundos | Sí, pero sobre features Baidu/ResNet que no existen para EFL ni SoccerTrack | Excluidos (§4). Comparables solo vía el `test` de SN-v2 (arriba) |
| E2E-Spot original (Hong et al. 2022) | Sí, 17 SN-v2 a 2 fps | Repo oficial | Misma arquitectura que el OSL ya medido; no aporta |

#### Resultados [M, 2026-11-08]

Todo medido con el evaluador oficial de T-DEED (`util.eval`), NMS suave de 1 s y candidatos ≥ 0,01; cachés
`<vídeo>.tdeed.<modelo>.npz`; resúmenes en `output/benchmark/bas/<modelo>/test/` (TV) y
`output/benchmark/bas_st2/<modelo>/<vista>/` (panorámica). «Tiros» y «Goles» son la AP de las clases SHOT y GOAL.

**TV: test oficial de SN-BAS** (Stoke + Reading, 3.980 acciones). Consolida §7.3, §7.5 y §7.8; las cifras salen de los
`summary.json` (la tabla de §7.3 daba 55,1 para AdaSpot-b por una evaluación anterior: la vigente es 54,9, como en §7.8).

| Modelo | ¿Limpio? | mAP@1 (12 cl.) | mAP@1 (10 cl.) | tight (12 cl.) | Tiros @1 | Goles @1 | Publicado [P] |
|---|---|---|---|---|---|---|---|
| **T-DEED `SoccerNetBall_test`** (RegNetY-200MF, BAS + SN-v2) | Sí | **61,2** | **62,5** | **65,9** | **73,9** | **24,2** | 58,91 (800MF, Revisited) |
| sn-teamspotting baseline (200MF + cabeza de equipo) | Sí | 55,5 | 56,4 | 60,1 | 59,7 | 17,4 | 53,59 |
| AdaSpot-b (10 clases) | Sí | — | 54,9 | — (10 cl. 59,8) | 52,8 | — | 56,24 ± 0,3 |
| AdaSpot-s (10 clases) | Sí | — | 54,2 | — | 51,0 | — | 53,12 ± 1,4 |
| E2E-Spot de OpenSportsLab (≈ 2 fps) | Sí [NV, split] | 34,5 | 31,1 | 48,5 | 15,4 | 18,2 | tight 47,98 |
| T-DEED `challenge2` (producción, 800MF) | **No** | *83,2* | — | — | *89,9* | *80,8* | — |
| T-DEED `SNB_pano1` (afinado en panorámica) | **No** | no se mide en test; en Middlesbrough (val) pierde 11 pt de tiros frente a `challenge2` (hoja de ruta §3.6) | | | | | |

- No hay cifra limpia del 800MF de producción en TV: exigiría entrenar un 800MF solo con `train` (≈ 1 jornada de GPU) o
  medir su cabeza de 17 eventos en el `test` de SoccerNet-v2 (arriba). Los dos partidos del split `challenge` (Cardiff–QPR,
  Wigan–Birmingham) no tienen etiquetas públicas ni los tenemos en disco.
- **La cifra casi limpia del modelo de producción es Middlesbrough** (`val`: no entrenó con él, pero eligió la época en él;
  1.604 acciones, 1 gol) [M, 2026-11-08, `output/benchmark/bas/SoccerNetBall_challenge2/val/summary.json`]:

  | Modelo en Middlesbrough | mAP@1 (12 cl.) | mAP@1 (10 cl.) | Tiros @1 (25) | Centro (24) | Cabezazo (127) | Fuera (75) | Bloqueo (28) |
  |---|---|---|---|---|---|---|---|
  | T-DEED `challenge2` (800MF, 6 partidos) | **77,2** | **82,6** | **88,4** | **91,2** | **92,1** | **60,3** | **78,7** |
  | T-DEED `SoccerNetBall_test` (200MF, 4 partidos) | 60,2 | 62,8 | 77,6 | 65,0 | 76,1 | 29,2 | 57,7 |
  | AdaSpot-b | — | 55,3 | 63,2 | 56,7 | 74,8 | 18,6 | 43,6 |

  El `SoccerNetBall_test` da en Middlesbrough (60,2) casi lo mismo que en el test (61,2) pese a haber elegido ahí la época,
  así que el sesgo de validación es de ~1 punto y el nivel real del modelo de producción en TV no vista está en torno a 75.
  La diferencia de 17 puntos con el 200MF no es solo la red grande (+2-3 según Revisited): también entrenó con 6 partidos
  en vez de 4. Sube sobre todo lo que el 200MF hace mal: fuera (+31), bloqueo (+21), centro (+26); la clase
  PLAYER SUCCESSFUL TACKLE se omite de la tabla a petición del usuario (inservible en todos los modelos salvo este, 56,8).

**Panorámica: eval oficial de SoccerTrack v2** (128057 + 132831, 4 partes, 4.372 acciones), con las dos cámaras
virtuales. Ningún modelo vio estos partidos. «≥ 10» = media sobre las clases con al menos 10 eventos (fuera HEADER, 5).

| Modelo | mAP@1 balón anotado | mAP@1 nuestra cámara | Δ | mAP@1 (≥ 10) anotado / nuestra | tight anotado / nuestra | Tiros @1 anotado / nuestra | Goles @1 (10) anotado / nuestra |
|---|---|---|---|---|---|---|---|
| **T-DEED `SNB_pano1`** (afinado con nuestra cámara, 6 partidos de train) | **38,4** | **38,3** | +0,1 | **41,6 / 41,5** | **42,0 / 42,3** | 34,5 / **39,0** | **12,3 / 6,2** |
| T-DEED `challenge2` (producción, TV) | 26,6 | 25,4 | +1,2 | 28,9 / 27,7 | 30,9 / 30,6 | 33,8 / 27,2 | 0,4 / 1,3 |
| T-DEED `SoccerNetBall_test` | 21,4 | 22,9 | −1,5 | 23,3 / 24,9 | 26,2 / 27,6 | 16,3 / 17,1 | 1,3 / 0,0 |
| sn-teamspotting baseline | 20,2 | 19,5 | +0,7 | 22,1 / 21,2 | 23,7 / 23,6 | 35,4 / 16,0 | 0,7 / 0,0 |
| AdaSpot-s (10 cl.: 13,7 / 16,3) | 11,4 | 13,6 | −2,2 | 12,5 / 14,8 | 14,1 / 16,6 | 4,1 / 9,3 | — |
| E2E-Spot de OpenSportsLab | 10,7 | 10,1 | +0,6 | 11,6 / 11,0 | 14,6 / 13,6 | 7,8 / 2,9 | 0,0 / 0,0 |
| AdaSpot-b (10 cl.: 3,2 / 5,3) | 2,7 | 4,4 | −1,7 | 3,0 / 4,8 | 3,1 / 5,1 | 0,0 / 0,0 | — |

Por partido (mAP@1 12 clases / Tiros @1; 128057 tiene ~27 tiros y 132831 ~30), los cuatro modelos de la familia T-DEED:

| Modelo | 128057 balón anotado | 128057 nuestra | 132831 balón anotado | 132831 nuestra |
|---|---|---|---|---|
| `SNB_pano1` | **47,3** / **53,4** | 44,1 / 38,7 | 31,3 / 22,8 | **35,4** / **50,1** |
| `challenge2` | 34,9 / 60,9 | 32,4 / 40,8 | 19,9 / 12,2 | 20,9 / 28,3 |
| `SoccerNetBall_test` | 27,6 / 29,2 | 28,8 / 30,7 | 16,9 / 4,3 | 20,5 / 16,2 |
| sn-teamspotting | 27,4 / 67,3 | 23,7 / 22,6 | 14,3 / 12,4 | 16,6 / 9,1 |

Lectura de la panorámica:

- **La cámara "ideal" (balón anotado) no cambia ni el nivel ni el orden**: en las 4 partes, los 7 modelos quedan a
  ±2,2 puntos de mAP@1 de lo que dan con nuestra cámara. El hueco de TV (61,2) a panorámica (38 con el afinado, 25 con
  el modelo de producción) **es de dominio** (fútbol universitario, jugadores de ~46 px, sin realización), no de la
  trayectoria de la cámara.
- Donde la cámara sí cuenta es en **128057** (pista de balón íntegra): la cámara ideal da **+2,5 a +3,7 de mAP@1** a los
  modelos que han aprendido la cámara de TV (`SNB_pano1`, `challenge2`, sn-teamspotting) y **+15 a +45 de AP en tiros**
  (sobre ~27 tiros, con ruido): seguir el balón en el golpeo es lo que hace un operador de TV. Es la misma conclusión
  que `docs/estudio_deteccion_tiros.md` §10 (AP ±3 s de tiros 60 frente a 79), ahora acotada: afecta a los tiros y vale
  ~3 puntos de mAP global.
- En **132831** la referencia es peor que la nuestra (−1 a −4): el render `vcam796` de ese partido (06-10) lleva la cámara al
  círculo central cuando la pista aparca el balón en (0,0) (17-18 % de los frames), y el montaje gran angular penaliza a
  todos. Ahí la comparación no mide la cámara.
- `SNB_pano1` es el único por encima de 30 en cualquiera de las dos cámaras; con la cámara ideal gana en 128057 (47,3) pero
  pierde en tiros en 132831 (22,8 frente a 50,1): está especializado en nuestra cámara.
- AdaSpot-s supera a AdaSpot-b en panorámica (13,6 frente a 4,4) y los dos colapsan; el E2E-Spot a 2 fps se queda en 10.
- **Goles**: ningún modelo pasa de 12 de AP con 10 goles; el gol real sigue dependiendo del marcador o del saque de centro
  (`analytics/goals.py`), no del spotter.

**Coste** (`spotter_cost.py`, GPU RTX 4090 de 24 GB **libre**, 0,5 GB ajenos; misma entrada para todos: clips de 100
frames muestreados a 796x448, uint8, lote 4; punta a punta = decodificar + redimensionar + inferir 5 min de Stoke,
7.502 frames). Sustituye a la tabla de §7.8, tomada con la GPU compartida (T-DEED 200MF daba 7,6 clips/s; aquí 21,4).

| Modelo | Parámetros | GFLOPs/clip (thop, ≈) | Solape / muestreo | Clips/s | s de vídeo por s de GPU | VRAM pico | Punta a punta 5 min | × tiempo real | Partido de 95 min (est.) |
|---|---|---|---|---|---|---|---|---|---|
| T-DEED 200MF `SoccerNetBall_test` | 12,3 M | 288 | 75 % / 1 de 2 | 21,4 | 42,8 | 8,6 GB | 23,9 s | 12,5× | ~7,6 min |
| sn-teamspotting (200MF + equipo) | 12,3 M | 288 | 75 % / 1 de 2 | 18,5 | 37,1 | 8,6 GB | 22,6 s | 13,3× | ~7,1 min |
| T-DEED 800MF `challenge2` = `SNB_pano1` (producción) | 46,2 M | 1.149 | 75 % / 1 de 2 | 9,7 | 19,5 | 14,0 GB | 24,8 s | 12,1× | ~7,9 min |
| AdaSpot-b | 10,6 M | 171 | 50 % / 1 de 2 | 20,9 | 83,7 | 6,3 GB | 21,6 s | 13,9× | ~6,8 min |
| AdaSpot-s | 7,6 M | 87 | 50 % / 1 de 2 | 28,8 | 115,3 | 5,7 GB | 21,8 s | 13,8× | ~6,9 min |
| E2E-Spot de OpenSportsLab (398x224) | 12,6 M | 291 [NV] | 50 % / 1 de 12 | 7,9 (incluye el resize en CPU) | 190,5 | 3,6 GB | **6,7 s** | **44,7×** | **~2,1 min** |

- **De punta a punta todos los modelos a 796x448 cuestan lo mismo** (22-25 s por 5 min, 12-14× tiempo real, ~7-8 min por
  partido): manda la decodificación y el resize (~300 fps), no la red. El 800MF de producción cuesta 2,2× el 200MF en
  inferencia pura y 14 frente a 8,6 GB de VRAM, pero solo un 4 % más en pared.
- AdaSpot-b es 2,0× más barato que el 200MF y 4,3× que el 800MF en inferencia pura (antes se midió 2,9× / 4,4× con la
  GPU compartida), con 7-8 puntos menos de mAP@1 en TV: no compensa.
- El E2E-Spot a 2 fps es el único que ahorra de verdad en pared (45×), porque decodifica 1 de cada 12 frames, pero no
  sirve a ±0,5 s (34,5 en TV).
- Los GFLOPs de thop para OSL (291 a 398x224, lo mismo que el 200MF a 796x448) no cuadran con su tamaño; se dejan como
  orientativos [NV].

**Conclusión de §7.11**: con pesos públicos no hay nada por encima de T-DEED en TV; en panorámica el único utilizable es
nuestro afinado `SNB_pano1`, y la palanca ya no es la cámara (salvo para tiros) sino más datos del dominio para afinar. El
siguiente paso de medida que sí falta es el `test` de SoccerNet-v2 (100 partidos limpios) para la cabeza de 17 eventos del
modelo de producción: piloto de 20 partidos (~32 GB, ~3 h) o los 100 (~160 GB, ~16 h), a decidir por el usuario.

### 7.12 Punto de trabajo (precisión, recall, falsos positivos) de T-DEED y AdaSpot en tres conjuntos, y metraje nuevo (SN-PCBAS-2026) [M, 2026-11-09]

**Por qué**: las tablas de §7.3, §7.6 y §7.11 dan **AP por clase** (Average Precision en %, con tolerancia ±0,5 s): el área
bajo la curva precisión-recall al recorrer todos los umbrales de confianza. Mide lo bien que el modelo *ordena* sus
detecciones y no depende de ningún umbral, pero **no dice cuántos falsos positivos emite en uso**. Para eso,
`scripts/benchmark/eval_prf.py` toma las predicciones ya post-procesadas por `eval_bas.py`
(`<out>/snms12/preds/<partido>/results_spotting.json`), las empareja con el GT igual que el mAP@1 oficial (±0,5 s, 1 a 1,
greedy por confianza) y barre un **umbral global** (el mismo para todas las clases) eligiendo el que maximiza la F1 micro
sobre las clases comunes (sin FREE KICK ni GOAL, que AdaSpot no predice). Informa P, R, F1, TP, FP y FN por clase a ese
umbral y a los umbrales fijos 0,15 (convención de PCBAS) y 0,5; escribe `prf.json` junto al `summary.json`. Los
umbrales óptimos son distintos por modelo (las confianzas de AdaSpot son más bajas), así que no vale comparar los tres a
un mismo umbral.

Formato de las celdas: **AP · P/R · FP** (AP a ±0,5 s; precisión y recall en % al umbral óptimo del modelo; falsos
positivos absolutos a ese umbral). Se comparan los tres modelos con test limpio y mismo protocolo (train 4 partidos, parada
en Middlesbrough): T-DEED 200MF `SoccerNetBall_test`, AdaSpot-b y AdaSpot-s. `challenge2` queda fuera (vio Stoke y Reading).

**1. TV, test oficial de SN-BAS (Stoke + Reading, 3.980 eventos).** Umbral óptimo: T-DEED 0,70, AdaSpot-b 0,35, AdaSpot-s 0,25.

| Clase (nº) | T-DEED 200MF | AdaSpot-b | AdaSpot-s |
|---|---|---|---|
| Pase (1.721) | 86 · 88/85 · 189 | 86 · 89/81 · 175 | 86 · 85/85 · 256 |
| Conducción (1.449) | 80 · 84/78 · 224 | 81 · 81/81 · 283 | 80 · 79/78 · 308 |
| Cabezazo (182) | 69 · 73/64 · 42 | 68 · 78/58 · 30 | 64 · 81/48 · 20 |
| Pase alto (181) | 80 · 87/72 · 20 | 76 · 79/67 · 32 | 77 · 88/58 · 14 |
| Fuera (145) | 37 · 53/37 · 48 | 21 · 32/8 · 26 | 24 · 44/12 · 23 |
| Banda (95) | 82 · 92/76 · 6 | 74 · 83/58 · 11 | 67 · 72/58 · 21 |
| Centro (60) | 81 · 81/72 · 10 | 67 · 81/57 · 8 | 69 · 89/40 · 3 |
| **Tiro (44)** | **74 · 79/59 · 7** | 53 · 68/30 · 6 | 51 · 75/34 · 5 |
| Bloqueo (67) | 34 · 62/12 · 5 | 23 · 56/8 · 4 | 22 · 60/9 · 4 |
| **Total (10 clases comunes)** | **mAP 62,5 · P 84,7 / R 76,7 · FP 552 · F1 80,5** | mAP 54,9 · P 83,5 / R 73,5 · FP 575 · F1 78,2 | mAP 54,2 · P 81,7 / R 73,4 · FP 654 · F1 77,3 |

En falsos positivos totales van casi iguales (552, 575, 654), casi todos pases y conducciones confundidos entre sí. La
diferencia de T-DEED está en el **recall de las clases raras**: tiro 59 frente a 30-34, banda 76 frente a 58, centro 72
frente a 40-57, fuera 37 frente a 8-12. (Fuera de la comparación, `challenge2` contaminado: umbral 0,55, F1 90,7, 345 FP.)

**2. Panorámica, eval de SoccerTrack v2 con nuestra cámara `vcam796vl1` (4.372 eventos).** Umbral óptimo: T-DEED 0,40,
AdaSpot-b 0,05, AdaSpot-s 0,05.

| Clase (nº) | T-DEED 200MF | AdaSpot-b | AdaSpot-s |
|---|---|---|---|
| Pase (1.928) | 66 · 72/67 · 510 | 16 · 75/6 · 36 | 54 · 70/51 · 427 |
| Conducción (1.703) | 63 · 66/61 · 550 | 9 · 66/4 · 36 | 45 · 50/44 · 742 |
| Pase alto (226) | 39 · 70/34 · 34 | 7 · 50/3 · 6 | 17 · 36/15 · 60 |
| Fuera (161) | 7 · 23/6 · 33 | 9 · 6/2 · 44 | 2 · 9/2 · 32 |
| Banda (83) | 41 · 76/30 · 8 | 11 · 30/4 · 7 | 29 · 38/31 · 43 |
| Centro (56) | 28 · 28/43 · 62 | 1 · 0/0 · 0 | 6 · 7/16 · 128 |
| **Tiro (57)** | **17 · 54/12 · 6** | 0 · 0/0 · 0 | 9 · 4/7 · 106 |
| Cabezazo (5) | 1 · 2/20 · 56 | 0 · 0/0 · 3 | 0 · 0/0 · 78 |
| Bloqueo (62) | 4 · 22/10 · 21 | 0 · 0/0 · 0 | 1 · 4/5 · 63 |
| **Total (10 comunes)** | **mAP 26,6 · P 65,8 / R 57,6 · FP 1.298 · F1 61,4** | mAP 5,3 · P 59,0 / R 4,4 · FP 132 · F1 8,2 | mAP 16,3 · P 51,8 / R 42,0 · FP 1.688 · F1 46,4 |

AdaSpot-b casi no emite nada (190 aciertos de 4.372); AdaSpot-s emite con la peor precisión (1.688 FP para 1.817 TP; en
tiro 106 FP por 4 TP). T-DEED acierta 2.494 con 1.298 FP. Fuera de la comparación (limpios aquí): `challenge2` F1 64,9 con
1.128 FP; `SNB_pano1` **F1 74,7, P 76,9 / R 72,6, 947 FP**.

**3. Metraje nuevo: SN-PCBAS-2026 (FOOTPASS), `val` (3 partidos de 2023/24, 6.070 eventos).**

Conjunto: `SoccerNet/SN-PCBAS-2026` en Hugging Face (acceso con el `HF_TOKEN` de `.env`, aprobado el 2026-11-08) = FOOTPASS
(Ochin et al., arXiv 2511.16183; repo `JeremieOchin/FOOTPASS`): 54 partidos completos de 2023/24 de Ligue 1, Bundesliga,
Serie A, LaLiga y Champions a 1080p/25 fps, con tracking táctico (22 filas por frame: jugador, lado, dorsal, rol, xy,
velocidad, caja) y un evento por **frame de contacto** con 8 clases (`class_names` de su `evaluation.py`: 1 drive, 2 pass,
3 cross, 4 throw-in, 5 shot, 6 header, 7 tackle, 8 block). Splits: train 48, **val 3** (game_18, game_24, game_47; vídeos
10,6 GB, etiquetas `val_tactical_data.h5`), challenge 3 (sin columna de clase). **Ningún modelo nuestro ha visto estos
partidos** (tampoco `challenge2`): es la primera medida en metraje nuevo (otras ligas, cuatro años después, 1080p) para
todos. Su métrica oficial es Macro-F1 a ±12 frames **con identidad del jugador** (ahí compite PAVE, §7.7); aquí se usa solo
como test de spotting de 8 clases: `eval_bas.py --dataset pcbas` convierte el h5 en `Labels-ball.json` (todo como primera
parte, ms desde el frame 0 del mp4) y evalúa con las 12 clases BAS (las 4 ausentes dan 0 y se excluyen de `mAP_pcbas8`);
`--merge-high-pass` funde PASS y HIGH PASS (FOOTPASS no distingue). Vídeos 1080p redimensionados a 796x448 por los dos
modelos. Datos en `datasets/SoccerNet/pcbas-2026/`, cachés `videos/game_XX.tdeed.<modelo>.npz`, runner reanudable
`scripts/benchmark/run_pcbas_val.sh`, resultados en `output/benchmark/bas_pcbas/<modelo>/{val,val_merged}/`.

Umbral óptimo: T-DEED 0,20, AdaSpot-b 0,20, AdaSpot-s 0,10. Sin fundir pase alto (fundiéndolo cambia ≤ 1 punto, abajo).

| Clase (nº) | T-DEED 200MF | AdaSpot-b | AdaSpot-s |
|---|---|---|---|
| Pase (3.059) | 70 · 72/70 · 807 | **74 · 87/67 · 306** | 76 · 79/73 · 611 |
| Conducción (2.470) | 61 · 55/68 · 1.382 | **71 · 73/71 · 635** | 69 · 64/70 · 991 |
| Cabezazo (162) | 43 · 62/33 · 33 | 53 · 62/45 · 44 | 53 · 74/41 · 23 |
| Centro (111) | **56 · 57/47 · 39** | 48 · 58/40 · 33 | 47 · 58/43 · 35 |
| Banda (97) | **70 · 41/75 · 104** | 40 · 46/45 · 51 | 32 · 30/40 · 90 |
| Bloqueo (78) | 29 · 29/42 · 81 | 15 · 36/14 · 20 | 22 · 26/23 · 52 |
| **Tiro (67)** | **59 · 41/79 · 76** | 26 · 36/25 · 30 | 35 · 34/43 · 56 |
| Entrada (26) | 11 · 8/27 · 78 | 2 · 0/0 · 2 | 1 · 0/0 · 11 |
| **Total (8 clases FOOTPASS)** | **mAP 50,0** · P 61,1 / R 67,2 · FP 2.600 · F1 64,0 (TP 4.078) | mAP 41,1 · **P 78,1 / R 65,9 · FP 1.121 · F1 71,5** (TP 4.003) | mAP 41,7 · P 69,1 / R 68,9 · FP 1.869 · F1 69,0 (TP 4.180) |

Variantes y comprobaciones:

- Fundiendo pase alto con pase (`val_merged`): T-DEED mAP 50,1 / F1 64,6 / FP 2.659; AdaSpot-b 41,2 / 72,8 / 1.141;
  AdaSpot-s 41,8 / 69,5 / 1.173 (su umbral óptimo pasa a 0,15). No cambia la lectura.
- Al umbral fijo 0,15 (convención PCBAS): T-DEED P 52,4 / R 75,8 / F1 61,9 / FP 4.181; AdaSpot-b P 74,0 / R 68,9 / F1 71,4 /
  FP 1.469; AdaSpot-s P 76,6 / R 62,7 / F1 68,9 / FP 1.163.
- Macro-F1 (media de las 8 clases, al umbral óptimo): T-DEED 47,6, AdaSpot-b 43,0, AdaSpot-s 42,8. No es comparable con
  el leaderboard de PCBAS (exige el jugador).
- Con tolerancia ±1 s el mAP de 8 clases sube a 55,2 / 45,8 / 45,8: la conducción gana ~9 puntos de AP en los tres
  modelos (T-DEED 61 → 70, AdaSpot-b 71 → 79), el tiro de T-DEED 59 → 67 y la banda 70 → 79. El instante de FOOTPASS
  (frame de contacto) no coincide del todo con el criterio de BAS, y afecta a todos por igual.

Lectura de PCBAS:

- **Quién hace más falsos positivos en metraje nuevo: T-DEED**, con diferencia: 2.600 frente a 1.121 (AdaSpot-b) y 1.869
  (AdaSpot-s) para un número de aciertos casi igual (4.078 / 4.003 / 4.180 de 6.070). 1.382 de sus falsos son conducciones
  (P 55) y 807 pases. En la EFL los tres emitían lo mismo (552-654).
- **T-DEED pierde mucho más punto de trabajo que AdaSpot al cambiar de metraje**: F1 80,5 → 64,0 (−16,5) frente a
  78,2 → 71,5 (−6,7) de AdaSpot-b; su umbral óptimo baja de 0,70 a 0,20 (las confianzas se hunden y se solapan). En AP
  los dos pierden parecido (mAP 62,5 → 50,0 y 54,9 → 41,1): **T-DEED sigue ordenando mejor, pero calibra peor fuera de
  la EFL**.
- Por clase se reparten: AdaSpot gana en las dos clases masivas (pase 74-76 frente a 70, conducción 69-71 frente a 61,
  cabezazo 53 frente a 43) y T-DEED en las raras (tiro 59 frente a 26-35, banda 70 frente a 32-40, centro 56 frente a 47-48,
  entrada 11 frente a 1-2). El mAP favorece a T-DEED porque pesa igual cada clase; la F1 micro favorece a AdaSpot porque
  pesa cada evento.
- Para producto, lo que importa en metraje ajeno es el punto de trabajo, no el AP: habría que **recalibrar el umbral de
  T-DEED por partido** (ya se hace en `main` con umbrales automáticos por partido, §7.3) o ganar robustez con datos de
  más ligas.
- **Modelo de producción `challenge2` (800MF), limpio en PCBAS** [M, 2026-11-09]: mAP@1 de 8 clases **53,1** (±1 s 57,0;
  tight 57,6); AP por clase: pase 71, conducción 67, cabezazo 61, centro 53, banda 74, tiro 66, bloqueo 27, entrada 5.
  Umbral óptimo 0,25 (en la EFL 0,55): **P 65,0 / R 65,4 / F1 65,2, 2.136 FP** (TP 3.970); conducción P 57 / R 74 con 1.374
  FP, pase P 80 / R 60 con 464. Con umbral por clase (`per_class_best`): tiro a 0,50 P 59 / R 69 (32 FP), banda a 0,60
  P 83 / R 71 (14 FP), cabezazo a 0,20 P 60 / R 63. Es decir: +3 de mAP sobre el 200MF (tiro 66 frente a 59, cabezazo 61
  frente a 43), pero **el mismo problema de calibración**: 1,9× los falsos positivos de AdaSpot-b y F1 6 puntos por debajo
  (65,2 frente a 71,5). En la EFL su cifra casi limpia era 77,2 de mAP@1 (Middlesbrough, §7.11): pierde ~24 puntos al
  salir de la EFL.

**En qué gana AdaSpot a T-DEED (respuesta consolidada, §7.8 + §7.11 + §7.12)**: (1) coste de inferencia pura, 2,0× menos
que el 200MF y 4,3× menos que el 800MF, con 6,3 GB de VRAM frente a 8,6 / 14 (aunque en pared da igual, manda la
decodificación); (2) **calibración y falsos positivos en metraje nuevo** (PCBAS: F1 71,5 frente a 64,0, 1.121 FP frente a
2.600), y las clases masivas pase/conducción/cabezazo en ese metraje; (3) su marco de comparación del paper (mismo
backbone a 398x224 sin SN-v2) y los benchmarks de PES/Tenis. Pierde en mAP en TV (−7 pt), en PCBAS (−9 pt), en todas las
clases raras (tiro, banda, centro, fuera, entrada) y colapsa en panorámica.

**Ganadores de los challenges 2025-2026** (comprobado 2026-11-08, para no buscarlos otra vez): Team BAS 2025 lo ganó
"dudek" (K. Czarnogórski, 60,03 Team-mAP@1 / 61,56 mAP@1 de acción) con el framework abierto `dude.k` (GitHub `int8/dude.k`,
PyPI) **sin pesos publicados**; el mejor en acción sola (68,92) fue UniBW Múnich, sin código. En 2026 no hay tarea de Team
BAS (tareas: Ball Action Anticipation, PCBAS [ganador PAVE, sin código, §7.7], Novel View Synthesis, Synloc, VQA). Nada
ejecutable, así que la tabla de pesos públicos de §7.11 sigue siendo completa.

La cabeza de **17 eventos de SoccerNet-v2** (córners incluidos, que no existen en BAS, SoccerTrack ni FOOTPASS) se mide en §7.13.

#### Versión resumida para presentar: nº de eventos, precisión / recall [M, 2026-11-09]

A petición del usuario, las mismas tablas con menos métricas: por clase, **nº de eventos reales** y **precisión / recall**
(en %, al umbral óptimo global de cada modelo, emparejamiento ±0,5 s). Sin AP ni falsos positivos. No se muestran entrada,
falta ni gol (casi nadie las detecta o AdaSpot no las predice); el total sí cuenta las clases comunes.

**SN-BAS, test oficial (TV, Stoke + Reading).** Umbrales 0,70 / 0,35 / 0,25.

| Clase | Nº | T-DEED 200MF | AdaSpot-b | AdaSpot-s |
|---|---|---|---|---|
| Pase | 1.721 | 88 / 85 | 89 / 81 | 85 / 85 |
| Conducción | 1.449 | 84 / 78 | 81 / 81 | 79 / 78 |
| Cabezazo | 182 | 73 / 64 | 78 / 58 | 81 / 48 |
| Pase alto | 181 | 87 / 72 | 79 / 67 | 88 / 58 |
| Fuera | 145 | 53 / 37 | 32 / 8 | 44 / 12 |
| Banda | 95 | 92 / 76 | 83 / 58 | 72 / 58 |
| Bloqueo | 67 | 62 / 12 | 56 / 8 | 60 / 9 |
| Centro | 60 | 81 / 72 | 81 / 57 | 89 / 40 |
| Tiro | 44 | 79 / 59 | 68 / 30 | 75 / 34 |
| **Total** | 3.972 | **85 / 77** | 84 / 74 | 82 / 73 |

**FOOTPASS (SN-PCBAS-2026), val (TV, 3 partidos de 2023/24).** Umbrales 0,20 / 0,20 / 0,10.

| Clase | Nº | T-DEED 200MF | AdaSpot-b | AdaSpot-s |
|---|---|---|---|---|
| Pase | 3.059 | 72 / 70 | 87 / 67 | 79 / 73 |
| Conducción | 2.470 | 55 / 68 | 73 / 71 | 64 / 70 |
| Cabezazo | 162 | 62 / 33 | 62 / 45 | 74 / 41 |
| Centro | 111 | 57 / 47 | 58 / 40 | 58 / 43 |
| Banda | 97 | 41 / 75 | 46 / 45 | 30 / 40 |
| Bloqueo | 78 | 29 / 42 | 36 / 14 | 26 / 23 |
| Tiro | 67 | 41 / 79 | 36 / 25 | 34 / 43 |
| **Total** | 6.070 | 61 / 67 | **78 / 66** | 69 / 69 |

**SoccerTrack v2, evaluación (panorámica, 128057 + 132831), cámara virtual que sigue el balón anotado (`vcam796`).**
Umbrales 0,30 / 0,05 / 0,05.

| Clase | Nº | T-DEED 200MF | AdaSpot-b | AdaSpot-s |
|---|---|---|---|---|
| Pase | 1.928 | 70 / 53 | 72 / 1 | 64 / 26 |
| Conducción | 1.703 | 62 / 56 | 46 / 1 | 44 / 26 |
| Pase alto | 226 | 49 / 44 | 43 / 1 | 49 / 10 |
| Fuera | 161 | 12 / 4 | 7 / 2 | 20 / 6 |
| Banda | 83 | 50 / 20 | 27 / 4 | 55 / 34 |
| Bloqueo | 62 | 17 / 6 | 0 / 0 | 9 / 5 |
| Tiro | 57 | 42 / 9 | 0 / 0 | 4 / 5 |
| Centro | 56 | 20 / 50 | 0 / 0 | 6 / 12 |
| **Total** | 4.331 | **60 / 49** | 35 / 1 | 46 / 23 |

Total con las dos cámaras virtuales (precisión / recall): balón anotado 60 / 49, 35 / 1 y 46 / 23; nuestra cámara
(`vcam796vl1`) 66 / 58, 59 / 4 y 52 / 42. Nuestro afinado `SNB_pano1` (entrenado con los 6 partidos de entrenamiento de
SoccerTrack): 73 / 68 con el balón anotado y 77 / 73 con nuestra cámara. La cámara del balón anotado sale peor en el total
porque en 132831 la pista aparca el balón en el centro (§7.11).

SoccerTrack v2 en cifras: 10 partidos universitarios con cámara panorámica fija, partición oficial del SoccerTrack Challenge
2025 (`docs/particion_datasets.md` §3): **6 de entrenamiento, 2 de validación, 2 de evaluación**. Etiquetas
(`bas/<partido>/<partido>_12_class_events.json`): las **12 clases de Ball Action Spotting**, cada evento con instante,
equipo (`left`/`right`) y jugador; sin córner, falta, tarjeta ni fuera de juego.

| Clase | Entrenamiento (6) | Validación (2) | Evaluación (2) |
|---|---|---|---|
| Pase | 5.360 | 2.027 | 1.928 |
| Conducción | 4.777 | 1.775 | 1.703 |
| Pase alto | 694 | 237 | 226 |
| Fuera | 465 | 145 | 161 |
| Centro | 269 | 69 | 56 |
| Banda | 232 | 70 | 83 |
| Bloqueo | 227 | 64 | 62 |
| Entrada | 207 | 50 | 50 |
| Tiro | 161 | 48 | 57 |
| Falta (libre) | 89 | 30 | 31 |
| Gol | 26 | 8 | 10 |
| Cabezazo | 19 | 7 | 5 |
| **Total** | **12.526** | **4.530** | **4.372** |

**Partidos exactos de cada conjunto de test** [2026-11-09] (ninguno usado para entrenar ni para elegir época en los tres
modelos comparados):

| Conjunto | Partido | Fecha | Eventos |
|---|---|---|---|
| SN-BAS test | Stoke City – Huddersfield Town (EFL Championship) | 2019-10-01 | 1.844 |
| SN-BAS test | Reading – Fulham (EFL Championship) | 2019-10-01 | 2.136 |
| FOOTPASS val | `game_18`: Napoli – Milan (Serie A) | 2023-10-29 [fecha deducida] | 1.879 |
| FOOTPASS val | `game_24`: OGC Nice en casa (Ligue 1), rival de blanco sin identificar (señal sin marcador) | 2023/24 | 1.976 |
| FOOTPASS val | `game_47`: Real Madrid – Napoli (Champions League) | 2023-11-29 [fecha deducida] | 2.215 |
| SoccerTrack v2 test | 128057 (M7): TSC (Tsukuba Syukyu Club) – Sakushin Gakuin University | 2024-03-18 | 1.932 |
| SoccerTrack v2 test | 132831 (M9): Chuo Gakuin University – University of Tsukuba C1 | 2024-04-20 | 2.440 |

FOOTPASS no publica el nombre de los partidos (solo `game_<n>`): los equipos de `game_18` y `game_47` están leídos del
marcador del vídeo y las fechas deducidas de la competición; `game_24` no lleva marcador. SN-BAS: entrenamiento Leeds–West
Bromwich, Hull–Sheffield Wednesday, Brentford–Bristol City y Blackburn–Nottingham Forest; validación Middlesbrough–Preston.
SoccerTrack: títulos y fechas de `raw/<id>/<id>_tracker_box_metadata.xml`.

**Por qué AdaSpot-s supera a AdaSpot-b en panorámica** [M, 2026-11-09]. Los dos son el mismo modelo y la misma receta; solo
cambia el backbone (`rny004_gsf` frente a `rny002_gsf`) y hay una única semilla publicada de cada uno. En TV el grande es
igual o mejor (SN-BAS 84 / 74 frente a 82 / 73; FOOTPASS 78 / 66 frente a 69 / 69). En panorámica el grande **deja de
disparar**: su probabilidad media de «hay acción» (1 − fondo) por frame cae de 0,060 en la EFL a 0,001-0,002 en SoccerTrack
(percentil 99: 0,81 → 0,01-0,03), mientras que el pequeño pasa de 0,080 a 0,036-0,042 y T-DEED no baja (0,29 → 0,24-0,30).
No es un fallo de nuestro arnés (mismo código en los cuatro conjuntos, sin NaN) ni del selector de recortes: conservando las
cabezas auxiliares del checkpoint, en 10 min de 128057 las dos ramas del grande caen por igual (baja resolución 0,058 →
0,001; recorte de alta resolución 0,062 → 0,005; recorte elegido 112×112 en todos los casos) y las del pequeño aguantan
(0,078 → 0,048; 0,081 → 0,071). Es un colapso de confianza fuera de dominio del backbone grande, entrenado con 4 partidos
de una sola liga; con una semilla por tamaño no se puede separar «más capacidad = más sobreajuste» de la suerte de la semilla.

**Qué parte de la imagen apaga a AdaSpot-b** [M, 2026-11-09] (hipótesis del usuario: zoom, movimientos bruscos o mal
seguimiento del balón). Medida: probabilidad media de acción por frame (1 − fondo) en 10 min de Stoke (TV) y de 128057
1.ª parte, transformando los frames antes de la red; AdaSpot-b / AdaSpot-s.

| Entrada | AdaSpot-b | AdaSpot-s |
|---|---|---|
| TV original | 0,053 | 0,075 |
| TV con zoom ×0,8 / ×1,25 / ×1,5 | 0,053 / 0,052 / 0,051 | 0,068 / 0,073 / 0,068 |
| TV desenfocada (σ = 2) o con el color de la panorámica | 0,053 / 0,056 | 0,070 / 0,076 |
| TV con el fondo (tercio superior) de la panorámica | 0,048 | 0,076 |
| **TV con el césped y los jugadores de la panorámica** | **0,003** | 0,046 |
| Panorámica original (nuestra cámara) | 0,002 | 0,049 |
| Panorámica con zoom ×0,6 / ×0,8 / ×1,25 / ×1,5 / ×2 | 0,000 / 0,005 / 0,003 / 0,002 / 0,002 | 0,033 / 0,047 / 0,045 / 0,038 / 0,024 |
| Panorámica con nitidez o con el color de TV | 0,001 / 0,001 | 0,034 / 0,047 |
| Panorámica con la cámara del balón anotado | 0,001 | 0,036 |

- **No es el zoom**: ningún factor entre ×0,6 y ×2 lo reactiva, y en TV el zoom no lo rompe. El tamaño de jugador ya es parecido
  (~46 px en la cámara virtual, ~52 px en TV a 796x448).
- **No es el seguimiento del balón ni los movimientos bruscos**: con la cámara del balón anotado está igual de apagado, y lo
  está a cualquier velocidad de cámara (0,0006 con la cámara casi quieta, 0,0012 a más de 10 px/frame; mediana 1,1 px/frame).
- **No es el color global, la nitidez ni el fondo** (gradas frente a árboles).
- **Es la zona de césped y jugadores**: basta pegar los dos tercios inferiores de la panorámica sobre la imagen de TV para
  que AdaSpot-b se apague (0,053 → 0,003). Lo que cambia ahí es el aspecto del juego: césped artificial a pleno sol con
  sombras duras, geometría de gran angular fijo y bajo (líneas curvadas, círculo central aplastado) y jugadores universitarios.
- T-DEED también pierde con la cámara virtual (85 / 77 → 60 / 49): la cámara arregla el encuadre, no el aspecto. Que aguante
  mejor es coherente con su preentrenamiento en SoccerNet-v2 (cientos de partidos y estadios) frente a los 4 partidos de
  AdaSpot desde ImageNet; no se ha aislado. La salida para AdaSpot sería afinarlo con panorámica, como `SNB_pano1`.

### 7.13 Cabeza de 17 eventos de SoccerNet-v2 (córners, faltas, tarjetas…): piloto de 20 partidos limpios del `test` [M, 2026-11-09]

**Qué se mide**: la segunda cabeza de T-DEED (18 salidas: fondo + los 17 eventos de SoccerNet-v2, `sn_probs` en las cachés
de `core.ball_spotting`), que `main` usa para córner, falta, libre directo/indirecto, tarjeta, saque inicial, penalti, cambio
y fuera de juego (`EVENT_SOURCES`), con umbrales por evento calibrados por partido (`analytics/engine.py`). T-DEED
preentrenó esa cabeza solo con `train`+`val` de SoccerNet-v2 (§7.11), así que el `test` oficial (100 partidos, en
`/mnt/datasets/SoccerNet`) está limpio. Piloto: **20 partidos** del test (`output/benchmark/sn_v2_pilot20.txt`; 19 de la
intersección limpia + Manchester United–Arsenal 2015; LaLiga 8, Serie A 4, EPL 3, Champions 3, Bundesliga 2, Ligue 1 1;
vídeos `{1,2}_720p.mkv` copiados a `datasets/SoccerNet/<liga>/<temporada>/<partido>/`, 32 GB, borrables), 4.586 eventos
(3.795 visibles). Por clase (total / visibles): fuera de juego de balón 1.353/1.303, banda 772/548, falta 459/447, libre
indirecto 427/219, despeje 343/177, tiro a puerta 254/253, córner 230/205, tiro fuera 219/218, cambio 113/94, saque
inicial 102/31, fuera de juego 81/76, amarilla 77/75, libre directo 70/63, gol 69/69, penalti 14/14, roja 2/2, segunda
amarilla 1/1 (las tres últimas no son medibles).

**Cómo**: `scripts/benchmark/eval_snv2.py` toma las cachés `<mitad>_720p.tdeed.<modelo>.npz`, aplica el NMS suave de T-DEED
(ventana 6 filas a stride 2 = 0,48 s), guarda todos los candidatos ≥ 0,01 como `results_spotting.json` (gameTime, position
en ms, half, confidence) y evalúa con el **evaluador oficial del paquete `SoccerNet`** (`label2vector`, `predictions2vector`,
`average_mAP` a 2 fps): Average-mAP *tight* (δ = 1…5 s), *loose* (δ = 5…60 s), total / visibles / no mostrados.
Comprobación: con la verdad como predicción el evaluador da 94,1 *tight* en 3 partidos (no 100: su propio NMS de ventana δ
funde eventos de la misma clase a menos de δ segundos). Runner reanudable `run_snv2_pilot.sh` (~3,7 min por mitad el
200MF, 31 min la evaluación de 855 k candidatos). Punto de trabajo con `eval_prf.py --dataset snv2 --tol-ms 1000` (±1 s
porque las etiquetas de SN-v2 van al segundo): umbral global y **umbral óptimo por clase** (`per_class_best`, como los
umbrales por evento de producción).

**Resultado del 200MF `SoccerNetBall_test`** (Average-mAP en %, todas las clases; P/R/FP al umbral óptimo de cada clase a
±1 s, 20 partidos; «≥0,3» = fracción de eventos reales con alguna predicción de su clase ≥ 0,3 a menos de 10 s, y desfase
absoluto mediano de esas predicciones):

| Evento (nº / visibles) | AP tight (visibles) | AP loose | Umbral · P/R · FP | ≥0,3 · desfase |
|---|---|---|---|---|
| **Córner (230 / 205)** | **56,8** (59,5) | 60,8 | 0,50 · 81/76 · 40 | 84 % · 0,07 s |
| Banda (772 / 548) | 49,5 (57,4) | 58,5 | 0,70 · 77/64 · 149 | 82 % · 0,08 s |
| Balón fuera (1.353 / 1.303) | 41,7 (42,6) | 51,0 | 0,75 · 69/60 · 369 | 91 % · 0,14 s |
| Libre directo (70 / 63) | 36,3 (44,3) | 38,2 | 0,55 · 71/60 · 17 | 70 % · 0,07 s |
| **Gol (69 / 69)** | 35,8 (37,7) | 37,5 | 0,25 · 49/55 · 40 | 45 % · 0,08 s |
| Despeje (343 / 177) | 31,2 (48,3) | 37,4 | 0,60 · 58/40 · 99 | 61 % · 0,10 s |
| Tiro fuera (219 / 218) | 29,1 (29,1) | 31,9 | 0,45 · 37/54 · 204 | 74 % · 0,09 s |
| Falta (459 / 447) | 22,4 (22,9) | 28,0 | 0,75 · 38/37 · 280 | 84 % · 0,16 s |
| **Tiro a puerta (254 / 253)** | 19,9 (20,0) | 24,0 | 0,35 · 30/61 · 373 | 73 % · 0,07 s |
| Libre indirecto (427 / 219) | 13,9 (19,8) | 21,9 | 0,70 · 14/23 · 592 | 70 % · 0,18 s |
| Saque inicial (102 / 31) | 12,4 (4,8) | 15,4 | 0,45 · 83/34 · 7 | 36 % · 0,01 s |
| Fuera de juego (81 / 76) | 5,0 (6,1) | 6,5 | 0,25 · 12/18 · 115 | 10 % · — |
| Cambio (113 / 94) | 4,4 (5,8) | 12,6 | 0,15 · 6/22 · 382 | 0 % |
| Amarilla (77 / 75) | 4,2 (4,5) | 13,4 | 0,10 · 2/25 · 995 | 0 % |
| Penalti (14 / 14) | 0,6 (1,6) | 1,0 | 0,15 · 2/57 · 510 | — |
| Roja (2), 2.ª amarilla (1) | 0,0 | 0,0 | — | — |
| **Average-mAP (17)** | **21,4** (visibles 23,8; no mostrados 12,3) | 25,8 (28,1; 18,8) | umbral global 0,65: P 45,3 / R 46,8 · FP 2.598 · F1 46,0 | |

Referencia externa [NV, de los papers, no reproducido]: los sistemas dedicados a SN-v2 publican en este mismo `test`
61,8 *tight* / 74,1 *loose* (E2E-Spot RegNetY-800MF; 61,2 el 200MF), 65,1 / 78,6 (Soares et al.) y 73,1 *tight* (COMEDIAN).

Lectura:

- **Esta cabeza no es un spotter de SoccerNet-v2 competitivo**: 21,4 frente a 62-73 de los sistemas dedicados. Es la cabeza
  auxiliar con la que T-DEED mejoró el BAS (los autores nunca publicaron su cifra en SN-v2), entrenada con clips de 8 s.
- **Cuando dispara, acierta el instante**: desfase mediano 0,07-0,18 s y 90-95 % de los aciertos a menos de 1 s en todas
  las clases con señal. El problema no es el *cuándo* sino el *si*: muchos falsos positivos a confianza alta (balón fuera,
  falta, libre indirecto) y clases que no se encienden.
- **Utilizables con umbral propio**: córner (P 81 / R 76, 2 falsos por partido), banda y balón fuera (P 69-77 / R 60-64),
  libre directo (P 71 / R 60), saque inicial (P 83 con R 34) y gol (P 49 / R 55; en `main` el gol real sale del marcador o
  del saque de centro, `analytics/goals.py`, no de aquí).
- **Flojas**: tiros (P 30-37 para R 54-61: el tiro de producto debe seguir saliendo de la cabeza de acciones, AP 74 en TV,
  §7.12), falta (P 38 / R 37), despeje (P 58 / R 40) y libre indirecto (P 14 / R 23).
- **Inservibles**: cambio, amarilla, fuera de juego, penalti y rojas: nunca superan 0,3 (0 % de los cambios y amarillas tienen
  predicción ≥ 0,3) y su AP es < 6. Son eventos de contexto (gesto del árbitro, banquillo, bandera) que un clip de 8 s
  centrado en el balón no ve. En `main` esos eventos (`card`, `substitution`, `offside`, `penalty`) salen de esta cabeza:
  **no son fiables ni con el 200MF ni con el 800MF de producción** (abajo).
- Lo *no mostrado* (acción fuera de plano o tapada por la repetición) no se detecta: despeje 48 → 9, banda 57 → 27, libre
  indirecto 20 → 3 (tight visibles → no mostrados). Eso no lo arregla ningún spotter de vídeo.
- **Margen**: córner, banda y balón fuera ya están a 50-60 de AP tight con la cabeza auxiliar; un spotter dedicado de SN-v2
  con pesos públicos (E2E-Spot, 61,8 tight de media sobre las 17) sería la vía para faltas, tarjetas, cambios y fueras de
  juego si el producto los necesita.

**Resultado del 800MF `challenge2` (producción)**, mismos 20 partidos (1,03 M de candidatos; ~3,7 min por mitad con la GPU
libre, 55 min la evaluación):

| Evento (nº / visibles) | AP tight 800MF (visibles) | AP tight 200MF | AP loose 800MF | Umbral · P/R · FP (800MF) | ≥0,3 · desfase (800MF) |
|---|---|---|---|---|---|
| **Córner (230 / 205)** | **57,8** (59,3) | 56,8 | 61,3 | 0,50 · 81/74 · 40 | 79 % · 0,07 s |
| **Penalti (14 / 14)** | 47,8 (52,8) | 0,6 | 55,6 | 0,10 · 83/36 · 1 | 0 % |
| Banda (772 / 548) | 45,0 (53,5) | 49,5 | 54,3 | 0,75 · 79/66 · 135 | 84 % · 0,08 s |
| Balón fuera (1.353 / 1.303) | 42,9 (43,7) | 41,7 | 52,2 | 0,75 · 75/58 · 258 | 91 % · 0,13 s |
| Libre directo (70 / 63) | 41,1 (49,5) | 36,3 | 43,1 | 0,45 · 72/61 · 17 | 67 % · 0,06 s |
| **Gol (69 / 69)** | 35,4 (37,1) | 35,8 | 36,9 | 0,40 · 55/51 · 29 | 62 % · 0,08 s |
| **Tiro a puerta (254 / 253)** | 32,8 (32,9) | 19,9 | 35,9 | 0,45 · 48/49 · 134 | 70 % · 0,12 s |
| Despeje (343 / 177) | 29,8 (48,6) | 31,2 | 34,7 | 0,50 · 54/37 · 110 | 50 % · 0,11 s |
| Tiro fuera (219 / 218) | 29,6 (29,8) | 29,1 | 32,8 | 0,45 · 50/48 · 104 | 63 % · 0,06 s |
| Falta (459 / 447) | 26,3 (26,8) | 22,4 | 32,6 | 0,75 · 45/48 · 268 | 85 % · 0,14 s |
| Libre indirecto (427 / 219) | 15,2 (21,8) | 13,9 | 22,3 | 0,65 · 19/25 · 455 | 70 % · 0,16 s |
| Saque inicial (102 / 31) | 10,5 (12,4) | 12,4 | 14,1 | 0,05 · 4/11 · 280 | 0 % |
| Amarilla (77 / 75) | 5,6 (5,9) | 4,2 | 17,0 | 0,25 · 3/22 · 581 | 29 % · 2,40 s |
| Fuera de juego (81 / 76) | 5,5 (7,6) | 5,0 | 7,2 | 0,30 · 15/10 · 45 | 12 % · 0,41 s |
| Cambio (113 / 94) | 4,5 (7,0) | 4,4 | 17,2 | 0,25 · 10/12 · 113 | 3 % |
| Roja (2), 2.ª amarilla (1) | 0,0 | 0,0 | 0,0 | — | — |
| **Average-mAP (17)** | **25,3** (visibles 28,8; no mostrados 13,6) | 21,4 | 30,4 (34,2; 18,9) | umbral global 0,70: P 59,2 / R 44,4 · FP 1.402 · F1 50,7 | |

- El modelo de producción gana **+3,9 de Average-mAP tight** (25,3 frente a 21,4) y emite la mitad de falsos positivos al umbral
  global (1.402 frente a 2.598) con el mismo recall. La mejora se concentra en penalti (48 frente a 1, sobre solo 14
  penaltis: 5 aciertos y 1 falso a umbral 0,10), tiro a puerta (33 frente a 20), libre directo y falta; pierde algo en banda
  y saque inicial.
- **Sigue a 35-45 puntos de los spotters dedicados** y el reparto por clases es el mismo: córner sólido (P 81 / R 74, 2 falsos
  por partido), banda, balón fuera y libre directo utilizables; tiros a medias (P 48-50 / R 48-49); **tarjeta, cambio, fuera
  de juego y saque inicial inservibles también en producción** (AP < 11; la amarilla dispara con 2,4 s de desfase y P 3).
- Consecuencia para `main`: de `EVENT_SOURCES`, `corner`, `free-kick` (directo) y `foul` (a medias) tienen respaldo medido;
  `card`, `substitution`, `offside` y `kick-off` no deberían mostrarse como fiables con esta cabeza; `penalty` promete pero
  necesita más penaltis para afirmarlo (los 100 partidos del test tendrían ~70).
