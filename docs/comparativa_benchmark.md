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
- **No merece la pena**: CALF, NetVLAD++, FOOTPASS como modelo, el tracker MOTIP de SoccerMaster y el detector de
  GTATrack. Los tres últimos no publican pesos.

---

## 1. Qué usa nuestro pipeline (`main`)

| Módulo | Modelo / algoritmo | Dónde |
|---|---|---|
| Detección de personas | SoccerMaster `SoccerNetGSR_Detection`: Deformable-DETR sobre SigLIP2, clips de 30 frames a 512x512, umbral 0,1. Da rol y dorsal. | `core/tracking.py:317-490` |
| Detección de balón | YOLO11x COCO (clase 32) a 1280 px, conf 0,1. El `yolo11s_snt_ball.pt` afinado existe pero no está por defecto. | `core/tracking.py:242-314`, `config.py:46` |
| Tracking | `supervision.ByteTrack`: activación 0,25, buffer 30, matching 0,8. Prefiltro conf ≥ 0,3 y área ≥ 300 px². Sin ReID ni GMC. | `core/tracking.py:610-647,787-796` |
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
Las medidas limpias en EFL deben hacerse con `SoccerNetBall_test`. Todo SoccerNet-v2 está dentro del entrenamiento de T-DEED.

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
| **FOOTPASS** ([repo](https://github.com/JeremieOchin/FOOTPASS), [arXiv 2511.16183](https://arxiv.org/abs/2511.16183)) | Dataset + 3 baselines | Vídeo 352x640 **más tracklets e identidades** | **No**: solo código. Datos en HF con NDA, 236 GB. | Contradictoria: Apache-2.0 en la etiqueta, CC BY-NC 4.0 en el README | Test Macro-F1 67,5 (TAAD+DST). Ganador de PCBAS 2026: 58,94. |
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
- La licencia real de FOOTPASS y su publicación en CVIU.
- Si los pesos del ganador de TBAS 2025 (dude.k) son públicos.
- El valor por defecto del tile de FutTraffic (1024 sale de un docstring) y la hoja del leaderboard que cita.
- El baseline GS-HOTA: 22,26 (nuestro `docs/datasets.md`) frente a 29,01 (informe del challenge 2025).
  Probablemente sean ediciones distintas.
- Si Stoke y Reading pertenecen al split *test* oficial de SN-BAS, condición para comparar con el 58,91 publicado.

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
