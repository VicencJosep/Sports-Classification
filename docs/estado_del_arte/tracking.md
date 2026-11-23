# Tracking de jugadores en fútbol de TV: estado del arte (2016-2026) y qué implica para el proyecto

Resumen a 2026-11-08. Cada paper se verificó en arXiv (abs/html), en la web del congreso o en el README oficial; las
notas de verificación están en `output/research_notes/tracking_sota_2026-11-08/` (fuera de git). Nuestro tracker actual
y sus medidas están en `docs/comparativa_benchmark.md` §7.10 (BoT-SORT + GMC + NMS + ReID OSNet, 59,4 HOTA en el test
de SoccerNet-Tracking con las detecciones de SoccerMaster; Deep-EIoU 62,4 y Deep-EIoU + GTA-Link 64,3 con las mismas cajas).

Convenciones: **[P]** cifra publicada por los autores (HOTA en *test* salvo que se diga), **[M]** medida por nosotros,
**[NV]** no verificado. Las cifras de distintos papers solo son comparables cuando comparten dataset, split y detector;
se indica cuando no es así. Nada de cámara panorámica: todo lo de abajo es cámara que sigue al juego (TV) o multideporte.

---

## 1. Dónde está el campo

### 1.1 Trackers por asociación heurística: de SORT a 2026

- **La familia sigue siendo SORT** ("Simple Online and Realtime Tracking", Bewley, ICIP 2016, arXiv 1602.00763): Kalman de
  velocidad constante + IoU + húngaro. De ahí salen tres ramas que aún compiten:
  - **DeepSORT → StrongSORT(++)** ("Simple Online and Realtime Tracking with a Deep Association Metric", Wojke, ICIP 2017,
    1703.07402; "StrongSORT: Make DeepSORT Great Again", Du, IEEE TMM 2023, 2202.13514): apariencia con EMA, Kalman NSA,
    compensación ECC y, en "++", dos post-procesos offline: AFLink (enlaza tracklets solo por posición, ventana ≤ 30
    frames) y GSI (interpolación gaussiana). Es el tracker del baseline de SoccerNet GSR.
  - **ByteTrack → BoT-SORT** ("ByteTrack: Multi-Object Tracking by Associating Every Detection Box", Zhang, ECCV 2022,
    2110.06864; "BoT-SORT: Robust Associations Multi-Pedestrian Tracking", Aharon, arXiv 2022, 2206.14651). ByteTrack
    aporta la segunda pasada con detecciones flojas; BoT-SORT se declara "integrado en ByteTrack" y añade la **GMC**
    (flujo óptico disperso Shi-Tomasi + LK + RANSAC afín, aplicado a la media y covarianza del Kalman), un Kalman con
    (w, h) y la fusión IoU-ReID. Es lo que usamos ahora.
  - **OC-SORT → Deep OC-SORT → Hybrid-SORT** ("Observation-Centric SORT", Cao, CVPR 2023, 2203.14360; Maggiolino, ICIP 2023,
    2302.11813; Yang, AAAI 2024, 2308.00783): re-actualizar el Kalman tras la oclusión con una trayectoria virtual
    (ORU), dirección del movimiento, CMC copiada de BoT-SORT, apariencia dinámica y pistas débiles (confianza, altura).
- **Lo que de verdad movió la aguja en deporte fue soltar el Kalman lineal y ensanchar la región de emparejamiento.** En
  SportsMOT [P]: ByteTrack 64,1 → OC-SORT 73,7 → **Deep-EIoU 77,2** ("Iterative Scale-Up ExpansionIoU and Deep Features
  Association for Multi-Object Tracking in Sports", Huang, WACV-W 2024, 2306.13074: sin Kalman, IoU expandida de forma
  iterativa + OSNet) → MeMoSORT 82,1 (Wang, arXiv 2025, 2508.09796: Kalman con memoria + Mo-IoU) y ARC-SORT 81,8 (Adari,
  arXiv 2026, 2610.22306: Kalman IMM + EIoU adaptativa a la incertidumbre, sin entrenamiento). La cadena de ideas es
  explícita en sus related work: **BIoU → EIoU → Mo-IoU / EIoU adaptativa / soft-BIoU**. El origen es C-BIoU ("Hard to
  Track Objects with Irregular Motions and Similar Appearances? Make It Easier by Buffering the Matching Space", Yang,
  WACV 2023, 2211.14317), segundo del challenge de tracking de SoccerNet 2022 (2211.13481) con 89,2 HOTA sobre
  detecciones oráculo.
- **La apariencia vale mucho en fútbol y poco en DanceTrack.** Con equipaciones distinguibles un ReID afinado al dominio
  da saltos grandes: Deep-EIoU con su OSNet "sports"; Deep HM-SORT 80,1 (Gran-Henriksen, arXiv 2024, 2406.12081: media
  armónica entre IoU y apariencia sobre Deep-EIoU); HAT 78,1-80,8 (Gao, ECCV 2026, 2503.12562: transforma el rasgo ReID
  con el historial, sin entrenar). En DanceTrack (ropa igual) MixSort baja respecto a ByteTrack y Deep OC-SORT solo gana
  haciendo la apariencia dinámica.
- **La compensación de cámara es necesaria pero ya no diferencial.** Está en BoT-SORT, Deep OC-SORT, BoostTrack
  (Stanojević, MVA 2024; "++" en Filomat 2025, 2408.13003), TrackTrack ("Focusing on Tracks for Online Multi-Object
  Tracking", Shim, CVPR 2025) y McByte++ (Stanczyk, arXiv 2026, 2608.15688, CMC **condicional**: descarta afines
  degeneradas y conserva la última fiable). Sus ablaciones la sitúan en +0,9 HOTA / +1,6 IDF1 en MOT17-val (BoT-SORT) y en
  cero con cámara fija (Deep OC-SORT en MOT20). Nosotros medimos lo mismo en TV: +1,1 HOTA y −27 % IDSW, concentrado en
  los paneos [M].
- **El techo lo mueven las asociaciones a nivel de tracklet, no el frame a frame.** GTA-Link ("Global Tracklet
  Association for Multi-Object Tracking in Sports", Sun, ACCV-W 2024, 2411.08216: offline, DBSCAN para partir tracklets
  impuros y clustering jerárquico con OSNet para unirlos) sube Deep-EIoU de 77,2 a 81,0 en SportsMOT y de 79,4 a 83,1 en
  SoccerNet con detecciones oráculo [P]; nosotros medimos +1,9 HOTA y +4,5 IDF1 con nuestro detector [M]. Encima de GTA se
  han construido GTATrack (ganador de SoccerTrack 2025, 2602.00484), McByte++ (81,5), ARC-SORT y SAM-Deep-EIoU (abajo).
- **Dos estudios sistemáticos de 2026** ordenan todo esto: el survey de tracking-by-detection de Yang, Shim, Ko y Kim
  (2609.08265, el grupo de TrackTrack) construye desde un SORT mínimo hasta 63,3 HOTA en DanceTrack añadiendo componentes
  (lo que más suma: suma ponderada espacial + apariencia +2,7, estado del Kalman +2,5, refuerzo por forma +2,1); "Beyond
  the Survey" (Ma, 2609.22291) concluye que **la detección domina** (más de 10 puntos) frente a la asociación (menos de 5)
  y que, con el mismo YOLOX + SBS50 en DanceTrack-val, TrackTrack 61,7 > Hybrid-SORT 59,9 > ByteTrack 46,5.

### 1.2 End-to-end y asociación aprendida: mejor en DanceTrack, por detrás en deporte

- **Cadena DETR**: Deformable DETR → TrackFormer (Meinhardt, CVPR 2022, 2101.02702) y MOTR (Zeng, ECCV 2022, 2105.03247) →
  MOTRv2 (Zhang, CVPR 2023, 2211.09791: anclas de YOLOX) → MeMOTR (Gao, ICCV 2023, 2307.15700: memoria larga) →
  **MOTIP** ("Multiple Object Tracking as ID Prediction", Gao, CVPR 2025, 2403.16848: abandona la propagación de consultas
  y predice el id con un decodificador) → **SoccerMaster + MOTIP** (Yang, CVPR 2026 oral, 2512.11016). Ramas: SambaMOTR
  (Segu, ICLR 2025, 2410.01806, Mamba), CO-MOT, MOTRv3, MATR (2509.21715), SelfMOTR (ECCV 2026, 2511.20279).
- **En SportsMOT los end-to-end van unos 10 puntos por detrás** [P]: MOTIP 72,6 y MATR 72,2 frente a 77-87 de los
  tracking-by-detection; el motivo principal es DetA (YOLOX afinado 88-89 frente a 83 del DETR). En DanceTrack el orden
  se iguala (MOTIP 72,0 con datos extra, MOTRv3 70,4, TDLP 70,1, CAMELTrack 69,3) y un zero-shot con SAM 2 (SAM2MOT,
  2504.04519) da 75,8 sin entrenar nada.
- **La asociación aprendida sí bate a las heurísticas con el mismo detector**: CAMELTrack ("Context-Aware Multi-cue
  ExpLoitation for Online Multi-Object Tracking", Somers, arXiv 2025, 2505.01257: dos transformers que sustituyen las
  reglas, se entrena en 1 h en una GPU de consumo, vive en TrackLab) da 80,4 en SportsMOT y TDLP (Adžemović, 2512.22105,
  predicción de enlaces track-detección) 81,9, frente a 77,2 de Deep-EIoU con YOLOX público. Pero MeMoSORT (82,1,
  heurístico) iguala, y la ganancia grande de 2026 viene de **SAM como árbitro**: Selective Mask Propagation
  (Holmberg, arXiv 2026, 2606.13033: Deep-EIoU + GTA y SAM 3 solo en las ventanas inciertas) llega a 87,2.
- **Coste**: todos los end-to-end entrenan en 8 GPUs (MOTIP y SambaMOTR 8×4090) e infieren a 7-19 fps; CAMELTrack ~13
  fps de pipeline; HAT, SAM2MOT, SMP y McByte++ no entrenan. **Ningún método publica pesos de fútbol**; CAMELTrack y
  MOTIP sí publican pesos de SportsMOT/DanceTrack.
- **En SoccerNet-Tracking la única comparación end-to-end publicada** es SoccerMaster + MOTIP 59,1 frente a YOLOv8 +
  PRTReID 59,8 (gana en DetA, pierde 4,7 de AssA; split no indicado) [P]: la misma banda que nuestros 59,4 [M]. TDLP da
  56,3 frente a 54,2 de CAMELTrack con detector público.

### 1.3 Fútbol con cámara de TV: datasets, challenges y la receta de los ganadores

- **Datasets**: SoccerNet-Tracking (Cioppa, CVPRW 2022, 2204.06918: 200 clips de 30 s, cámara principal 1080p25; test 49
  clips; baselines con detecciones GT ByteTrack 71,5 / DeepSORT 69,6, con detecciones propias 47,2 / 36,7); SportsMOT
  (Cui, ICCV 2023, 2304.05170: fútbol, básquet y voleibol; MixSort como baseline, 74,1); SoccerNet Game State
  Reconstruction (Somers, CVPRW 2024, 2404.11335: tracking + rol + equipo + dorsal + posición en el campo, métrica
  GS-HOTA con tolerancia gaussiana de 5 m); SoccerTrack v2 (Scott, 2508.01802) es panorámica fija y queda fuera.
- **Challenges de tracking** (informes 2210.02365 y 2309.06006): 2022, solo asociación sobre detecciones GT, ganó Kalisteo
  (CEA List, "TrackMerger") con 93,64; 2023, detección + asociación, Kalisteo 75,61 con YOLOX + Kalman con GMC + corte de
  tracklets al cruzarse + ReID MGN + fusión iterativa offline; MOT4MOT (2308.16651) 66,27. Sobre el mismo test con
  detecciones GT: Deep-EIoU 85,4 (GTA lo reproduce en 79,4), PRT-Track 90,8, SportsSUSHI 90,9; con detector propio,
  SportsSUSHI 71,4 (YOLOX afinado). Es decir: **el detector mueve 20-30 puntos; la asociación, 5-10.**
- **PRTReID** ("Multi-task Learning for Joint Re-identification, Team Affiliation, and Role Classification for Sports
  Visual Tracking", Mansourian, MMSports 2023, 2401.09942) es un BPBreID (Somers, WACV 2023, 2211.03679, ReID por partes
  del cuerpo) con cabezas de equipo y rol; es el ReID del baseline GSR, de SoccerFactory (SoccerMaster) y el que usamos
  para equipos. CAMELTrack también parte de BPBreID.
- **Game State Reconstruction** (informes 2409.10587 y 2508.19182): baseline 22,3 → 29,01 GS-HOTA; ganadores
  Constructor.tech 63,81 (2024; "From Broadcast to Minimap", Golovkin, CVPRW 2025, 2504.06357) y KIST-GSR 63,90 (2025);
  lianyou 62,76 y Playbox&MIXI 61,64 (**BoT-SORT + GTA** dentro de TrackLab). La receta común: detector afinado (YOLOv5/X,
  RF-DETR), tracker online simple (DeepSORT en coordenadas de campo, Deep-EIoU, BoT-SORT), ReID OSNet/CLIP, calibración
  por frame suavizada en el tiempo, **y enlace offline de tracklets usando dorsal + equipo + plausibilidad física como
  identidad a largo plazo** (Constructor.tech: partir tracklets con más de un dorsal/equipo, fusionar por dorsal, luego
  por ReID bajo velocidad plausible, interpolar; −90 % de tracklets). Resultado: GS-AssA 82 con GS-DetA 50; el cuello de
  GSR es ya detección y calibración, no asociación.
- **Dorsal como identidad a largo plazo**: "General-Purpose Jersey Number Recognition" (Koshkina & Elder, CVPRW 2024,
  2405.13896: STR PARSeq + legibilidad + torso por pose, 87,4 % por tracklet en fútbol) lo usan SportsSUSHI (Koshkina,
  2502.21242: SUSHI de grafos jerárquicos + dorsal + equipo + coordenadas de campo), Playbox&MIXI y SAM-Deep-EIoU; Balaji
  (MMSports 2023, 2309.06285) usa keyframes. SoccerMaster saca el dorsal de la propia cabeza de detección (lo que ya
  tenemos sin votación por tracklet).
- **2026**: SoccerNet no convoca tracking ni GSR (2607.07320: anticipación, PC-BAS, síntesis de vistas, SynLoc, VQA). Lo
  nuevo en fútbol broadcast es SoccerMaster (CVPR 2026) y Broadcast2Pitch (WACV 2026 oral, homografía + ReID + tracklets
  refinados con un VLM; preprint y cifras [NV]).

### 1.4 La cámara: tres maneras de tratarla, complementarias

- **Compensar el movimiento en la imagen.** El término CMC nace en Tracktor (Bergmann, ICCV 2019, 1903.05625); ECC
  (Evangelidis & Psarakis, TPAMI 2008) lo usan StrongSORT, BoostTrack y UCMCTrack+; la GMC por flujo óptico disperso de
  BoT-SORT la heredan Deep OC-SORT, SparseTrack (+1,3 HOTA en MOT17-val) y TrackTrack; McByte++ la hace condicional.
  Ataca solo el error de predicción del Kalman durante el paneo: barata, online, sin calibración, y **no dice nada de quién
  vuelve al plano**.
- **Trackear en el plano del campo.** UCMCTrack ("Multi-Object Tracking with Uniform Camera Motion Compensation", Yi, AAAI
  2024, 2312.08952) pasa el Kalman al suelo con una homografía fija por secuencia y modela la cámara como ruido de
  proceso (DanceTrack 63,4): **no sirve en TV con paneo y zoom**, donde la homografía cambia cada frame. Lo que sí
  funciona es calibrar por frame y suavizar: TVCalib (Theiner & Ewerth, WACV 2023, 2207.11709, el del baseline GSR) →
  NBJW/PnLCalib (Gutiérrez-Pérez & Agudo, CVPRW 2024 → CVIU 2026, 2404.08401: keypoints + líneas + optimización, JaC5 80,6
  frente a 54,8 de TVCalib; lo usan UPCxMobius y lianyou; ya lo tenemos en `core/homography.py`) → BroadTrack (Magera,
  WACV 2025, 2412.01721: seguimiento temporal de la cámara con modelo cámara + trípode, heredero del PTZ-SLAM de Lu, Chen
  y Little, BMVC 2019, 1907.08816; lo extiende Playbox&MIXI). Con posiciones en metros se puede exigir velocidad plausible
  al asociar y al fusionar tracklets, y saber por qué borde salió un jugador (Constructor.tech, DCTracker en KBS 2024
  [cifras NV], FieldMOT en CVPRW 2025 [NV]).
- **Reidentificar a largo plazo.** La única familia que ataca de frente la reentrada: AFLink no llega (ventana de 30
  frames y 75 px); GTA-Link no tiene límite temporal salvo el no-solape y fue diseñado para reentradas; Deep HM-SORT
  conserva todos los tracklets indefinidamente en memoria; McByte++ añade memoria de identidad online; SAM-Deep-EIoU separa
  explícitamente "identidad en pantalla" (SAM 3) de "reidentificación fuera de pantalla" (GTA con dorsal + equipo + veto a
  transiciones por bordes opuestos). Nuestro diagnóstico encaja: las reentradas duran 4,4 s de mediana, el 97 % salen por
  el borde, y ningún tracker online pasa del 4 % de recuperación; Deep-EIoU 15 %, GTA 44 % [M].
- **Pseudo-profundidad** (SparseTrack 2306.05238, CAMOT WACV 2024, GRASPTrack 2508.08117, DepthMOT 2404.05518) está
  pensada para oclusiones entre peatones; en fútbol el orden por el pie ya lo da la cámara elevada y no toca las
  reentradas. **No hay ningún trabajo 2025-2026 que estime pan/tilt/zoom de la retransmisión para asociar**; lo más
  cercano es BroadTrack como módulo de localización.

---

## 2. Mapa de relaciones (verificado en los related work)

- SORT → DeepSORT → StrongSORT(++) [AFLink, GSI] → baseline SoccerNet GSR (TrackLab).
- SORT → OC-SORT → Deep OC-SORT [CMC de BoT-SORT] → MeMoSORT; OC-SORT → Hybrid-SORT; OC-SORT → MixSort-OC.
- SORT/ByteTrack → **BoT-SORT [GMC]** → **Deep-EIoU** → Deep HM-SORT / **GTA-Link** / ARC-SORT / SAM-Deep-EIoU;
  Deep-EIoU + GTA → GTATrack (SoccerTrack 2025) y KIST-GSR (ganador GSR 2025, con OSNet); BoT-SORT + GTA → Playbox&MIXI.
- C-BIoU [BIoU] → Deep-EIoU [EIoU] → Mo-IoU (MeMoSORT), EIoU adaptativa (ARC-SORT); BIoU → soft-BIoU (BoostTrack++).
- ByteTrack → SparseTrack, MixSort-Byte, DiffMOT (movimiento por difusión), UCMCTrack (ablación), McByte → McByte++ [+GTA].
- Tracktor [CMC] → ECC en StrongSORT/BoostTrack/UCMCTrack+; GMC de BoT-SORT → Deep OC-SORT → TrackTrack → McByte++ (condicional).
- DETR → TrackFormer / MOTR → MOTRv2 → MeMOTR → MOTIP → SoccerMaster + MOTIP; grupo de MOTIP → HAT (solo ReID).
- BPBreID → PRTReID → baseline GSR y SoccerFactory (SoccerMaster); BPBreID → CAMELTrack; CAMELTrack → TDLP (lo compara y
  supera); SUSHI (Cetintas, CVPR 2023) → SportsSUSHI [dorsal de Koshkina & Elder + calibración de Gutiérrez-Pérez].
- TVCalib → baseline GSR; NBJW → PnLCalib → UPCxMobius, lianyou; PTZ-SLAM → BroadTrack → Playbox&MIXI.
- Koshkina & Elder [dorsal] → SportsSUSHI, Playbox&MIXI (PARSeq), SAM-Deep-EIoU; GSR 2024-2025 → dorsal + equipo como
  clave de fusión offline.
- SAM 2/3 → SAM2MOT (zero-shot), McByte/McByte++ (máscaras como pista), SMP (SAM solo en ventanas inciertas).

---

## 3. Tablas de referencia

### 3.1 Tracking-by-detection heurístico (HOTA test [P]; † tomada de la tabla de otro paper)

| Paper (1er autor) | Venue/año | arXiv | Qué cambia | SportsMOT | DanceTrack |
|---|---|---|---|---|---|
| SORT (Bewley) | ICIP 2016 | 1602.00763 | Kalman + IoU + húngaro | 56,3† | 47,9† |
| DeepSORT (Wojke) | ICIP 2017 | 1703.07402 | + ReID y cascada por edad | — | 45,6† |
| ByteTrack (Zhang) | ECCV 2022 | 2110.06864 | 2.ª pasada con detecciones flojas | 62,8 / 64,1 | 47,7 |
| StrongSORT++ (Du) | TMM 2023 | 2202.13514 | ECC, EMA, NSA; AFLink + GSI offline | — | 55,6 |
| BoT-SORT (Aharon) | arXiv 2022 | 2206.14651 | GMC + Kalman (w,h) + IoU-ReID | 68,7† | — |
| OC-SORT (Cao) | CVPR 2023 | 2203.14360 | ORU / OCM / OCR | 71,9 / 73,7 | 55,1 |
| C-BIoU (Yang) | WACV 2023 | 2211.14317 | IoU bufferizada en cascada, sin Kalman | — (SoccerNet 89,2 oráculo) | 60,6 |
| Deep OC-SORT (Maggiolino) | ICIP 2023 | 2302.11813 | CMC + apariencia dinámica | — | 61,3 |
| MixSort (Cui) | ICCV 2023 | 2304.05170 | apariencia tipo MixFormer | 65,7 / 74,1 | 46,7 (val) |
| SparseTrack (Liu) | arXiv 2023 | 2306.05238 | cascada por pseudo-profundidad | — | 55,5 |
| **Deep-EIoU (Huang)** | WACV-W 2024 | 2306.13074 | sin Kalman, EIoU iterativa + OSNet | **77,2** | — |
| Hybrid-SORT (Yang) | AAAI 2024 | 2308.00783 | pistas débiles (confianza, altura) | — | 62,2 / 65,7 |
| UCMCTrack (Yi) | AAAI 2024 | 2312.08952 | Kalman en el suelo, CMC uniforme | — | 63,4 / 63,6 |
| DiffMOT (Lv) | CVPR 2024 | 2403.02075 | movimiento por difusión | 72,1 / 76,2 | 62,3 |
| BoostTrack++ (Stanojević) | Filomat 2025 | 2408.13003 | una etapa, soft-BIoU | — | — (MOT17 66,6) |
| Deep HM-SORT (Gran-Henriksen) | arXiv 2024 | 2406.12081 | media armónica IoU/apariencia | 80,1 | — |
| **GTA-Link (Sun)** | ACCV-W 2024 | 2411.08216 | post-proceso offline de tracklets | **81,0** | — |
| TrackTrack (Shim) | CVPR 2025 | — | asociación centrada en la pista + CMC | — | 66,5† |
| McByte++ (Stanczyk) | arXiv 2026 | 2608.15688 | máscaras SAM, CMC condicional, memoria | 79,9 / 81,5 (+GTA) | — |
| HAT (Gao) | ECCV 2026 | 2503.12562 | ReID transformado por historial | 78,1-80,8 | 58,6 / 61,2 |
| MeMoSORT (Wang) | arXiv 2025 | 2508.09796 | Kalman con memoria + Mo-IoU | 82,1 | 67,9 |
| ARC-SORT (Adari) | arXiv 2026 | 2610.22306 | Kalman IMM + EIoU adaptativa, sin entrenar | 81,8 (80,3 con dets públicas) | 62,1 |
| SMP / SAM-Deep-EIoU (Holmberg) | arXiv 2026 | 2606.13033 | SAM 3 solo en ventanas inciertas + GTA | **87,2** | — |

### 3.2 End-to-end y asociación aprendida (HOTA test [P])

| Paper (1er autor) | Venue/año | arXiv | Tipo | SportsMOT | DanceTrack |
|---|---|---|---|---|---|
| TrackFormer (Meinhardt) | CVPR 2022 | 2101.02702 | E2E, track queries | — | — (MOT17 57,3) |
| MOTR (Zeng) | ECCV 2022 | 2105.03247 | E2E | — | 54,2 |
| MOTRv2 (Zhang) | CVPR 2023 | 2211.09791 | E2E + anclas YOLOX | — | 69,9 (73,4 ensemble) |
| MeMOTR (Gao) | ICCV 2023 | 2307.15700 | E2E + memoria | 70,0 | 68,5 |
| MOTIP (Gao) | CVPR 2025 | 2403.16848 | DETR + decodificador de id | 72,6 | 69,6 / 72,0 |
| SambaMOTR (Segu) | ICLR 2025 | 2410.01806 | E2E + Mamba | 69,8 / 72,1 | 67,2 / 69,0 |
| GeneralTrack (Qin) | CVPR 2024 | 2406.00429 | asociación aprendida | 74,1 | 59,2 |
| CAMELTrack (Somers) | arXiv 2025 | 2505.01257 | asociación aprendida (TrackLab) | 80,4 | 66,1 / 69,3 |
| TDLP (Adžemović) | arXiv 2025-26 | 2512.22105 | predicción de enlaces | 81,9 | 70,1 |
| SAM2MOT (Jiang) | AAAI 2026 [NV] | 2504.04519 | zero-shot Grounding-DINO + SAM 2 | — | 75,8 |
| SoccerMaster + MOTIP (Yang) | CVPR 2026 | 2512.11016 | fundacional fútbol + MOTIP | — (SoccerNet-Tracking 59,1) | — |

### 3.3 Fútbol broadcast y cámara

| Paper (1er autor) | Venue/año | arXiv | Aporta | Cifra clave [P] |
|---|---|---|---|---|
| SoccerNet-Tracking (Cioppa) | CVPRW 2022 | 2204.06918 | dataset TV | ByteTrack 71,5 (GT) / 47,2 (propias) |
| Challenges 2022 / 2023 (Giancola / Cioppa) | arXiv | 2210.02365 / 2309.06006 | ganadores | Kalisteo 93,64 (GT) / 75,61 |
| BPBreID (Somers) | WACV 2023 | 2211.03679 | ReID por partes | base de PRTReID y CAMELTrack |
| PRTReID (Mansourian) | MMSports 2023 | 2401.09942 | ReID + equipo + rol | SoccerNet 90,8 (GT) / 59,8 (YOLOv8) |
| SN-GSR (Somers) | CVPRW 2024 | 2404.11335 | tarea + GS-HOTA | baseline 22,3; HOTA 57,6 |
| From Broadcast to Minimap (Golovkin) | CVPRW 2025 | 2504.06357 | ganador GSR 2024 | GS-HOTA 63,81 |
| SoccerNet 2025 (Giancola) | arXiv 2025 | 2508.19182 | KIST-GSR, Playbox&MIXI (BoT-SORT+GTA) | 63,90 / 61,64 |
| SportsSUSHI (Koshkina) | arXiv 2025 | 2502.21242 | grafos + dorsal + campo | SoccerNet 90,9 (GT) / 71,4 |
| Jersey number recognition (Koshkina & Elder) | CVPRW 2024 | 2405.13896 | dorsal por STR | 87,4 % por tracklet |
| Tracktor (Bergmann) | ICCV 2019 | 1903.05625 | origen del CMC | — |
| TVCalib (Theiner & Ewerth) | WACV 2023 | 2207.11709 | calibración por frame | AC@5 57,6 |
| PnLCalib (Gutiérrez-Pérez & Agudo) | CVPRW 2024 → CVIU 2026 | 2404.08401 | keypoints + líneas | JaC5 80,6 |
| BroadTrack (Magera) | WACV 2025 | 2412.01721 | cámara de TV en el tiempo | +15 % Jaccard |
| GTATrack (Jian) | MMSports 2025 | 2602.00484 | Deep-EIoU + GTA (panorámica fija) | SoccerTrack 0,60 |
| Surveys 2026 (Yang; Ma) | arXiv 2026 | 2609.08265 / 2609.22291 | qué componente suma | detección > 10 pt, asociación < 5 |

---

## 4. Qué significa para nosotros

- **Estamos donde está la literatura con este detector.** Con las cajas de SoccerMaster: ByteTrack 56,4 → BoT-SORT + GMC +
  NMS + OSNet 59,4 [M]; SoccerMaster + MOTIP 59,1 y YOLOv8 + PRTReID 59,8 [P] están en la misma banda. Deep-EIoU (62,4) y
  Deep-EIoU + GTA (64,3) [M] marcan lo que la asociación puede dar sin cambiar el detector. El hueco hasta SportsSUSHI
  (71,4) o Kalisteo 2023 (75,6) viene de dos cosas que la literatura señala: **DetA** (ellos afinan YOLOX en SoccerNet;
  nuestro DetA es 69 frente a 97,5 con oráculo [M]) y **enlace a largo plazo con dorsal + equipo + campo**, que nadie de la
  banda 59-64 hace.
- **La GMC ya la tenemos y es lo que la literatura dice que vale**: imprescindible con cámara móvil, residual después. El
  siguiente escalón de asociación online es la línea BIoU → EIoU: Deep-EIoU, Deep HM-SORT o ARC-SORT (sin entrenamiento,
  80,3 en SportsMOT con detecciones y embeddings públicos) sustituyendo al Kalman + IoU de BoT-SORT.
- **Las reentradas se resuelven a nivel de tracklet, no de frame**: GTA-Link (ya medido, +1,9 HOTA / +4,5 IDF1 y 44 % de
  reentradas recuperadas [M]) y, encima, dorsal (SoccerMaster ya lo da por detección) y equipo (PRTReID ya lo da) como
  clave dura, como hacen Constructor.tech, KIST-GSR y SAM-Deep-EIoU. Encaja con nuestro modo partido completo.
- **La calibración por frame que ya tenemos (PnLCalib) vale para el tracking**, no solo para el mapa: posiciones en metros
  para exigir velocidad plausible al fusionar tracklets y para saber por qué borde salió el jugador. Hace falta
  suavizado temporal tipo BroadTrack; PnLCalib solo calibra el 34-71 % de los frames de TV [M].
- **Lo que no aplica**: UCMCTrack (homografía fija), pseudo-profundidad (SparseTrack y derivados), los end-to-end
  (10 puntos por detrás en deporte, 8 GPUs de entrenamiento, sin pesos de fútbol), y entrenar una asociación aprendida
  (CAMELTrack/TDLP) sin verdad propia de tracking en nuestro dominio.
- **Licencias**: Deep-EIoU, GTA-Link, su OSNet "sports" y GTATrack no tienen licencia; ARC-SORT, McByte++, SMP y TDLP
  habría que comprobarlas; ultralytics es AGPL-3.0; TrackLab/CAMELTrack y PnLCalib son abiertos (MIT / GPL-3.0 según módulo).

## 5. Siguientes pasos

| # | Paso | Coste | Resultado esperado |
|---|---|---|---|
| 1 | GTA-Link como post-proceso en modo partido completo sobre los tracklets de BoT-SORT, con dorsal + equipo como veto/clave (receta Constructor.tech / SAM-Deep-EIoU) | 1-2 d; offline, ~0 GPU extra (OSNet ya calculado) | HOTA 59,4 → ~62-64 y reentradas 4 % → 30-40 % en el test [M a medir]; luego medir en Espanyol/Mallorca con `evaluate_match.py` |
| 2 | Asociación EIoU en lugar de Kalman + IoU: portar Deep-EIoU / ARC-SORT a `BotSortTracker` manteniendo GMC y el ReID actual; ajustar en train, una pasada en test | 1-2 d | +2-3 HOTA online (Deep-EIoU 62,4 medido con las mismas cajas) |
| 3 | Detector: afinar un YOLO11/YOLOX en SoccerNet-Tracking train frente a SoccerMaster (prioridad 4 de `comparativa_benchmark.md` §4) | 1 d GPU | la palanca mayor según los surveys de 2026 (DetA 69 → ?) |
| 4 | Calibración temporal (suavizado tipo BroadTrack sobre PnLCalib) y plausibilidad física en la fusión de tracklets | 2-3 d | tracklets fusionados solo si la velocidad en metros es plausible; base para GS-HOTA propio |
| 5 | Guardarraíl de la GMC (McByte++): descartar afines degeneradas en zooms bruscos | 0,5 d | menos IDSW en cortes de plano/zoom; medir en test |

## 6. Referencias clave (arXiv)

SORT 1602.00763 · DeepSORT 1703.07402 · Tracktor 1903.05625 · PTZ-SLAM 1907.08816 · TrackFormer 2101.02702 · MOTR 2105.03247 ·
ByteTrack 2110.06864 · StrongSORT 2202.13514 · OC-SORT 2203.14360 · SoccerNet-Tracking 2204.06918 · BoT-SORT 2206.14651 ·
TVCalib 2207.11709 · SoccerNet 2022 results 2210.02365 · BPBreID 2211.03679 · MOTRv2 2211.09791 · 2.º SoccerNet 2022
2211.13481 · C-BIoU 2211.14317 · Deep OC-SORT 2302.11813 · SportsMOT/MixSort 2304.05170 · SparseTrack 2306.05238 ·
Deep-EIoU 2306.13074 · MeMOTR 2307.15700 · Hybrid-SORT 2308.00783 · MOT4MOT 2308.16651 · SoccerNet 2023 results 2309.06006 ·
Balaji dorsales 2309.06285 · UCMCTrack 2312.08952 · PRTReID 2401.09942 · DiffMOT 2403.02075 · MOTIP 2403.16848 · PnLCalib
2404.08401 · SN-GSR 2404.11335 · Koshkina & Elder 2405.13896 · GeneralTrack 2406.00429 · Deep HM-SORT 2406.12081 ·
BoostTrack++ 2408.13003 · SoccerNet 2024 results 2409.10587 · SambaMOTR 2410.01806 · GTA-Link 2411.08216 · BroadTrack
2412.01721 · SportsSUSHI 2502.21242 · HAT 2503.12562 · SAM2MOT 2504.04519 · From Broadcast to Minimap 2504.06357 ·
CAMELTrack 2505.01257 · McByte 2506.01373 · SoccerNet 2025 results 2508.19182 · MeMoSORT 2508.09796 · SoccerTrack v2
2508.01802 · MATR 2509.21715 · SelfMOTR 2511.20279 · SoccerMaster 2512.11016 · TDLP 2512.22105 · GTATrack 2602.00484 · SMP
2606.13033 · SoccerNet 2026 results 2607.07320 · McByte++ 2608.15688 · Survey TbD 2609.08265 · Beyond the Survey 2609.22291 ·
ARC-SORT 2610.22306.
