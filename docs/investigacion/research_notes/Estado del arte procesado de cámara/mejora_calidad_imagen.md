# Mejora y restauración de la calidad de imagen/vídeo como preprocesado para análisis automático (fútbol)

Notas de investigación, noviembre de 2026. Marcas: **[R]** = reciente (2024-2026), **[A]** = antecedente (2016-2023).
Nota de método: todas las fuentes citadas se abrieron en esta sesión (arXiv `abs`/`html`, ar5iv, actas). Las cifras se
extrajeron de las tablas HTML con un lector automático; van con su número de tabla para poder recomprobarlas, y las
incoherencias detectadas en los propios artículos se señalan. Lo que solo apareció en un resultado de búsqueda y no se pudo
abrir (CVF Open Access, MDPI, DOAJ y Preprints.org devolvieron 403) figura como **«sin verificar»** en *Gaps*. Cuando la
página abierta no nombra la sede, se dice.

## 1. Superresolución (imagen y vídeo): ¿mejora detección de objetos pequeños, dorsales, re-identificación o acciones?

### Takeaway
La SR «para el ojo» puesta delante de un modelo congelado da ganancias pequeñas y solo cuando la entrada es muy baja
(≤480p); a 720p-1080p la única medida en fútbol es de +0,8 a +1,1 puntos de mAP@[.5:.95] con un detector sin afinar, y con
el mAP@0,5 bajando. Las ganancias grandes y consistentes aparecen solo cuando la SR se entrena con la pérdida de la tarea
(TDSR, SR4IR, EDTR) o se usa como rama auxiliar durante el entrenamiento; no hay ninguna evidencia medida de que la SR
mejore el balón, los dorsales, la re-identificación o el action spotting en fútbol.

### Cited Findings

**Métodos de referencia (qué es el estado del arte)**
- [A] Real-ESRGAN (Wang, Xie, Dong, Shan; informe técnico, 2021): SR ciega de imagen entrenada con degradaciones
  sintéticas de alto orden (desenfoque, ruido, reescalado, JPEG); el resumen solo reclama «superior visual performance»,
  sin métricas de tarea — [arXiv 2107.10833](https://arxiv.org/abs/2107.10833)
- [A] BasicVSR++ (Chan, Zhou, Xu, Loy; arXiv 2021, la página no nombra sede): SR de vídeo recurrente; 7,3 M de parámetros
  y 77 ms de tiempo de ejecución con entrada LR de 180×320 (Tabla 1) —
  [ar5iv 2104.13371](https://ar5iv.labs.arxiv.org/html/2104.13371)
- [A] RealBasicVSR (Chan, Zhou, Xu, Loy; arXiv 2021, la página no nombra sede): SR de vídeo real; 6,3 M de parámetros y
  63 ms para salida de 720×1280 en una V100, frente a 149 ms de Real-ESRGAN/BSRGAN/RealSR (16,7 M) y 1082 ms de RealVSR
  (Tabla 2). Advierte de que «temporal propagation is a double-edged sword» y de que propagar por toda la secuencia «leads
  to outputs with severe artifacts» (§3.1) — [ar5iv 2111.12704](https://ar5iv.labs.arxiv.org/html/2111.12704)
- [R] SeedVR2 (Wang et al., 13 autores; arXiv jun. 2025, «Camera Ready of ICLR2026»): restauración de vídeo por difusión
  en un solo paso con post-entrenamiento adversario; el resumen no da cifras de velocidad —
  [arXiv 2506.05301](https://arxiv.org/abs/2506.05301)
- [R] FlashVSR (Zhuang, Guo, Cai, Li, Liu, Yuan, Xue; arXiv oct. 2025): SR de vídeo por difusión en streaming, un paso;
  «∼17 FPS for 768×1408 videos on a single A100 GPU». Su Tabla 2 compara con los demás métodos de difusión (Upscale-A-Video,
  STAR, DOVE, SeedVR2-3B); cifras en la pregunta 7 — [arXiv 2510.12747](https://arxiv.org/html/2510.12747v1)

**Evidencia en fútbol y deporte**
- [R] Seweryn, Chęć, Łukasik, Wróblewska (arXiv, ene. 2024, sin sede): SR con RLFN (×2/×3/×4/×6) sobre 100 clips de
  SoccerNet y Faster R-CNN ResNet50 **sin afinar** («in its original form, without any additional training»). Tabla III,
  mAP@[.5:.95] sin SR → con RLFN ×4: 320×240: 24,3 → 27,3 (≈ +12 % relativo); 640×480: 27,6 → 28,5; 1280×720: 29,5 → 30,3
  (SR entrenada en SoccerNet) o 30,6 (entrenada en UHDSR8K). ×6 no supera a ×4. **El mAP@0,5 baja con SR en todas las
  resoluciones** (320×240: 46,7 → 41,4; 1280×720: 56,3 → 42,4) y los autores lo reconocen («the mAP@IoU=0.50 metric shows
  a decline»). Clases fundidas en «ball» y «person» y **sin resultados por clase**: no se sabe qué pasa con el balón. No se
  informa del coste — [arXiv 2402.00163](https://arxiv.org/html/2402.00163v1)
- [A] Tabla I del mismo trabajo, sin SR: el detector apenas gana por encima de 720p (mAP@[.5:.95] 29,5 a 1280×720 y 29,5 a
  1920×1080) — [arXiv 2402.00163](https://arxiv.org/html/2402.00163v1)
- [R] Martini, Zolezzi, Iacono, Vercelli (arXiv mar. 2025; Scientific Reports 16:1882, 2026): reescalador generativo por
  difusión (img2img + ControlNet + LoRA) de recortes de retransmisión de 64×64 a 1024×1024; el resumen no evalúa ninguna
  tarea posterior, solo calidad visual — [arXiv 2503.11181](https://arxiv.org/abs/2503.11181)
- [A] Dorsales, SoccerNet 2023 (Cioppa, Giancola, Somers, Magera, Zhou et al., 104 autores): el único equipo que declara SR
  (AIBrain, J7: «we upscale the image using ESRGAN-based model») quedó 7.º con 75,18 % —
  [ar5iv 2309.06006](https://ar5iv.labs.arxiv.org/html/2309.06006)
- [R] Koshkina y Elder (arXiv may. 2024): 87,4 % en tracklets de SoccerNet **sin SR ni deblurring**; lo que usan es un
  clasificador de legibilidad (ResNet34) que descarta fotogramas ilegibles, filtrado del sujeto principal y agregación por
  tracklet. Quitar el filtrado del sujeto principal cuesta 2,89 puntos (87,45 → 84,56, Tabla 8) —
  [arXiv 2405.13896](https://arxiv.org/html/2405.13896)
- [R] Balaji, Bright, Rambhatla, Chen, Wong, Zelek, Clausi (arXiv mar. 2024): en vez de restaurar, preentrenan un MAE cuyo
  enmascarado aplica desenfoque de movimiento a los parches; SoccerNet test 77,31 % frente a 75,83 % con enmascarado a cero
  y 76,47 % con desenfoque gaussiano (Tabla II(b)). No comparan con deblurring ni SR —
  [arXiv 2403.11328](https://arxiv.org/html/2403.11328v1)

**SR orientada a la tarea (literatura general)**
- [A] Haris, Shakhnarovich, Ukita (arXiv 2018): TDSR, la red de SR se entrena con la pérdida del detector; el resumen dice
  que mejora «consistently and significantly» la detección en baja resolución y que SR y detector optimizados por separado
  es lo habitual — [arXiv 1803.11316](https://arxiv.org/abs/1803.11316)
- [R] SR4IR (Kim, Oh, Lee; arXiv abr. 2024; CVPR 2024 según el listado de CVF, no abierto): detección en VOC2012 con Faster
  R-CNN, oráculo HR 36,7 mAP (Tabla 2). A ×4: bicúbico 27,5; SR con pérdida de píxel y luego detector (S→T) 30,3 (EDSR) /
  31,5 (SwinIR); entrenamiento conjunto ingenuo 28,3 / 31,4; SR entrenada contra un detector ya fijado (T→S) 27,2 / 29,7
  (peor que bicúbico con EDSR); SR4IR 32,4 / 33,9. A ×8: bicúbico 18,9; S→T 21,9 / 23,7; SR4IR 25,5 / 27,8 —
  [arXiv 2404.01692](https://arxiv.org/html/2404.01692v2)
- [R] EDTR (Kim, Oh, Lee; arXiv jul. 2025; ICCV 2025 según el listado de la conferencia, no abierto): restauración guiada
  por la tarea con *prior* de difusión, bajo degradaciones mixtas (Mixture-B = desenfoque gaussiano + submuestreo ×1-16 +
  ruido + JPEG 50-100). Tabla 1, detección mAP en Mixture-A / Mixture-B: sin restaurar 14,5 / 0,0; SwinIR 21,0 / 22,1; TDSR
  16,2 / 14,6; SR4IR 22,3 / 22,6; EDTR 1 paso 31,2 / 30,6; EDTR 4 pasos 33,4 / 31,9. Más pasos de difusión empeoran
  (clasificación 68,4 con 4 pasos, 66,8 con 50; Tabla 3) — [arXiv 2507.22459](https://arxiv.org/html/2507.22459v2)
- [A] Shermeyer y Van Etten (EarthVision 2019): en satélite, la SR ayuda sobre todo en la mejor resolución nativa (30 cm →
  15 cm: mAP +13-36 %); en resoluciones más bastas «helps less» — [arXiv 1812.04098](https://arxiv.org/abs/1812.04098)
- [A] SuperYOLO (Zhang, Lei, Xie, Fang, Li, Du; IEEE TGRS): rama auxiliar de SR **solo en entrenamiento** («the SR branch
  is discarded in the inference stage»), coste de inferencia nulo; 75,09 % mAP50 en VEDAI (el resumen no aísla la aportación
  de la rama) — [arXiv 2209.13351](https://arxiv.org/abs/2209.13351)
- [A] SAHI (Akyon, Altinuc, Temizel; ICIP 2022), alternativa sin SR para objetos pequeños: inferencia por teselas, +6,8 /
  +5,1 / +5,3 AP (FCOS / VFNet / TOOD) sin reentrenar y +12,7 / +13,4 / +14,5 acumulado afinando con teselas, en VisDrone y
  xView — [arXiv 2202.06934](https://arxiv.org/abs/2202.06934)

**Reconocimiento de acciones y re-identificación**
- [A] Zhang, Liu, Xiong (arXiv 2019; ICCV 2019 según el listado de CVF, no abierto), Tabla 4, UCF101, TSN, vídeo ×4: HR
  86,02 / 87,63 / 93,49 (espacial / temporal / fusión); bicúbico 71,25 / 81,08 / 87,87; ESRGAN 82,97 / 75,32 / 87,75 (sube
  la rama espacial 11,7 puntos y **baja la temporal 5,8**: fusión por debajo del bicúbico); VDSR 67,09 / 79,81 / 86,84
  (peor que bicúbico en todo); su SR orientada a reconocimiento 92,13 en fusión. «The SR methods optimized for visual
  quality are not optimal for action recognition task» — [ar5iv 1903.05577](https://ar5iv.labs.arxiv.org/html/1903.05577)
- [A] Chen, Qiao, Wang (arXiv 2022, reto Tiny Actions de ActivityNet): RealBasicVSR como generador fijo para crear vídeos SR
  de entrenamiento; ir-CSN F1 0,469 → 0,484 con SR y 0,492 con SR + destilación (Tabla 2) —
  [arXiv 2209.14711](https://arxiv.org/html/2209.14711v2)
- [A] Mao, Zhang, Yang (IJCAI 2019): en re-ID de baja resolución la SR no va sola, sino entrenada conjuntamente con un
  extractor invariante a la resolución; Rank-1 36,4 % en CAVIAR y 73,3 % en MLR-CUHK03 —
  [arXiv 1906.09748](https://arxiv.org/abs/1906.09748)

**Alucinaciones**
- [R] Ren, Goyal, Hu, Aumentado-Armstrong, Mohomed, Levinshtein (arXiv jul. 2025, v2 ene. 2026): las alucinaciones de la SR
  generativa (detalles que no casan con la imagen LR ni con la real) «are not well-characterized with existing image
  metrics»; proponen una puntuación con un LLM multimodal — [arXiv 2507.14367](https://arxiv.org/abs/2507.14367)
- [R] EDTR: el *prior* de difusión sin diseño específico «leads to excessive generation and produces non-existent content»
  (§4.2.1) y «can produce undesirable patterns, leading to misdetection results» (apéndice S8) —
  [arXiv 2507.22459](https://arxiv.org/html/2507.22459v2)

### Inferences
- Para nuestra entrada (720p-1080p) la SR previa tiene poco recorrido: en la única medida futbolística la ganancia a 720p es
  de ~1 punto con un detector de COCO sin afinar, y el propio detector no gana nada de 720p a 1080p. Con un detector afinado
  al dominio la ganancia esperable es menor (SR4IR: la mejor opción «SR fija + tarea» se queda 2-3 puntos por debajo de la SR
  entrenada con la tarea, y la ganancia absoluta se reduce cuanto menos falta resolución).
- Para el balón de 5-15 px, la literatura apunta antes a dar más píxeles reales al detector (inferir a resolución nativa o
  por teselas, SAHI) y a mapas de características de alta resolución que a inventar píxeles; la caída del mAP@0,5 en Seweryn
  et al. es compatible con falsos positivos o pérdida de objetos pequeños tras la SR, pero el artículo no lo desglosa.
- La SR generativa (GAN o difusión) es la que más riesgo tiene para modelos temporales: mejora la apariencia por fotograma y
  rompe la coherencia entre fotogramas (Zhang et al. 2019), que es justo lo que usan T-DEED/AdaSpot y el tracking.
- Si se quisiera SR, las dos formas con respaldo son (a) entrenarla con la pérdida de la tarea y (b) usarla como rama
  auxiliar de entrenamiento con coste cero en inferencia; ninguna se ha medido en fútbol.
- En dorsales, la palanca medida es elegir los fotogramas legibles y agregar por tracklet, no restaurar los ilegibles.

### Gaps
- No encontré ningún trabajo que mida SR sobre detección del **balón** por separado, sobre **action spotting**, sobre
  **tracking/re-ID de jugadores** ni sobre cámaras amateur o panorámicas.
- No encontré ninguna medida del efecto de la SR de vídeo por difusión (SeedVR2, FlashVSR, DOVE, STAR) sobre tareas
  posteriores; solo se evalúan con métricas perceptuales.
- Sin verificar: «Evaluating and Preserving High-level Fidelity in Super-Resolution» (arXiv 2512.07037), «OSRO» (SPIE 2024,
  SR en una etapa orientada a reconocimiento de acciones) y la ficha de proyecto del equipo ESRGAN de dorsales; solo
  vistos en resultados de búsqueda.
- La caída del mAP@0,5 con SR en Seweryn et al. no está explicada en el artículo y convendría contrastarla con el PDF.
- En EDTR el valor «0,0» de detección sin restaurar en Mixture-B es llamativo; recomprobar contra la tabla original.

## 2. Desenfoque de movimiento, obturación y fps: deblurring, detectores conscientes del desenfoque, interpolación y rolling shutter

### Takeaway
El deblurring como preprocesado es inconsistente y a veces catastrófico para el detector (COCO limpio 58,5 → 55,5; GOPRO
borroso 19,6 → 2,2), mientras que entrenar con desenfoque y con etiquetas adaptadas gana siempre; en deporte, la línea que
funciona es tratar el desenfoque como señal (BlurBall) o como aumento (pose en béisbol, dorsales), no quitarlo. No hay
evidencia medida de que interpolar fotogramas ayude a action spotting o a seguir el balón: lo único medido es ≈ +1 punto en
segmentación de objetos a bajo fps.

### Cited Findings

**Deblurring frente a entrenar con desenfoque**
- [A] Sayed y Brostow (CVPR, según los comentarios de arXiv; 2020-2021), Faster R-CNN en COCO minival, mAP@0,5, exposición
  de leve (E1) a fuerte (E5), Tablas 1-2: detector original 58,50 limpio y 50,95 / 32,59 / 15,75 / 8,75 / 4,58; **deblurring
  (Nah et al. 2017) + detector original** 55,50 limpio y 49,18 / 42,13 / 30,31 / 12,72 / 6,26; **detector entrenado con
  aumento de desenfoque** 56,51 limpio y 54,93 / 52,44 / 46,85 / 37,56 / 31,37; su mejor método (etiquetas expandidas +
  especialización por exposición) 58,62 limpio y 58,01 / 56,40 / 50,97 / 46,37 / 43,78 —
  [ar5iv 2011.14448](https://ar5iv.labs.arxiv.org/html/2011.14448)
- [A] Mismo trabajo, desenfoque real (nítido / borroso, pseudo-verdad de DetectoRS): en GOPRO el deblurring previo hunde el
  detector (35,85 / 19,64 → 10,58 / 2,24); en REDS ayuda (24,38 → 31,84 en borroso); el aumento de desenfoque mejora en todas
  las columnas borrosas (GOPRO 26,54, RealBlur 35,63, REDS 29,24). Resumen: el deblurring «only partly improves object
  detection»; y «There is no benefit in augmenting for blur, and then using either minibatch statistics or a deblurer
  network» (Fig. 5) — [ar5iv 2011.14448](https://ar5iv.labs.arxiv.org/html/2011.14448)
- [A] NAFNet (Chen, Chu, Zhang, Sun; ECCV 2022), referencia de deblurring de imagen: 33,69 dB en GoPro con el 8,4 % del coste
  del anterior estado del arte; solo métricas de píxel — [arXiv 2204.04676](https://arxiv.org/abs/2204.04676)

**Balón y jugadores en deporte**
- [R] BlurBall (Gossard, Radovic, Ziegler, Zell; arXiv sep. 2025, v2 feb. 2026; tenis de mesa): el balón rápido pasa «from a
  distinct dot into a streak»; etiquetar el **punto medio** de la estela en vez del frente mejora a todos los detectores
  (WASB: F1 95,58 → 96,00; AP 95,10 → 96,85; Tabla 2); el modelo que además estima longitud y ángulo del desenfoque llega a
  F1 96,52 y AP 98,23 (Tabla 3), y usar el desenfoque en la predicción de trayectoria baja el error de 84,4 a 53,0 px
  (§4.5). No compara con deblurring previo ni da resultados en fútbol —
  [arXiv 2509.18387](https://arxiv.org/html/2509.18387v2)
- [A] DeepBall (Komorowski, Kurzejamski, Sarwas; arXiv 2019): «When a ball is kicked and moves at high velocity, its image
  becomes blurry and elliptical»; lo resuelven con arquitectura (hipercolumnas) y aumentos, sin restauración; AP 0,877 en
  ISSIA-CNR — [ar5iv 1902.07304](https://ar5iv.labs.arxiv.org/html/1902.07304)
- [A] FootAndBall (mismos autores; arXiv 2019): en plano general el balón «can appear as small as 8 pixels» y hasta ~20 px en
  la banda cercana; FPN con mapa de confianza a 1/4 de resolución, sin restauración; AP de balón 0,909 en ISSIA-CNR, 37 fps a
  1920×1080 en una GTX 1060 con ~199 k parámetros — [ar5iv 1912.05445](https://ar5iv.labs.arxiv.org/html/1912.05445)
- [A] SoccerNet 2023, tracking: el equipo ICOST (T4) declara «we first run a state-of-the-art image deblurring method to
  pre-preprocess images» antes de entrenar el detector; quedó 4.º (HOTA 65,67), sin ablación que aísle el deblurring —
  [ar5iv 2309.06006](https://ar5iv.labs.arxiv.org/html/2309.06006)
- [A] Bright, Chen, Zelek (MMSports 2023): a 30 fps la retransmisión produce «partial body motion blur during fast actions»
  que limita la pose; lo atacan con **aumento sintético de desenfoque** y vídeo real añadido, no con deblurring: pérdida
  −54,2 % (2D) y −36,2 % (3D) y mejora media del 29,2 % en estimadores de pose existentes —
  [arXiv 2309.01010](https://arxiv.org/abs/2309.01010)
- [R] Otsubo, Sawafuji, Saito (MMSports 2025; fútbol americano con cámara a pulso): «captured using handheld cameras,
  introducing motion blur that affects tracking accuracy»; BoT-SORT preentrenado HOTA 29,6 / IDF1 34,3; **afinando solo el
  detector** 44,9 / 55,3; solo el ReID 29,9 / 34,5; ambos 45,7 / 56,1 (Tablas 5-6) —
  [arXiv 2511.09455](https://arxiv.org/html/2511.09455v1)
- [A] Robustez de modelos de acción al desenfoque: en Kinetics-400P el desenfoque de movimiento es el peor desenfoque para
  R3D (robustez absoluta/relativa 0,73 / 0,63 frente a 0,79 / 0,71 del zoom y 0,83 / 0,77 del desenfoque óptico; Tabla 4)
  — [ar5iv 2207.01398](https://ar5iv.labs.arxiv.org/html/2207.01398); en Mini SSV2-C el desenfoque de movimiento cuesta
  15,4-28,4 % de acierto (§4.3) — [ar5iv 2110.06513](https://ar5iv.labs.arxiv.org/html/2110.06513)

**Interpolación de fotogramas y fps**
- [A] RIFE (Huang, Zhang, Heng, Shi, Zhou; ECCV 2022): 16 ms por fotograma a 640×480, 31 ms a 720p y 68 ms a 1080p en una
  TITAN X (Pascal) (§6.4, Tabla 3); 4-27 veces más rápido que SuperSlomo y DAIN —
  [ar5iv 2011.06294](https://ar5iv.labs.arxiv.org/html/2011.06294)
- [A] SportsSloMo (Chen y Jiang; arXiv 2023): banco de interpolación centrado en personas, >130 k clips y 1 M de fotogramas
  de cámara lenta deportiva ≥720p; los métodos del estado del arte reentrenados rinden peor que en otros conjuntos; no
  evalúa tareas posteriores — [arXiv 2308.16876](https://arxiv.org/abs/2308.16876)
- [A] FLAVR (Kalluri, Pathak, Chandraker, Tran; arXiv 2020, la página no nombra sede): única medida encontrada de
  «interpolar y luego analizar». Segmentación de objetos en DAVIS submuestreado: 15 → 30 fps, J&F 65,5 → 66,6; 8 → 30 fps,
  61,9 → 62,8. Como preentrenamiento autosupervisado para acciones: UCF101 63,10 frente a 50,02 de inicialización aleatoria
  y 87,70 del supervisado en Kinetics — [ar5iv 2012.08512](https://ar5iv.labs.arxiv.org/html/2012.08512)
- [A] Feng, Bai, Yao, Yu, Ouyang (arXiv 2022, v3 2023): la precisión de «all recent state-of-the-art trackers drops
  dramatically when the input frame rate changes»; lo arreglan en el **tracker** (asociación consciente del fps y
  entrenamiento periódico), no interpolando — [arXiv 2209.11404](https://arxiv.org/abs/2209.11404)
- [A] Need for Speed (Galoogahi, Fagg, Huang, Ramanan, Lucey; arXiv 2017): con captura real a 240 fps (100 vídeos, 380 k
  fotogramas) «simple trackers such as correlation filters outperform complex methods based on deep networks» —
  [arXiv 1703.05884](https://arxiv.org/abs/1703.05884)

**Rolling shutter**
- [A] Hu, Datta, Beerel, Beerel (arXiv 2023; datos sintéticos de Unreal Engine 5, peatones, objetos lentos): obturador global
  y rodante son «remarkably congruent for coarse-grained detection» pero difieren en medidas finas de localización; «ML
  pipelines might not need explicit correction for RS for many object detection applications» —
  [arXiv 2309.08136](https://arxiv.org/abs/2309.08136)

### Inferences
- La interpolación no añade información sobre el balón: si está borroso u oculto en los dos fotogramas de origen, el
  fotograma intermedio es una hipótesis del modelo de flujo; con objetos de 5-15 px y movimiento no lineal (botes, toques)
  el riesgo de balones duplicados o desplazados es alto. La ganancia medida en otra tarea es ≈ 1 punto.
- Subir los fps **en captura** sí aporta (menos desplazamiento entre fotogramas y, con obturación corta, menos desenfoque),
  pero es una decisión de grabación, no de preprocesado; con vídeo de YouTube a 25-30 fps no es accionable.
- El desenfoque del balón es información de velocidad y dirección (BlurBall): la vía con mejor respaldo es etiquetar el
  centro de la estela, entrenar con aumento de desenfoque de movimiento y, si acaso, predecir sus atributos.
- Con cámara a pulso, afinar el detector al dominio vale +15 HOTA (Otsubo et al.); es el orden de magnitud contra el que
  habría que comparar cualquier restauración.
- El rolling shutter solo sería relevante para medidas finas (posición subpíxel del balón, calibración) con paneos bruscos;
  para detección gruesa la evidencia disponible dice que no hace falta corregirlo.

### Gaps
- No encontré ningún detector de balón consciente del desenfoque medido en fútbol, ni una comparación deblurring frente a
  aumento sobre balones.
- No encontré ninguna medida de interpolación de fotogramas sobre action spotting, tracking de jugadores o balón.
- Sin verificar: suplemento de «Precise Event Spotting» (Santra et al., CVPR 2025), cuyo fragmento de búsqueda dice que más
  fotogramas aumentan el cómputo sin mejorar el spotting; y un trabajo de IET Image Processing que entrena conjuntamente
  interpolación implícita y tracking. Ninguno se pudo abrir.
- No encontré estudios del efecto del rolling shutter sobre objetos rápidos reales ni sobre tracking; el único trabajo es
  sintético y de objetos lentos.
- No encontré estudios que relacionen tiempo de obturación con precisión de detección del balón en fútbol.

## 3. Baja luz, focos, contraluz, sombras duras, lluvia y niebla: mejora (Retinex, Zero-DCE y sucesores) y efecto medido

### Takeaway
La mejora de baja luz pensada para el ojo no mejora de forma fiable la detección ni el reconocimiento: en ExDark, un
detector afinado sobre las imágenes oscuras iguala o supera a los afinados sobre imágenes mejoradas (49,2 frente a 48,5-49,2),
y en acciones a oscuras solo la corrección gamma ayuda de forma consistente (la ecualización de histograma empeora 3 de 4
modelos). Con lluvia y niebla pasa lo mismo o peor: casi todos los métodos de eliminación de lluvia empeoran la detección.
Lo que sí gana son los módulos entrenados con la pérdida de la tarea (IA-YOLO, FeatEnHancer, YOLA).

### Cited Findings

**Métodos de referencia y su coste**
- [A] Zero-DCE (Guo, Li, Guo, Loy, Hou, Kwong, Cong; arXiv 2020, la página no nombra sede): 79 416 parámetros; 0,0025 s por
  imagen de 1200×900 en una 2080 Ti, frente a 0,0078 s de EnlightenGAN, 0,12 s de RetinexNet y 0,49 s de LIME en CPU
  (Tabla 3). En caras a oscuras (DARK FACE, detector DSFD) la precisión «increases considerably» respecto a la imagen sin
  mejorar (Fig. 8, sin cifras en texto) — [ar5iv 2001.06826](https://ar5iv.labs.arxiv.org/html/2001.06826)
- [A] Retinexformer (Cai, Bian, Lin, Wang, Timofte, Zhang; arXiv 2023, la página no nombra sede): 1,61 M de parámetros y
  15,57 GFLOPs (Tabla 1). En ExDark, con YOLO-v3 **entrenado desde cero sobre cada versión mejorada**: Retinexformer 66,1
  AP; SCI 65,6; SNR-Net 65,3; Zero-DCE 65,1; KinD 65,0; RetinexNet 64,0; MIRNet 63,6 (Tabla 3(b)). **No da la cifra sin
  mejora** — [ar5iv 2303.06705](https://ar5iv.labs.arxiv.org/html/2303.06705)

**¿Ayuda a la detección?**
- [R] LIME-Eval (Li, Zhao, Guo; arXiv oct. 2024), ExDark, YOLOX-medium, Tabla 1. Afinando el detector con aumentos antes de
  la mejora: imagen oscura 49,2; Zero-DCE 49,2; IAT 49,0; Bread 48,9; RetinexFormer 48,5. Mejorando primero: Zero-DCE 45,9;
  Bread 45,9; IAT 47,6; RetinexFormer 46,1 (todos por debajo de 49,2). Con detector preentrenado sin afinar: imagen oscura
  35,0; Bread 35,4; IAT 35,3; RetinexFormer 34,2; Zero-DCE 34,0. Concluyen que «higher detection accuracy does NOT
  necessarily correlate with superior enhancement quality» y que afinar confunde la calidad del mejorador con la
  adaptabilidad del detector — [arXiv 2410.08810](https://arxiv.org/html/2410.08810v1)
- [R] Wu, Lai, Jie, Gao, Hou, Zhang, Shen (arXiv sep. 2024): el efecto de los métodos de baja luz sobre clasificación y
  detección es «inconsistent and can sometimes be harmful»; hay «a disconnect between image enhancement for human visual
  perception and for machine analysis» — [arXiv 2409.14461](https://arxiv.org/abs/2409.14461)
- [R] YOLA (Hong, Cheng, Huang, Fan, Liu; arXiv oct. 2024, la página no nombra sede), mAP50 en ExDark con YOLOv3 / TOOD
  (Tabla 1): línea base 71,0 / 72,5; con mejora previa KinD 69,4 / 72,6, SMG 68,5 / 71,5, NeRCo 68,5 / 71,8; orientados a
  la tarea DENet 71,3 / 73,5, MAET 72,5 / 74,3, YOLA 72,7 / 75,2. En DARK FACE (Tabla 2): línea base 60,0 / 62,1; SMG 48,9 /
  55,8; NeRCo 49,1 / 55,6; YOLA 61,5 / 67,4. «The LLIE methods fail to achieve satisfactory performance due to inconsistency
  between human visual and machine perception» — [arXiv 2410.18398](https://arxiv.org/html/2410.18398)
- [A] FeatEnHancer (Hashmi, Kallempudi, Stricker, Afzal; ICCV 2023): módulo que mejora características jerárquicas optimizado
  con la pérdida de la tarea; +5,7 mAP en ExDark, +1,5 en DARK FACE, +5,1 mIoU en ACDC y +1,8 mAP en detección en vídeo
  (DarkVision), según el resumen — [arXiv 2308.03594](https://arxiv.org/abs/2308.03594)
- [A] IA-YOLO (Liu, Ren, Yu, Guo, Zhu, Zhang; AAAI 2022): módulo diferenciable de procesado de imagen cuyos parámetros
  predice una CNN pequeña entrenada de extremo a extremo con YOLOv3, para niebla y baja luz —
  [arXiv 2112.08088](https://arxiv.org/abs/2112.08088)
- [A] Sun, Ren, Wang, Cao (NeurIPS 2022): parten de que los métodos de restauración existentes «cannot improve the object
  detector performance» y a veces la reducen; proponen generar la pseudo-verdad de la restauración con un ataque adversario
  dirigido para que la imagen restaurada sirva al detector (deshaze y baja luz) —
  [NeurIPS 2022](https://neurips.cc/virtual/2022/poster/54700)

**¿Ayuda al reconocimiento de acciones?**
- [A] ARID (Xu, Yang, Cao, Mao, Yin, See; arXiv 2020), top-1 en acciones a oscuras, Tabla 3 (C3D / I3D-RGB / 3D-ResNet-101 /
  3D-ResNeXt-101): sin mejora 40,34 / 68,29 / 71,57 / 74,73; **corrección gamma 44,09 / 69,14 / 75,15 / 78,06** (única que
  sube en los cuatro); ecualización de histograma 39,49 / 63,67 / 65,49 / 75,82; LIME 39,61 / 73,02 / 75,45 / 77,40; BIMEF
  45,23 / 68,89 / 68,28 / 73,39; KinD 46,64 / 67,55 / 70,59 / 69,62. Concluyen que las mejoras «are rather limited» y que
  «Some frame enhancement methods even deteriorate classification accuracy» —
  [ar5iv 2006.03876](https://ar5iv.labs.arxiv.org/html/2006.03876)
- [A] Xu, Yang, Cao, Yin, Chen, Li, Li, Xu (IJCV, resumen del reto UG2 2021 Track 2): reconocimiento de acciones a oscuras
  con líneas base de modelos y métodos de mejora; la tarea sigue con «substantial room for improvements» —
  [arXiv 2202.09545](https://arxiv.org/abs/2202.09545)
- [A] Yi, Yang, Li, Tan, Kot (arXiv 2021): niebla y contraste son las corrupciones que más dañan a los modelos de vídeo en
  Mini Kinetics-C (S3D: 69,4 limpio → 47,6 con niebla y 46,4 con contraste; Tabla 1); en Mini SSV2-C la lluvia baja S3D de
  58,2 a 27,5 — [ar5iv 2110.06513](https://ar5iv.labs.arxiv.org/html/2110.06513)

**Lluvia y niebla**
- [A] Li, Araujo, Ren, Wang, Tokuda, Hirata Junior, Cesar-Junior, Zhang, Guo, Cao (arXiv 2019): detectores preentrenados en
  COCO sobre lluvia real, Tabla 6. Conducción (RID), sin quitar la lluvia → tras cada método: YOLO-V3 27,84 → 23,75-26,72;
  RetinaNet 23,92 → 19,28-21,71; SSD-512 17,71 → 16,69-17,06. Vigilancia (RIS): FRCNN 22,68 → 18,02-21,41; YOLO-V3 23,27 →
  18,71-21,80. Única excepción: FRCNN en RID (16,52 → hasta 23,42 con CGAN). «Almost all existing deraining algorithms will
  deteriorate the detection performance»; «No existing deraining method seems to directly help detection» —
  [ar5iv 1903.08558](https://ar5iv.labs.arxiv.org/html/1903.08558)
- [A] RESIDE (Li, Ren, Fu, Tao, Feng, Zeng, Wang; IEEE TIP 2019), niebla real RTTS, Tabla X: sin quitar la niebla → mejor
  método: FRCNN 37,58 → 41,56; YOLO-V2 40,37 → 40,76; SSD-300 50,26 → 51,82; SSD-512 55,55 → 57,17; pero varios métodos
  empeoran (GRM: 28,90 / 29,41 / 35,59 / 39,18). «The results are not perfectly consistent among four different detection
  models» — [ar5iv 1712.04143](https://ar5iv.labs.arxiv.org/html/1712.04143)
- [A] Pei, Huang, Zou, Lu, Wang (arXiv 2018): los métodos de eliminación de niebla «cannot improve much the
  image-classification performance» y a veces la reducen — [arXiv 1810.05716](https://arxiv.org/abs/1810.05716)

### Inferences
- La medida interna (gamma automática + CLAHE ayuda poco al modelo de eventos de noche) encaja con ARID: la gamma es la única
  corrección que sube de forma consistente (+0,9 a +3,8 puntos) y la ecualización de histograma tiende a restar. Habría que
  separar internamente el efecto de la gamma del de CLAHE (ARID mide ecualización global, no CLAHE: es una analogía).
- El patrón de LIME-Eval sugiere que, si se van a tener datos nocturnos para afinar, el mejorador sobra; solo aporta (y
  poco, ±1 punto) cuando el modelo va congelado, y depende del mejorador.
- Para medio campo al sol y medio a la sombra, y para focos y contraluz, no hay evidencia específica: lo más cercano son las
  corrupciones de contraste y brillo de los bancos de robustez, que apuntan a aumentos fotométricos y adaptación de
  estadísticas por vídeo (pregunta 4) antes que a una mejora previa.
- Si se quisiera un módulo de mejora, los que tienen respaldo medido son los entrenados con la pérdida de la tarea y de coste
  mínimo (del orden de los 79 k parámetros de Zero-DCE o un módulo diferenciable tipo IA-YOLO).

### Gaps
- No encontré ningún estudio de mejora de baja luz, focos o sombras medido sobre vídeo deportivo (detección, tracking o
  eventos). La búsqueda de partidos nocturnos no devolvió ningún conjunto de datos específico.
- Sin verificar: Pan et al., «Unsupervised Clustering in Football Analysis: A Color-Segmentation and Lighting Adaptation
  Approach» (IEEE Access, 2024), que según el fragmento de búsqueda corrige sombras e iluminación desigual para agrupar
  equipos y gana ≥0,2 de índice de Rand ajustado en iluminación difícil; la ficha devolvió 403.
- Retinexformer no informa de la detección sin mejora, así que su tabla ordena mejoradores pero no dice si mejorar compensa.
- La tabla de YOLA atribuye en el texto GDIP e IAT a una cita distinta de la de la tabla; revisar antes de citar esas filas.
- No encontré trabajos sobre lluvia o niebla en fútbol ni sobre vídeo (solo imagen fija).

## 4. Compresión y resolución baja (streams de YouTube): robustez de los modelos y si compensa restaurar frente a entrenar con degradaciones

### Takeaway
La compresión moderada daña poco a los modelos de apariencia (detección estable de CRF 22 a 37; Kinetics casi intacto) y
mucho a los que dependen del movimiento (SSv2: robustez relativa 0,41 con MPEG; flujo óptico hundido a partir de CRF 25).
Reentrenar o afinar con datos degradados recupera gran parte de la caída en imagen, y adaptar estadísticas en inferencia
recupera otra parte sin tocar los pesos; no encontré ninguna comparación directa «restaurar antes» frente a «entrenar con
degradaciones» en vídeo, y los aumentos de una sola corrupción no generalizan.

### Cited Findings

**Cuánto cae cada tipo de modelo**
- [A] Schiappa, Biyani, Kamtam, Vyas, Palangi, Vineet, Rawat (CVPR 2023): 90 perturbaciones sobre 6 modelos de acción. En
  Kinetics-400P la compresión es leve (R3D: JPEG 0,93 / 0,91 y MPEG2 0,89 / 0,85 de robustez absoluta / relativa; Tabla 4);
  en SSv2-P es dura para las CNN (I3D: JPEG 0,74 / 0,58 y MPEG2 0,63 / 0,41) y menos para Timesformer (0,85 / 0,74 y 0,82 /
  0,67). El ruido es la peor categoría para las CNN (R3D gaussiano 0,61 / 0,47; MViT 0,90 / 0,87). «Transformer based models
  are consistently more robust compared to CNN based models» — [arXiv 2207.01398](https://arxiv.org/abs/2207.01398) y
  [ar5iv](https://ar5iv.labs.arxiv.org/html/2207.01398)
- [A] Yi, Yang, Li, Tan, Kot (arXiv 2021): 12 corrupciones de vídeo, entre ellas H.265 ABR y CRF, error de bit, pérdida de
  paquetes y conversión de fps. TimeSformer es el más robusto (82,2 limpio y 71,4 de media corrupta en Mini Kinetics-C,
  frente a 70,5 y 57,7 de I3D; Tabla 1); la robustez crece con la capacidad, y los modelos eficientes (SlowFast, X3D-M) son
  peores en corrupciones temporales — [ar5iv 2110.06513](https://ar5iv.labs.arxiv.org/html/2110.06513)
- [A] Poyser, Atapour-Abarghouei, Breckon (ICPR 2020): cinco tareas bajo JPEG y H.264; la caída es significativa por debajo
  de calidad JPEG 15 y por encima de CRF 40. Acciones (dos ramas, UCF101, Tabla V): rama espacial 78,87 (CRF 23), 78,46
  (CRF 30), 74,57 (CRF 40), 44,20 (CRF 50); **rama de movimiento 70,12 (CRF 23) → 44,92 (CRF 25) → 37,36 (CRF 30)** —
  [arXiv 2007.14314](https://arxiv.org/abs/2007.14314) y [ar5iv](https://ar5iv.labs.arxiv.org/html/2007.14314)
- [A] O'Byrne, Vibhoothi, Sugrue, Kokaram (arXiv 2022): YOLOv5 sobre 50 vídeos de vigilancia en H.264 a CRF 22-47; de CRF 22
  a 37 la detección no cae de forma apreciable; a compresión mayor cae, sobre todo con poca luz y objetos rápidos; reentrenar
  con vídeo comprimido da hasta +1 % de F1 — [arXiv 2211.05805](https://arxiv.org/abs/2211.05805)
- [A] Tanaka, Harell, Bajić (ISCAS 2022): con vídeo sin comprimir como referencia, el parámetro de cuantización afecta a
  MOTA con un 95 % de confianza; el rango de búsqueda de movimiento no — [arXiv 2202.00892](https://arxiv.org/abs/2202.00892)
- [R] Tian, Lu, Yan, Zhai, Chen, Gao (TPAMI 2024): la compresión «deteriorates downstream video understanding tasks,
  especially at low-bitrate settings»; banco con tres tareas y ocho conjuntos —
  [arXiv 2202.02813](https://arxiv.org/abs/2202.02813)
- [R] VSE-MOT (Du, Xing, Li, Yu; Pattern Recognition, 2025): DanceTrack degradado con la tubería de Real-ESRGAN aplicada dos
  veces (desenfoque, reescalado 0,15-1,5, ruido, JPEG 20-40). Caída de HOTA limpio → degradado **con modelos ya entrenados
  en la mezcla limpio + degradado**: MOTRv2 58,4 → 51,4; MOTIP 62,1 → 52,2; MeMOTR 59,3 → 49,1; Hybrid-SORT 44,5 → 37,3; su
  método (guiado por un modelo visión-lenguaje) 64,1 → 59,9. No comparan con restaurar primero; dicen que «Image enhancement
  tracking often fall short when dealing with real-world complexity» sin cifras —
  [arXiv 2509.14060](https://arxiv.org/html/2509.14060v1)
- [A] Michaelis, Mitzkus, Geirhos, Rusak, Bringmann, Ecker, Bethge, Brendel (arXiv 2019): los detectores caen al 30-60 % de
  su rendimiento con corrupciones; estilizar las imágenes de entrenamiento mejora la robustez en todos los tipos y
  severidades — [arXiv 1907.07484](https://arxiv.org/abs/1907.07484)

**Reentrenar o aumentar con degradaciones**
- [A] Poyser et al., detección (Faster R-CNN, VOC 2007, Tabla III), mAP preentrenado → reentrenado con datos comprimidos:
  JPEG 15: 0,545 → 0,647; JPEG 10: 0,442 → 0,627; JPEG 5: 0,187 → 0,559 (0,703 a calidad 95). En cambio, en el modelo de
  acciones de dos ramas reentrenar **empeora** la rama de movimiento (CRF 40: 38,96 → 9,26) y solo compensa a CRF 50 —
  [ar5iv 2007.14314](https://ar5iv.labs.arxiv.org/html/2007.14314)
- [A] Yi et al.: entrenar con ruido gaussiano baja la media corrupta de 53,3 a 43,8 y «degrades the performance of models on
  10 out of 12 types of corruption» (Tabla 3); de 12 modelos entrenados con una sola corrupción, 9 quedan por debajo del
  modelo limpio en media corrupta (Tabla 4) — [ar5iv 2110.06513](https://ar5iv.labs.arxiv.org/html/2110.06513)
- [A] Schiappa et al. (apéndice B.1, Fig. 12): entrenar con perturbaciones ayuda a las CNN (ResNet50, X3D) y perjudica a
  MViT, que «performs much worse when trained on perturbations» — [ar5iv 2207.01398](https://ar5iv.labs.arxiv.org/html/2207.01398)
- [A] Martins y Brito (arXiv 2021): seguimiento de jugadores de fútbol con conjuntos de detección y tracking para tres
  calidades de vídeo; el resumen no da cifras — [arXiv 2105.10700](https://arxiv.org/abs/2105.10700)

**Adaptación en inferencia**
- [A] AdaBN (Li, Wang, Shi, Liu, Hou; arXiv 2016): recalcular las estadísticas de BatchNorm en el dominio destino, «does not
  require additional components, and is parameter-free» — [arXiv 1603.04779](https://arxiv.org/abs/1603.04779)
- [A] Schneider, Rusak, Eck, Bringmann, Brendel, Bethge (NeurIPS 2020): sustituir las estadísticas de BN por las de las
  imágenes corruptas baja el error medio de ResNet-50 en ImageNet-C de 76,7 % a 62,2 %, y el del mejor modelo robusto
  (DeepAugment + AugMix) de 53,6 % a 45,4 %; mejora 25 modelos — [arXiv 2006.16971](https://arxiv.org/abs/2006.16971)
- [A] ViTTA (Lin, Mirza, Kozinski, Possegger, Kuehne, Bischof; arXiv 2022, la página no nombra sede): 12 corrupciones de
  vídeo a severidad 5, incluidas JPEG y H.265 ABR. Media top-1 en UCF101 / SSv2 / K400 con TANet (Tabla 1): sin adaptar
  51,35 / 24,31 / 37,16; adaptar BN con lote de 8 vídeos 65,77 / 28,18 / 39,32, **con lote de 1 no gana o pierde** (51,59 /
  17,21 / 31,40); TENT lote 8: 72,92 / 31,57 / 41,13; ViTTA 78,33 / 38,07 / 48,94. Solo H.265 ABR en Kinetics-400 con
  TANet: 13,91 sin adaptar, 12,29 con BN, 14,24 con ViTTA (casi irrecuperable); JPEG en UCF101: 83,40 → 89,19 (BN) → 91,70
  (ViTTA) — [ar5iv 2211.15393](https://ar5iv.labs.arxiv.org/html/2211.15393)

**Restauración de artefactos y preprocesado para máquinas**
- [A] Galteri, Seidenari, Bertini, Del Bimbo (ICCV 2017): la eliminación de artefactos con GAN «can be used as a
  pre-processing step for object detection» cuando la compresión hace fallar al detector, y rinde mejor que redes entrenadas
  con MSE o SSIM (solo resumen, sin cifras) — [arXiv 1704.02518](https://arxiv.org/abs/1704.02518)
- [A] Reto NTIRE 2022 de SR y mejora de vídeo comprimido (Yang, Timofte et al.): tres pistas sobre vídeo HEVC (mejora, ×2 y
  ×4), conjunto LDV 2.0 de 335 vídeos; evaluación por calidad, no por tarea — [arXiv 2204.09314](https://arxiv.org/abs/2204.09314)
- [R] Zhao, Guo, Zhao, Li, Zhang, Xie (arXiv dic. 2025; comentario «DCC 2024»): preprocesador neuronal **antes** del códec
  estándar, entrenado con un códec virtual diferenciable y la tarea; ahorra más del 15 % de tasa a igual precisión. Requiere
  controlar la codificación — [arXiv 2512.15331](https://arxiv.org/abs/2512.15331)

**Dominio amateur**
- [R] Cabado, Cioppa, Giancola, Villa, Guijarro-Berdiñas, Padrón, Ghanem, Van Droogenbroeck (CVPR Workshops 2024, pp.
  3386-3398): «transferring across leagues is challenging, with the most impactful feature being broadcasting editing
  quality» — [ML Anthology](https://mlanthology.org/cvprw/2024/cabado2024cvprw-beyond)

### Inferences
- Un stream de YouTube a 720p-1080p está en la zona de compresión que los modelos de apariencia toleran; lo que más sufre es
  lo que depende del movimiento fino, que es el caso del action spotting de precisión y del balón. Eso sugiere medir
  internamente la caída por compresión sobre eventos de balón antes de decidir nada.
- El hallazgo interno de AdaBN coincide con Schneider et al. y con ViTTA: adaptar estadísticas recupera una parte grande de
  la caída, pero necesita estadísticas estables (un lote de 1 vídeo corto no basta). Un partido de 90 min da de sobra, así
  que la variante «por vídeo» es la favorable; métodos tipo ViTTA (alinear estadísticas de características y coherencia
  temporal) son el siguiente escalón si AdaBN se queda corto.
- La evidencia favorece entrenar o afinar con la degradación real del dominio (Poyser en detección, Otsubo con cámara a
  pulso) frente a aumentos sintéticos de una sola corrupción (Yi et al.) y avisa de que los transformers de vídeo pueden
  empeorar con aumentos de perturbación (Schiappa et al.).
- Según Cabado et al., lo que más pesa al pasar de retransmisión a amateur es la realización, no la calidad de píxel: otra
  razón para no esperar mucho de restaurar la imagen.
- Quitar artefactos de compresión antes del modelo solo tiene respaldo (antiguo y de resumen) cuando la compresión es tan
  fuerte que el detector falla y no se puede reentrenar.

### Gaps
- No encontré ningún estudio de robustez a la compresión de modelos de action spotting ni de detección/tracking en fútbol
  con cifras; el de Martins y Brito no da números en el resumen.
- No encontré ninguna comparación directa, sobre el mismo banco de vídeo, entre restaurar antes, entrenar con degradaciones
  y adaptar en inferencia.
- Las cifras de Cabado et al. no se pudieron leer (el PDF de CVF devolvió 403); solo se abrió la ficha con el resumen, que
  describe la transferencia entre ligas y entre cámara automática y realización profesional.
- En VSE-MOT no hay fila de «tracker entrenado solo en limpio», así que la caída real sin entrenamiento mixto es
  desconocida; además CO-MOT sube en MOTA con datos degradados (44,9 → 53,3), lo que pide cautela con esa tabla.
- Los valores numéricos de la Fig. 12 de Schiappa et al. (robustez relativa >1) no se interpretan bien fuera del texto; se
  cita solo la conclusión cualitativa.

## 5. Estabilización de vídeo: métodos clásicos y aprendidos 2022-2026 y efecto medido sobre tracking, calibración o acciones

### Takeaway
No encontré evidencia de que estabilizar los píxeles antes del modelo mejore tracking o reconocimiento de acciones: la única
medida controlada dice que «stabilization does not seem to help activity recognition much» (+1,0 mAP con 16 fotogramas, −0,8
con 64). Lo que sí está medido es compensar el movimiento de cámara **dentro del tracker** (BoT-SORT: +1,6 IDF1 y +0,9 HOTA
en MOT17) con coste despreciable frente al detector.

### Cited Findings

**Métodos**
- [R] Ali, Im, Kim, Kim, Gupta, Luo, Li (arXiv ago. 2025): estabilización a fotograma completo con meta-aprendizaje y
  adaptación por vídeo; compara con L1, Bundled, Adobe Premiere Pro 2020, StabNet, Yu y Ramamoorthi, FuSta, Zhang et al.,
  DMBVS y DIFRINT (Tabla VII); no da tiempos por fotograma. Propone métricas a nivel de objeto en NUS (persistencia media e
  IoU temporal; Tabla X): DMBVS 33,909 y 0,7771; Bundled 33,434 y 0,7969; su mejor variante 35,020 y 0,8143 —
  [arXiv 2508.18859](https://arxiv.org/html/2508.18859)
- [R] Liu, Wan, Ren, Wen (CVPR 2026, según los comentarios de arXiv): estabilización en línea, no supervisada y sin
  fotogramas futuros. Tabla 7, tiempo por fotograma en una Jetson AGX Orin (sin resolución declarada): MeshFlow 168,95 ms;
  StabNet 179,88 ms; NNDVS 340,02 ms; Liu et al. 36,41 ms; el suyo 78,94 ms (~13 fps). Recorte / distorsión / estabilidad en
  NUS (Tabla 2): MeshFlow 0,78 / 0,83 / 0,85; StabNet 0,66 / 0,88 / 0,82; NNDVS 0,92 / 0,98 / 0,87; el suyo 0,95 / 0,98 /
  0,90. No habla de tareas posteriores — [arXiv 2602.23141](https://arxiv.org/html/2602.23141)
- [A] Zhang, Liu, Tan, Zeng, Liu (ICCV 2023): estabilización profunda en línea de latencia mínima; optimiza la trayectoria de
  cámara con una ventana deslizante sin fotogramas futuros — [arXiv 2212.02073](https://arxiv.org/abs/2212.02073)

**Efecto sobre tareas posteriores**
- [A] Sigurdsson, Gupta, Schmid, Alahari (arXiv 2020, la página no nombra sede): estabilizar el vídeo antes de un 3D-ResNet50.
  Charades-Ego: con 64 fotogramas 23,6 → 22,8 mAP (**empeora**); con 16 fotogramas 17,5 → 18,5. Charades con movimiento de
  cámara sintético 13,1 → 13,5; Olympic-Sports 96,4 → 97,2 (Tabla 2). «Stabilization does not seem to help activity
  recognition much»; modelar características y transformaciones a la vez evita «compounding errors do to stabilization over
  many frames, and loss of resolution» — [ar5iv 2003.05614](https://ar5iv.labs.arxiv.org/html/2003.05614)
- [A] BoT-SORT (Aharon, Orfaig, Bobrovsky; arXiv 2022), ablación en MOT17 validación (Tabla 1): sin compensación de
  movimiento de cámara MOTA 77,67 / IDF1 79,89 / HOTA 68,12; con ella 78,31 / 81,51 / 69,06. Se estima con puntos clave,
  flujo óptico disperso y RANSAC (módulo de estabilización de OpenCV, transformación afín) y su «run time is negligible
  compared to the detector inference time» — [ar5iv 2206.14651](https://ar5iv.labs.arxiv.org/html/2206.14651)
- [R] McByte++ (Stanczyk, Yoon, Bremond; arXiv ago. 2026), tracking deportivo sin entrenamiento, compensación afín con ORB.
  Frente al McByte original, estimarla sobre la imagen reducida ×4 cuesta ~1 punto y el tracker va mucho más rápido (las
  tablas no separan otras optimizaciones): SoccerNet-tracking 2022 con detecciones oráculo, HOTA 85,0 / IDF1 79,9 a 1,04 fps
  → 84,1 / 78,9 a 10,51 fps (Tabla 1); SportsMOT 76,9 / 77,5 a 3,60 fps → 75,9 / 76,3 a 18,65 fps (Tabla 2). Reducir ×6 ya
  degrada (SportsMOT HOTA 72,2). **No hay fila sin compensación** —
  [arXiv 2608.15688](https://arxiv.org/html/2608.15688)
- [A] Schiappa et al. (CVPR 2023), perturbaciones de cámara sobre modelos de acción en Kinetics-400P (Tabla 4): la rotación
  estática es la peor (R3D 0,76 / 0,67 de robustez absoluta / relativa) y la rotación variable entre fotogramas se tolera
  mucho mejor (0,94 / 0,92) — [ar5iv 2207.01398](https://ar5iv.labs.arxiv.org/html/2207.01398)
- [R] Otsubo, Sawafuji, Saito (MMSports 2025): con cámara a pulso, «rapid camera transition make it difficult to consistently
  track» en pases y despejes; no hacen ablación de estabilización ni de compensación —
  [arXiv 2511.09455](https://arxiv.org/html/2511.09455v1)
- [R] Yang, Shim, Ko, Kim (arXiv sep. 2026): revisión con experimentos del tracking por detección que evalúa por separado
  medidas de similitud, asociación, compensación de movimiento de cámara e interpolación desde un tracker mínimo; el resumen
  no da cifras — [arXiv 2609.08265](https://arxiv.org/abs/2609.08265)

### Inferences
- Para tracking, la respuesta estándar ya está en el proyecto (GMC de BoT-SORT): compensa el movimiento global sin remuestrear
  la imagen, sin recorte y sin acumular error, y se puede abaratar estimándolo a menor resolución (hasta ×4 sin pérdida
  apreciable, según McByte++).
- La estabilización de píxeles cuesta campo de visión (recorte de 0,66 a 0,95 según el método) y remuestrea la imagen, lo
  que perjudica justo a los objetos pequeños; con ventanas temporales largas puede incluso restar (Sigurdsson et al.).
- La vibración de alta frecuencia de un mástil se parece más a la «rotación variable» de Schiappa et al. (bien tolerada por
  modelos de clasificación) que a una inclinación fija; pero eso es clasificación de acciones, no localización temporal
  precisa ni balón, así que no pasa de indicio.
- Para paneos bruscos, la evidencia apunta a tratarlos en el tracker y en los datos de entrenamiento (afinar con vídeo a
  pulso) antes que a estabilizar.

### Gaps
- No encontré ninguna medida del efecto de estabilizar sobre action spotting, calibración del campo ni detección de balón.
- No encontré ninguna ablación «con y sin compensación de cámara» en bancos deportivos: McByte++ solo compara variantes de
  compensación; el fragmento de búsqueda que atribuía +4,1 HOTA en SportsMOT a la compensación no se confirma en las tablas
  abiertas.
- Sin verificar (403 o solo fragmento): Xu et al., «Video Stabilization: A Comprehensive Survey from Classical Mechanics to
  Deep Learning Paradigms» (Modelling 6(2):49, 2025); Wang et al., «Real-Time Tracking With Stabilized Frame» (CVPR
  Workshops 2020); la afirmación de la revisión de Yang et al. de que la compensación mejora en todos los conjuntos (solo se
  abrió el resumen); MoveSORT (arXiv 2407.00738).
- Los tiempos de los métodos aprendidos fuera de línea (FuSta, DIFRINT, variantes de meta-aprendizaje) no aparecen en las
  páginas abiertas; los de LightStab son en una Jetson y sin resolución declarada.
- Los métodos clásicos (L1, Bundled, MeshFlow) solo se citan a través de las tablas comparativas de los trabajos recientes;
  no abrí sus artículos originales.

## 6. Principio general: «mejorar para el ojo» frente a «mejorar para la máquina»; cuándo empeora y qué alternativas hay

### Takeaway
La evidencia es consistente desde 2018 hasta 2025: la calidad perceptual y el rendimiento de la máquina son objetivos
distintos, y una restauración genérica delante de un modelo congelado da ganancias pequeñas, inconsistentes o negativas. Las
tres alternativas con respaldo medido son restaurar con la pérdida de la tarea, entrenar o afinar con la degradación real, y
adaptar el modelo en inferencia; las dos últimas son además las baratas.

### Cited Findings

**Fundamento**
- [A] Blau y Michaeli (CVPR 2018): «distortion and perceptual quality are at odds with each other»; el resultado vale para
  cualquier medida de distorsión — [arXiv 1711.06077](https://arxiv.org/abs/1711.06077)
- [A] VidalMata et al. (24 autores; arXiv 2019, DOI de TPAMI 2020), reto UG2: las capacidades de la fotografía computacional
  «have not always translated in a useful way to visual recognition tasks»; «we are in the early days of building a bridge»
  — [arXiv 1901.09482](https://arxiv.org/abs/1901.09482)
- [A] Vidal, Banerjee, Grm, Struc, Scheirer (WACV 2018 / reto CVPR 2018), banco UG2 para medir el impacto de restauración y
  mejora en el reconocimiento automático: «there is plenty of room for algorithmic innovation» —
  [arXiv 1710.02909](https://arxiv.org/abs/1710.02909)
- [A] Diamond, Sitzmann, Julca-Aguilar, Boyd, Wetzstein, Heide (arXiv 2017, rev. 2021): entrenar de extremo a extremo el
  procesado y la clasificación produce imágenes que conservan detalle «at the cost of increased noise and artifacts», es
  decir, peores para el ojo, y «substantially improves perception in low light» —
  [arXiv 1701.06487](https://arxiv.org/abs/1701.06487)
- [A] Liu, Wen, Liu, Wang, Huang (IJCAI 2018): eliminación de ruido en cascada con la red de la tarea y pérdida conjunta que
  actualiza solo el eliminador de ruido; compensa la caída de las tareas de alto nivel —
  [arXiv 1706.04284](https://arxiv.org/abs/1706.04284)

**Cuándo empeora la restauración (casos medidos)**
- Deblurring: baja el detector en imagen nítida (58,50 → 55,50) y lo hunde en GOPRO (19,64 → 2,24) —
  [ar5iv 2011.14448](https://ar5iv.labs.arxiv.org/html/2011.14448)
- Lluvia: casi todos los métodos bajan la detección (YOLO-V3 27,84 → 23,75-26,72) —
  [ar5iv 1903.08558](https://ar5iv.labs.arxiv.org/html/1903.08558)
- Niebla: algunos métodos bajan 9-16 puntos (GRM) y la mejora del mejor es de 0,4-4 puntos —
  [ar5iv 1712.04143](https://ar5iv.labs.arxiv.org/html/1712.04143); en clasificación a veces reduce —
  [arXiv 1810.05716](https://arxiv.org/abs/1810.05716)
- Baja luz: SMG y NeRCo bajan DARK FACE de 60,0 a ~49 con YOLOv3 — [arXiv 2410.18398](https://arxiv.org/html/2410.18398);
  ecualización de histograma y KinD bajan las acciones a oscuras hasta 6 y 5 puntos —
  [ar5iv 2006.03876](https://ar5iv.labs.arxiv.org/html/2006.03876)
- SR perceptual: ESRGAN baja la rama temporal 5,8 puntos y VDSR queda por debajo del bicúbico —
  [ar5iv 1903.05577](https://ar5iv.labs.arxiv.org/html/1903.05577); en fútbol baja el mAP@0,5 —
  [arXiv 2402.00163](https://arxiv.org/html/2402.00163v1)
- SR entrenada contra una red de tarea fija (T→S) queda por debajo del bicúbico (27,2 frente a 27,5 mAP) —
  [arXiv 2404.01692](https://arxiv.org/html/2404.01692v2)
- Generativos: propagación temporal larga amplifica artefactos — [ar5iv 2111.12704](https://ar5iv.labs.arxiv.org/html/2111.12704);
  la difusión produce «non-existent content» y más pasos bajan la precisión —
  [arXiv 2507.22459](https://arxiv.org/html/2507.22459v2); las alucinaciones no las captan las métricas habituales —
  [arXiv 2507.14367](https://arxiv.org/abs/2507.14367)
- Estabilización: resta 0,8 mAP con ventanas de 64 fotogramas — [ar5iv 2003.05614](https://ar5iv.labs.arxiv.org/html/2003.05614)

**Alternativas propuestas**
- Restauración guiada por la tarea: TDSR (2018), SR4IR (2024: 32,4-33,9 frente a 27,5 del bicúbico), EDTR (2025: 31,2-33,4
  frente a 21,0 de SwinIR), IA-YOLO (2022), FeatEnHancer (2023: +5,7 mAP en ExDark), YOLA (2024), Sun et al. (2022) —
  enlaces en las preguntas 1 y 3; p. ej. [arXiv 2404.01692](https://arxiv.org/html/2404.01692v2)
- Aumentos o afinado con la degradación: desenfoque (31,37 frente a 6,26 del deblurring en E5) —
  [ar5iv 2011.14448](https://ar5iv.labs.arxiv.org/html/2011.14448); compresión (JPEG 10: 0,442 → 0,627) —
  [ar5iv 2007.14314](https://ar5iv.labs.arxiv.org/html/2007.14314); estilización —
  [arXiv 1907.07484](https://arxiv.org/abs/1907.07484); cámara a pulso (+15 HOTA afinando el detector) —
  [arXiv 2511.09455](https://arxiv.org/html/2511.09455v1). Límite: una sola corrupción no generaliza —
  [ar5iv 2110.06513](https://ar5iv.labs.arxiv.org/html/2110.06513)
- Adaptación en inferencia: estadísticas de BN (76,7 → 62,2 % de error en ImageNet-C) —
  [arXiv 2006.16971](https://arxiv.org/abs/2006.16971); AdaBN — [arXiv 1603.04779](https://arxiv.org/abs/1603.04779); ViTTA
  en vídeo (51,35 → 78,33 en UCF101 corrupto) — [ar5iv 2211.15393](https://ar5iv.labs.arxiv.org/html/2211.15393)
- Seleccionar en vez de restaurar: clasificador de legibilidad y agregación por tracklet en dorsales —
  [arXiv 2405.13896](https://arxiv.org/html/2405.13896)
- Usar la degradación como señal: atributos del desenfoque del balón — [arXiv 2509.18387](https://arxiv.org/html/2509.18387v2)
- Evaluar los mejoradores con un detector preentrenado y sin afinar, porque afinar oculta las diferencias —
  [arXiv 2410.08810](https://arxiv.org/html/2410.08810v1)

### Inferences
- Regla práctica que se desprende: restaurar solo compensa si (a) el modelo posterior no se puede tocar, (b) la degradación
  es severa y de un tipo concreto, y (c) el restaurador se valida con la métrica de la tarea, no con PSNR ni a ojo. En cuanto
  el modelo se puede afinar o adaptar, la restauración genérica deja de aportar.
- Para este producto, donde los modelos son propios y se pueden afinar, el orden que sugiere la evidencia es: adaptación por
  vídeo (ya validada internamente con AdaBN) → afinado y aumentos con la degradación real del dominio → correcciones
  globales baratas e inocuas (gamma) → y solo al final, restauración entrenada con la pérdida de la tarea.
- Las correcciones globales y monótonas (gamma, balance) son las únicas «para el ojo» que no han mostrado daño; cualquier
  operador local o generativo cambia la estadística de textura y movimiento de la que dependen los modelos.
- Los métodos generativos añaden un riesgo específico para un producto de estadísticas: pueden inventar un balón, un dorsal
  o un contacto verosímiles, y las métricas habituales no lo detectan.
- La adaptación de dominio entre retransmisión y amateur es la otra alternativa que la literatura menciona (Sun et al. la
  usan como línea base); queda fuera de estas notas.

### Gaps
- No encontré ninguna revisión (survey) dedicada a «restauración para visión por máquina»; las revisiones recientes
  localizadas (restauración todo en uno, arXiv 2410.15067; ultra alta definición, arXiv 2505.16161) son de bajo nivel y no
  se abrieron.
- No hay ningún banco deportivo que compare restauración, aumentos y adaptación con el mismo modelo; el principio general se
  apoya en imagen fija, vigilancia, conducción y acciones genéricas.
- De VidalMata et al., Sun et al., Liu et al. y Diamond et al. solo se abrió el resumen: las frases son suyas, las cifras de
  sus tablas no se han verificado.

## 7. Coste: tiempo por fotograma y viabilidad para partidos de 90 min a 25 fps en una GPU de 24 GB

### Takeaway
Un partido son 135 000 fotogramas. Las correcciones ligeras (Zero-DCE, compensación de cámara en el tracker) cuestan minutos
o nada; la interpolación y la SR recurrente cuestan del orden de la duración del partido o más; la SR por difusión va de
~2 h (FlashVSR reducido, a 768×1408) a cientos de horas, y varios métodos no caben en 24 GB. Las alternativas robustas
(adaptar BN, aumentos) no añaden coste por fotograma en inferencia.

### Cited Findings
- Zero-DCE: 0,0025 s por imagen de 1200×900 en una 2080 Ti; 79 416 parámetros (Tabla 3) —
  [ar5iv 2001.06826](https://ar5iv.labs.arxiv.org/html/2001.06826)
- Retinexformer: 1,61 M de parámetros, 15,57 GFLOPs (Tabla 1) — [ar5iv 2303.06705](https://ar5iv.labs.arxiv.org/html/2303.06705)
- RIFE: 31 ms por fotograma interpolado a 720p y 68 ms a 1080p en una TITAN X (Pascal) (§6.4) —
  [ar5iv 2011.06294](https://ar5iv.labs.arxiv.org/html/2011.06294)
- RealBasicVSR 63 ms y Real-ESRGAN 149 ms, ambos para salida de 720×1280 (×4) en una V100 (Tabla 2) —
  [ar5iv 2111.12704](https://ar5iv.labs.arxiv.org/html/2111.12704)
- BasicVSR++: 77 ms con entrada LR de 180×320 (Tabla 1; GPU no indicada) —
  [ar5iv 2104.13371](https://ar5iv.labs.arxiv.org/html/2104.13371)
- SR de vídeo por difusión, 101 fotogramas a 768×1408 en una A100 (Tabla 2 de FlashVSR), fps y memoria pico: Upscale-A-Video
  0,12 fps y 18,39 GB; STAR 0,15 fps y 24,86 GB; DOVE 1,39 fps y 25,44 GB; SeedVR2-3B 1,43 fps y 52,88 GB; FlashVSR completo
  6,52 fps y 18,33 GB; FlashVSR reducido 16,92 fps y 11,13 GB — [arXiv 2510.12747](https://arxiv.org/html/2510.12747v1)
- EDTR: 1 paso de difusión basta y 4 pasos es el máximo útil (Tabla 3) — [arXiv 2507.22459](https://arxiv.org/html/2507.22459v2)
- Estabilización en línea en una Jetson AGX Orin: Liu et al. 36,41 ms; LightStab 78,94 ms; MeshFlow 168,95 ms; StabNet
  179,88 ms; NNDVS 340,02 ms por fotograma (Tabla 7) — [arXiv 2602.23141](https://arxiv.org/html/2602.23141)
- Compensación de movimiento de cámara en el tracker: «negligible compared to the detector inference time» —
  [ar5iv 2206.14651](https://ar5iv.labs.arxiv.org/html/2206.14651); en un tracker deportivo, con la estimación a 1/4 de
  resolución el tracker completo pasa de 1,04 fps (McByte original) a 10,51 fps en SoccerNet-tracking; la tabla no separa
  otras optimizaciones — [arXiv 2608.15688](https://arxiv.org/html/2608.15688)
- AdaBN no añade componentes ni parámetros — [arXiv 1603.04779](https://arxiv.org/abs/1603.04779); la rama auxiliar de SR de
  SuperYOLO se descarta en inferencia — [arXiv 2209.13351](https://arxiv.org/abs/2209.13351)
- Referencia de un detector de balón y jugadores: FootAndBall, 37 fps a 1920×1080 en una GTX 1060 —
  [ar5iv 1912.05445](https://ar5iv.labs.arxiv.org/html/1912.05445)

### Inferences
Proyección propia a 90 min × 25 fps = 135 000 fotogramas, multiplicando los tiempos citados (GPU distintas de la nuestra:
orden de magnitud, no medida):

| Técnica | Base citada | Tiempo por partido | ¿Cabe en 24 GB? |
|---|---|---|---|
| Zero-DCE | 2,5 ms a 1200×900 | ~6 min | sí |
| Compensación de cámara en el tracker | despreciable | ~0 | sí |
| AdaBN por vídeo | una pasada de estadísticas | fracción de una inferencia | sí |
| Estabilización en línea (Liu et al. / LightStab) | 36 / 79 ms en Jetson | ~1,4 h / ~3,0 h en Jetson | sí |
| RIFE 25 → 50 fps | 31 ms (720p) / 68 ms (1080p) | ~1,2 h / ~2,6 h | sí |
| RealBasicVSR ×4, salida 720p | 63 ms | ~2,4 h | sí |
| BasicVSR++ ×4, entrada 180×320 | 77 ms | ~2,9 h | sí |
| Real-ESRGAN ×4, salida 720p | 149 ms | ~5,6 h | sí |
| FlashVSR reducido, salida 768×1408 | 16,92 fps | ~2,2 h | sí (11 GB) |
| FlashVSR completo | 6,52 fps | ~5,8 h | sí (18 GB) |
| DOVE | 1,39 fps | ~27 h | no (25,4 GB) |
| SeedVR2-3B | 1,43 fps | ~26 h | no (52,9 GB en esa configuración) |
| STAR | 0,15 fps | ~250 h | no (24,9 GB) |
| Upscale-A-Video | 0,12 fps | ~310 h | sí (18,4 GB) |

- Las cifras de SR son para salidas de ~720p (entrada de 180×320 a 192×352). Aplicar ×4 a nuestra entrada de 720p multiplica
  los píxeles por 16: si el coste escala linealmente, RealBasicVSR pasaría a ~38 h y Real-ESRGAN a ~89 h por partido, y la
  salida (2880×5120) habría que volver a reducirla para el modelo. Es decir, la SR solo es viable sobre recortes (jugador,
  dorsal, zona del balón), no sobre el fotograma completo.
- La SR por difusión a fotograma completo queda fuera de alcance salvo FlashVSR, y aun así cuesta más que todo el resto del
  pipeline sin evidencia de beneficio para la tarea.
- Lo único con coste compatible con procesar cada partido de forma rutinaria es: corrección global (gamma, Zero-DCE),
  compensación de cámara en el tracker y adaptación de estadísticas; coincide con lo que la evidencia de las preguntas 3-6
  señala como útil o, al menos, inocuo.

### Gaps
- Ningún tiempo citado está medido en una GPU de 24 GB de la generación actual; las proyecciones son órdenes de magnitud.
- No encontré tiempos por fotograma de NAFNet ni de deblurring de vídeo en las páginas abiertas (solo coste relativo), ni de
  los estabilizadores fuera de línea, ni el coste añadido de ViTTA.
- No encontré la memoria pico de RealBasicVSR/BasicVSR++ sobre secuencias largas (son recurrentes y habría que trocear el
  partido).
- La Tabla 2 de FlashVSR es la única fuente abierta para los tiempos de Upscale-A-Video, STAR, DOVE y SeedVR2; no se
  contrastaron con los artículos originales.
