# Percepción que decide «dónde mirar» en vídeo deportivo gran angular o de alta resolución: balón pequeño, señales alternativas de RoI y consumo de alta resolución (2018-2026)

Notas de investigación web, cerradas el 17-11-2026. Convenciones:

- **[R año]** = trabajo reciente (2024-2026). **[A año]** = antecedente (2015-2023).
- Solo se cita lo que se abrió en esta sesión (arXiv/ar5iv, actas, repositorios). Lo que no se pudo abrir va marcado como **«sin verificar»** y recogido en *Gaps*.
- Las cifras se extrajeron de las páginas con una herramienta que resume el HTML; van con su tabla o sección de origen para poder cotejarlas con el PDF antes de publicarlas. Donde la propia fuente se contradice, se indica.
- «Contexto del proyecto» = lo que decía el encargo (panorámica 4096×1080, balón ~8 px, YOLO11 por ventanas, Viterbi, 7/28 tiros sin detección, RoI real en AdaSpot sin mejora). No es una fuente; solo se usa en *Inferences*.

---

## 1. Detección y seguimiento de balón pequeño en cámara fija gran angular: métodos, cifras y código

### Takeaway

Las redes de mapa de calor con varios fotogramas de entrada (TrackNetV2 → WASB → TOTNet / TrackNetV5) son el estado del arte; en el único banco de fútbol con cámara fija gran angular comparado en igualdad de condiciones (ISSIA-CNR reanotado por WASB) logran F1 85-88 frente a 36-45 de los detectores sin esa receta. Ninguna novedad de 2024-2026 (TrackNetV4/V5, TOTNet, BlurBall) se ha evaluado en fútbol y no existe verdad-terreno pública de balón por fotograma en panorámica 4K: SoccerTrack v2 no la trae.

### Cited Findings

**Fútbol con cámara fija gran angular (cifras)**

- [A 2023] WASB (Tarashima, Haq, Wang, Tagawa; NTT Communications / Tokyo Metropolitan Univ.; arXiv 2311.05237). Su conjunto «Soccer» procede de D'Orazio et al. 2009 (seis vídeos sincronizados de cámara fija), 1920×1080 a 25 fps, 11 994 fotogramas de entrenamiento y 5 999 de test, con el balón **reanotado a mano** porque las anotaciones originales estaban «colapsadas»; desplazamiento medio del balón entre fotogramas 10,4±10,0 px (train) y 15,7±13,0 px (test) (§4.1, Tabla 1) — [WASB, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237). La sede BMVC 2023 aparece en la página de actas listada por el buscador, no en la página abierta — [BMVC, sin abrir](https://bmvc2022.mpi-inf.mpg.de/BMVC2023/310).
- [A 2023] Resultados en ese conjunto de fútbol con umbral de distancia τ = 4 px, todos los métodos reentrenados por los autores (Tabla 2; F1 / Acc. / AP / FPS): DeepBall 44,5 / 92,7 / 26,3 / 44,6; DeepBall-Large 44,9 / 89,5 / 34,0 / 42,0; BallSeg 36,1 / 92,6 / 20,0 / 64,8; TrackNetV2 86,6 / 97,7 / 77,2 / 66,0; ResTrackNetV2 84,6 / 97,4 / 75,5 / 56,2; MonoTrack 85,2 / 97,4 / 78,6 / 58,0; **WASB (paso 3) 88,3 / 97,9 / 83,6 / 55,7; WASB (paso 1) 88,2 / 97,9 / 86,2 / 23,6** — [WASB, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237).
- [A 2023] Ablación de WASB en fútbol: quitar las reducciones de paso (*stride*) del tronco HRNet para conservar resolución sube F1 de 81,7 a 88,3 y AP de 71,7 a 83,6, a costa de bajar de 85,7 a 55,7 FPS (Tabla 3); el entrenamiento consciente de la posición (HLSM) aporta +0,5 F1 / +1,0 AP; el centro del mapa de calor en inferencia (CoH) +0,5 F1 / +2,5 AP; **el seguimiento en línea aporta 0 en fútbol** («online tracking does not contribute on the Soccer and Badminton datasets»); el sobremuestreo con paso 1 da +2,6 AP (Tabla 4) — [WASB, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237).
- [A 2023] WASB usa 3 fotogramas consecutivos concatenados por canales, redimensionados a **288×512**; 1,5 M de parámetros; medido en un servidor con 4 GPU V100 (§5.2) — [WASB, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237).
- [A 2020] FootAndBall (Komorowski, Kurzejamski, Sarwas; DOI 10.5220/0008916000470056, SciTePress): detector totalmente convolucional con FPN que saca mapa de confianza de balón a 1/4 y de jugadores a 1/16. En ISSIA-CNR (seis cámaras Full-HD fijas, 25 fps, 20 000 fotogramas anotados) el balón «puede medir tan poco como 8 píxeles» en el lado lejano y unos 20 en el cercano. Tabla 2: AP de balón 0,909 y de jugador 0,921 con 199 k parámetros a 37 FPS en 1080p (GTX 1060); DeepBall 0,877 de AP de balón; Faster R-CNN 0,866 a 8 FPS — [FootAndBall, ar5iv](https://ar5iv.labs.arxiv.org/html/1912.05445); código y modelo preentrenado en `github.com/jac99/FootAndBall`, enlazado desde [la página de arXiv](https://arxiv.org/abs/1912.05445). La fuente se contradice en dos puntos (120 k frente a 199 k parámetros; AP de jugador de Faster R-CNN 0,774 en el texto y 0,874 en la tabla) — [FootAndBall, ar5iv](https://ar5iv.labs.arxiv.org/html/1912.05445).
- [A 2019] DeepBall (Komorowski, Kurzejamski, Sarwas; arXiv 1902.07304): red de 48 658 parámetros con hipercolumnas y mapa de confianza a 1/4; en ISSIA-CNR (7 000 fotogramas con balón, 13 000 sin balón u ocluido; **radio** del balón 8-16 px según este artículo) AP 0,877 a 190 FPS (Titan X), 0,792 sin aumento de datos y 0,833 sin hipercolumnas (Tabla 2) — [DeepBall, ar5iv](https://ar5iv.labs.arxiv.org/html/1902.07304).
- [R 2025] SoccerNet-v3D e ISSIA-3D (Gutiérrez-Pérez y Agudo, IRI CSIC-UPC; arXiv 2504.10106): con YOLOv11-l, AP@0,5 de balón en retransmisión (SoccerNet-v3D) 0,65 con las cajas originales y 0,81 con cajas optimizadas; **en ISSIA-3D (cámara fija) el mismo detector da 0,04 y 0,39, y solo llega a 0,65 tras entrenarlo en ISSIA** (Tabla 2); código en `github.com/mguti97/SoccerNet-v3D` según el propio artículo — [SoccerNet-v3D, arXiv](https://arxiv.org/html/2504.10106v1).
- [R 2025-2026] SoccerTrack v2 (Scott, Uchida, Kuroda, Kim, Fujii; arXiv 2508.01802, v1 de agosto de 2025 y v2 de octubre de 2026 con 39 páginas y líneas base): diez partidos universitarios, 932 min, panorámica fija cosida de unos 4K de ancho (3840 o 4096 px) y 1080-1906 px de alto, 25 fps — [SoccerTrack v2, arXiv abs](https://arxiv.org/abs/2508.01802), [v2 HTML](https://arxiv.org/html/2508.01802v2).
- [R 2026] SoccerTrack v2 **no mide el balón por fotograma**: la pista de balón publicada es «una reconstrucción» con tramos rectos a velocidad constante entre anclas (casi todas, eventos anotados), no sirve como verdad-terreno de detector, y los autores describen la detección de balón en panorámica como difícil (§3, §4.9) — [SoccerTrack v2, v2 HTML](https://arxiv.org/html/2508.01802v2).
- [R 2026] SoccerTrack Challenge 2025 (panorámica fija de ojo de pez, 4096×1080, seis vídeos de ~6 000 fotogramas): el reto y la solución ganadora GTATrack tratan **solo jugadores**; el balón no se menciona (Tabla 2, §5.1) — [GTATrack, arXiv](https://arxiv.org/html/2602.00484).
- [A 2023] En SoccerNet-tracking 2023, el tercer clasificado (MOT4MOT) trató el balón como detección de objeto único con YOLOv8l afinado más un «filtrado propietario» sin detallar; HOTA global 66,27, sin métrica propia del balón — [MOT4MOT, arXiv](https://arxiv.org/abs/2308.16651).

**Familia TrackNet y derivados (ninguno evaluado en fútbol)**

- [A 2019] TrackNet (Huang, Liao, Chen, İk, Peng; arXiv 1907.03698): mapa de calor sobre uno o varios fotogramas consecutivos de tenis a 640×360; precisión 99,7 %, exhaustividad 97,3 %, F1 98,5 % en un vídeo, y 95,3 / 75,7 / 84,3 % en validación cruzada de 10 pliegues con más vídeos — [TrackNet, arXiv](https://arxiv.org/abs/1907.03698).
- [A 2023] TrackNetV3 (DOI ACM 10.1145/3595916.3626370): añade una **imagen de fondo estimada como entrada auxiliar**, aumento *mixup* y un módulo de rectificación de trayectoria (InpaintNet). En volante de bádminton: TrackNetV2 Acc 94,98 / P 99,64 / R 94,56 / F1 97,03 a 27,70 FPS; TrackNetV3 97,51 / 97,79 / 99,33 / 98,56 a 25,11 FPS; YOLOv7 57,82 / 78,53 / 59,96 / 68,00 — [TrackNetV3, repositorio](https://github.com/qaz812345/TrackNetV3).
- [R 2024] TrackNetV4 (Raj, Wang, Gedeon; arXiv 2409.14543): mapas de atención de movimiento a partir de diferencias de fotogramas en valor absoluto con una capa de solo dos parámetros aprendibles, fusionados con las características visuales. Tabla I: tenis F1 97,1 → 97,5 y 96,4 → 97,0 (dos protocolos); volante 90,6 → 91,4 sobre TrackNetV2 y 97,5 → 97,9 sobre TrackNetV3; coste de 1-2 FPS. Sin fútbol — [TrackNetV4, arXiv](https://arxiv.org/html/2409.14543v1). El buscador lo da como publicado en ICASSP 2025; la página abierta no lo indica (sin verificar).
- [R 2025-2026] TrackNetV5 (Tang, Chen, Jiang, Li, Guo; arXiv 2512.02789 v4, enero de 2026): separa la diferencia de fotogramas en mapas de polaridad positiva y negativa (conserva la dirección del movimiento) y añade una cabeza Transformer que refina un «borrador» con atención espacial y temporal. En el conjunto de TrackNetV2 (tenis), Tabla 1: F1 0,9677 (V2), 0,9581 (V4) y **0,9859 (V5)**; falsos negativos 937 → 344; entrada 512×288; 117,09 GFLOPs. Sin fútbol y sin enlace a código — [TrackNetV5, arXiv](https://arxiv.org/html/2512.02789).
- [R 2025] TOTNet (Xu, Agrahari Baniya, Wells, Bouadjenek, Dazeley, Aryal; Deakin Univ.; arXiv 2508.09650): convoluciones 3D que conservan la dimensión temporal, pérdida ponderada por visibilidad y aumento que tapa el balón; 5 fotogramas a 288×512; 28 FPS (12 FPS con flujo óptico RAFT). Solo deportes de raqueta; código en `github.com/AugustRushG/TOTNet` según el propio artículo — [TOTNet, arXiv](https://arxiv.org/html/2508.09650v1). Cifras de oclusión en la pregunta 2.
- [R 2025-2026] BlurBall (Gossard, Radovic, Ziegler, Zell; Univ. de Tubinga; arXiv 2509.18387 v2): WASB con atención *Squeeze-and-Excitation* que estima a la vez posición y desenfoque; solo tenis de mesa — [BlurBall, arXiv](https://arxiv.org/html/2509.18387v2). Cifras en la pregunta 2.

**Inferencia por ventanas y detectores de objetos diminutos**

- [A 2022] SAHI (Akyon, Altinuc, Temizel; ICIP 2022; arXiv 2202.06934): trocear en inferencia sube el AP +6,8 / +5,1 / +5,3 puntos (FCOS / VFNet / TOOD) en VisDrone y xView, y +12,7 / +13,4 / +14,5 acumulado si además se afina con recortes; código en `github.com/obss/sahi` — [SAHI, arXiv](https://arxiv.org/abs/2202.06934).
- [A 2023 / R 2025] Revisión de Transformers para objetos pequeños (Rekavandi et al.; arXiv 2309.04902): COCO define «pequeño» como caja ≤ 32×32 px; el mejor AP de pequeños recogido es 48,4 (Group DETR v2, ViT-Huge); en vídeo el mejor resultado en objetos pequeños de ImageNet VID es 13,2 mAP, que los autores llaman «una laguna de investigación significativa», y señalan que la información temporal es especialmente útil en fotogramas con oclusión o fondo recargado — [Rekavandi et al., ar5iv](https://ar5iv.labs.arxiv.org/html/2309.04902).
- [R 2026] En la panorámica 4096×1080 del SoccerTrack Challenge 2025, un detector DETR especializado en objetos pequeños rindió **peor** que YOLOv11x afinado para jugadores: HOTA 0,405 (SO-DETR + OSNet) frente a 0,491 (YOLOv11x + OSNet), Tabla 6 — [GTATrack, arXiv](https://arxiv.org/html/2602.00484).

**Código disponible**

- WASB: licencia MIT; el *model zoo* ofrece pesos de los siete modelos (DeepBall, DeepBall-Large, BallSeg, TrackNetV2, ResTrackNetV2, MonoTrack, WASB) para los cinco deportes, fútbol incluido; el README solo anuncia código de evaluación — [WASB, repositorio](https://github.com/nttcom/WASB-SBDT), [MODEL_ZOO](https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md).

### Inferences

- WASB reduce 1920×1080 a 288×512 (factor 3,75): un balón de 8-20 px queda en unos 2-5 px a la entrada y aun así se llega a F1 88 con tres fotogramas. La palanca principal de su ablación es **no perder resolución en el tronco** (+6,6 F1), no el seguidor. En una panorámica de 4096 px con balón de ~8 px, reducir el fotograma entero a 512 px dejaría el balón en ~1 px (coincide con lo que afirma SoccerTrack v2), así que las ventanas a resolución nativa son obligatorias; lo que la literatura sugiere añadir a un YOLO por fotograma es la entrada de 3-5 fotogramas y la salida en mapa de calor de alta resolución, por ventana.
- La brecha de ~40 puntos de F1 entre DeepBall/BallSeg y la familia TrackNet en fútbol no es atribuible solo a la entrada temporal: DeepBall tiene 0,1 M de parámetros y BallSeg ya usa dos fotogramas. Es indicio fuerte, no prueba limpia.
- La caída de AP de 0,65-0,81 (retransmisión) a 0,04-0,39 (cámara fija) con el mismo YOLO indica que un detector de balón no se transfiere entre tipos de cámara sin datos del dominio; respalda la vía interna de pseudoetiquetas en panorámica.
- No hay prueba de que los detectores «para objetos diminutos» de 2024-2026 mejoren a un YOLO afinado en este dominio; la única comparación abierta (jugadores, SO-DETR) fue negativa.
- Todas las cifras de fútbol gran angular provienen de un único estadio profesional (ISSIA-CNR, Serie A). No hay cifra publicada para campos de fútbol base.

### Gaps

- **No se encontró ningún banco público con balón anotado por fotograma en panorámica 4K.** SoccerTrack v1 (Scott et al., CVPRW 2022; ojo de pez 8K y dron 4K) no se pudo abrir (CVF devolvió 403); que incluya pista de balón y con qué cifras queda **sin verificar**.
- Los artículos originales de TrackNetV2 (IEEE 9302757) y MonoTrack (IEEE 9857202) no se abrieron; sus cifras aquí son las reimplementaciones de WASB, TrackNetV3, V4 y V5.
- No se halló ningún estudio riguroso de SAHI aplicado a balones deportivos; solo documentación de proyectos y blogs (no citados).
- La revisión «A comprehensive review of ball detection techniques in sports» (PeerJ CS, 2025) devolvió 403: sin verificar.
- Una variante adaptativa de SAHI (ASAHI, arXiv 2604.19233) apareció en el buscador; no se abrió: sin verificar.
- Velocidades: los FPS publicados son a 288×512 o 1080p; no hay cifras de coste por ventanas sobre 4096×1080.

---

## 2. Balón rápido, borroso, aéreo u ocluido: qué se hace y qué ganancia medida da sobre un detector por fotograma

### Takeaway

Hay tres vías con ganancia medida: (i) redes multi-fotograma entrenadas expresamente para la oclusión (TOTNet sube la exactitud en fotogramas totalmente ocluidos de 0,63 a 0,80; la rectificación de TrackNetV3 sube la exhaustividad de 94,6 a 99,3 %), (ii) tratar el desenfoque como señal y no como ruido (BlurBall: ganancia pequeña en detección, pero −37 % de error al predecir la trayectoria), y (iii) seguidores con estados de juego (vuelo, posesión, «espera tras el golpeo», fuera) o imputación condicionada a los jugadores, que rellenan huecos con ~1 m de error usando solo el 10-20 % de las observaciones del balón. En fútbol, un seguidor genérico no añadió nada sobre un buen detector multi-fotograma (WASB): la ganancia viene de modelar el estado del juego, no de suavizar.

### Cited Findings

**Entrada multi-fotograma y oclusión**

- [R 2025] TOTNet, Tabla 2 (RMSE en px / exactitud; umbral 5 px, 10 px en oclusión total). Tenis de mesa TTA, oclusión total: TrackNetV2 100,89 / 0,51; monoTrack 39,55 / 0,67; WASB 37,30 / 0,63; TOTNet 12,31 / 0,74; **TOTNet con flujo óptico 7,19 / 0,80**. Tenis, oclusión total: WASB 264,45 / 0,17; TOTNet 27,98 / 0,67. Tenis, balón visible: WASB 16,58 / 0,92; TOTNet 6,07 / 0,95 — [TOTNet, arXiv](https://arxiv.org/html/2508.09650v1).
- [R 2025] Ablación de TOTNet en TTA (Tabla 3), exactitud en oclusión total: línea base 0,54; + pérdida ponderada por visibilidad 0,61; + aumento de oclusión solo 0,56 (y el RMSE empeora a 54,26); **ambos juntos 0,74**; + flujo óptico 0,80 — [TOTNet, arXiv](https://arxiv.org/html/2508.09650v1). La propia tabla tiene incoherencias (en bádminton monoTrack supera a TOTNet en «no visible» pese a lo que dice el texto) — [TOTNet, arXiv](https://arxiv.org/html/2508.09650v1).
- [A 2023] TrackNetV3: el módulo de rectificación genera máscaras de reparación donde el volante está ocluido y restaura la trayectoria por *inpainting* (secuencias de 16, 30 % de máscara); junto con la imagen de fondo y *mixup*, la exhaustividad pasa de 94,56 % (V2) a 99,33 % — [TrackNetV3, repositorio](https://github.com/qaz812345/TrackNetV3).
- [R 2026] TrackNetV5: la cabeza de refinamiento residual, que usa el contexto de los borradores de fotogramas vecinos, reduce los falsos negativos de 937 a 344 en el conjunto de TrackNetV2; no hay banco específico de oclusión — [TrackNetV5, arXiv](https://arxiv.org/html/2512.02789).
- [A 2019] BallSeg (Van Zandycke y De Vleeschouwer; MMSports '19): entrada de 6 canales = imagen RGB + diferencia absoluta con el fotograma anterior; concatenar los dos fotogramas «dio malos resultados» y la diferencia «mejora significativamente» la exactitud (Fig. 4, sin cifra); punto de operación TPR 0,66 con 0,33 falsos candidatos por escena; coste: 38,39 → 24,67 FPS a 1024×512 (Tabla 1, GTX 1080 Ti) — [BallSeg, ar5iv](https://ar5iv.labs.arxiv.org/html/2007.11876).
- [A 2023] En fútbol, el seguimiento en línea de WASB (consistencia temporal en inferencia) no cambia F1, Acc. ni AP (Tabla 4) — [WASB, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237).

**Desenfoque de movimiento como señal**

- [R 2025-2026] BlurBall: en su conjunto (26 grabaciones, 64 119 fotogramas, cámaras fijas) etiquetar el balón en el **punto medio de la estela** en lugar del borde delantero mejora a todos los modelos (Tabla 2): WASB F1 95,58 → 96,00 y AP 95,10 → 96,85; TrackNetV2 F1 90,37 → 91,61; MonoTrack 93,60 → 94,97; TrackNetV3 93,74 → 95,93 — [BlurBall, arXiv](https://arxiv.org/html/2509.18387v2).
- [R 2025-2026] BlurBall frente a WASB (Tabla 3, 1 paso): F1 96,00 → 96,52, Acc 92,56 → 93,47, AP 96,85 → 98,23; error de longitud de la estela 3,1 → 1,5 px y de ángulo 13,5° → 6,5°. Ajustar la trayectoria con posición + desenfoque baja el error medio de predicción de 84,4 ± 136,6 px a 53,0 ± 87,1 px en 95 trayectorias (§4.5) — [BlurBall, arXiv](https://arxiv.org/html/2509.18387v2).
- [A 2019] DeepBall cita como falsos negativos la oclusión severa y el balón que se confunde con equipación o fondo blancos — [DeepBall, ar5iv](https://ar5iv.labs.arxiv.org/html/1902.07304).

**Seguimiento con estados físicos y posesión (grafo / programación entera / haz)**

- [A 2016] Maksai, Wang y Fua (EPFL; arXiv 1511.06181, CVPR 2016 según el buscador): programa entero mixto sobre posiciones 3D, estados (vuelo, en posesión, rodando, no presente) y restricciones físicas dependientes del estado (parábola en vuelo, velocidad constante rodando); el balón «en posesión» debe estar a menos de una distancia fija de un jugador. En fútbol (ISSIA, seis cámaras a 25 fps, 1 000 fotogramas de evaluación) bastan dos estados porque el balón suele salir de plano al volar; el texto afirma que supera a los demás métodos pero **no da cifras** (solo curvas, Fig. 4) — [Maksai et al., ar5iv](https://ar5iv.labs.arxiv.org/html/1511.06181).
- [R 2025] Vorobev, Prosvetov y Elhadji Daou (arXiv 2506.07981): filtro recursivo con estado híbrido y búsqueda en haz (K ≈ 1000 hipótesis) sobre detecciones 2D del balón (RTMDet-t), posiciones de jugadores y calibración. Cuatro modos: **J** (vuelo balístico con botes), **P** (posesión: el balón va con un jugador), **W** («espera tras la posesión»: tras un golpeo y **antes de que el balón se vuelva a detectar**, guarda instante y jugador) y **O** (fuera del campo). Sin detección, el estado J se extrapola con varianza creciente y una penalización fija — [Vorobev et al., arXiv](https://arxiv.org/html/2506.07981v1).
- [R 2025] Resultados de Vorobev et al. (Tabla 2; 10 partidos × 3 min de la liga rusa, 6144×3240, verdad-terreno por triangulación con cuatro cámaras): fracción de fotogramas a menos de 1 m, 0,61 (latencia 1 fotograma) a 0,66 (latencia 50), frente a 0,56 del EKF de Ribnick 2009, 0,49 de Metzler 2013, 0,39 de Kim 1998 y 0,64 del MIP multicámara de Maksai 2016; F1 de detección de golpeo 0,59-0,74 (±12 fotogramas); 53 fps en un núcleo de CPU. El resumen promete «precisión centimétrica», pero solo el 59 % de los fotogramas queda a ≤ 0,5 m; conjunto propietario, sin código — [Vorobev et al., arXiv](https://arxiv.org/html/2506.07981v1), [abs](https://arxiv.org/abs/2506.07981).

**Imputación de la trayectoria a partir de los jugadores**

- [A 2023] Kim, Choi, Kim, Yoon y Ko (KDD 2023; arXiv 2306.08206): infiere primero el poseedor y después la trayectoria del balón a partir de las trayectorias de los jugadores (Set Transformer + Bi-LSTM jerárquico). Sin ninguna observación del balón: error 3,66 m y acierto de poseedor 64,7 % (Tabla 2). **Con el 20 % de las posiciones del balón dadas: error 1,41 m y acierto de poseedor 91,25 %; F1 de pases 0,87** (Tablas 3-4) — [Kim et al., ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206).
- [R 2024] TranSPORTmer (Capellera, Ferraz, Rubio, Agudo, Moreno-Noguer; ACCV 2024 según el buscador; arXiv 2410.17785): en 283 partidos de LaLiga 2022-23, inferencia del balón con el 100 % enmascarado 2,71 m de error medio frente a 3,89 m de Kim et al.; **con el 80 % enmascarado 0,80-0,88 m y con el 90 % 1,09-1,23 m** (Tabla 4); clasificación del estado del juego 80,84 % — [TranSPORTmer, arXiv](https://arxiv.org/html/2410.17785v2).
- [R 2025-2026] Peral, Capellera, Ferraz, Rubio y Agudo (WACV 2026; arXiv 2512.19528): Transformer multimodal sobre trayectorias, tipo de jugador y **recortes de imagen de cada jugador** en cámara táctica (283 partidos, secuencias de 9,6 s a 6,25 Hz), sin información del balón en inferencia. Tabla 1: error medio 1,17 m y máximo 4,77 m, estado del balón 85,85 %, poseedor 89,56 %, frente a 1,72 m / 79,77 % / 83,79 % de TranSPORTmer y 1,86 m de BallRadar — [Peral et al., arXiv](https://arxiv.org/html/2512.19528v1), [abs](https://arxiv.org/abs/2512.19528).

**Reconstrucción 3D monocular del balón**

- [A 2022] Van Zandycke y De Vleeschouwer (CVSports 2022; arXiv 2204.00003): estima el diámetro del balón en píxeles con una CNN sobre recortes y lo convierte a profundidad con el diámetro real. Baloncesto, balones de 14-45 px: error de diámetro 1,6 px → error 3D de 1,8-2,3 m, **≈ 10 % de la distancia a la cámara** (Tabla 2) — [Van Zandycke y De Vleeschouwer, ar5iv](https://ar5iv.labs.arxiv.org/html/2204.00003), [abs](https://arxiv.org/abs/2204.00003).
- [R 2025] En fútbol (SoccerNet-v3D, misma fórmula): error 3D medio de 4,2 m con las mejores cajas y 15,3 m con las originales; una variación del 10 % en el tamaño en píxeles produce errores de unos 6 a 14 m, mientras que desplazar el centro un 2 % de la diagonal produce 0,6-1,6 m (Tabla 2, Fig. 6) — [SoccerNet-v3D, arXiv](https://arxiv.org/html/2504.10106v1).
- [A 2023] Caio, Van Zandycke y De Vleeschouwer (MMSports '23; arXiv 2309.03640): en vez del diámetro, predice la proyección del balón sobre el suelo (altura en píxeles) a partir de la imagen y el contexto; baloncesto, «mejora sustancial» sin cifra en el resumen — [Caio et al., arXiv](https://arxiv.org/abs/2309.03640).
- [R 2025] «Where Is The Ball» (Ponglertnapakorn y Suwajanakorn; CVsports en CVPR 2025; arXiv 2506.05763): LSTM que eleva secuencias de seguimiento 2D a trayectoria 3D con una representación canónica independiente de la cámara, entrenado solo con simulación y evaluado en cuatro conjuntos sintéticos y tres reales; el resumen no da cifras — [Ponglertnapakorn y Suwajanakorn, arXiv](https://arxiv.org/abs/2506.05763).

### Inferences

- El fallo medido en el proyecto (7 de 28 tiros sin detección en ±2 s) corresponde exactamente al hueco que cubren el modo **W** de Vorobev et al. y la imputación condicionada a jugadores: el Viterbi interno rechaza saltos imposibles pero no plantea la hipótesis «el jugador X golpeó en t y el balón vuela sin verse». Las cifras de imputación (0,8-1,4 m con el 10-20 % de observaciones) sugieren que, tras un golpeo, bastan pocas detecciones dispersas más las trayectorias de los jugadores para acotar el balón dentro de una ventana razonable.
- Unas pseudoetiquetas generadas por un detector por fotograma están sesgadas hacia balones nítidos y lentos; BlurBall muestra que la convención de etiquetado de la estela importa (+0,4 a +2,2 F1 según el modelo). Reetiquetar o sintetizar estelas es una vía barata no explorada internamente.
- El aumento de oclusión solo funciona junto con la pérdida ponderada por visibilidad (TOTNet); aplicado solo, empeora. Requiere etiquetas de visibilidad.
- La profundidad por diámetro es inviable con balón de ~8 px: 1 px de error ya es un 12,5 % del tamaño, por encima del 10 % que en SoccerNet-v3D produce 6-14 m. Para el balón aéreo, la vía defendible es ajustar la parábola a la pista 2D (modo J, LSTM de trayectoria), no medir el tamaño.
- Las ganancias de la vía (i) están medidas en deportes de raqueta con cámara fija cercana; trasladarlas a fútbol es una hipótesis.

### Gaps

- No hay ninguna medición en fútbol de TOTNet, BlurBall ni TrackNetV4/V5.
- Maksai et al. no dan cifras en el texto; la comparación numérica con ellos procede de la reimplementación de Vorobev et al.
- Wang et al., «Take your eyes off the ball» (CVIU 2014, posesión primero y balón después) y Ren et al. (CVIU 2009, varias cámaras fijas) solo aparecieron en el buscador: **sin verificar**.
- Un artículo de Procedia Computer Science (2025) que según el buscador mide la ganancia de un filtro de Kalman sobre YOLOv8x para el balón devolvió 403: sin verificar.
- Ninguno de los trabajos de imputación desde jugadores analiza por separado pases largos o balones aéreos; sus errores máximos (4,8-7 m) sugieren que ahí fallan, pero no está medido.
- No se encontró ningún trabajo que cuantifique la tasa de detección del balón en los segundos posteriores a un tiro.

---

## 3. Falsos positivos persistentes (balones de reserva, conos, otro campo): qué técnicas se usan

### Takeaway

No se encontró ningún trabajo dedicado a balones de reserva o conos; la literatura lo trata de forma indirecta con tres mecanismos: meter el movimiento o el fondo como entrada de la red, verificar los candidatos en una segunda etapa y dar al seguidor estados explícitos de «no presente / fuera / en posesión». Los autores de DeepBall ya señalaron en 2019 los objetos estáticos parecidos al balón («balones de reserva fuera del campo») como trabajo futuro que requiere información temporal.

### Cited Findings

- [A 2019] DeepBall: falsos positivos observados en medias blancas de jugadores y en el fondo fuera del campo (publicidad); mitigaciones: minado de negativos difíciles (3:1), aumento de datos y contexto multiescala. Trabajo futuro declarado: usar información temporal para separar del balón en movimiento los objetos estáticos parecidos, citando expresamente partes de la publicidad y **balones de reserva fuera del campo** — [DeepBall, ar5iv](https://ar5iv.labs.arxiv.org/html/1902.07304).
- [A 2023] TrackNetV3 concatena a la entrada una **imagen de fondo estimada** del vídeo (cámara fija) para localizar el volante pese a interferencias visuales (`--bg_mode concat`) — [TrackNetV3, repositorio](https://github.com/qaz812345/TrackNetV3).
- [R 2024-2026] TrackNetV4 y V5 introducen el movimiento como atención a partir de diferencias de fotogramas (valor absoluto en V4; polaridad con signo en V5); V5 escoge la variante combinada por tener mayor precisión y menos falsos positivos (ablación, Tablas 4-5) — [TrackNetV4, arXiv](https://arxiv.org/html/2409.14543v1), [TrackNetV5, arXiv](https://arxiv.org/html/2512.02789).
- [A 2019] BallSeg: los falsos positivos «no son espacialmente consistentes» entre mapas de calor de recortes distintos, de modo que promediar varios recortes aleatorios los suprime; con solo dos recortes promediados la tasa de detección sube ~10 % a baja tasa de falsos positivos (§4.3, Fig. 9-10) — [BallSeg, ar5iv](https://ar5iv.labs.arxiv.org/html/2007.11876).
- [A 2022] Verificación en segunda etapa: la CNN de recortes de Van Zandycke y De Vleeschouwer devuelve una confianza de «hay balón» que, usada para filtrar los *k* mejores candidatos del detector, mejora la detección «por un amplio margen» frente a quedarse con el primero (§4.4, Fig. 5; sin cifra en el texto) — [Van Zandycke y De Vleeschouwer, ar5iv](https://ar5iv.labs.arxiv.org/html/2204.00003).
- [A 2016 / R 2025] Estados del seguidor: Maksai et al. incluyen el estado *not_present* y exigen cercanía a un jugador en posesión; Vorobev et al. añaden el modo O (fuera del campo, p. ej. saques de banda) y puntúan cada detección según su coherencia con la hipótesis de estado — [Maksai et al., ar5iv](https://ar5iv.labs.arxiv.org/html/1511.06181), [Vorobev et al., arXiv](https://arxiv.org/html/2506.07981v1).
- [R 2024-2026] El estado del juego (pase, posesión, fuera de juego/pausa) se puede clasificar por fotograma solo con los jugadores: 80,84 % (TranSPORTmer) y 85,85 % con recortes de imagen (Peral et al.) — [TranSPORTmer, arXiv](https://arxiv.org/html/2410.17785v2), [Peral et al., arXiv](https://arxiv.org/html/2512.19528v1).
- [R 2026] Para jugadores en panorámica de ojo de pez, reentrenar el detector con pseudoetiquetas de alta confianza redujo los falsos positivos de 4 913 a 494 y los falsos negativos de 40 046 a 16 186 (Tabla 4) — [GTATrack, arXiv](https://arxiv.org/html/2602.00484).
- [A 2023] El equipo MOT4MOT necesitó un filtrado adicional (no descrito) sobre YOLOv8l para subir la precisión del balón en SoccerNet-tracking — [MOT4MOT, arXiv](https://arxiv.org/abs/2308.16651).

### Inferences

- El filtro interno «dentro del campo y no estático» es una versión por reglas de lo que la literatura hace dentro de la red (fondo o diferencia como canal). La ventaja de hacerlo en la red es que no descarta el balón real cuando está parado (saques, faltas): la red ve a la vez apariencia y movimiento, y el seguidor decide con el estado.
- Lo que la literatura sugiere y no consta como hecho internamente: (a) canal de fondo estimado por mediana temporal en la entrada del detector propio; (b) clasificador de candidatos «balón en juego» con contexto de jugadores (distancia al jugador más cercano, estado del juego inferido); (c) minado de negativos difíciles con los objetos fijos ya identificados por el filtro actual, que salen gratis como negativos etiquetados.
- Con cámara fija, un objeto estático persistente es trivial de catalogar por posición a lo largo del partido (mapa de ocupación de detecciones inmóviles); no se encontró esta técnica publicada para balones, así que es una propuesta, no un hallazgo.

### Gaps

- **No se halló ningún trabajo sobre balones de reserva, conos ni campos adyacentes en fútbol base** (tres búsquedas distintas). Es una laguna real de la literatura, no solo de esta búsqueda.
- Halbinger y Metzler (icSPORTS 2013, sustracción de fondo + verificación con una cámara estática) y el clasificador de verificación de UT Austin Villa (RoboCup) solo aparecieron en el buscador: **sin verificar**.
- La observación de que los detectores clásicos fallan sobre todo cuando el balón está parado (detector por isofotas) procede de un resumen del buscador: sin verificar.
- No hay cifras de cuánto reduce los falsos positivos el canal de fondo de TrackNetV3 por separado (su tabla mezcla los tres cambios).

---

## 4. Señales alternativas al balón para localizar la acción

### Takeaway

Las trayectorias de los jugadores son el sustituto mejor respaldado: solo con ellas el balón se infiere con 2,7 m de error medio (1,2 m si se añaden recortes de imagen de los jugadores), y una región de 600×600 centrada en esa estimación contiene el balón real el 96 % de los fotogramas en un vídeo panorámico fijo. La pose y la mirada por sí solas localizan mal (los humanos aciertan un 20-34 % y un modelo solo con contexto se queda en 6 mAP), pero como complemento del detector añaden unos 4 puntos; y las señales sin balón pierden la precisión temporal de los eventos (mAP a 1 s de 0,22 sin balón frente a 0,60 con él en SoccerTrack v2).

### Cited Findings

**Posiciones y trayectorias de jugadores**

- [A 2023] Kim et al. (KDD 2023), §5.3 y Tabla 5: en el único partido de test con GPS, eventos y **vídeo de campo completo con cámaras fijas**, una región centrada en el balón estimado solo con las trayectorias de los jugadores («sin usar ningún resultado de detección de balón») contiene el balón real el 36,37 % de los fotogramas con caja de 100, 66,11 % con 200, 79,97 % con 300, 92,82 % con 500 y **96,25 % con 600**. El artículo no da la resolución del vídeo ni las unidades de la caja (presumiblemente píxeles) ni el número de fotogramas — [Kim et al., ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206).
- [A 2023] Los autores proponen ese uso expresamente: *zoom* automático para retransmisión, imputación de tramos perdidos, anotación semiautomática de pases — [Kim et al., arXiv abs](https://arxiv.org/abs/2306.08206). Dato práctico: el modelo completo usa seis características por jugador a 10 Hz; con solo la posición el error sube de 3,66 a 4,65 m y con posición y velocidad queda en 3,92 m (Tabla 7) — [Kim et al., ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206).
- [R 2024] TranSPORTmer: 2,71 m con solo posiciones (x, y) y tipo de agente, un 30 % menos que Kim et al. (3,89 m) en el mismo conjunto (Tabla 4) — [TranSPORTmer, arXiv](https://arxiv.org/html/2410.17785v2).
- [R 2025-2026] Peral et al. (WACV 2026), ablación de modalidades (Tabla 2): solo trayectorias 1,97 m y poseedor 82,12 %; trayectorias + tipo 1,78 m; trayectorias + recortes 1,68 m y 86,38 %; las tres 1,45 m y 87,67 %; con el preentrenamiento CropDrop 1,17 m y 89,56 %. **Solo recortes de imagen, sin trayectorias: 23,95 m** — [Peral et al., arXiv](https://arxiv.org/html/2512.19528v1).
- [R 2026] SoccerTrack v2, detección de acciones a partir de trayectorias reales (red convolucional temporal a 5 Hz, validación cruzada de 5 pliegues, Tabla 5): solo jugadores (82 características) mAP 0,222 ± 0,021 a 1 s y 0,524 ± 0,048 a 5 s; **jugadores + pista de balón (19 características más) 0,599 ± 0,064 y 0,666 ± 0,045**. Por clase a 1 s (Tabla 6): pase 0,286 → 0,853, conducción 0,227 → 0,830. Los autores avisan de que la condición con balón puede estar inflada porque la pista de balón se reconstruyó a partir de los eventos — [SoccerTrack v2, v2 HTML](https://arxiv.org/html/2508.01802v2).
- [R 2025] FOOTPASS (Ochin, Chekroun, Stanciulescu, Manitsaris; CVIU; arXiv 2511.16183): añadir contexto táctico multiagente a un detector puramente visual sube el F1 global de 35,9 (TAAD) a 52,1 (TAAD + GNN) y 67,5 (TAAD + DST); el modelo con contexto recupera el 33,4 % de los eventos cuyo jugador no tiene caja, frente al 3,9 % del visual (Tabla 2, Fig. 6) — [FOOTPASS, arXiv](https://arxiv.org/html/2511.16183v1).

**Pose, orientación y mirada**

- [R 2025] «Spot The Ball» (Balamurugan, Wu, Chun, Gaw, Eyzaguirre, Gerstenberg; arXiv 2511.00261): localizar un balón borrado en imágenes de fútbol, baloncesto y voleibol. Los humanos aciertan un 20-34 % apoyándose en la dirección de la mirada y la pose; cuatro modelos de visión-lenguaje se quedan en ≤ 17 % y recurren a heurísticas (centro de la imagen, cerca de jugadores) — [Spot The Ball, arXiv](https://arxiv.org/abs/2511.00261).
- [R 2025] PLUCC (Salass, Bright, Nazemi, Chen, Zelek, Clausi; Univ. de Waterloo; arXiv 2506.04365): detector de disco de hockey de un solo fotograma con un codificador de contexto que recibe la segmentación de todos los jugadores (detector + SAM 2). En casos difíciles (borroso, ocluido), Tabla 1: mAP 83,5 frente a 79,6 de la misma red sin codificador de contexto (FCN-ResNet152) y 71,3 de Faster R-CNN; error medio en coordenadas de pista 1,05 m frente a 3,89 m de esa misma red sin contexto (Tabla 2). **Una red entrenada solo con el contexto obtiene 6,0 mAP y 178,5 px de error medio**, con mapas de calor que apuntan en la dirección de la mirada de los jugadores. El preprocesado baja la velocidad de 53,3 a 6,0 FPS — [PLUCC, arXiv](https://arxiv.org/html/2506.04365v1).
- [A 2021] Vats, Fani, Clausi y Zelek (CVSports 2021; arXiv 2105.10563): localización del disco con contexto temporal y un mecanismo de atención alimentado con mapas gaussianos en las posiciones de los jugadores; AUC 73,1 %, 5 fps en 720p; el aprendizaje multitarea con la posición del disco mejora el reconocimiento de eventos — [Vats et al., arXiv](https://arxiv.org/abs/2105.10563).
- [A 2020] Arbués-Sangüesa, Martín, Fernández, Ballester y Haro (CVSports en CVPR 2020; arXiv 2004.07209): la orientación corporal estimada por pose y proyectada al campo, combinada con posiciones, predice el pase más factible con exactitud top-3 > 0,7 sobre más de 6 000 pases — [Arbués-Sangüesa et al., arXiv](https://arxiv.org/abs/2004.07209).

**Encuadre de operadores humanos y «actionness»**

- [A 2015] Chen y Carr, «Mimicking Human Camera Operators» (WACV 2015): estiman la configuración pan-tilt-zoom de la cámara de un operador humano en cada fotograma y entrenan un regresor que predice el ángulo de paneo a partir de posiciones ruidosas de jugadores; demostrado en un partido de baloncesto de instituto, con mejor resultado que el método anterior según el resumen (sin cifras en las páginas abiertas) — [Chen y Carr, MLAnthology](https://mlanthology.org/wacv/2015/chen2015wacv-mimicking), [Disney Research](https://la.disneyresearch.com/?p=12419).
- [A 2020] Quiroga, Carrillo, Maldonado, Ruiz y Zapata, «As Seen on TV» (CVPRW 2020): producción automática de baloncesto con una sola cámara fija 4K; detecta jugadores, balón y árbitros para fijar las coordenadas de la acción principal y el estado del juego. Solo se pudo abrir una ficha de catálogo con el resumen; los detalles de la «actionness» gaussiana sobre posiciones de jugadores más regla de balón proceden del buscador — [ficha en SPONET](https://sponet.de/sponet/Record/4072946?lng=en).
- [A 2017] Deep 360 Pilot (Hu, Lin, Liu, Cheng, Chang, Sun; arXiv 1705.01759): agente que elige el ángulo de visión en vídeo deportivo de 360° seleccionando el **objeto principal** entre candidatos detectados (RNN selector + regresor), supervisado con el objeto que anotadores humanos marcaron como más saliente. Tabla 2 (solape medio): 0,66-0,76 frente a 0,47-0,73 de AUTOCAM, **0,20-0,25 de puntuar las cajas con un mapa de saliencia** y 0,42-0,72 de puntuarlas con movimiento — [Deep 360 Pilot, ar5iv](https://ar5iv.labs.arxiv.org/html/1705.01759).

**Saliencia aprendida dentro del modelo posterior**

- [R 2026] AdaSpot obtiene su RoI sin supervisión: promedia por canales los mapas de características de la rama de baja resolución, normaliza, suaviza en espacio y tiempo y umbraliza. En SN-BAS las RoI «suelen seguir al balón», pero los autores reconocen que la saliencia se dispersa cuando no hay eventos cerca y el balón puede quedar fuera, y que el evento «centro» depende de un contexto más amplio que la RoI — [AdaSpot, arXiv](https://arxiv.org/html/2602.22073v1).
- [R 2026] La selección de RoI **aprendida** es inestable en detección precisa de eventos (Tabla 5, tres semillas, rama de alta resolución sola en SN-BAS): AdaFocus-v2 23,1 ± 15,1, Uni-AdaFocus 39,4 ± 3,3, AdaSpot 52,1 ± 0,7 — [AdaSpot, arXiv](https://arxiv.org/html/2602.22073v1).

### Inferences

- Orden de fiabilidad que sale de las cifras para «dónde está el juego» cuando el balón no se ve: (1) trayectorias de jugadores con modelo aprendido (1,2-2,7 m; 96 % dentro de una caja de 600); (2) lo mismo más apariencia de los jugadores (pose implícita en los recortes: de 1,97 a 1,68 m); (3) pose o mirada sola (muy por debajo: 6 mAP, 20-34 % en humanos). No se encontró ninguna comparación directa con el centroide o la densidad de jugadores ni con el flujo óptico como señal de RoI.
- El resultado interno «dar a AdaSpot la RoI real no mejora a su saliencia» es coherente con la literatura: en SN-BAS la saliencia ya sigue al balón y el límite está en otro sitio (eventos que requieren contexto, RoI única). La señal de jugadores es más útil **antes** del modelo posterior (decidir el recorte de la panorámica) que dentro de él.
- Hay una asimetría clara: las señales de jugadores bastan para **encuadrar** (tolerancia de metros) pero no para **fechar** eventos de balón (mAP a 1 s 0,22 sin balón). El balón sigue siendo necesario para la detección de acciones; lo que cambia es que no hace falta verlo en todos los fotogramas.
- Todos los modelos de inferencia desde jugadores se entrenaron con seguimiento profesional (óptico o GPS de 22 jugadores con identidad y equipo). En fútbol base con seguimiento ruidoso y jugadores sin identificar, el error real será mayor; no hay medición.
- En vídeo de 360°, seleccionar un objeto detectado con un modelo secuencial aprendido triplica a la saliencia genérica; apoya usar detecciones (jugadores, balón) y no saliencia de bajo nivel para decidir el encuadre.

### Gaps

- **Ninguna comparación publicada entre centroide/densidad de jugadores, flujo óptico y balón como señal de RoI en fútbol.** El flujo óptico solo aparece como entrada auxiliar (TOTNet; E2E-Spot, pregunta 5).
- Chen et al., «Learning Online Smooth Predictors for Realtime Camera Planning» (CVPR 2016) y la tesis de J. Chen (UBC, con fútbol) no se pudieron abrir (403): **sin verificar**. Tampoco se obtuvieron cifras de error de paneo de Chen y Carr 2015 (el PDF supera el límite de descarga).
- El PDF de «As Seen on TV» devolvió 403: su método de «actionness» y sus cifras quedan sin verificar más allá del resumen.
- Pano2Vid / AutoCam (Su, Jayaraman, Grauman) solo vistos en el buscador: sin verificar.
- No se encontró predicción de la mirada de espectadores (seguimiento ocular) aplicada a fútbol panorámico.
- Kim et al. no documentan cómo proyectan la estimación al vídeo ni la resolución; el 96 % es de un único partido.

---

## 5. Alta resolución en los modelos posteriores: ventanas, reducción de escala y zoom o RoI adaptativa; trabajos sobre panorámica completa

### Takeaway

No se encontró ningún trabajo que haga detección de acciones a partir del vídeo sobre la panorámica completa: SoccerTrack v2 (octubre de 2026) lo deja explícitamente como experimento pendiente y solo publica una línea base sobre trayectorias. Lo medido es que (a) la resolución pesa más que el tamaño del modelo cuando los atletas son pequeños (SynLoc: pasar de reducir 6× a 4× da +11,5 puntos; los ganadores de 2026 usan ventanas adaptativas guiadas por un detector grueso), (b) en detección de acciones de balón los mejores sistemas trabajan a 796×448-1280×736 sobre vídeo de retransmisión de 720p, y (c) una rama de RoI de alta resolución aporta +5 mAP sobre baja resolución con un coste tres veces menor que subir toda la imagen (AdaSpot).

### Cited Findings

**Sobre panorámica completa**

- [R 2026] SoccerTrack v2 no publica ninguna línea base de detección de acciones basada en vídeo; §3 discute los obstáculos y llama «primeros experimentos naturales» a reducir el fotograma completo o a recortar centrado en el juego, y advierte de que reducir la panorámica 4K al tamaño de entrada habitual de un *backbone* de vídeo deja un balón de 22 cm en aproximadamente **un píxel**, por lo que hace falta un recorte. Trabajo futuro declarado: estudiar el encuadre que sigue al balón generando ventanas desde la panorámica — [SoccerTrack v2, v2 HTML](https://arxiv.org/html/2508.01802v2).
- [R 2026] Reconstrucción del estado del juego sobre la panorámica (SoccerTrack v2, §4.7-4.8, Tabla 3 y tablas suplementarias): RF-DETR + ViTPose + PRTReID + BoT-SORT + ViTSTR para dorsales + asociación global de *tracklets*; **GS-HOTA 12,25 sobre la parte completa (21,66 sin exigir atributos)**; en los primeros 30 s con las predicciones fijas 23,21 (55,33 sin atributos), cayendo de forma monótona (1 min 21,30; 5 min 18,47; 10 min 15,40). La Tabla 3 da además 31,35 (55,72 sin atributos) como media de los 30 s iniciales; la diferencia con 23,21 no queda explicada en el texto leído. Sin atributos, parte completa: DetA 43,24, AssA 11,11, LocA 81,67; primeros 30 s: DetA 49,87, AssA 62,96. Coste: unas 37 h por parte en una GPU de 16 GB. No consta si la panorámica se trocea ni el tamaño de entrada del detector — [SoccerTrack v2, v2 HTML](https://arxiv.org/html/2508.01802v2).
- [R 2026] Seguimiento de jugadores en panorámica de ojo de pez 4096×1080 (SoccerTrack Challenge 2025, ganador GTATrack; taller MMSports de ACM Multimedia 2025): YOLOv11x afinado con **1280 px en el lado largo y sin ventanas**, Deep-EIoU + asociación global; HOTA 0,60 (DetA 0,76, AssA 0,47) frente a 0,42 de ByteTrack; las pseudoetiquetas subieron el HOTA de 0,380 a 0,491 (Tablas 3-4). El texto y las tablas no coinciden en algunas cifras (0,511 frente a 0,491; ByteTrack 0,52 frente a 0,42) — [GTATrack, arXiv](https://arxiv.org/html/2602.00484), [abs](https://arxiv.org/abs/2602.00484).
- [R 2025] Spiideo SoccerNet SynLoc (Ardö, Nilsson, Cioppa, Magera, Giancola, Liu, Ghanem, Van Droogenbroeck; VISAPP 2025): localización de atletas en coordenadas de campo desde cámaras estáticas 4K que cubren medio campo cada una (17 instalaciones, ~70 k imágenes con atletas sintéticos). Línea base YOLOX-pose, Tabla 2 (mAP-LocSim): a 640 px (reducción 6×) tiny 60,6 / s 63,6 / m 67,8; **a 960 px (reducción 4×) tiny 72,6 / s 76,3 / m 79,3**. Los autores: subir la resolución da «un aumento más significativo que aumentar el tamaño del modelo» porque los atletas reducidos tienen muy pocos píxeles — [SynLoc, SciTePress](https://www.scitepress.org/publishedPapers/2025/131082/pdf/index.html), [ficha](https://www.scitepress.org/Papers/2025/131082); datos y código en [github.com/Spiideo/sskit](https://github.com/Spiideo/sskit).
- [R 2026] SynLoc en SoccerNet 2026 (Tabla 4; mAP-LocSim / exactitud por fotograma): línea base 77,30 / 33,74; ganador 97,67 / 81,91 con **ventanas adaptativas conscientes de los bordes** (amplía las ventanas de una rejilla fija alrededor de las detecciones gruesas de un YOLO26-Large) y RTMPose-X de dos puntos clave; el tercero usa ventanas fijas con NMS (RF-DETR + ViTPose); el cuarto pasa el detector global y lo repite sobre recortes reescalados de cada jugador; el quinto reentrena a mayor resolución. No se dan tamaños de ventana — [SoccerNet 2026, arXiv](https://arxiv.org/html/2607.07320).
- [R 2026] El artículo del ganador (Nguyen, Nguyen, Huynh, Tran; arXiv 2609.02705) da LocSim 97,44 y mAP 0,9128, «más de un 21 %» sobre la línea base, y afirma que las ventanas adaptativas mitigan la pérdida de exhaustividad con variación extrema de escala — [Nguyen et al., arXiv](https://arxiv.org/abs/2609.02705).
- [R 2024] Cabado et al., «Beyond the Premier» (CVPRW 2024): evalúa modelos de detección de acciones entre ligas (de aficionados a profesionales) y entre calidades de realización (de cámara pilotada por IA a realización profesional); concluye que la transferencia es difícil y que **la calidad de la realización es el factor más influyente** (resumen; las cifras no se pudieron leer) — [Cabado et al., MLAnthology](https://mlanthology.org/cvprw/2024/cabado2024cvprw-beyond).

**Resolución de entrada en detección de acciones de balón (vídeo de retransmisión, 720p)**

- [A 2023] Ganador de SoccerNet Ball Action Spotting 2023 (repositorio lRomul): fotogramas en escala de grises a **1280×736**, 15 fotogramas apilados de tres en tres, EfficientNetV2-B0 con fusión lenta 2D→3D; 86,47 % mAP@1 en el conjunto del reto. La fusión lenta frente a fusión temprana 2D supuso de ~65 % a ~80 %; no hay ablación de resolución — [lRomul, repositorio](https://github.com/lRomul/ball-action-spotting).
- [R 2024] SoccerNet 2024, detección de acciones de balón (mAP@1): T-DEED 73,39 con clips de 100 fotogramas a **796×448**; segundo (UniBw Munich) 71,35 con 21 fotogramas en gris a 720p; línea base 56,15. Los vídeos son de 720p — [SoccerNet 2024, arXiv](https://arxiv.org/html/2409.10587).
- [R 2025-2026] En SoccerNet 2025 (acciones de balón por equipo; ganador 60,03 de Team-mAP@1 reimplementando T-DEED) y 2026 (acciones centradas en el jugador; ganador PAVE con 58,94 de F1 y características X3D alineadas a la RoI de cada jugador) los resúmenes no indican resolución de entrada; en 2026 el seguimiento del balón para extraer características centradas en él es «un tema recurrente» — [SoccerNet 2025, arXiv](https://arxiv.org/html/2508.19182), [SoccerNet 2026, arXiv](https://arxiv.org/html/2607.07320).
- [A 2022] E2E-Spot (Hong, Zhang, Gharbi, Fisher, Fatahalian; arXiv 2207.10213), Tabla 3(c): bajar la altura de 224 a 112 px cuesta −7,6 mAP en tenis, −5,6 y −4,2 en patinaje, −3,5 en saltos y −2,6 en gimnasia (tolerancia de 1 fotograma); el flujo óptico solo es peor que RGB (tenis 58,2 frente a 96,1) pero suma algo en conjunto (96,3) — [E2E-Spot, ar5iv](https://ar5iv.labs.arxiv.org/html/2207.10213).
- [A 2024] El artículo de T-DEED usa 224×224 en patinaje y saltos y no incluye ni SoccerNet ni ablación de resolución — [T-DEED, arXiv](https://arxiv.org/html/2404.05392v2).

**Zoom o RoI adaptativa**

- [R 2026] AdaSpot (Xarles, Escalera, Moeslund, Clapés; arXiv 2602.22073, febrero de 2026): rama global a baja resolución (398×224 en SN-BAS, la mitad del fotograma de 796×448) + una RoI por fotograma recortada del fotograma de alta resolución y llevada a 112×112. Tabla 3 (AdaSpot-s, mAP en SN-BAS a 0,5 s / 1 s): completo 53,02 / 56,43; **solo baja resolución 47,98 / 51,69**; solo rama RoI 52,13 / 55,95; sin suavizado 49,99 / 53,84 — [AdaSpot, arXiv](https://arxiv.org/html/2602.22073v1), [abs](https://arxiv.org/abs/2602.22073).
- [R 2026] AdaSpot, Tabla 4: tamaño del recorte en SN-BAS 56² → 51,53; 84² → 51,16; **112² → 53,02**; 168² → 52,08; 224² → 50,60; **dos RoI por fotograma empeoran (49,19)**; RoI de relación de aspecto adaptativa empeora (51,61). En tenis, la rama RoI añade +3,93 / +2,14 / +2,12 mAP a resoluciones base de 112² / 168² / 224² por unos +6 GFLOPs, más de lo que daría subir toda la imagen con el mismo coste (Fig. 3, §4.3) — [AdaSpot, arXiv](https://arxiv.org/html/2602.22073v1).
- [R 2026] Coste en SN-BAS (Tabla 6, 398×224): AdaSpot-s 46,18 GFLOPs y AdaSpot-b 90,04 frente a 150,02 de E2E-Spot-800MF y 151,31 de T-DEED-800MF — [AdaSpot, arXiv](https://arxiv.org/html/2602.22073v1).
- [A 2018] Learning to Zoom (Recasens, Kellnhofer, Stent, Matusik, Torralba; ECCV 2018): capa diferenciable de muestreo guiada por saliencia que amplía las regiones útiles de una imagen de varios megapíxeles antes de la red; probada en estimación de la mirada y clasificación fina; degrada a muestreo uniforme si hay incertidumbre — [Learning to Zoom, arXiv](https://arxiv.org/abs/1809.03355).
- [A 2021] FOVEA (Thavamani, Li, Cebron, Ramanan; arXiv 2108.12102, ICCV 2021 según el buscador): deformación guiada por saliencia tomada de un *prior* espacial del conjunto o de **las detecciones del fotograma anterior**. Argoverse-HD tras afinar (Tabla 1; AP / AP de pequeños / latencia): línea base a 0,5× 24,2 / 4,9 / 50,9 ms; FOVEA 28,1 / 10,3 / 50,5 ms; resolución completa 33,3 / 16,8 / 135,0 ms. Duplica el AP de objetos pequeños con ~2 ms extra, pero se queda lejos de la resolución completa — [FOVEA, ar5iv](https://ar5iv.labs.arxiv.org/html/2108.12102).
- [A 2023] Learning to Zoom and Unzoom (Thavamani, Li, Ferroni, Ramanan; CVPR 2023): amplía, calcula características y «deshace» la deformación con un mapeo bilineal por tramos invertible, de modo que sirve para cualquier tarea con salida espacial (detección 2D, segmentación, detección 3D monocular) — [LZU, arXiv](https://arxiv.org/abs/2303.15390).
- [A 2022] SAHI como alternativa de ventanas sin tocar el modelo (cifras en la pregunta 1) — [SAHI, arXiv](https://arxiv.org/abs/2202.06934).

**Poda o reducción de *tokens* en Transformers de vídeo**

- [R 2024] Run-Length Tokenization (Choudhury, Zhu, Liu, Niinuma, Kitani, Jeni; NeurIPS 2024): fusiona en un solo *token* los parches que se repiten en el tiempo. Reduce un 30 % el tiempo de afinado sin pérdida, sube el rendimiento un 35 % sin reentrenar con 0,1 % de caída de exactitud, y recorta hasta un 80 % de los *tokens* en vídeos largos — [RLT, NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3181db351fd3ced43cd589b0b572675d-Abstract-Conference.html).

**Recortes por jugador en lugar de fotograma completo**

- [R 2025] FOOTPASS reduce la retransmisión Full-HD a **352×640** y su línea base visual (TAAD: X3D-L + RoI Align sobre las trayectorias de los actores, clips de 50 fotogramas) obtiene F1 35,9; el salto a 67,5 viene del contexto táctico, no de más resolución (Tabla 2) — [FOOTPASS, arXiv](https://arxiv.org/html/2511.16183v1).

### Inferences

- Para la panorámica, la evidencia favorece un esquema en dos pasos (detector grueso → ventanas o recortes a resolución nativa) frente a reducir todo: SynLoc lo demuestra para atletas (+20 puntos entre la línea base reducida y los ganadores con ventanas) y SoccerTrack v2 lo razona para el balón (1 px tras reducir).
- La cámara de vídeo fija y casi estática hace que la reducción de *tokens* por repetición temporal sea especialmente atractiva: gran parte de la panorámica es césped inmóvil. No se encontró ninguna aplicación a deporte; es hipótesis.
- FOVEA aporta una idea trasladable: usar las detecciones del fotograma anterior (jugadores, balón) como mapa de saliencia para deformar o recortar, sin entrenar un selector. Coincide con la conclusión de AdaSpot de que los selectores aprendidos son inestables.
- AdaSpot degrada con dos RoI por fotograma, así que su diseño actual no cubre una panorámica donde balón y jugada relevante pueden estar separados; el recorte previo de la panorámica sigue siendo necesario antes de AdaSpot.
- Los sistemas de detección de acciones de balón operan con el fotograma de retransmisión a 796×448-1280×736, donde el operador ya ha hecho el *zoom*. Para reproducir esa escala desde una panorámica de 4096 px hay que recortar del orden de un tercio a una quinta parte del ancho; no hay medición publicada de cuánto rinde un modelo entrenado en retransmisión sobre esos recortes.
- La caída del GS-HOTA de 55 a 22 (sin atributos) entre 30 s y la parte completa indica que en panorámica el problema dominante del seguimiento es la asociación a largo plazo (AssA 63 → 11), no la detección (DetA 50 → 43).

### Gaps

- **No existe (o no se encontró) ninguna medición de T-DEED, AdaSpot ni E2E-Spot sobre panorámica completa o recortes de panorámica.**
- Las cifras de Cabado et al. (cámara pilotada por IA frente a realización profesional) no se pudieron leer (PDF ilegible para la herramienta; CVF 403): sin verificar.
- vid-TLDR (arXiv 2403.13347) y FrameFusion (arXiv 2501.01986) solo vistos en el buscador: sin verificar.
- SynLoc no publica tamaños de ventana ni tiempos por etapa; el artículo de Nguyen et al. solo se leyó en el resumen.
- SoccerTrack v2 no indica si su detector trocea la panorámica.
- No se encontraron ablaciones de resolución por encima de 224 px en detección precisa de eventos salvo las de AdaSpot.

---

## 6. Requisitos de tamaño: altura mínima de jugador y tamaño de balón por tarea

### Takeaway

No hay un estudio único que dé umbrales por tarea; las cifras medidas, reunidas, apuntan a que con jugadores de ~25×55 px (mediana en la panorámica de SoccerTrack v2, torso de 12×11 px) la detección funciona, la identidad y el dorsal no (GS-HOTA cae de 21,7 a 12,3 al exigir atributos), y la pose queda en la zona donde el AP se reduce a cerca de la mitad (personas de 44 ± 19 px: AP 35,9 frente a 62,4 a resolución completa). Para el balón, las redes multi-fotograma trabajan con 2-5 px a la entrada si hay movimiento, pero la localización 3D por tamaño necesita decenas de píxeles.

### Cited Findings

**Qué tamaños hay en panorámica y en cámara fija**

- [R 2026] SoccerTrack v2 (panorámica de 3840-4096 px de ancho): caja mediana de jugador **25×55 px** y región mediana del torso **12×11 px** (§4.7) — [SoccerTrack v2, v2 HTML](https://arxiv.org/html/2508.01802v2).
- [A 2019-2020] ISSIA-CNR (Full-HD fija): balón de unos 8 px en el lado lejano y ~20 px en el cercano según FootAndBall; «radio de 8-16 px» según DeepBall (las dos fuentes no coinciden en si es radio o diámetro) — [FootAndBall, ar5iv](https://ar5iv.labs.arxiv.org/html/1912.05445), [DeepBall, ar5iv](https://ar5iv.labs.arxiv.org/html/1902.07304).
- [A 2022] Baloncesto con cámaras fijas de 2-5 Mpx (DeepSport): balón de 14-45 px; APIDIS (800×600): 5-27 px (Tabla 1) — [Van Zandycke y De Vleeschouwer, ar5iv](https://ar5iv.labs.arxiv.org/html/2204.00003).
- [R 2025] En SynLoc, un error de un píxel en la imagen 4K original equivale como máximo a 30 cm en el campo, incluso en el lado lejano (§7.2, Fig. 5) — [SynLoc, SciTePress](https://www.scitepress.org/publishedPapers/2025/131082/pdf/index.html).

**Detección de balón**

- [A 2023] WASB, TOTNet, BlurBall y TrackNetV5 operan a 288×512 (o 512×288) desde vídeo de 720p-1080p, es decir, con el balón reducido 2,5-3,75 veces, y con 3-5 fotogramas de entrada — [WASB, ar5iv](https://ar5iv.labs.arxiv.org/html/2311.05237), [TOTNet, arXiv](https://arxiv.org/html/2508.09650v1), [BlurBall, arXiv](https://arxiv.org/html/2509.18387v2), [TrackNetV5, arXiv](https://arxiv.org/html/2512.02789).
- [A 2019-2020] Los detectores por fotograma de fútbol que trabajan a resolución nativa de 1080p con balón de 8-20 px dan AP 0,877 (DeepBall) y 0,909 (FootAndBall) en ISSIA — [FootAndBall, ar5iv](https://ar5iv.labs.arxiv.org/html/1912.05445).
- [A 2019] BallSeg entrena con aumento de escala que mantiene el balón entre 15 y 45 px — [BallSeg, ar5iv](https://ar5iv.labs.arxiv.org/html/2007.11876).
- [A 2022] Estimación del diámetro: los recortes de 64×64 px dieron la mejor exactitud (Fig. 7) y, entrenando a media resolución (balones de 7-19 px), la distribución del error conserva una dispersión similar (Fig. 6; sin cifras en el texto) — [Van Zandycke y De Vleeschouwer, ar5iv](https://ar5iv.labs.arxiv.org/html/2204.00003).
- [R 2025] Localización 3D por tamaño: sensibilidad de 6-14 m ante un 10 % de variación del tamaño en píxeles (Fig. 6) — [SoccerNet-v3D, arXiv](https://arxiv.org/html/2504.10106v1).

**Detección y localización de jugadores**

- [R 2025] Con atletas en 4K reducidos 6× (640 px) el mAP-LocSim es 60,6-67,8; reducidos 4× (960 px), 72,6-79,3; un modelo *tiny* a 960 supera a uno *m* a 640 (Tabla 2) — [SynLoc, SciTePress](https://www.scitepress.org/publishedPapers/2025/131082/pdf/index.html).
- [A 2021] Detección de personas (Faster R-CNN, COCO): AP 0,545 a escala original, 0,507 a 1/2 y 0,387 a 1/4; para personas pequeñas (área ≤ 32² px) AP 0,394 a 1/2 y 0,322 a 1/4 (Tablas 1-3) — [Hardy et al., ar5iv](https://ar5iv.labs.arxiv.org/html/2107.02108).

**Pose**

- [A 2019] PifPaf (Kreiss, Bertoni, Alahi; EPFL): limitando el lado largo a 321 px «para emular un recorte de una cámara 4K», las personas medianas de COCO quedan en **44 ± 19 px de alto**; ahí el AP de personas medianas es 35,9 (PifPaf), 28,2 (Mask R-CNN reentrenado) y 25,0 (OpenPose), frente a 62,4, 58,0 y 57,1 a resolución completa (Tablas 1-2). La introducción sitúa el caso difícil en alturas de 30-90 px — [PifPaf, ar5iv](https://ar5iv.labs.arxiv.org/html/1903.06593).
- [A 2021] Hardy, Dasmahapatra y Kim: con cajas reales y HRNet, el AP de puntos clave baja de 0,722 (COCO a 1/2) a 0,538 (a 1/4); la superresolución lo recupera a 0,649 a 1/4 (Tablas 5-6), ayuda por debajo de unos 3 500 px² de área de segmentación y perjudica por encima — [Hardy et al., ar5iv](https://ar5iv.labs.arxiv.org/html/2107.02108).

**Dorsal e identidad**

- [R 2026] En la panorámica de SoccerTrack v2, exigir atributos (dorsal, rol, equipo) baja el GS-HOTA de la parte completa de 21,66 a 12,25 y el de los primeros 30 s de 55,33 a 23,21, con torso mediano de 12×11 px (Tabla 3, Tabla suplementaria 2) — [SoccerTrack v2, v2 HTML](https://arxiv.org/html/2508.01802v2).
- [R 2024] Koshkina y Elder (arXiv 2405.13896): clasificador de legibilidad + recorte de torso guiado por pose + PARSeq. En hockey solo ~5-6 % de las imágenes son legibles (7 787 de 132 983); exactitud por *tracklet* en SoccerNet 87,45 % (test) y 79,31 % (reto), en retransmisión. No analiza exactitud frente a tamaño; anota que las imágenes de cámara estacionaria universitaria son «de menor resolución y con números más difíciles de descifrar» — [Koshkina y Elder, arXiv](https://arxiv.org/html/2405.13896).
- [A 2019] Re-identificación a baja resolución (Mao, Zhang, Yang; arXiv 1906.09748): con imágenes de 8 a 32 px de ancho (VR-Market1501) un ResNet50 da 57,0 % de rank-1 y el mejor método 66,9 %; con consultas reducidas a 1/2-1/4 (MLR-CUHK03), 67,4 % y 73,3 % (Tabla 3) — [Mao et al., ar5iv](https://ar5iv.labs.arxiv.org/html/1906.09748).

**Reconocimiento y detección de acciones**

- [A 2020] TinyVIRAT (Demir, Rawat, Shah; UCF): clips de vigilancia reales de 10×10 a 128×128 px (media 70×70); F1 de 28,7-35,1 según el modelo (Tabla II). En UCF-101 reducido (Tabla IV): 112×112 → 84,72 %; **28×28 → 66,72 %** con interpolación bicúbica y 82,87 % con superresolución aprendida; **14×14 → 14,14 %** y 70,55 % — [TinyVIRAT, ar5iv](https://ar5iv.labs.arxiv.org/html/2007.07355).
- [A 2022] Detección precisa de eventos: 112 px frente a 224 px de alto, −2,6 a −7,6 mAP según el deporte (Tabla 3c) — [E2E-Spot, ar5iv](https://ar5iv.labs.arxiv.org/html/2207.10213).
- [R 2026] AdaSpot: la RoI de alta resolución recortada del fotograma de 796×448 y llevada a 112×112 es el óptimo en SN-BAS; recortes de 56² o de 224² rinden 1,5-2,4 mAP menos (Tabla 4) — [AdaSpot, arXiv](https://arxiv.org/html/2602.22073v1).
- [R 2024] Los sistemas ganadores de acciones de balón consumen el fotograma de retransmisión de 720p entero a 796×448 (T-DEED) o a resolución casi nativa en gris (1280×736, 720p) — [SoccerNet 2024, arXiv](https://arxiv.org/html/2409.10587), [lRomul, repositorio](https://github.com/lRomul/ball-action-spotting).

**Región para «encuadrar» sin ver el balón**

- [A 2023] Una caja de 300×300 centrada en el balón inferido desde los jugadores lo contiene el 80 % de las veces y una de 600×600 el 96 % (Tabla 5; unidades no declaradas) — [Kim et al., ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08206).

### Inferences

Tabla de trabajo (síntesis propia a partir de las cifras citadas; los umbrales son órdenes de magnitud, no mediciones directas en fútbol base):

| Tarea | Lo que indican las cifras | Situación en panorámica (~25×55 px por jugador, balón ~8 px) |
|---|---|---|
| Detección de jugador | Funciona con pocos píxeles si no se reduce más la imagen; la resolución pesa más que el modelo (SynLoc) | Viable a resolución nativa o por ventanas |
| Seguimiento a corto plazo | DetA 43-50, AssA 63 en 30 s (SoccerTrack v2) | Viable |
| Identidad a largo plazo / re-identificación | AssA 11 en la parte completa; rank-1 de 57-67 % con 8-32 px de ancho | No fiable sin más píxeles o sin señales no visuales |
| Dorsal | Torso de 12×11 px; con atributos el GS-HOTA cae a la mitad; incluso en retransmisión solo ~5 % de imágenes legibles | Inviable sin *zoom* óptico |
| Pose | 44 ± 19 px de alto → AP ~36 (frente a ~62 a resolución completa); ganancia de superresolución por debajo de ~3 500 px² | Zona degradada; útil como señal gruesa (orientación), no fina |
| Detección de acciones | 112 → 224 px de alto ya da +3 a +8 mAP; mejores sistemas a 448-736 px de alto sobre encuadre de retransmisión | Requiere recorte de la panorámica; el fotograma completo reducido deja el balón en ~1 px |
| Detección de balón | Redes multi-fotograma operan con 2-5 px a la entrada; detectores por fotograma, con 8-20 px | Viable por ventanas a resolución nativa, mejor con 3-5 fotogramas |
| Balón 3D por tamaño | 14-45 px y aun así ~10 % de la distancia | Inviable con 8 px; usar física de la trayectoria |

- La superresolución aprendida recupera mucho en reconocimiento de acciones (14×14: de 14 % a 71 %) y algo en pose; es una vía que la literatura respalda para personas pequeñas, pero la mejora de calidad de imagen está fuera del alcance de estas notas.
- Ninguna fuente abierta da la altura en píxeles de los jugadores en los vídeos de retransmisión con los que se entrenan T-DEED o AdaSpot, así que no se puede calcular con datos publicados qué tamaño de recorte de la panorámica iguala esa escala; hay que medirlo en los datos propios.

### Gaps

- **No se encontró ningún estudio que mida rendimiento frente a altura del jugador en píxeles para re-identificación, dorsal o detección de acciones en deporte.** Las cifras de re-identificación y de acciones a baja resolución son de vigilancia y de conjuntos genéricos.
- Koshkina y Elder no dan el tamaño de los recortes de SoccerNet ni la fracción legible en fútbol.
- SoccerTrack v2 no publica la exactitud de dorsales por separado ni la exhaustividad del detector.
- La definición de «diminuto» de AI-TOD (y su media de ~12 px) no se verificó en la fuente original.
- No hay cifra publicada del tamaño mínimo de balón para detección por fotograma en fútbol por debajo de 8 px.
- FOOTPASS y los informes de SoccerNet 2025-2026 no dan el tamaño de los recortes por jugador (tubos RoI) que usan TAAD o PAVE.
