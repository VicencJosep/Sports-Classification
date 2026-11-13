# PAVE (ganador de PCBAS 2026): qué es, si podemos replicarlo y qué trasladar a T-DEED y AdaSpot [2026-11-12]

Valoración sin experimentos (0 GPU). Fuentes: `docs/papers/SoccerNet-2026-results-2607.07320v1.md` §3 y §8.2, el informe
técnico de PAVE ([arXiv 2606.28389](https://arxiv.org/abs/2606.28389), 2 páginas), el paper de FOOTPASS
([arXiv 2511.16183](https://arxiv.org/abs/2511.16183)), el repo de los baselines (`JeremieOchin/FOOTPASS`) y nuestras
medidas de §7.7, §7.12, §7.14 y §7.15 de `docs/comparativa_benchmark.md`. Convenciones: [P] publicado, [M] medido por
nosotros, [NV] no verificado / estimado.

## 0. Respuesta corta

- **PAVE no es un detector de eventos sobre vídeo ni un tracker**: es la cadena TAAD → DST del propio dataset con tres
  retoques (transformer temporal por tracklet, atención entre jugadores en el DST y ensemble con filtro de acuerdo). Su
  entrada es el **tracking certificado de FOOTPASS** (posición métrica, velocidad, equipo, dorsal verificado contra la
  alineación y rol táctico de los 22 jugadores, frame a frame). No aporta nada al tracking: lo asume resuelto.
- **Lo que da los puntos no es la visión por jugador sino el estado del juego** [P]: en el test de FOOTPASS el TAAD visual
  solo hace Macro-F1 23,8; con el GNN de posiciones 34,5; con el DST 46,4; PAVE 58,9. Nuestros spotters de imagen completa,
  fuera de dominio y sin jugador, ya están en 43-48 de Macro-F1 en los mismos 3 partidos de validación [M].
- **Replicar PAVE entero no compensa**: código base público pero sin pesos, detalles clave sin publicar (pérdida,
  muestreo, ventanas), 1-2 días de GPU por pasada completa [NV], y el resultado exige en inferencia un estado del juego
  que nuestro pipeline no produce con esa calidad (equipo 62-68 % en juveniles [M], dorsal sin medir, sin roles).
- **Lo que sí se traslada**, por orden de coste: (1) el filtro de acuerdo a nivel de evento con excepción para clases
  raras en nuestra fusión T-DEED + AdaSpot-b (0 GPU, cachés existentes); (2) **estado del juego como entrada de la
  segunda etapa** (`stack_scores.py`), primero con los datos tácticos de FOOTPASS val (0 GPU) y luego con nuestro
  tracking; (3) la misma segunda etapa entrenada con el train de FOOTPASS (ya propuesto en §7.15; ~12 h de GPU de
  inferencia [NV]). Todo ataca el cuello de botella medido: la **etiqueta** de eventos ya detectados (§7.15 §6).
- Del tracking nos llevamos otra cosa: FOOTPASS val son 3 partidos de TV completos con cajas, equipo, dorsal y posición
  métrica por jugador, ya en disco: la primera verdad (plateada) para medir equipo, dorsal y homografía de nuestra cadena
  en partidos enteros.

## 1. Qué es PAVE [P]

| Etapa | Baseline de FOOTPASS | Lo que cambia PAVE |
|---|---|---|
| **Entrada** | Vídeo 352x640 + datos tácticos del dataset: por frame y jugador `left_to_right, shirt_number, role_id, x, y, speed_x, speed_y, roi_*` (cajas de un solo jugador). Sin balón. | Igual. Los tracklets, el equipo, el dorsal y el rol **vienen del dataset** (pipeline propio de los autores, «FIFA EPTS certified», dorsales contrastados con la alineación). |
| **TAAD** (visión por tracklet) | X3D-L, clips de 50 frames (2 s), RoI-Align sobre la caja de cada jugador, TCN, softmax de 9 clases por jugador y frame. ≤ 500 clips por clase y época, 4-5 tracklets de fondo por acción. | Transformer temporal (4 capas, d = 256, 8 cabezas, ≤ 50 frames) sobre las features RoI; peso del fondo 0,05; 4 → 6 tracklets por clip; arreglo de un bug del warmup. |
| **DST** (estado del juego) | Transformer codificador-decodificador (6 capas, 512) sobre ventanas de 750 frames (30 s): entrada = 26 huecos de rol × (5 canales de estado + 9 logits de TAAD); el decodificador genera tuplas (frame, acción, rol) de forma autorregresiva. | Antes del codificador, **atención entre los 26 jugadores en cada frame** (d = 64, 4 cabezas) y después atención temporal por jugador; el orden espacial-primero es el que funciona. |
| **Ensemble** | — | 4 DST (capacidades y atención distintas). Fusión por evento: agrupar predicciones de la misma (equipo, dorsal, clase) a ±12 frames; puntuación = media × √(n/N); **descartar lo que apoya un solo modelo**, salvo tackle. Umbrales por clase descartados (sobreajustan a las 6 mitades de validación: 48,0 frente a 48,6 en challenge). |

Cifras [P], Macro-F1 (validación = las 6 mitades de game_18/24/47, las mismas que usamos; challenge = game_0/36/50):

| Paso | Val | Challenge |
|---|---|---|
| Baseline TAAD + DST (organizadores) | — | 46,4 |
| DST con el TAAD mejorado | 54,1 | 48,6 |
| + atención por jugador | 54,9 | — |
| + orden espacial-primero | 56,7 | 55,1 |
| Ensemble de 4 | 60,9 | 58,9 |

Lo que el informe **no** da: pérdidas, muestreo de clips, longitud de ventana del TAAD, horas de GPU, optimizador del
TAAD, y ningún código ni pesos (comprobado en §7.7). Las ablaciones se apoyan en 6 mitades de validación; la distancia
val → challenge pasa de ~6 puntos (modelo solo) a ~2 (ensemble), es decir, el ensemble es sobre todo reducción de varianza.

## 2. La comparación en los mismos partidos [M] frente a [P]

Novedad respecto a §7.7 (que comparaba sobre partidos distintos): la validación de PAVE son **nuestros 3 partidos de
FOOTPASS val**, con la misma tolerancia (±12 frames = ±0,48 s frente a nuestros ±0,5 s). F1 por clase; la nuestra sale de
P/R de §7.12 al umbral global óptimo de cada modelo (ligeramente optimista) y **sin exigir jugador** (cota superior);
la de PAVE exige equipo y dorsal (pesimista para ellos).

| Clase (nº en val) | PAVE ensemble (con jugador) | T-DEED 200MF (solo clase) | AdaSpot-b (solo clase) |
|---|---:|---:|---:|
| Pase (3.059) | 84,0 | 71 | 76 |
| Conducción (2.470) | 83,9 | 61 | 72 |
| Centro (111) | 76,4 | 52 | 47 |
| Saque de banda (97) | 82,8 | 53 | 45 |
| Tiro (67) | 67,7 | 54 | 30 |
| Cabezazo (162) | 45,0 | 43 | 52 |
| Bloqueo (78) | 35,4 | 34 | 20 |
| Entrada (26) | 12,3 | 12 | 0 |
| **Macro-F1** | **60,9** | 47,6 | 43,0 |

Lectura, con la cautela de que PAVE entrena **en dominio** (48 partidos de las mismas ligas y temporada) y nosotros con 4
partidos de la EFL de 2019:

- PAVE gana de largo donde **el estado del juego lleva la clase**: saque de banda (+30-38; jugador fuera del campo),
  centro (+24-29; posición en banda), conducción frente a pase (+12-23; quién recibe), tiro (+14-38; posición y dirección
  hacia portería).
- **Empata o pierde donde la clase es visual y de contacto**: cabezazo (45 frente a 43-52), bloqueo (35 frente a 20-34),
  entrada (12 frente a 0-12). Son los límites que los propios autores y el resumen del challenge reconocen (tackle, oclusiones,
  huecos en los tracklets), y coinciden con nuestras clases raras.
- Los baselines de FOOTPASS en test (Macro-F1 [P], calculada de sus F1 por clase): TAAD 23,8 → +GNN 34,5 → +DST 46,4. La
  visión por tracklet sola queda **por debajo de nuestros spotters de imagen completa** (43-48 fuera de dominio); las
  posiciones de los 22 jugadores valen +23 puntos. Esto es lo que decide qué trasladar.

## 3. Replicar PAVE desde cero: qué hay y qué falta

| Pieza | Tenemos | Falta | Coste [NV] |
|---|---|---|---|
| Datos de entrenamiento | Token HF aprobado; val (3 partidos, Full HD + tácticos) en disco | Train: `videos_352x640_TRAIN.zip` 22,8 GB + `tactical_data_TRAIN.zip` 2,5 GB (basta para TAAD, que entrena a 352x640); Full HD son 184 GB | descarga; 803 GB libres |
| Evaluación | `evaluation.py` oficial y las etiquetas de val | Challenge sin etiquetas (h5 de 13 columnas): solo se evalúa en val, los mismos 3 partidos que se usan para elegir época | — |
| Código de TAAD y DST | Público (`train_TAAD_Baseline.py`, `train_DST.py`, conversores), sin pesos | Entrenar TAAD (X3D-L, A6000 en el paper), pasarlo por los 48 partidos (`run_TAAD_on_matches.py`) y entrenar DST | TAAD ~2-4 h; inferencia sobre 96 mitades 4-8 h; DST 1-3 h por variante; **1-2 días de GPU por pasada completa**, y harán falta varias |
| Retoques de PAVE | Descripción de 2 páginas | Transformer temporal en TAAD, atención por jugador en DST, fusión por evento: ~500 líneas sobre el baseline, con la pérdida y el muestreo adivinados | 1-2 semanas de ingeniería |
| Referencia para saber si está bien | Las tablas del informe | Sin pesos de referencia; con 6 mitades de validación una diferencia de 2-3 puntos es ruido | — |

Veredicto: **no**. Lo esperable es quedarse entre el baseline (46) y el ensemble (59) sin saber por qué, y el producto
final pide en inferencia un estado del juego (equipo, dorsal y rol de los 22, posiciones métricas sin huecos) que nuestro
pipeline no da: equipo por k-means 68 % en Middlesbrough y 62 % en el Espanyol [M], dorsal por detección sin votación y
sin medir, roles solo portero/jugador/árbitro, posiciones de PnLCalib sin medir contra verdad. La etapa que vale +23
puntos (DST) recibiría una entrada muy distinta de la que la hizo funcionar. Replicar solo el TAAD tampoco: con tracklets
perfectos se queda en 23,8.

## 4. Qué partes sí se trasladan a T-DEED y AdaSpot

Ordenadas por coste. Ninguna toca el backbone de los spotters: §7.14 y §7.15 ya mostraron que el recorte (AdaSpot) no es
lo que frena y que detectar está casi resuelto (AP 92 sin clase); se pierden ~20 puntos de mAP en la etiqueta.

1. **Fusión por evento con filtro de acuerdo y excepción para clases raras** (regla del ensemble de PAVE). Coste: 0 GPU,
   ~medio día sobre `fuse_scores.py` (hoy promedia frame a frame; PAVE agrupa eventos ya decodificados y descarta los que
   apoya un solo modelo). Lo que ya sabemos [M]: la media simple da +4,2 de mAP y −48 % de falsos positivos en FOOTPASS;
   la media geométrica (acuerdo frame a frame) empeora (50,4). Diferencia con PAVE: su acuerdo es por evento y exceptúa la
   clase rara; aquí habría que exceptuar lo que solo T-DEED ve (tiro 54 % frente a 0 %, saque de banda 30 % frente a 0 %
   en FOOTPASS, §7.15 §3). Expectativa honesta: falsos positivos −10-20 %, mAP ±1-2 [NV]. Rápido de descartar.
2. **Estado del juego como entrada de la segunda etapa** (es lo que hace el DST, y la propuesta 3 de §7.15 §8). Hoy
   `stack_scores.py` lee solo curvas de probabilidad (±5 s). Añadir por frame: posiciones y velocidades de los jugadores
   (o resumen: nº de jugadores por zona, velocidad y posición del más cercano al pico, distancia al rival más próximo,
   jugador fuera de banda), y en nuestro pipeline además el balón y la posesión (FOOTPASS no trae balón). Dos medidas, en
   este orden:
   - **Cota superior con los datos tácticos de FOOTPASS val** (h5 ya en disco, 2 partidos entrenan y 1 predice, como en
     §7.15 §7): 0 GPU, 1 día. Si saque de banda, centro, conducción y tiro suben como en la tabla de §2, el estado del
     juego es la palanca; si no suben, se cierra la vía antes de descargar nada.
   - **Con nuestro tracking** sobre los mismos 3 partidos: dice cuánto se pierde con nuestras posiciones (sin rol ni
     dorsal: la atención espacial de PAVE no necesita identidad, solo la salida «quién» la necesita).
   Límite conocido [M]: con 2 partidos la segunda etapa no aprende clases con < 100 ejemplos; con estado del juego cada
   ejemplo informa más, pero las clases de contacto (bloqueo, entrada, cabezazo) tampoco mejoran en PAVE.
3. **Segunda etapa con el train de FOOTPASS** (propuesta 1 de §7.15 §8, pendiente de tu visto bueno): puntuar los 48
   partidos con los spotters congelados (~12 h de GPU a 4-7 GB [NV]) y entrenar en CPU con curvas + datos tácticos del
   train (2,5 GB, sin vídeo). Es el equivalente exacto de PAVE para nosotros: nuestros spotters hacen de TAAD (mejor que
   él) y la segunda etapa de DST. Antes, un test de 20-30 min de GPU: puntuar un partido de val a 352x640 (zip de 1,4 GB)
   y comparar con Full HD (T-DEED mAP 50,0 / F1 64,0); decide entre descargar 25 GB o 184 GB.
4. **Pesos de clase y muestreo** (peso del fondo 0,05 en PAVE; raíz de la frecuencia en PCBAS-7; ≤ 500 clips por clase en
   TAAD): es la propuesta 2 de §7.15 §8 (cabeza de T-DEED con pérdida equilibrada y SoccerTrack v2 mezclado). PAVE no
   aísla su ganancia (entra en el +2,2 de 46,4 → 48,6 junto con bugs y scheduler). Modesto; va detrás de 2 y 3.
5. **Contexto largo y decodificación por secuencia** (ventanas de 30 s y tuplas autorregresivas del DST, frente a ±5 s y
   convolución en `stack_scores.py`): ampliar el contexto de la segunda etapa a 20-30 s con un GRU bidireccional o un
   transformer pequeño es barato en CPU; la gramática implícita (un pase lo sigue una recepción) es parte de lo que el DST
   aprende. [NV], después de 2.
6. **Lo que no se traslada**: el transformer temporal por tracklet (requiere la rama por jugador de TAAD, que no tenemos y
   que sola rinde peor que nuestros spotters), la atención sobre 26 huecos de rol (sin roles), umbrales por clase
   (PAVE los descarta; coincide con §7.15: pesos por clase ±0,5), y ensembles de variantes del mismo modelo (§7.15: un
   segundo T-DEED no aporta; la ganancia viene de mezclar arquitecturas).

## 5. Y el tracking

- **De PAVE, nada**: consume el tracking del dataset. El resumen del challenge sí dice qué falta con tracking certificado:
  huecos en los tracklets e identidad a través de oclusiones (tackle). Es el mismo problema que tenemos (reentradas,
  §7.10) y la vía ya está en la lista: Deep-EIoU y GTA-Link (+1,9 HOTA offline, §7.4).
- **De FOOTPASS, un banco de pruebas de partidos enteros**: val trae, para 3 partidos de TV completos en 1080p, caja,
  equipo, dorsal, rol y posición métrica de cada jugador en cada frame. Es verdad **plateada** (salida de su pipeline,
  posiciones imputadas en los huecos, sin corrección manual declarada), pero sirve para medir por primera vez, en partidos
  enteros y no en clips de 30 s, la exactitud de equipo, dorsal y `field_xy` de nuestra cadena, y los cambios de identidad
  a lo largo de 90 min. Coste: una pasada del pipeline por partido (la homografía es la etapa cara; a medir), 0 descargas.
- Con el `tactical_data_TRAIN.zip` (2,5 GB, sin vídeo) se puede entrenar en CPU un clasificador de **rol táctico** (13
  roles) a partir de posiciones, y un modelo de asignación de equipo/posesión por geometría. Útil para «quién» en los
  eventos; fuera del objetivo de esta valoración.

## 6. Orden propuesto

| # | Qué | GPU | Decide |
|---|---|---|---|
| 1 | Fusión por evento con filtro de acuerdo (§4.1) sobre las cachés de SN-BAS test y FOOTPASS val | 0 | si el acuerdo quita falsos sin perder tiro y banda |
| 2 | Segunda etapa con datos tácticos de FOOTPASS val, 2 entrenan y 1 predice (§4.2) | 0 | si el estado del juego es la palanca (banda, centro, conducción, tiro) |
| 3 | Test 352x640 frente a Full HD en un partido de val | 20-30 min | 25 GB o 184 GB de descarga |
| 4 | Train de FOOTPASS puntuado con los spotters + segunda etapa con estado del juego (§4.3) | ~12 h [NV] | la versión «PAVE para nosotros» |
| 5 | Medir equipo, dorsal y homografía de nuestra cadena en FOOTPASS val (§5) | una pasada por partido | base para «quién» y para la métrica oficial de PCBAS con nuestra cadena (§7.7, punto 3) |

Los pasos 3-5 usan la GPU compartida: no se lanzan sin visto bueno.
