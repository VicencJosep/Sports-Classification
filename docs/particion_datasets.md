# Partición de los datasets: qué partidos entrenan y cuáles evalúan

Documento de referencia, vigente desde el 8 de octubre de 2026. **Las listas de aquí son fijas y comunes a
todos los modelos** (detección de balón, acciones de balón, eventos, ajuste fino de T-DEED, etc.). Un
partido tiene un único papel por dataset:

| Papel | Para qué sirve | Qué NO se puede hacer con él |
|---|---|---|
| **Entrenamiento** | Entrenar pesos | Informar resultados sobre él |
| **Validación** | Elegir épocas, hiperparámetros y umbrales; desarrollar y depurar | Informar resultados "finales" (son optimistas) |
| **Evaluación** | Solo medir y informar | Entrenar, elegir umbrales, mirar errores para ajustar el modelo |

Si un modelo nuevo necesita más datos de entrenamiento, se toman de la lista de entrenamiento o de partidos
nuevos; **nunca de validación ni evaluación**. Al añadir un dataset se añade aquí su partición antes de entrenar.
Las listas largas están en `docs/particiones/*.txt`.

## 1. SoccerNet-v2 (eventos: tiros, goles, centros, córners, faltas, tarjetas, sustituciones…)

500 partidos de TV (2014-2017, 6 ligas) con `Labels-v2.json`. Los vídeos (720p) estaban en
`/mnt/datasets/SoccerNet/` (a 8-oct-2026 ese montaje no está disponible); las etiquetas, en `datasets/SoccerNet/`.

| Papel | Partidos | Fichero |
|---|---|---|
| Entrenamiento | Los **100** de los que salen los 11.401 clips de 6 s del SVC (`datasets/dataset_vjepa_backup/`) | `particiones/soccernet_v2_entrenamiento_svc.txt` |
| Evaluación | Los **6** partidos completos del benchmark de tiros (Levante–R.Madrid, Lazio–Roma, Bayern–Augsburg, PSG–Angers, Liverpool–Swansea, Arsenal–PSG) + los **57** partidos de `datasets/clips_analisis/` que no están en el entrenamiento (Arsenal–PSG está en los dos grupos: 62 partidos distintos) | `particiones/soccernet_v2_evaluacion.txt` |
| Contaminados | **13** partidos de `clips_analisis/` que también están en el entrenamiento del SVC: sus clips no valen para evaluar el SVC | `particiones/soccernet_v2_clips_analisis_contaminados.txt` |
| Sin asignar | Los ~380 restantes: disponibles para entrenamiento futuro | — |

Hechos que condicionan su uso:
- Las cifras de `clips_analisis/` publicadas hasta septiembre de 2026 (SVC, VLM) incluían esos 13 partidos
  contaminados (13 de 70). Para el SVC son optimistas; para los VLM (sin entrenar) no afecta.
- **T-DEED (`SoccerNetBall_test` y `challenge2`) preentrenó con SoccerNet-v2** (ver `estudio_deteccion_tiros.md`
  §6): ningún partido de v2 sirve para evaluar T-DEED ni sus ajustes finos. Esta partición solo vale para
  modelos entrenados por nosotros desde los clips (SVC, futuros).

## 2. SoccerNet Ball Action Spotting 2024 (7 partidos EFL: pases, conducciones, cabezazos, saques, tiros, goles…)

7 partidos completos de la EFL Championship 2019-20 (`datasets/SoccerNet/england_efl/2019-2020/`,
`Labels-ball.json` con resolución de ms). De aquí salen también `datasets/dataset_ball_actions/` (clips T=32
del R3D-18, con el nombre del partido en el fichero) y `ground_truth/`.

| Papel | Partidos |
|---|---|
| **Entrenamiento** | Blackburn Rovers–Nottingham Forest, Brentford–Bristol City, Hull City–Sheffield Wednesday, Leeds United–West Bromwich |
| **Validación** | Middlesbrough–Preston North End (incluye `ground_truth/`: sus primeros 20 min, 405 anotaciones revisadas a mano) |
| **Evaluación** | Stoke City–Huddersfield Town, Reading–Fulham |

Es exactamente la partición del checkpoint `SoccerNetBall_test` de T-DEED (entrenó con los 4; Middlesbrough
eligió la época), así que ese checkpoint se evalúa limpio en Stoke y Reading. Estado de los modelos actuales
respecto a esta partición:

| Modelo | Entrenó con | ¿Evaluación limpia? |
|---|---|---|
| T-DEED `SoccerNetBall_test` | los 4 de entrenamiento (val Middlesbrough) | Sí: Stoke, Reading (AP tiros 75,6 %) |
| T-DEED `SoccerNetBall_challenge2` (producción) | 6 (todos menos Middlesbrough) | **No**: no queda ningún partido EFL fuera de su entrenamiento; solo validación en Middlesbrough |
| R3D-18 `r3d18_ball_actions-v2.pt` (producción) | 6 (todos menos Middlesbrough, split `seed=0`) | **No**: entrenó con Stoke y Reading. El próximo reentreno debe usar solo los 4 de entrenamiento |
| V-JEPA2 + SVC | ninguno (clips de SoccerNet-v2) | Sí: los 7 |

`ground_truth/` es validación: sirve para desarrollar y comparar versiones (`scripts/evaluate_pipeline.py`),
no para la cifra final. Las cifras finales de acciones de balón y tiros en TV se dan sobre Stoke y Reading.

## 3. SoccerTrack v2 (cámara panorámica fija: 12 acciones de balón, posiciones de jugadores, pista de balón)

10 partidos universitarios (`datasets/soccertrack-v2/`, 130 GB con vídeo y GSR). Se adopta **tal cual la
partición oficial del SoccerTrack Challenge 2025**, que es la que usan las publicaciones del dataset:

| Papel | Partidos (id = etiqueta del paper) |
|---|---|
| **Entrenamiento** | 117092 (M1), 118575 (M3), 118576 (M4), 118577 (M5), 118578 (M6), 128058 (M8) |
| **Validación** | 117093 (M2), 132877 (M10) |
| **Evaluación** | 128057 (M7), 132831 (M9) |

Notas:
- Todo lo medido en octubre de 2026 sobre el 117093 (cámara virtual, detección de balón, cámara guiada por
  YOLO) es **validación**: sirve para desarrollar; la cifra final debe salir de M7 + M9.
- Los experimentos "T-DEED sin entrenar sobre los 10 partidos" no contaminan nada (no se entrenó), pero si se
  ajusta T-DEED con SoccerTrack, solo con los 6 de entrenamiento.
- Montajes de cámara: 117092 (3840×1906) y 132831 (3840×1504) difieren de los otros 8 (4096×1080). Hay uno de
  cada en entrenamiento y evaluación, lo que es deseable.
- Las pseudoetiquetas de balón (pista `ball/*.npz` proyectada a píxeles) solo se generan para entrenamiento y
  validación; en M7 y M9 la pista del dataset se usa únicamente como verdad de referencia.
- `ball/*.npz` está reconstruida entre eventos anotados: nunca como entrada de un modelo (filtra la cronología
  de los eventos).

## 4. SoccerNet Tracking (cajas de balón en TV)

Ya se usó: el detector de balón `models/ball/yolo11s_snt_ball.pt` se entrenó con él (octubre de 2026). Se adopta la
partición oficial del dataset (`train` / `test`, por secuencias de 30 s de partidos de la Swiss Super League): se
entrena con `train` y se mide en `test`. Sus partidos no coinciden con ninguno de los de evaluación de la sección 1
(otra liga). Si se vuelve a entrenar el detector único (TV + panorámica), `test` sigue siendo solo de medida.

Nota sobre el **Clásico 2016/17** (Real Madrid 2-3 Barcelona, `datasets/clasico_2017/`, también en
`datasets/SoccerNet/spain_laliga/2016-2017/2017-04-23 - 21-45 Real Madrid 2 - 3 Barcelona/Labels-v2.json`): está en
SoccerNet-v2, con el que T-DEED preentrenó, así que para T-DEED es **in-sample** (cifras optimistas). Se usa como
única verdad de córner, falta y banda en TV y para desarrollar highlights (validación), nunca como evaluación.

## 6. Partidos propios (juveniles, cámara única; vídeos de YouTube)

Son el dominio del producto y no están en ningún dataset público. Papel fijado el 2026-10-21:

| Papel | Partidos |
|---|---|
| **Validación** | Mallorca–Barcelona juvenil (`433f62c92b727870` en el backend; 3 goles) y Espanyol–Damm juvenil (`f2c9315d3394c6ee`; 0 goles; 30 fps). Ya sirvieron para fijar umbrales de highlights (`goal_audio_min_db`, anclas) y para mirar errores: no valen como evaluación. La anotación manual de acciones y eventos (`output/annotations/<partido>_Ground_Truth.csv`, lector `scripts/annotation_csv.py`) se hace sobre estos dos. |
| **Evaluación** | Dos juveniles nuevos, sin abrir ni mirar hasta que el pipeline esté cerrado; solo se miden (acciones, eventos, goles y highlights) con el marcador o la anotación como verdad. Pendiente de elegirlos. |
| **Entrenamiento** | Ninguno de momento. Si se afina T-DEED con juveniles (`SNB_tv1`), los partidos anotados para ello serán nuevos y distintos de los dos de validación y los dos de evaluación. |

## 5. Resumen por modelo que se vaya a entrenar

| Modelo | Entrena con | Valida con | Evalúa (cifra final) en |
|---|---|---|---|
| Detector de balón (panorámica) | SoccerTrack entrenamiento (6) | 117093, 132877 | 128057, 132831 |
| Detector de balón único (TV + panorámica) | lo anterior + SoccerNet Tracking (train) | ídem + SoccerNet Tracking (val) | ídem + SoccerNet Tracking (test) |
| Acciones de balón (R3D-18, reentreno) | EFL: Blackburn, Brentford, Hull, Leeds | Middlesbrough / `ground_truth/` | Stoke, Reading |
| T-DEED ajustado con SoccerTrack | SoccerTrack entrenamiento (6) | 117093, 132877 | 128057, 132831 |
| Eventos (SVC o sucesor) | SoccerNet-v2: los 100 del SVC (+ sin asignar) | — (fijar al reentrenar) | los 6 + 57 de `soccernet_v2_evaluacion.txt` |
