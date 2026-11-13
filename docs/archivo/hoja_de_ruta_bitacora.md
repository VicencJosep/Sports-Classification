# Bitácora de la hoja de ruta

> Entradas completas de [docs/hoja_de_ruta.md](../hoja_de_ruta.md) §7, de la más antigua a la más reciente. Cada sesión
> añade aquí su entrada al final y una línea en el §7 de la hoja de ruta.

- **2026-10-19**: hoja de ruta inicial (workflow de 12 agentes). Hallazgos: los CSV de `output/annotations` eran volcados
  del pipeline; el Clásico está en SoccerNet-v2 con `Labels-v2.json`; T-DEED ciego en juveniles (no es umbral); equipo de las
  acciones ~68 %; GT principal fuera de git y sin generador; `/mnt/datasets` montado en el host.
- **2026-10-20**: segunda ronda (5 agentes) sobre las ideas del usuario: futtraffic no es cámara virtual (sí: `autocal`,
  `pancheck`, puertas de balón); duración de clips diagnosticada (5/19 en 12-30 s; causa: replays de TV + tope 45 s); plan
  de homografía fija + tracking por ventanas medido con GSR; experimento barato `SNB_pano1` en TV definido; literatura VLM:
  sin evidencia de que un VLM mayor puntúe jugadas mejor que reglas + marcador. El usuario anota el Espanyol 25:00-30:00
  (`output/annotations/espanyol_damm_Ground_Truth.csv`, formato `tiempo,etiqueta,dorsal(=tracker_id),equipo`). Decisiones del
  usuario: main debe funcionar bien; panorámica sigue en rama; clips 12-30 s con excepciones. Documento creado; resumen del
  mecanismo de selección añadido a `docs/highlights_evaluacion.md`.
- **2026-10-21** (sesión en `feat/highlights`; el usuario anota el Espanyol en paralelo): carril B, semana 1. (1) Lector del
  CSV (`scripts/annotation_csv.py`), arnés único (`scripts/evaluate_match.py`) y generador de GT reescrito
  (`scripts/build_ground_truth.py`: Labels-ball, Labels-v2 con desfase por mitad, CSV; fps real por `video.json`/ffprobe);
  `ground_truth/*.gt.json` ya van a git. Middlesbrough reproduce 87,7/89,1; equipo 68,1 %. Espanyol 25-30 (hoja a medias,
  126 filas): F1 82,9, equipo 62,1 % (los clústeres de prtreid son impuros: 51 % y 76 %), tracker 91,5 %: el fallo de
  equipo está en el agrupamiento, no en el poseedor. (2) Política de duración de clips implementada y medida sin GPU
  (`--vlm-cache-only`): 20/20 en rango, 8/8 goles, 0 falsos; dos arreglos de paso (gol al candidato cercano al rótulo;
  nada de re-anclar por celebración en TV). (3) `foul` marcado `estimated` en telemetría y API; origen del gol y estado
  degradado del VLM en `HighlightsResponse` (backend: 22 tests OK). (4) Partición de juveniles y nota del Clásico en
  `docs/particion_datasets.md`. (5) Carril C: cadena `run_pano1_tv.sh` lanzada y esperando VRAM (GPU a 23/24 GB ajena).
  Hallazgo [M]: etiqueta spotter frente a combinada, +2,7 F1 en Middlesbrough y −8,5 en Espanyol (ver §2.1). Sin medir
  aún: córner/falta/tiro en el juvenil (la hoja no tiene eventos todavía). Commits del usuario: 61be4fe (arnés y GT) y
  8a56011 (duración de clips). Más tarde la GPU se liberó y la cadena §3.6 corrió sola (52 min): `SNB_pano1` ve los 3 goles
  del Mallorca como tiro (> 0,4) pero hunde la EFL (tiros −11 pt, header desaparece, cabeza SN sin "Goal"); `SNB_tv1` queda
  condicionado a mezclar EFL en cada lote y a tener tiros anotados en juveniles (tabla en §3.6).
- **2026-11-02** (sesión desatendida): **modelo de SoccerHigh portado e integrado** (`docs/highlights_soccerhigh.md`).
  VideoMAEv2-Giant K710 + resumidor DSNet-AF en torch puro (`analytics/highlights/soccerhigh.py`, `videomaev2.py`); baseline
  reproducido con sus rasgos (45 test: F1 0,412 / F1@T 0,395 frente a 0,396 / 0,388) y con rasgos propios del 720p en 3 de
  test (0,368 / 0,365 frente a 0,353 / 0,346 con los suyos); `video.ini` sustituido por alineación de rasgos
  (`scripts/soccerhigh_align.py`, Clásico +968/+277 s, sin deriva). Clip = 16 fotogramas consecutivos a 2 fps (8 s), no a
  ritmo nativo. En el pipeline: etapa `soccerhigh` (rasgos cacheados junto al vídeo, ~8 min/partido), anclas por tramo,
  rasgo `soccerhigh` en reglas (peso 2,5), red de seguridad del VLM (≥ 0,75), `soccerhigh_score` en la API y badge en el
  frontend. Los 4 partidos regenerados: 8/8 goles, 0 falsos, 21/21 en rango; SoccerHigh ve los 5 goles del Clásico (0,64-0,88)
  y 2/3 del Mallorca, y nada en Espanyol ni Middlesbrough (sin goles). Contra la verdad de SoccerHigh en el Clásico (115 s de
  resumen): tramos del modelo F1 0,316, clips del frontend 0,256 (la política de 12-45 s los hace largos frente a planos de
  3-10 s). Arreglo de paso: el ranking del VLM se truncaba con ~30 candidatos (`rank_candidates`). Pendiente: afinar el
  resumidor con rasgos propios (train de SoccerHigh + juveniles anotados) y medir el pipeline entero en más partidos de test.
- **2026-11-06** (tracking, `main`): **ByteTrack sustituido por BoT-SORT** (ultralytics) en `core/tracking.py`: GMC
  sparseOptFlow, NMS previo 0,7, ReID OSNet x1.0 (`core/osnet.py`, pesos "sports" sin licencia en `models/reid/`) o PRTReId,
  cajas detectadas en vez de las del Kalman; `TrackingConfig` nuevo (`tracker`, `reid`, `nms_iou`, `lost_track_buffer=72`,
  `reid_lost_without_gate`, `appearance_thresh=0.95`); tests `tests/test_tracking_botsort.py`. Arnés: `track_mot.py` con
  variantes `ours_*`, `id_analysis.py` (IDSW por tercil de paneo, reentradas, fragmentación) y
  `run_snmot_train_tuning.sh` (12 seq de train; el test se usó una vez por candidata). Resultado [M], test 49 seq, mismas
  detecciones: ByteTrack 56,4 / 66,0 / 4.892 IDSW → GMC 57,5 / 67,0 / 3.565 (−32 % IDSW en el tercil con más paneo) →
  GMC+NMS+OSNet+perdidos por apariencia 0,95 **59,4 / 70,5 / 4.199** (por defecto). Hallazgos: el ReID de BoT-SORT no
  recupera reentradas (hueco mediano 4,4 s, 97 % por el borde; recupera 1-4 %, Deep-EIoU 15 %, GTA-Link offline 44 %); el
  detector a conf 0,1 duplica el 12 % de cajas (el NMS quita 16 k FP); PRTReId como ReID 58,7 a 98 ms/frame, descartado.
  Coste: tracker 0,8 → ~32 ms/frame (pipeline ~28 → ~15 fps; `reid="none"` ~20 fps). Pendiente: borrar la caché
  `*.tracks.jsonl` y reanalizar los partidos de TV para ver el efecto en equipo/ejecutor con `evaluate_match.py`; enlace
  offline tipo GTA-Link para las reentradas; licencia de los pesos OSNet antes de producto.
- **2026-11-08** (estado del arte en tracking): `docs/estado_arte_tracking.md`, con 60+ papers verificados en arXiv por 4 agentes
  (notas en `output/research_notes/tracking_sota_2026-11-08/`). Conclusiones: nuestros 59,4 HOTA están en la banda de
  SoccerMaster+MOTIP (59,1) y YOLOv8+PRTReID (59,8); el hueco hasta SportsSUSHI/Kalisteo (71-76) es detector + enlace a largo plazo
  con dorsal y equipo; la línea BIoU → EIoU (Deep-EIoU, ARC-SORT) es el siguiente escalón online; GTA-Link + dorsal/equipo para las
  reentradas; UCMCTrack, pseudo-profundidad y end-to-end descartados para TV. Pasos en §5 del documento.
- **2026-11-08** (benchmark de eventos, `docs/comparativa_benchmark.md` §7.11): tabla única de los spotters con pesos públicos
  (T-DEED `SoccerNetBall_test` / `challenge2` / `SNB_pano1`, sn-teamspotting, AdaSpot-b/-s, E2E-Spot de OSL) en el test
  oficial de SN-BAS (Stoke + Reading, 3.980 acciones; contaminación verificada en los splits de cada repo) y en la eval de
  SoccerTrack v2 con dos cámaras virtuales: la que sigue el balón anotado (`vcam796`) y la nuestra (`vcam796vl1`). Runner
  reanudable `scripts/benchmark/run_bas_views.sh`; `SNB_pano1` se carga ya desde `core.ball_spotting`. Resultado [M]: en TV
  nadie supera a T-DEED (61,2 mAP@1); en panorámica la cámara ideal da lo mismo que la nuestra (±2 pt en los 7 modelos;
  `SNB_pano1` 38,4 / 38,3), con +3 pt de mAP y +15 AP de tiros solo en 128057 (en 132831 la referencia está degradada): el
  hueco TV → panorámica es de dominio, la cámara solo cuenta en tiros. Coste con GPU libre: 12-14× tiempo real de punta a
  punta para todos los modelos a 796x448 (decodificación manda; el 800MF de producción solo +4 % en pared, 14 GB de VRAM).
  Búsqueda SOTA: SoccerMaster, UniSoccer y sn-gamestate no hacen spotting; ganadores de 2025-26 sin pesos; lRomul 2023
  solo PASS/DRIVE. Hallazgo colateral: T-DEED preentrenó solo con `train`+`val` de SoccerNet-v2, así que su `test` (100
  partidos, en /mnt) está limpio para la cabeza de 17 eventos (y los 6 del benchmark de tiros están en él); corregido en
  `docs/particion_datasets.md`. Decisión pendiente del usuario: piloto de 20 partidos de ese test (~3 h) o los 100 (~16 h).
- **2026-11-09** (punto de trabajo y metraje nuevo, `docs/comparativa_benchmark.md` §7.12): el usuario preguntó qué eran los
  porcentajes de las tablas (AP por clase a ±0,5 s, no precisión) y quién hace más falsos positivos. Nuevo
  `scripts/benchmark/eval_prf.py` (umbral global óptimo por modelo, P/R/F1/TP/FP/FN por clase, emparejamiento ±0,5 s 1 a 1)
  y tres tablas «AP · P/R · FP» para T-DEED 200MF, AdaSpot-b y AdaSpot-s: TV (SN-BAS test), panorámica (ST2 eval, nuestra
  cámara) y **SN-PCBAS-2026/FOOTPASS val** (3 partidos de 2023/24 de otras ligas, 6.070 eventos de 8 clases, descargado con el
  token del usuario; metraje nuevo para todos los modelos, también `challenge2`). Resultado [M]: en TV los tres emiten los
  mismos FP (552-654) y T-DEED gana por recall en clases raras (F1 80,5 vs 78,2); en panorámica AdaSpot-b casi no emite y
  AdaSpot-s es el más impreciso (1.688 FP); **en PCBAS T-DEED hace 2,3× más falsos positivos que AdaSpot-b** (2.600 vs 1.121,
  casi todos conducciones y pases) y pierde más punto de trabajo (F1 80,5 → 64,0 frente a 78,2 → 71,5; umbral óptimo 0,70 →
  0,20), aunque sigue ganando en mAP (50,0 vs 41,1) por tiro (59 vs 26), banda, centro y entrada. Lectura: T-DEED ordena mejor
  pero calibra peor fuera de la EFL → recalibrar umbral por partido o entrenar con más ligas. Cabeza de 17 eventos de SN-v2 (§7.13, piloto de 20 partidos
  limpios del test, evaluador oficial, `eval_snv2.py`): el 200MF da Average-mAP tight 21,4 / loose 25,8 frente a 62-73 de los
  sistemas dedicados; acierta el instante (desfase mediano < 0,2 s) pero dispara de más; córner (P 81 / R 76), banda, balón
  fuera y libre directo son utilizables con umbral propio; tiros flojos (el tiro debe salir de la cabeza de acciones); cambio,
  amarilla, fuera de juego y penalti inservibles en el 200MF (nunca pasan de 0,3). Modelo de producción `challenge2` (medido la misma noche): en SN-v2
  25,3 tight / 30,4 loose (+3,9; penalti 48 con 14 casos, tiro a puerta 33 frente a 20), mismas clases inservibles (tarjeta,
  cambio, fuera de juego, saque inicial: no mostrarlas como fiables en `main`); en PCBAS mAP@1 53,1 y F1 65,2 con 2.136 FP
  (umbral óptimo 0,25 frente a 0,55 en la EFL): también calibra mal fuera de la EFL, ~24 puntos menos de mAP que en
  Middlesbrough. Vídeos temporales: 32 GB de SN-v2 en `datasets/SoccerNet/<liga>/` y 10,6 GB de zip en
  `datasets/SoccerNet/pcbas-2026/raw/`, borrables cuando el usuario decida.
- **2026-11-09** (limpieza del repo, pedida por el usuario para gastar menos contexto; sin cambios de comportamiento,
  43 tests igual antes y después). Medido en 44 sesiones: el 59 % de los tokens de herramientas son salidas de Bash, el
  26 % búsquedas web y los docs solo el 1,5 %. (1) `basketball/` (3,7 GB) fuera de `main`, en
  `/media/dd3/ner-training/basketball_Sports-Classification`; la rama `basketball` conserva el código. (2) Docs partidos
  en resumen + detalle: `comparativa_benchmark.md` 119 → 26 KB (un fichero por resultado en
  `docs/comparativa_benchmark/`), `datasets.md` 119 → 24 KB (catálogo en `docs/datasets_catalogo/`; no puede llamarse
  `docs/datasets/` porque el `.gitignore` ignora cualquier carpeta `datasets/`), bitácora completa aquí;
  `estudio_deteccion_tiros.md` y `Propuestas-desarrollo-Highlights.md` a `docs/archivo/`. (3) README 74 → 6 KB; el uso por
  módulo pasa a `docs/uso/` y los problemas de dependencias a `docs/dependencias_y_problemas.md`. (4) Borrados
  `orquestrator/` (vacío) y `utils/legacy_inference_demo.py`. (5) `CLAUDE.md` nuevo con el mapa de lectura. (6) 19,8 GB de
  `tmp_pack_*` huérfanos de julio fuera de `.git` (38 → 18 GB). No se tocó: SoccerMaster (es el detector de personas por
  defecto, `--detector hybrid`; pesos en `Soccer_Master/sn-gamestate/pretrained_models`), `scripts/shot_benchmark/` y
  `scripts/ball_detector/` (`tune_tdeed_event_thresholds.py` y el afinado de T-DEED dependen de ellos, y sus lanzadores
  llevan rutas fijas) ni `automation/` (pendiente de que el usuario diga si lo usa).
- **2026-11-09** (artículos a Markdown, pedido por el usuario). Los tres PDF de `docs/papers/` tienen ahora un `.md`
  del mismo nombre: T-DEED (arXiv 2404.05392v2), AdaSpot (2602.22073v1) y SoccerNet 2026 Challenges Results
  (2607.07320v1); figuras originales en `docs/papers/img/<nombre>/` (20 MB). Se convirtieron desde la versión HTML de
  arXiv (generada del LaTeX: ecuaciones en LaTeX, tablas completas, secciones en orden) y no desde el PDF, porque la
  extracción directa del PDF rompía ecuaciones y mezclaba las dos columnas. Cotejado con el texto de los PDF: todos los
  números decimales y todos los grupos de citas coinciden. Tres diferencias del HTML corregidas con el PDF: en AdaSpot la
  numeración de las referencias 51↔52 y 59↔60 venía intercambiada (las entradas de su bibliografía quedan con el formato
  del HTML, iniciales en vez de nombres, pero con la numeración del PDF); las citas agrupadas se ordenan y comprimen como
  en el PDF; y en SoccerNet los índices de afiliación de los 105 autores y los separadores de las palabras clave se
  recuperaron del PDF. Observación del propio artículo de AdaSpot: los GFLOPs de AdaSpot-b son 56,78 en su Tabla 1 y 36,78
  en su Tabla 6.
  Más tarde, a petición del usuario, de SoccerMaster salió lo que no se usa: 16 GB a `/media/dd3/ner-training/Soccer_Master_sin_uso_Sports-Classification`
  (Qwen2.5-VL-7B de `sn-gamestate/pretrained_models/jn/`, código de sn-gamestate, tracklab salvo `trackeval`, sam2,
  images) y se borraron `infer_mp4.py`, `eval_captions.py`, `eval_classification_head.py` y las constantes
  `SOCCER_MASTER_*` de `config.py`. Criterio: traza en ejecución del detector (abre `backbone.pt`, las 5 cabezas y
  `google/siglip2-large-patch16-512`; nada más). Quedan 5 GB en `Soccer_Master/`.
- **2026-11-09** (RoI anotada frente a saliencia en AdaSpot, `docs/comparativa_benchmark.md` §7.14; rama
  `exp/adaspot-new-roi`). Encargo del usuario («Fase 1»): medir la «ceguera espacial» del selector de RoI no supervisado
  de AdaSpot y el techo con RoI perfecta, con los pesos congelados, en SN-BAS test, FOOTPASS val y SoccerTrack v2 eval.
  Tres premisas del encargo no se cumplen en los datos [M]: **SN-BAS no trae coordenadas de nada** (solo instante, clase
  y equipo; el artículo de AdaSpot lo reconoce en su §G.2), **FOOTPASS trae jugadores pero no balón**, y **la pista de
  balón de SoccerTrack v2 no sirve para centrar un recorte de 112 px** (reconstruida entre eventos y anclada en la
  posición de evento del proveedor: a 32 px de mediana de los pies del ejecutor y a más de 56 px en el 31 % de los
  eventos). La RoI GT se define igual en FOOTPASS y SoccerTrack v2: caja del jugador que ejecuta cada evento, interpolada
  entre ejecutores consecutivos (83 % y 92-99 % de los eventos cubiertos); en SN-BAS solo caben un control (recorte
  central) y un pseudo-GT con el balón de nuestro detector (41-71 % de los eventos). Código, sin tocar el repo de
  AdaSpot: `scripts/benchmark/adaspot_gt_roi.py` (pistas, `RoIOverride` en el sitio de `roi_selector`, `with_rois` sobre
  el dataset de inferencia), `adaspot_scores.py --use_gt_roi / --roi saliency gt center det` (varias variantes por
  lectura de vídeo y centros de RoI guardados en la caché), `eval_f1_pcbas.py --dataset pcbas`, `adaspot_roi_report.py`
  y `run_adaspot_gt_roi.sh`. Comprobaciones: el modo saliencia reproduce las cachés bit a bit; devolverle al modelo sus
  propios centros da diferencia 0; la ventana de la cámara virtual se reconstruye a ±1 px. De paso, `adaspot_scores.py`
  usaba 2 workers sobre un dataset iterable y puntuaba cada clip dos veces (mismo resultado, 25 % más de tiempo): ahora 1.
  La cola se lanzó a las 11:27 dentro del contenedor (`docker exec -d`) y una segunda tanda de sensibilidad (RoI en los
  pies del ejecutor; saliencia desplazada lo mínimo para contenerlo) queda en cola detrás. Primer resultado [M],
  FOOTPASS con AdaSpot-b: saliencia 41,2 mAP@1 / 44,5 Macro-F1@0,15, RoI GT 39,8 / 42,7, recorte central 36,2 / 39,9. El
  selector deja fuera al ejecutor en el 23 % de los eventos (la mitad de los tiros) y esos se detectan 22 puntos peor,
  pero apuntarle el recorte no los recupera (recall 57,1 → 56,3): con pesos congelados el selector no es el cuello de
  botella. Mismo patrón con AdaSpot-b en SoccerTrack v2 (selector ciego en el 62-66 % de los eventos, pero 5,3 → 5,1 de
  mAP con RoI GT: su apagón en panorámica no es del selector) y en SN-BAS (recorte central 50,4 frente a 54,9; pseudo-GT
  54,5). A las 12:12 la cola se quedó esperando GPU (otros procesos ocupaban 16 GB): se relanzó con lotes de 2 clips
  (4 GB en vez de 7,4), que mueven el mAP ±0,1 (medido re-puntuando la saliencia en SN-BAS: 54,92 → 55,03). AdaSpot-s y
  la segunda tanda quedan en cola; la tabla (`output/benchmark/adaspot_gt_roi/tabla.md`) está pendiente de volcar entera.
- **2026-11-11** (cierre de §7.14 y §7.15: qué frena a los spotters y fusión; rama `exp/adaspot-new-roi`, todo sin GPU).
  Las dos colas de la RoI de AdaSpot acabaron el 2026-11-09 a las 14:30 y se volcaron al documento 7.14 (§5.4 y §6):
  AdaSpot-s repite el patrón de AdaSpot-b (RoI GT 1-2 pt de mAP por debajo de la saliencia en TV, 2-2,4 en panorámica), la
  RoI en los pies no cambia nada y la saliencia corregida empata; la Fase 2 (reentrenar con RoI supervisada) queda
  descartada. El usuario pidió entonces por dónde mejorar T-DEED o AdaSpot en todas las clases y en cualquier metraje, sin
  panorámica. Dos scripts nuevos: `scripts/benchmark/spotter_errors.py` (clasifica cada fallo y cada predicción sobrante
  al umbral de trabajo; `--versus` cruza dos modelos) y `scripts/benchmark/fuse_scores.py` (media ponderada de las cachés
  de probabilidad de varios modelos, escrita como una caché más; `--tune` barre pesos en Middlesbrough). En `eval_bas.py`
  el post-proceso y el evaluador pasan a `evaluate_candidates()` (mismas cifras, comprobado). Hallazgos [M]: (1) no hay
  ceguera: T-DEED deja sin respuesta el 1-2 % de los eventos; pierde por confianza baja (13 % en SN-BAS test, 17 % en
  FOOTPASS), etiqueta equivocada (5-7 %) e instante (3-6 %), y en FOOTPASS el 21 % de sus predicciones no tiene ningún
  evento a ±0,5 s; (2) las clases que hunden la media casi no tienen ejemplos de entrenamiento (34 entradas, 100 tiros,
  128 bloqueos en 4 partidos); (3) T-DEED y AdaSpot-b se complementan (en FOOTPASS cada uno acierta un 14 % de eventos
  que el otro falla); (4) la fusión 0,5 / 0,5, que es también el óptimo del barrido en validación, da en SN-BAS test
  62,5 → 64,0 de mAP y F1 80,5 → 82,0, y en FOOTPASS val 50,1 → 54,3 de mAP, F1 64,6 → 75,4 y 2.659 → 1.377 falsos
  positivos; con el modelo de producción en FOOTPASS, 53,2 → 56,1, F1 65,9 → 77,2 y 2.170 → 1.072; (5) pesos por clase
  y AdaSpot-s como tercer modelo no aportan; (6) la fusión no sube tiro, saque de banda, bloqueo ni entrada. Quedan
  propuestas en §2.1, sin lanzar: integrar la fusión en el pipeline y afinar con el train de FOOTPASS (48 partidos).
  Cachés fusionadas: `<vídeo>.tdeed.Fusion_{T200_AdaB,T200_AdaB_pc,T200_AdaBS,C2_AdaB}.npz`; resultados en
  `output/benchmark/{bas,bas_pcbas}/Fusion_*` y `output/benchmark/errors/`.
  Segunda pasada el mismo día, a petición del usuario («seguro que hay algún ajuste de arquitectura o de combinación»),
  también sin GPU. (7) Separando detección de etiqueta en las cachés por frame: la AP de «hay una acción» sin clase es
  92,0 con la fusión en SN-BAS test y 86,2 en FOOTPASS, y si cada detección llevase la etiqueta correcta el mAP por clase
  pasaría de 66,5 a 87,0 y de 56,3 a 78,5; tiros, centros y bloqueos se detectan en el 93-100 % de los casos y se
  etiquetan bien en el 61 %, 77 % y 15 %. (8) Combinaciones descartadas: T-DEED + sn-teamspotting 62,5 (no suma nada
  frente a 62,5 de T-DEED solo), tres modelos 63,7, media geométrica 61,8; en FOOTPASS dos T-DEED suben los falsos
  positivos (2.541). (9) `scripts/benchmark/stack_scores.py`, segunda etapa aprendida (TCN de 75.000 parámetros sobre las
  curvas de los spotters congelados, CPU): en FOOTPASS dejando un partido fuera, T-DEED solo 50,1 → 53,3 de mAP y F1
  64,6 → 71,9; sobre la fusión, mAP igual (54,3), F1 75,4 → 77,7 y 1.377 → 1.026 falsos positivos; con la cabeza de SN-v2
  como entrada 55,1 / 78,1; con producción + AdaSpot-b, F1 79,4 y 969 falsos positivos. Mejora las clases con más de
  ~100 ejemplos (cabezazo +5) y empeora las que tienen menos (entrada, tiro); en SN-BAS test, entrenada con un solo
  partido, no gana a la media (62,5; 64,1 añadiendo FOOTPASS en las clases comunes). Propuestas en §2.1, sin lanzar.

- **2026-11-11 (anotación desde el visor)**: el usuario quiere seguir la verdad de un juvenil (min 10-60) corrigiendo
  en el visor lo que detecta el pipeline en vez de teclear la hoja CSV. Hecho, sin GPU, en `backend/` (rama
  `exp/adaspot-new-roi`, sin commit) y en el frontend (rama `feat/clip-editor`, sin commit):
  (1) **copia revisada**: `backend/app/services/review_service.py` + rutas `…/parts/{part}/review[/items|/export]`. La
  primera corrección copia `memory/analysis.json` a `review/analysis.json` y todo se guarda ahí (más `edits.jsonl`); el
  análisis del pipeline no se toca. Corregir = etiqueta (vocabulario de `annotation_csv.py`), instante, `tracker_id`
  del ejecutor y lado local/visitante (una vez por tracklet); borrar con deshacer; añadir lo no detectado.
  (2) **Editor de clips** del frontend: abierto sobre una acción o un evento (tarjeta del vídeo, lápiz de la lista o
  tecla `E`; `N` añade en el instante actual) tiene, además de «Save clip», «Apply correction» y «Delete». Teclas 1-0
  para el tipo, `T` instante, `H`/`A` equipo, Enter guarda y el vídeo sigue; el jugador se elige con un clic sobre él en
  el vídeo. Una barra bajo el reproductor cuenta lo corregido/añadido/borrado y exporta el tramo.
  (3) **Hoja exportada** = formato de `scripts/annotation_csv.py` (`review/ground_truth_<desde>-<hasta>.csv`); lo no
  tocado en el tramo cuenta como correcto; acción + evento de la misma jugada a < 1,5 s = una fila. Comprobado de
  punta a punta en Middlesbrough con una instancia aislada: el lector la lee sin avisos y `evaluate_match --csv` da
  las diferencias esperadas [M: una acción reetiquetada, otra añadida y un evento pasado a córner → drive 24/25, high_pass 3/4].
  Verificación: 30 tests del backend, 83 del frontend, recorrido real en Chromium (32 comprobaciones). Avisos de
  método: la verdad así obtenida hereda los candidatos y los instantes del pipeline del que parte (favorece a
  producción frente a otras variantes; sembrar con umbral bajo vía `respot_telemetry.py` lo reduce) y el equipo solo
  queda en las filas cuyo tracklet se ha marcado. Las 126 filas ya anotadas del Espanyol 25-30 no hace falta
  importarlas: `evaluate_match --csv` acepta varias hojas.

- **2026-11-12 (el editor atascaba el túnel)**: anotando el Mallorca (`433f62c92b727870`) por el reenvío de puertos de
  VS Code, «Apply correction» tardaba y al rato la web dejó de cargar. Medido: guardar son 37-85 ms en el backend y
  1 KB en el cable [M]; lo que atascaba era la tira de miniaturas del editor de clips, que sacaba los fotogramas con un
  `<video preload="auto">` aparte saltando por el fichero: **38 MB por apertura** [M] y peticiones de rango que el
  túnel no cerraba; en el servidor había 15 descargas de vídeo abiertas con 50 MB en cola de envío (`ss`, Send-Q) y
  detrás de ellas esperaban el PATCH y la propia página. Arreglo: `GET /api/videos/{id}/thumbnail?t=&h=` (ffmpeg,
  3-4 KB, 100-160 ms la primera vez y 2 ms desde `videos/<id>/thumbs/`; `backend/app/services/thumbnails.py`) y
  `Filmstrip` con `<img>` e instantes en rejilla (mismas URL al reabrir); abrir el editor baja ahora 40 KB de
  miniaturas (más 580 KB de tracking la primera vez en cada zona) [M]. Segundo coste, de CPU: cada cambio de estado
  del visor repintaba las ~1.500 filas de «Ball actions» (350-415 ms de Enter a editor cerrado en el servidor, con el
  modo desarrollo de React); con la lista y sus filas memorizadas (`BallActionList`, `useStableCallback`) son
  123-142 ms [M]. Tests: 31 del backend, 85 del frontend, recorrido en Chromium 32/32. Al reiniciar el backend con
  descargas de vídeo abiertas, uvicorn se queda esperando a que acaben: hay que matarlo con `-9`. Pendiente si el
  túnel se vuelve a atascar con el reproductor: acotar las respuestas a rangos abiertos (`bytes=N-`) a unos pocos MB.
  Segunda parte del mismo día: «Tracking unavailable» y recargas de la página mientras el usuario anotaba. El
  backend respondía bien (ni un error desde el reinicio); lo provoqué yo: dos reinicios del backend con su página
  abierta y ediciones del frontend en caliente (cada guardado se recarga en su navegador), más un cargador de tracking
  que no reintentaba. Desde entonces el frontend se prueba en una copia aparte (otro Vite, otro backend) y se aplica de
  una vez. Preparado y probado ahí: (a) respuestas `Range` del vídeo acotadas a 2 MB
  (`BACKEND_VIDEO_RANGE_CHUNK_BYTES`): Chromium reproduce y salta igual, 12 MB en toda la prueba y como mucho una
  conexión abierta de 2 MB, frente a 4 conexiones colgadas con 4 MB en cola cada una [M]; (b) tracking con coordenadas
  redondeadas (−25 % comprimido) y ventanas de 10 s en vez de 30 s: abrir el editor en una zona nueva baja 171 KB en
  total (eran 620 KB) [M]; (c) el cargador de tracking reintenta solo (1, 2, 4... 15 s) y solo dice «unavailable» tras
  tres fallos seguidos, y se recupera sin recargar. Tests: 32 del backend, 88 del frontend, recorrido 32/32.
- **2026-11-12 (valoración de PAVE, PCBAS 2026)**: el usuario pidió analizar el paper de resultados de SoccerNet 2026 y
  valorar si replicar PAVE desde cero o aplicar partes a T-DEED/AdaSpot (sin GPU). Leídos: §3 y §8.2 del paper de
  resultados, el informe técnico de PAVE (arXiv 2606.28389), el paper de FOOTPASS (arXiv 2511.16183) y el README de
  `JeremieOchin/FOOTPASS`; listado del repo HF `SoccerNet/SN-PCBAS-2026` con nuestro token (train 352x640 22,8 GB +
  tácticos 2,5 GB; Full HD 184 GB; sin posición del balón). Hallazgos: (1) PAVE es TAAD → DST del dataset con transformer
  temporal por tracklet, atención entre los 26 jugadores y ensemble de 4 con filtro de acuerdo (descarta lo que apoya un
  solo modelo, salvo tackle); su entrada es el tracking certificado de FOOTPASS, no vídeo ni tracking propio. (2) La
  palanca es el estado del juego: baselines en test Macro-F1 TAAD 23,8 → +GNN 34,5 → +DST 46,4 (calculadas de sus F1 por
  clase [P]); PAVE 58,9. (3) La validación de PAVE son nuestros 3 partidos de FOOTPASS val: en los mismos partidos PAVE
  (con jugador) hace 60,9 frente a 47,6 de T-DEED y 43,0 de AdaSpot-b (solo clase, umbral óptimo) [M]; gana en banda,
  centro, conducción y tiro, empata en cabezazo, bloqueo y entrada. (4) Replicar: código público sin pesos, pérdida y
  muestreo sin publicar, 1-2 días de GPU por pasada [NV], y en inferencia pediría equipo/dorsal/rol de calidad EPTS que
  no tenemos (equipo 62-68 % en juveniles): descartado. Escrito `docs/pave_valoracion.md` (§0 resumen, §2 tabla por
  clase, §3 réplica, §4 qué trasladar, §5 tracking, §6 orden); dos filas nuevas en la hoja de ruta §2.1 a 0 GPU
  (fusión por evento con filtro de acuerdo; estado del juego del h5 táctico de FOOTPASS val en `stack_scores.py`) y
  un test de 20-30 min de GPU (352x640 frente a Full HD) antes de decidir la descarga del train. Del tracking: PAVE no
  aporta nada; FOOTPASS val sirve como verdad plateada de equipo, dorsal y homografía en partidos enteros. Enlaces en
  §7.7 y en la tabla de `CLAUDE.md`. Nada lanzado en GPU.
