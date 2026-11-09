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
