> Parte de [docs/datasets.md](../datasets.md): catálogo a 2026-11-01, con la numeración original del documento.

## 5. Baloncesto

DeepSportradar (ficha en 2.4) es el único conjunto público con máscaras de instancia, calibración de cancha y balón sobre la misma cámara fija lateral de pabellón (Keemotion, vista desde el centro de la pista), un encuadre parecido al de una cámara amateur fija. Su utilidad para `basketball/` es alta en calibración y segmentación de jugadores, media en balón (solo recortes con posición-oráculo) y nula en eventos (tiro, canasta) o marcador. Las licencias CC BY-NC-ND 4.0 y CC BY-NC-SA 4.0 impiden el uso comercial y la redistribución de derivados. Sus datos siguen descargables (Kaggle + GitHub); la organización no tiene actividad desde 2023.

| Dataset (año, enlace) | Qué contiene | Licencia | Para qué nos sirve |
|---|---|---|---|
| [TrackID3x3](https://github.com/open-starlab/TrackID3x3) (2025, Nagoya / open-starlab; MMSports 2025) | Baloncesto 3x3 a cancha completa con vídeo: subconjuntos Indoor y Outdoor (cámaras fijas) y Drone (127.119 frames a 3840x2160, no verificado en el PDF); cajas de los 6 jugadores en todos los frames, 10 keypoints de pose en algunos, identidades; Google Drive | "Creative Commons Attribution 4.0 International (CC BY 4.0), unless otherwise stated" (el módulo de dorsales es solo no comercial) | Alta: cámara fija interior y exterior con licencia permisiva; salvedad: 3x3 (6 jugadores) |
| [TeamTrack](https://atomscott.github.io/TeamTrack/), parte de baloncesto (2024) | 65,5 min y 1.296.900 cajas en tres vistas: ojo de pez 6K (Z CAM E2-F6, 17,5 min), dron lateral 4K (24 min) y dron cenital 4K (24 min); IDs persistentes; Kaggle + Google Drive | Kaggle "MIT"; la web del proyecto no la indica | Alta-media: cámara fija gran angular de partidos universitarios |
| [DeepSportradar / basketball-instants](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset) (2019-2023) | ~364 instants en 15 pabellones (LNB Pro A); 408 imágenes con máscaras; 728 vistas calibradas; 18.703 recortes de re-id; balón 3D con posición-oráculo | CC BY-NC-ND 4.0 / CC BY-NC-SA 4.0 | Alta en calibración y segmentación, solo investigación |
| [SportsMOT](https://github.com/MCG-NJU/SportsMOT), parte de baloncesto (2023) | Clips de retransmisión (Olympics, NCAA, NBA) a 720p y 25 fps; 845,4 frames y 9,1 jugadores por clip de media; solo jugadores | CC BY-NC 4.0, sin redistribución | Media: tracking en TV, cámara móvil |
| Roboflow Universe: `basketball-player-detection-3-ycjdo` (654 imágenes NBA, 10 clases), `basketball-court-detection-2` (keypoints), `persistence/basketball-hoop-player` (216 imágenes), `tomas-34ray/basketball-detection-cjjxp` | Pequeños, de origen NBA; los dos primeros ya los usa el pipeline como modelos hospedados | CC BY 4.0 (los cuatro, según `README.dataset.txt` y las fichas de Universe; páginas de Universe no legibles hoy: 403) | Media: aro y balón; licencias heterogéneas |
| Eventos en retransmisión NBA: [NBA dataset GAR](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530205.pdf) (181 partidos, 9.172 clips, 9 clases con resultado, "available upon request"), [FineSports](https://github.com/PKU-ICST-MIPL/FineSports_CVPR2024) (~10.000 vídeos, 16.000 instancias, 52 subtipos, Release Agreement por correo), [NSVA](https://github.com/jackwu502/NSVA) (>32K clips con subtítulos; CC BY-NC), [BARD](https://bodai.unibs.it/?p=1198) (60 partidos NBA, 14.676 clips a 720p; licencia sin confirmar) | Tiro, canasta, pase, rebote etiquetados a nivel de clip | bajo petición o no comercial | Media-baja: única fuente de eventos de tiro/canasta, pero TV y acceso restringido |
| [SpaceJam](https://github.com/simonefrancia/SpaceJam) (2018-2020, tesis Univ. Bolonia) | ~32.560 clips de 16 frames de un solo jugador + articulaciones 2D; 10 clases (Block, Pass, Run, Dribble, Shoot, Ball in Hand, Defence, Pick, No Action, Walk) | MIT (enlace en el repo) | Baja-media: clasificar "tiro" por pose del jugador recortado |

Lo que no existe en baloncesto: ningún dataset público con jugadores + balón + aro + cancha + eventos desde una cámara única amateur; los eventos de tiro y canasta solo están etiquetados sobre retransmisiones NBA/NCAA. Para la rama basketball la vía realista es TrackID3x3 + los sets de aro y balón de Roboflow para detección, DeepSportradar solo para investigación en calibración, y anotación propia o reglas (trayectoria balón-aro) para eventos. "Basketball-51", "HoopTrack" y "BasketballGAR" no existen como datasets; "NBA SportVU 2015-16" es tracking posicional sin vídeo.

### Inventario local (antes §3.5 de `docs/datasets.md`)

La carpeta `basketball/` salió del repo el 2026-11-09 y ahora está en `/media/dd3/ner-training/basketball_Sports-Classification`;
las rutas de la tabla son relativas a su subcarpeta `datasets/`. La rama `basketball` conserva el código versionado.

| Dataset (ruta) | Origen | Tamaño | Papel | Licencia | Modelos |
|---|---|---|---|---|---|
| Roboflow `basketball-player-detection-3-ycjdo` v4 (654 imágenes 464/96/94, 10 clases: ball, ball-in-basket, number, player, player-in-possession, player-jump-shot, player-layup-dunk, player-shot-block, referee, rim; 1280x1280) | `basketball/scripts/download_datasets.py` | 139M | train/valid/test de Roboflow | CC BY 4.0 (`README.dataset.txt`) | modelo hospedado en Roboflow usado por el pipeline (`inference.get_model`) |
| Roboflow `basketball-court-detection-2` v14 (1.460 imágenes, keypoints COCO, 640x640) | ídem | 122M | ídem | CC BY 4.0 | modelo hospedado de keypoints de cancha |
| `dataset_fusionado` (Roboflow `ugiat-basketball_merged_dataset` v1: 5.593 imágenes 4.583/486/524, 17 clases → 6) | propio en Roboflow | 2,5G | | marcado "Public Domain" en Roboflow, pero el README del derivado avisa de que incluye fuentes CC BY 4.0 (mal marcado) | ninguno |
| `basketball_final` (1.305/294/289 imágenes, 10 clases) y `basketball_final_5cls` | `build_final_dataset.py` (2026-10-16), partición por clip de origen NBA | 378M + 15M | entrenamiento sugerido, no ejecutado | hereda las anteriores | ningún modelo propio en disco; `fashion-clip` para equipos |

En disco: `basketball/datasets` · ~3,2G · dataset_fusionado 2,5G, basketball_final 378M, player-detection 139M, court-detection 122M.

Hallazgo del inventario: ningún modelo propio entrenado y un dataset fusionado con la licencia mal marcada.
