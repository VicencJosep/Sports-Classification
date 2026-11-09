> Parte de [docs/datasets.md](../datasets.md): catálogo a 2026-11-01, con la numeración original del documento.

## 8. Huecos de esta investigación

Lo que los investigadores no pudieron verificar a 2026-11-01:

- Licencias declaradas en Roboflow Universe (`football-players-detection-3zvbc`, `football-ball-detection-rejhg`, `football-field-detection-f07vi`, los dos de baloncesto que usa el pipeline y sus homónimos): las páginas devolvieron HTTP 403; los recuentos proceden de tutoriales y fichas de workspace.
- Repositorio, tamaño y licencia del dataset xGHub (CVIU 2026): ScienceDirect, UPF e IRI devolvieron 403; solo consta el registro Crossref/Semantic Scholar y un post del autor.
- Dominio true-think.com (Fu et al. 2023): no se comprobó si sigue sirviendo el dataset ni sus condiciones.
- Resolución, fps y tamaño total de los vídeos del DFL Shootout (indicios de 1280x720 y 25 fps); contenido exacto del write-up del 1.º puesto (CAPTCHA de Kaggle).
- Licencia de Spiideo SynLoc (no consta en sskit, en el paper ni en research.spiideo.com) y número de imágenes reales frente a sintéticas.
- Licencias de los datasets de Kaggle leídos solo por metadatos o fuentes secundarias: "Football Events", European Soccer Database (ODbL según espejos), "Soccer View and Event Score", "football match actions video dataset", "FOOTBALL-SOCCER-VIDEOS-DATASET"; fecha de corte de "Football Events" en 2016-17.
- Enlaces de descarga de SoccerDB (repo sin enlaces) y de Comprehensive Soccer/SSET (HUST devolvía 404); tamaño, clases y enlace del dataset de vídeo "SoccerEvent" (Karimi et al.); URL de descarga del dataset de Huang et al. 2022.
- Composición de OSL-loc-SPL-highlight (liga, resolución, licencia) y de OpenSportsLab/PFF (326 GB; si trae partidos completos); términos reales de PFF FC (requieren registro); texto del NDA de SoccerReplay-1988.
- Si las copias locales de BAS-2024 y SoccerNet Tracking están cubiertas por un NDA vigente a nombre de alguien del equipo; licencia de `Labels-v2.json`/`Labels-ball.json` (ningún documento de SoccerNet la fija); cláusulas sobre pesos de modelos (no existen en el NDA).
- Reparto exacto train/valid/test/challenge de los 200 clips de GSR y liga de origen de los 12 partidos de Tracking ("Swiss Super League" solo aparece en notas internas del proyecto); nombres de las 10 clases de SoccerNet-GAR; tamaños de la mayoría de repos de HF (la API devolvió listas vacías).
- TeamTrack: si el balón conserva cajas y la licencia del dataset en la web del proyecto; frames y resolución de los subconjuntos indoor/outdoor de TrackID3x3.
- Transfermarkt datasets: si incluye la EFL Championship 2019-20 y cuántos partidos de UCL 2014-17 tienen `game_events` completos.
- StatsBomb: fecha de publicación original (2023) de las temporadas 2015/16 completas (los `match_updated` son de 2024-2026); tamaño total del repositorio; número de ficheros de 360.
- Alfheim: resolución y fps de las cámaras (el PDF de MMSys 2014 no se obtuvo); Signality: ninguna fuente primaria de acceso; Impect: número de partidos, licencia (`LICENSE.pdf`) y precisión temporal; MMSys'22 Eliteserien: datos no accesibles; SVW: lista de deportes no legible.
- Páginas bloqueadas o caídas: deeperaction.github.io/multisports (404), media.hust.edu.cn/dataset (404), datasets.simula.no (certificado), mdpi.com (403), link.springer.com (login), kaggle.com (solo JavaScript), sciencedirect.com (403), repositori.upf.edu (acceso denegado), cvlab.cse.msu.edu (conexión rechazada), mmsys2022.ie (403), atomscott.github.io/SoccerTrack (404), research.spiideo.com (sin contenido estático), bundesliga.com/en/aws-bundesliga-data-shootout (404), fbref.com (403).
