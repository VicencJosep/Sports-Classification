> Parte de [docs/datasets.md](../datasets.md): catálogo a 2026-11-01, con la numeración original del documento.

## 6. Licencias y uso comercial

| Dataset | Licencia | ¿Permite producto comercial? | Nota |
|---|---|---|---|
| SoccerNet: vídeos (500 + 50 partidos, Tracking RAW, BAS 2023-2025, PCBAS-2026, GAR) | NDA con KAUST: "Use only for non-commercial research purposes and do not distribute to tiers" | No | Cláusula 8: no crea licencia alguna; vigencia 2 años; ley saudí |
| SoccerNet `Labels-v2.json` / `Labels-ball.json` (`SN-Labels`) | sin licencia declarada en HF | No verificado | sin licencia explícita no hay permiso de reutilización más allá del uso de investigación |
| SN-echoes | CC BY 4.0 | Sí (las transcripciones) | su valor depende de vídeos bajo NDA |
| SN-GSR-2024/2025, SN-Depth-2025, SN-MVFouls-2024/2025 | GPL-3.0 (etiqueta HF) | Ambiguo | licencia de software aplicada a datos; alcance sobre modelos sin resolver |
| SN-VQA-2026 | CC BY-SA 4.0 | Sí, con share-alike | |
| SoccerNet-v3 | "open licence", "no restriction whatsoever" | Sí (las anotaciones) | frames de vídeo bajo NDA |
| FOOTPASS / SN-PCBAS-2026 (anotaciones) | CC BY-NC 4.0 (README); LICENSE del repo Apache-2.0; paper CVIU CC BY 4.0 | No | vídeo bajo NDA, en HF a 352x640 y Full HD |
| SoccerTrack v2 | CC BY 4.0 | Sí, con atribución y cita | la única licencia explícita en disco; acceso HF restringido |
| TeamTrack | Kaggle "MIT" | En principio sí | confirmar por escrito con los autores |
| SoccerTrack v1 | Kaggle "GNU Lesser General Public License 3.0" | Atípica | licencia de software sobre datos |
| TrackID3x3 | CC BY 4.0 (salvo el módulo de dorsales) | Sí | baloncesto 3x3 |
| SoccerHigh | CC BY-NC-SA 4.0 | No | solo evaluación e I+D |
| StatsBomb Open Data | StatsBomb Public Data User Agreement | No ("commercially exploit the data or any analysis derived" prohibido) | sin redistribución; logo obligatorio; registro |
| SkillCorner Open Data | MIT | Sí | sin vídeo; inútil para visión |
| DFL - Bundesliga Data Shootout | "Competition-Use Only" (Kaggle §7.A) | No | ya no se distribuye |
| DeepSportradar | CC BY-NC-ND 4.0 / CC BY-NC-SA 4.0 | No | fuentes discrepan entre ND y SA |
| MultiSports, SportsMOT, NSVA | CC BY-NC 4.0 | No | SportsMOT además prohíbe redistribuir |
| SoccerReplay-1988, SoccerBench | CC BY-NC-SA 4.0 (+ NDA en SoccerReplay) | No | |
| MatchTime | CC BY-SA 4.0 | Sí (anotaciones) | sobre vídeo SoccerNet |
| SoccerChat (anotaciones) | MIT | Sí (anotaciones) | vídeo bajo NDA |
| SJN-210k | CC BY-SA 4.0 | Sí, con share-alike | pesos: zona gris |
| SoccerSynth-Detection | Apache-2.0 | Sí | sintético, cámara TV |
| Spiideo SynLoc | no consta | Desconocido | pedir condiciones a Spiideo |
| Alfheim | "The data may only be used for non-commercial research purposes." | No | prohibida la reidentificación |
| ISSIA-CNR, Fu et al. 2023, WC14, TS-WorldCup, Metrica | no consta / sin licencia formal | Desconocido | Metrica: "please aknowledge the source" |
| IDSSE (Sportec/DFL) | CC BY 4.0 con autorización de la DFL | Sí | sin vídeo |
| Wyscout 2017/18 | CC BY 4.0 | Sí | sin solape con nuestro vídeo |
| Transfermarkt datasets | CC0-1.0 | Sí | |
| openfootball | dominio público | Sí | solo resultados |
| Kaggle Football Events, European Soccer Database | no verificable (JS); ESD ODbL según espejos | Desconocido | texto de comentario con derechos de BBC/ESPN/onefootball |
| Understat, FBref, WhoScored | sin términos / propietario | No (scraping) | zona gris legal |
| PFF FC Open Data | términos no públicos (formulario) | Desconocido | espejo OpenSportsLab/PFF GPL-3.0 |
| EIGD-S | CC BY-NC-SA 4.0 | No | |
| TACDEC, SoccerSum | CC BY-NC-ND 4.0 | No | permiso escrito para competiciones o uso comercial |
| Roboflow baloncesto (los dos del pipeline) | CC BY 4.0 | Sí, con atribución | `dataset_fusionado` marcado "Public Domain" por error |
| Roboflow derivados del DFL Shootout | no verificable (403) | No recomendable | procedencia "Competition-Use Only" |
| T-DEED (código) y pesos `SNB_pano1`/`SNB_tv1` | GPL-3.0; pesos entrenados con vídeo bajo NDA | Decidir antes de distribuir | ver párrafo siguiente |

El NDA de SoccerNet es un acuerdo bilateral entre el solicitante ("Collaborator", con nombre y afiliación) y King Abdullah University of Science and Technology, vinculado desde todos los repositorios gated ([texto](https://docs.google.com/document/d/1Efz8yP-baa8CtcMB7SYIID7Bn5sH7sGTB4-u1YdVRf4/export?format=txt)). Su Exhibit A fija el uso permitido en una frase: "Use only for non-commercial research purposes and do not distribute to tiers" (sic, por third parties). La cláusula 3.c prohíbe revelar la información "to any third party except authorized representatives, consultants, or subcontractors engaged to assist in pursuing the Purposes"; la 6 obliga a devolver las copias a petición de KAUST; la 8 dice: "No license is created under this Agreement, nor shall any be implied therefrom under any patent, trademark, application, copyright, trade secret, or other intellectual property right of either Party, other than the use of Confidential Information for the Purposes"; la 10 fija la vigencia en dos años y la 11 la ley de Arabia Saudí. El formulario de Hugging Face añade tres casillas: "I understand that the use of this dataset is for research and non-commercial use only", "I understand that this dataset and the videos are protected by copyright" y "I agree not to share the dataset with anyone else". Los 7 partidos EFL de Ball Action Spotting tienen el mismo gating, el mismo `license: other` y el mismo NDA que el resto.

Zona gris: ni el NDA ni las reglas de los retos mencionan los pesos de modelos entrenados con los datos. Afecta a los T-DEED que usamos en producción (preentrenados por sus autores con SoccerNet-v2 y los 7 EFL), a `SNB_pano1` (derivado de `challenge2`, además GPL-3.0), al SVC (clips de vídeo SoccerNet), al R3D-18 (`dataset_ball_actions`) y a PRTReId/PnLCalib (SoccerNet GSR y Calibration). No es posible verificar desde el repositorio si las copias locales de BAS-2024 y Tracking están cubiertas por un NDA vigente a nombre de alguien del equipo. Este documento no saca conclusiones jurídicas: antes de distribuir un producto que incluya cualquiera de esos pesos conviene consultar asesoría legal, y mientras tanto tratar los datos con NDA y las licencias NC como material de evaluación e I+D, y preferir para entrenar lo que es CC BY 4.0 (SoccerTrack v2, IDSSE, SN-echoes), CC0 (Transfermarkt), MIT (TeamTrack, por confirmar) o Apache-2.0 (SoccerSynth-Detection) y datos propios.
