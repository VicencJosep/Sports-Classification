# Estudio: detección de tiros y acciones de balón en partido completo

Resumen de todas las mediciones hechas entre el 28 de septiembre y el 5 de octubre de 2026
(rama `feat/T-DEED`). Objetivo: **estadística automática de tiros** (da igual si van dentro o
fuera), es decir, manda la precisión. Todos los números de este documento se pueden reproducir
con los scripts de `scripts/shot_benchmark/`; los ficheros de resultados están en
`output/shot_benchmark/` (no versionado).

## 1. Conjuntos de evaluación

| Nombre | Qué es | Para qué sirve | Limpio para… |
|---|---|---|---|
| **7 EFL** | Los 7 partidos completos de SoccerNet Ball Action Spotting (EFL Championship 2019-20, `datasets/SoccerNet/england_efl/`), `Labels-ball.json` con resolución de ms: 169 tiros, 13 goles, 261 centros, 713 cabezazos… | Partido completo, tiros y acciones de balón | V-JEPA2 + SVC (entrenado con clips de SoccerNet-v2, no EFL); VLM (no se entrenan); T-DEED NO (entrenó con 4 o 6 de ellos) |
| **3 EFL limpios** | Middlesbrough–Preston, Stoke–Huddersfield, Reading–Fulham: 69 tiros | La comparación justa de los tres métodos | Todos. Para T-DEED solo con el checkpoint `SoccerNetBall_test` (entrenado con los otros 4; Middlesbrough es su partido de validación: no entrena con él pero se usó para elegir la mejor época) |
| **Banco de 150 candidatos difíciles** | Candidatos propuestos por V-JEPA2 en los 7 EFL: 60 tiros reales, 60 centros, 30 "sin anotar" (casi todos repeticiones) | ¿Reconoce un tiro cuando lo tiene delante? ¿Lo confunde con un centro? | Idem; el subconjunto de los 3 limpios son 44 candidatos (21 tiros, 13 centros, 10 repeticiones) |
| **447 candidatos limpios** | Todos los candidatos de V-JEPA2 (umbral 0,02) en los 3 EFL limpios: 61 tiros, 52 centros, 118 sin anotar, 216 otras acciones | Medir un verificador en el mismo punto en que lo usaría el pipeline | Todos |
| **`ground_truth/`** | Primeros 20 min de Middlesbrough–Preston, 405 anotaciones (367 acciones de balón puntuables) | Pases, conducciones, cabezazos, saques de banda (±1 s) con `scripts/evaluate_pipeline.py` | R3D-18 (es su partido de validación) y T-DEED (idem) |
| **`clips_analisis/`** | Clips ya recortados por evento (61 de gol/tiro; 212 de 8 categorías) | Comparativas de septiembre de los VLM | Todos, pero **no representa el problema real**: los clips de tiro duran 2-3 s y están cortados en el golpeo |
| **6 partidos SoccerNet-v2** | Levante–R.Madrid, Lazio–Roma, Bayern–Augsburg, PSG–Angers, Liverpool–Swansea, Arsenal–PSG (130 tiros) | Generalización a otras ligas | **Contaminado para T-DEED** (ver §6) |

Métricas: **AP** (área bajo precisión-recall, un tiro cuenta si el pico cae a ±3 s de la anotación,
emparejado 1 a 1), **P@R70** (precisión cuando el umbral se ajusta para encontrar el 70 % de los tiros)
y **AUC tiro/centro** (probabilidad de que un tiro real puntúe por encima de un centro; 0,5 = azar).

## 2. Resultado principal: los tres métodos sobre el mismo conjunto limpio

### 2a. ¿Reconoce un tiro cuando lo tiene delante? (44 candidatos de los 3 EFL limpios)

| Método | Acepta tiros reales (de 21) | Acepta centros (de 13) | Acepta repeticiones (de 10) | Precisión | Recall | F1 | AUC tiro/centro |
|---|---|---|---|---|---|---|---|
| Qwen3-Omni-30B (vídeo 224p + razonamiento) | 21 | 13 | 10 | 48 % | 100 % | 65 % | 0,50 |
| VideoARM + Qwen3.5-9B (orquestador) | 10 | 5 | 8 | 43 % | 48 % | 45 % | 0,65 |
| **T-DEED** (`SoccerNetBall_test`, umbral 0,3) | **18** | **4** | **0** | **82 %** | **86 %** | **84 %** | **0,87** |
| V-JEPA2 + SVC (pipeline anterior, umbral 0,3) | 13 | 8 | 3 | 54 % | 62 % | 58 % | 0,39 |

El F1 del Omni (65 %) es el de "aceptar todo": su precisión es la proporción de tiros del banco.

### 2b. Detección de tiros en partido completo (3 EFL limpios, 69 tiros)

| Método | AP ±3 s | P@R70 | Alarmas por partido a R70 |
|---|---|---|---|
| V-JEPA2 + SVC, etiqueta ganadora por ventana (pipeline anterior) | 31,6 %* | 29 %* | 61* |
| V-JEPA2 + SVC, picos por clase (`core/spotting.py`) | 34,5 % | 26 % | 64 |
| V-JEPA2 → verificador Qwen3.5-9B (sí/no, 16 frames) | 22,8 % | 22 % | 76 |
| V-JEPA2 → verificador VideoARM + Qwen3.5-9B | pendiente (`output/shot_benchmark/vlm_clean/videoarm.jsonl`) | | |
| V-JEPA2 → verificador Qwen3-Omni-30B | pendiente (`vlm_clean/omni_video.jsonl`) | | |
| V-JEPA2 → verificador T-DEED | 74,7 % | 74 % | 22 |
| **T-DEED solo** (`SoccerNetBall_test`) | **75,6 %** | **74 %** | **23** |

\* medido sobre los 7 EFL (`output/shot_benchmark/vjepa2_efl.txt`), no sobre los 3 limpios.
T-DEED además localiza el tiro con precisión de ±1 s (AP ±1 s 70,0 %; V-JEPA2: 6,3 %).
Techo de recall de T-DEED en estos partidos: 90 % (el 10 % de los tiros no aparece ni bajando el umbral).

## 3. Acciones de balón (pase, conducción, pase alto, cabezazo, saque de banda)

`ground_truth/` (Middlesbrough 20 min, 367 acciones, tolerancia ±1 s), `scripts/evaluate_pipeline.py`:

| Cuándo se dispara | Quién pone la etiqueta | Precisión | Recall | F1 |
|---|---|---|---|---|
| Cambio de posesión (pipeline anterior) | R3D-18 propio | 28 % | 57 % | 38 % |
| T-DEED (`challenge2`, umbral 0,5) | R3D-18 propio | 80 % | 80 % | 80 % |
| T-DEED | Combinación R3D-18 × T-DEED (**por defecto**) | 87 % | 88 % | 88 % |
| T-DEED | T-DEED | 90 % | 92 % | 91 % |

Conclusión: el R3D-18 propio clasifica bien; lo que fallaba era el disparador (solo el 34 % de los
cambios de posesión caía sobre una acción real). El umbral 0,5 se eligió sobre este mismo partido,
así que es algo optimista. No está medido si el **jugador** asignado por la posesión es el correcto.

## 4. Lo que se probó para mejorar V-JEPA2 y no funcionó

| Idea | Resultado (AP tiros, 7 EFL, candidatos de V-JEPA2) |
|---|---|
| Picos por clase en vez de etiqueta ganadora | 31,6 % → 34,4 % |
| Re-puntuador (regresión logística) sobre las probabilidades del SVC en ±3 ventanas, leave-one-match-out | 38,2 %, solo mejora la zona de alta confianza |
| Cabeza nueva sobre el embedding completo de V-JEPA2 (3072-d) con los negativos difíciles reales | 31-32 % (no supera al SVC) |
| Quitar el recorte central de V-JEPA2 (solo ve el 49 % del ancho; frame entero a 256×256 o 256×448 con RoPE parcheado) | 31 % / 30 %: **no mejora** |
| Curva de aprendizaje (1→6 partidos de entrenamiento) | La cabeza sobre embedding sigue subiendo +5 pp por duplicación: el límite era el número de partidos etiquetados |

## 5. Lo que se probó con los VLM y no funcionó

Banco de 150 candidatos difíciles (7 EFL), AUC tiro/centro:

| Variante | Dice tiro en tiros / centros / repeticiones | AUC |
|---|---|---|
| Qwen3.5-9B, sí/no con 16 frames | 72 / 79 / 100 % | 0,54 |
| Qwen3.5-9B, fotogramas + análisis guiado + JSON | 98 / 97 / 93 % | 0,54 |
| Qwen3.5-9B, vídeo nativo 224p 4 fps | 90 / 78 / 87 % | 0,54 |
| Qwen3.5-9B, fotogramas + razonamiento | 100 / 100 / 100 % | 0,49 |
| Qwen3.5-9B, vídeo + razonamiento | 98 / 93 / 100 % | 0,57 |
| VideoARM + Qwen3.5-9B (clasificación de 10 etiquetas, tiro = shot∪goal) | 57 / 40 / 83 % | 0,63 |
| Qwen3-Omni-30B AWQ-4bit, vídeo 224p + razonamiento | 100 / 100 / 100 % | 0,52 |

Diagnóstico: con fotogramas o vídeo a 224p el balón ocupa 1-2 píxeles; el modelo ve "golpeo en el
área" y responde "tiro". Tampoco detectan repeticiones de forma fiable. Las cifras de septiembre en
`clips_analisis/` (VideoARM 49 %, híbrido 61-69 %, Omni 55 % de acierto de clase) no son
comparables: clips cortos ya recortados, sin centros ni repeticiones como negativos.

Donde sí pueden aportar (no medido): describir en texto un evento ya detectado, eventos sin modelo
específico (en `clips_analisis/` el Omni acertó 79 % de tarjetas y 73 % de sustituciones) y
preguntas abiertas sobre el partido.

## 6. Fuga de datos en la "generalización" de T-DEED

Con `SoccerNetBall_challenge2` en 6 partidos de SoccerNet-v2 de otras ligas salió AP 78 %. **No vale
como medida limpia.** Prueba: en partidos que T-DEED vio al entrenar, el 89 % de sus picos cae a
±0,15 s de la anotación; en partidos nunca vistos (Stoke, Reading), el 55 %; en esos 6 partidos de
SoccerNet-v2, el 89 %. Las anotaciones de v2 tienen resolución de ms, así que no es un artefacto.
Conclusión: el checkpoint publicado preentrenó con ellos. No existe ningún partido de SoccerNet-v2
con etiquetas públicas que se pueda garantizar fuera de su entrenamiento. La generalización a otras
ligas y realizaciones solo se podrá medir con partidos anotados por nosotros.

## 7. Modelos y checkpoints

| Modelo | Dónde | Entrenado con | Notas |
|---|---|---|---|
| V-JEPA2 ViT-L + SVC | `models/svc_classifier_more_penalties.joblib` | Clips de 100 partidos de SoccerNet-v2 | 16 clases; 38 min/partido en la 4090 |
| R3D-18 acciones de balón v2 | `models/r3d18_ball_actions-v2.pt` | 6 EFL (val Middlesbrough) | 6 clases, sin tiro ni centro |
| T-DEED `SoccerNetBall_test` | `models/tdeed/SoccerNetBall_test/` | 4 EFL, val Middlesbrough | RegNetY-200MF; el de evaluación limpia |
| T-DEED `SoccerNetBall_challenge2` | `models/tdeed/SoccerNetBall_challenge2/` | 6 EFL, val Middlesbrough | RegNetY-800MF; el de producción; 9-15 min/partido |
| Qwen3.5-9B (vLLM fp8) | `scripts/serve_events_vllm.sh`, puerto 8010 | — | VideoARM encima |
| Qwen3-Omni-30B-A3B-Thinking AWQ-4bit | `scripts/serve_events_omni.sh`, puerto 8002 | — | Vídeo solo a 224p (OOM del encoder a más) |

T-DEED es GPL-3.0 (sin problema para uso interno; revisar si se distribuye). Ambos checkpoints
tienen una segunda cabeza con las 17 clases de SoccerNet-v2 (tarjetas, sustituciones, saque
inicial…), no evaluada limpiamente por el mismo motivo del §6.

## 8. Estado del pipeline (rama `feat/T-DEED`)

`python -m main --mode full-match` usa por defecto T-DEED para tiros y goles (`--shot-backend
tdeed`), T-DEED como disparador de acciones de balón (`--ball-actions-trigger tdeed`) con etiqueta
combinada (`--ball-actions-label combined`), posesión propia para el jugador y V-JEPA2 para el resto
de eventos. Las cachés de T-DEED se guardan junto al vídeo (`<vídeo>.tdeed.<modelo>.npz`).

Demos: `output/tdeed_demo/final.mp4` (20 min de Middlesbrough con banners),
`output/tdeed_demo/tiros_Stoke_SoccerNetBall_test.mp4` (un clip por detección en un partido no visto:
20 detecciones, 15 aciertos, 5 falsas alarmas, 6 tiros no detectados de 21).

## 9. Pendiente

- Rellenar las dos filas de VideoARM y Omni en §2b.
- Cuando T-DEED marca un tiro, la acción de balón de ese instante debe ser "tiro" (hoy sale "conducción").
- Medir si el jugador asignado a cada acción es el correcto.
- Viabilidad en vídeo panorámico (SoccerTrack v2, `datasets/soccertrack-v2/`): T-DEED tal cual y con "cámara virtual".
- Tarjetas, sustituciones y saque inicial: necesita anotar eventos en partidos propios.
- Ajuste fino de T-DEED con datos propios cuando se sepa qué falla en producción.
