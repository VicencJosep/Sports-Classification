# Highlights de fútbol: estado del arte (2024-2026) y qué implica para el proyecto

Resumen a 2026-11-01. Fuentes verificadas (paper, código y datos abiertos) en
`output/research_notes/highlights_sota_verificacion_2026-11-01.md`; nuestro mecanismo actual en
`docs/highlights_evaluacion.md`.

## 1. Dónde está el campo

- **Verdad de highlight = el resumen oficial alineado al partido completo.** Así lo hacen SoccerHigh (MMSports 2025:
  237 partidos de SoccerNet, F1 del baseline 0,40), SVHighlights (KDD 2026: 320 retransmisiones de 8 deportes, 40 de
  fútbol) y MOMENTS (2026: 3.954 momentos binarios). Nadie publica importancia graduada por anotadores. Métricas: F1 por
  frame, F1@T (seleccionar hasta la duración del resumen) y HIT@K (acertar K jugadas).
- **Los VLM localizan y describen, pero no puntúan jugadas mejor que las señales clásicas.** El pipeline sin entrenar de
  SVHighlights (planos + ASR + VLM + LLM que puntúa) acierta el primer highlight el 27 % de las veces; en MOMENTS un
  Qwen2.5-VL-7B queda en 0,75 de acierto en un problema binario balanceado y Gemini en 0,80; en SoccerBench pasar de 7B a
  72B da +6 puntos. Lo que funciona es extraer señales y dejar que el LLM componga (DIAMOND, WSC, Spiideo).
- **El audio vale más de lo que usábamos.** En SVHighlights un modelo pequeño sobre embeddings de audio supera a la
  visión sola (mAP 41,8 frente a 38,0) y la fusión llega a 46,9; la narración es la mejor pista, pero el sonido de fondo
  solo conserva señal (33,5). La sonoridad por umbral fijo no separa nada.
- **Las posiciones de los jugadores cuentan lo que el balón no deja ver.** Una GNN diminuta sobre tracking reconoce goles
  con recall 73 % frente al 17 % de un modelo de vídeo ("Pixels or Positions", 2026); PathCRF (KDD 2026) infiere posesión
  y golpeos sin balón con F1 76 %; SoccerTrack v2 hace spotting de tiros y goles solo con rasgos de trayectoria en
  cámara panorámica fija; la regla de saque de centro de FIFA/MIT (2022) confirma goles solo con posiciones.
- **Lo que no aplica.** Los modelos de grounding temporal (UniTime, TEMPURA, TimeRefine) son de dominio general, sin
  fútbol; los modelos fundacionales de fútbol (SoccerMaster, MatchVision) se apoyan en rótulos, repeticiones y
  celebraciones de TV (SoccerLens), lo que explica su caída en cámara única.

## 2. Qué significa para nosotros

- Nuestra arquitectura (T-DEED + tracking + marcador + VLM que elige y corta) es la que la literatura valida. No hay
  motivo para sustituirla por un VLM mayor.
- Lo que nos falta es medir con una verdad externa: hoy solo medimos recall de goles en 3 partidos. SoccerHigh lo da
  gratis para TV; para juveniles no existe nada público.
- Las palancas con evidencia son tres: gol sin rótulo desde posiciones (juveniles), señal de peligro sin balón desde
  trayectorias (juveniles) y audio aprendido + detección de planos y repeticiones (TV).

## 3. Siguientes pasos

| # | Paso | Coste | Resultado esperado |
|---|---|---|---|
| 1 | Evaluación en SoccerHigh (45 partidos de test, ya en disco): resolver desfases HQ/720p, reproducir su baseline, evaluador propio con F1@T y HIT@K, oráculos "solo goles", "todos los tiros" y "acciones repetidas" | Sin GPU; bloqueo: fichero `video.ini` (acceso HF a SoccerNet) o alineación con los resúmenes de YouTube | Primera cifra comparable con el estado del arte y techo de las señales simples |
| 2 | Audio y planos en SVHighlights con sus features (PANN, CLIP, fronteras de plano, ASR) sobre el subconjunto de fútbol; después trasladar a SoccerHigh | Sin vídeo, GPU pequeña | Jerarquía de señales en TV y modelo de audio para sustituir el umbral de sonoridad |
| 3 | Nuestro pipeline sobre SoccerHigh: variante ligera (T-DEED + audio + marcador + VLM) en los 45 y completo en 10 | ~18 h + ~30 h de GPU desatendida | Dónde estamos frente al baseline y cuánto aporta el tracking |
| 4 | Posiciones: PathCRF con sus pesos y la regla de saque de centro sobre Sportec Open (7 partidos, abierto) y SoccerTrack v2 (posiciones reales y nuestro tracking del mismo vídeo) | CPU | Saber si la posesión y el gol sin balón aguantan con nuestro tracking antes de entrenar nada |
| 5 | Anotación de juveniles: un "resumen de referencia" por partido (jugadas con inicio, momento clave, fin, tipo, obligatoria, nota) en los 2 partidos de evaluación | Trabajo manual | Única forma de medir fuera de TV y de fijar la política de corte |

Decisiones pendientes: acceso a SoccerNet en Hugging Face para `video.ini`; reparto de GPU entre los pasos 2 y 3;
orden de los pasos 2 y 4.

## 4. Referencias clave

SoccerHigh arXiv 2509.01439 · SVHighlights arXiv 2606.06926 · MOMENTS arXiv 2601.16333 · Xu et al. (audio) arXiv
2609.17923 · Pixels or Positions arXiv 2511.12606 · PathCRF arXiv 2602.12080 · SoccerTrack v2 arXiv 2508.01802 ·
Vidal-Codina et al. arXiv 2202.00804 · Sportec Open DOI 10.1038/s41597-025-04505-y · SoccerLens arXiv 2605.09598 ·
UniTime arXiv 2506.18883 · SoccerBench arXiv 2505.03735.
