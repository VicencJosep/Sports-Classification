# Uso — mejores jugadas (`analytics/highlights`)

> Sección sacada del README el 2026-11-09, sin cambios de contenido.

Tras analizar un partido, selecciona 3-6 jugadas (goles, ocasiones, carreras, rachas de tiros) y las recorta en
clips con la jugada entera: arranque de la posesión, momento clave y celebración/repeticiones. No es "gol ± 5 s":
los candidatos salen de reglas sobre lo que ya calculó el pipeline, y un VLM decide cuáles valen y dónde cortar.

```bash
# Partido analizado por el backend (rutas resueltas desde backend/storage):
docker exec -w /app Sports-Classification python -m analytics.highlights \
    --backend-match 433f62c92b727870 --out output/highlights/mallorca
# Rutas explícitas (lo que deja `python -m main --mode full-match`):
docker exec -w /app Sports-Classification python -m analytics.highlights \
    --video partido.mp4 --pipeline-output output/mipartido --out output/mipartido/highlights
# --no-vlm: solo reglas (19 s/partido).  --no-cut: sin clips.  --force: rehacer señales y veredictos.
```

Salida en `--out`: `highlights.json` (clips con inicio/fin, etiqueta, título y descripción en castellano,
puntuaciones y candidatos), `clip_NN.mp4` (resolución original, recodificado para corte exacto, fundidos de
0,4 s) y las cachés `signals.npz` / `candidates.json` / `vlm_cache.json`.

**Cómo funciona** (`config.HighlightsConfig` tiene todos los umbrales y pesos):

1. **Señales** (`signals.py`, rejilla de 0,5 s): curvas de T-DEED de las dos cabezas SIN umbral (un gol de
   juvenil puntúa SHOT 0,1-0,2 y no llega a `telemetry.json`), excitación del audio (RMS sobre la mediana móvil
   de 5 min, sostenida 15 s: una celebración dura, un silbato no), poseedor/equipo y posiciones en metros.
2. **Candidatos** (`candidates.py`): anclas = picos de tiro/gol, picos de audio con el balón en juego y eventos
   emitidos; la jugada crece hacia atrás por la cadena de acciones del equipo atacante hasta el balón parado
   anterior. Puntuación por reglas (tiro, gol, audio, pases, metros de conducción, tiros en 60 s, cercanía a
   portería, saque de centro después).
3. **VLM** (`vlm.py`, Qwen3.5 con visión; por defecto el Ollama de `localhost:11434`, también vLLM u otro
   servidor OpenAI-compatible con `vlm_api="openai"`): juicio por jugada con frames densos (1/s) en el
   momento clave, **ranking relativo** de todos los candidatos en una sola llamada de texto, y clasificación
   de los frames posteriores (acción / celebración / grada / repetición / vuelta / juego) para fijar el fin.
4. **Gol = VLM + contexto**: el modelo confunde gol con parada; lo fiable es lo que pasa después (saque de
   centro = gol; córner o banda = no) y el griterío. Sin audio, un gol que solo afirma el VLM se queda en
   "ocasión". Dos jugadas casi seguidas (parada → córner → gol) van en un mismo clip.

Medido (2026-10-16): Mallorca–Barcelona juvenil, los 2 goles seleccionados (ninguno estaba en la telemetría)
y ningún gol falso; Espanyol–Damm (0 goles) y Middlesbrough (20 min, sin audio) sin goles falsos. Con VLM
tarda ~2 min por partido (3-4 s por consulta con la GPU libre; Ollama y el pipeline no caben a la vez en la
4090, por eso el backend los serializa).

Desde el backend: casilla "Generate highlights automatically" al subir el vídeo (`POST /api/matches/{id}/analysis`
con `{"highlights": true}`) o pestaña *Highlights* del partido (`POST .../parts/{part}/highlights`); los clips se
reproducen y descargan desde ahí. Subirlos a YouTube con `automation/molmo_publisher.py` queda como siguiente paso.
