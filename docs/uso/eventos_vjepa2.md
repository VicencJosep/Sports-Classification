# Uso — módulo de eventos (`core/events.py`, standalone)

> Sección sacada del README el 2026-11-09, sin cambios de contenido.

Clasifica un clip .mp4 cualquiera (sin necesidad de etiqueta ni nombre especial):
extrae embeddings de V-JEPA2 (`facebook/vjepa2-vitl-fpc64-256`, 64 frames muestreados
uniformemente por timestamp) y los pasa por un `Pipeline(StandardScaler + SVC)` ya
entrenado (`svc_classifier_more_penalties.joblib`).

```bash
# Un único clip -> resultado por stdout
python -m core.events --source clip.mp4

# Carpeta de clips -> un JSONL con una predicción por línea
python -m core.events --source clips_analisis/goal/ --output events.jsonl
# --classifier / --model-id / --device permiten sobreescribir los defaults
```

Cada predicción (`EventPrediction.to_telemetry()`):

```json
{
  "video_path": "clips_analisis/goal/corner100.mp4",
  "label": "goal",
  "confidence": 0.874,
  "probabilities": {"card": 0.01, "corner": 0.02, "...": "...", "goal": 0.874, "...": "..."},
  "is_background": false
}
```

El clasificador reconoce **16 clases**: 9 de evento (`card`, `corner`, `foul`, `goal`,
`kick-off`, `penalty`, `shot`, `substitution`, `throw-in`) y 7 de "fondo" — planos sin
evento (primeros planos, público, cámara principal...) que evitan que el modelo dispare
eventos falsos durante callejones del partido. La taxonomía es deliberadamente más
gruesa que la de `clips_analisis/`/`dataset_vjepa_backup/` (que sí distinguen
`shots_on_target`/`shots_off_target` y `yellow`/`red_card`): se fusionaron a propósito
en `shot`/`card` porque esa frontera era difusa para el modelo y perjudicaba la
precisión — es una decisión de diseño, no una limitación pendiente.

**Limitación de diseño importante**: `sample_frames` reparte los 64 frames por igual a
lo largo de TODA la duración del vídeo de entrada. El módulo espera un **clip ya
recortado en torno a un único evento** (unos pocos segundos, como los de
`clips_analisis/`), no un partido completo — si le pasas 90 minutos sin trocear, la
señal del evento se diluye entre miles de frames irrelevantes y la predicción no será
útil. Trocear un partido completo en clips (sliding window / detección de escenas) es
tarea de una capa superior (`analytics/engine.py` o `main.py`), pendiente de implementar.

## Instalación (nota que estaba en la sección de instalación del README)

El modelo `facebook/vjepa2-vitl-fpc64-256` se descarga de Hugging Face en la primera ejecución (~1,2 GB) y requiere un
`transformers` reciente (V-JEPA2 se integró en la librería en 2025; si `from transformers import VJEPA2Model` falla,
`pip install -U transformers` dentro del contenedor). Desde el 2026-10-15 el pipeline no usa este módulo por defecto
(`--event-source tdeed`); se activa con `--event-source mixed` o `vjepa2`.
