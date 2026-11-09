# Uso — homografía / calibración de campo (`core/homography.py`)

> Sección sacada del README el 2026-11-09, sin cambios de contenido.

Convierte el tracking de píxeles a **metros sobre un campo canónico de 105x68**, que es lo que
hace comparables las posiciones entre frames, entre cámaras y entre partidos.

### Dependencias (no versionadas, hay que traerlas a mano)

```bash
# 1) Código de PnLCalib (solo el código; no lo tocamos, por eso va entero al .gitignore)
git clone https://github.com/mguti97/PnLCalib.git models/homography/PnLCalib

# 2) Pesos (releases v1.0.0 del mismo repo) -> models/homography/weights/
mkdir -p models/homography/weights && cd models/homography/weights
curl -L -o SV_kp    https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp
curl -L -o SV_lines https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines
```

### Ejecución

```bash
# Integrado en el pipeline (etapa entre tracking y analytics)
python -m main --source partido.mp4 --output-dir output/partido --mode full-match --radar

# Standalone (requiere tracks.jsonl previo)
python -m core.homography --source partido.mp4 --output tracks_2d.jsonl --radar-video radar.mp4
```

Salidas: `<video>.homography.jsonl` (caché de matrices H, junto al vídeo como tracks/possession),
`tracks_2d.jsonl` (tracking + `field_xy` en metros) y opcionalmente `radar.mp4` (vista cenital).

### Sistema de coordenadas — ojo con esto

PnLCalib trabaja **centrado** (X ∈ [-52.5, 52.5], Y ∈ [-34, 34]): su `keypoint_world_coords_2D`
aplica `[x - 52.5, y - 34]`. Este módulo NO propaga esa convención — traslada a **origen en la
esquina** (X ∈ [0, 105], Y ∈ [0, 68]) antes de devolver nada. Todo lo que consuma `field_xy` debe
asumir esquina. Si algún día los puntos salen todos en el cuadrante negativo, este es el motivo.

### Rendimiento (medido en la RTX 4090 del contenedor)

Es la etapa más cara del pipeline y es GPU-bound: son dos HRNet-W48 a 960x540 por frame. El solve
de la homografía en CPU (RANSAC) es 0.5 ms/frame, despreciable.

| | batch=1 | batch=4 | batch=8 |
|---|---|---|---|
| fp32 | 58 ms/frame | 41 ms/frame (6.1 GiB) | 42 ms/frame (11.6 GiB) |
| **fp16** | 43 ms/frame | **26 ms/frame (4.7 GiB)** | 24 ms/frame (8.7 GiB) |

Defaults: `half=True`, `batch_size=4`, `stride=5`. El parámetro que manda es `stride`: a 25 fps un
partido de 90 min son 135k frames, o sea ~1 h a stride=1 y ~12 min a stride=5. Una calibración
cada 0.2 s sobra para el paneo suave de una cámara de retransmisión.

### Qué esperar de la calidad (medido sobre `ground_truth/`, 20 min de partido real)

- **74% de los frames** acaban con homografía (el resto: repeticiones, planos cortos, pocas líneas
  visibles). Se marcan con `homography_ok: false` en vez de arrastrar una H de otro plano.
- **95% de los jugadores proyectados caen dentro del campo**, 99.3% con 5 m de margen.
- Error de reproyección mediano ~1.9 px.

⚠️  **La velocidad instantánea NO es usable todavía**: mediana 5.1 m/s, con 21% de muestras por
encima de 12 m/s (físicamente imposible). No es culpa de la calibración — se comprobó que los
pares de frames que comparten la MISMA H tienen peor velocidad (5.51 m/s) que los que cruzan H
distintas (4.20), así que el jitter no viene de la homografía. Viene de arriba: el `anchor_xy` del
tracker se mueve 3.7 px/frame de mediana y la perspectiva amplifica eso a metros en el fondo del
campo. **Antes de publicar cualquier métrica física hace falta suavizado temporal en el espacio de
campo** (Kalman/savgol sobre `field_xy` por `tracker_id`), que es el siguiente paso natural.

### Proyecciones descartadas

`project_tracks` no escribe `field_xy` cuando el punto no es proyectable (3.4% de los casos):
`w < 0` (el punto cae detrás del plano de cámara y la división proyectiva devuelve un punto
reflejado de aspecto plausible) o `w ≈ 0` (sobre la línea del horizonte, la homografía diverge y
proyecta a cientos de km). Se verificó que estos descartes **no tienen sesgo de profundidad**
(mediana de `y_img` 356.8 en descartados vs 358.5 en aceptados), así que no se está perdiendo
sistemáticamente al equipo del fondo: son detecciones fuera del campo (banquillo, público).

El balón se proyecta asumiendo z=0 y se marca con `field_xy_assumes_ground: true` — cuando está
en el aire la proyección es incorrecta por construcción.
