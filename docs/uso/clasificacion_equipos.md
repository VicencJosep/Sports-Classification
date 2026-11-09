# Uso — clasificación de equipos (`core/classification.py`, PRTReId + K-Means)

> Sección sacada del README el 2026-11-09, sin cambios de contenido.

Asigna cada `tracker_id` de un vídeo ya trackeado a uno de 2 equipos, por apariencia. Puerto del
algoritmo real de `Soccer_Master/sn-gamestate/sn_gamestate/team/tracklet_team_clustering_api.py`
(`TrackletTeamClustering`) fuera de tracklab/Hydra/pandas — no es una reimplementación desde cero.

```bash
python -m core.tracking --source partido.mp4 --output tracks.jsonl --detector hybrid --device cuda

python -m core.classification --source partido.mp4 --tracks tracks.jsonl --output teams.json --device cuda
# --embedder stub corre el mismo pipeline sin GPU ni prtreid instalado (color medio del
# crop en vez de un embedding aprendido) — útil para probar el resto del pipeline en seco.
```

Algoritmo (idéntico al original, verificado línea a línea contra su código fuente):

1. Recorta cada detección con `role=="player"` de `tracks.jsonl` (goalkeeper/referee/ball quedan
   fuera del clustering, igual que en `TrackletTeamClustering`).
2. Extrae un embedding de apariencia de 256-d por crop vía **PRTReId** (BPBReID + backbone
   HRNet32, entrenado en SoccerNet) — o, con `--embedder stub`, el color medio BGR del crop.
3. Promedia los embeddings de cada `tracker_id` (media, no mediana ni "frame más nítido" —
   así lo hace el original) y separa esas medias con `KMeans(n_clusters=2, random_state=0)`
   (sin normalizar antes, igual que el original).

`teams.json` (un único objeto, no JSONL — es un resultado a nivel de vídeo completo):

```json
{
  "source_video": "partido.mp4", "tracks_jsonl": "tracks.jsonl",
  "embedder": "prtreid", "embedding_dim": 256, "n_clusters": 2, "random_state": 0,
  "teams": [{"tracker_id": 7, "team_cluster": 0}, {"tracker_id": 12, "team_cluster": 1}],
  "excluded": [{"tracker_id": 3, "reason": "never_player"}, {"tracker_id": 45, "reason": "no_valid_crops"}],
  "n_frames_processed": 812, "n_crops_embedded": 5123, "n_crops_skipped_degenerate": 2
}
```

Join trivial para `analytics/engine.py`: `{t["tracker_id"]: t["team_cluster"] for t in teams["teams"]}`.

### Vídeo anotado por equipo (`--annotated-output`)

```bash
python -m core.classification --source partido.mp4 --tracks tracks.jsonl --output teams.json \
       --annotated-output teams.mp4 --device cuda
```

Escribe un `.mp4` con cajas + etiqueta coloreadas por **equipo** (rojo/azul; gris para quien no
tenga equipo asignado — árbitros, porteros, tracks excluidos) en vez de por `tracker_id` como hace
`core/tracking.py --annotated-output`. Es una **segunda pasada** sobre el vídeo (decodifica de
nuevo, barato, sin re-ejecutar el detector ni PRTReId) — necesaria porque el equipo de cada
jugador no se conoce hasta que se ha visto el vídeo entero (K-Means necesita todos los tracklets a
la vez). La reutiliza `utils/visualizers.py::render_annotated_video`, que lee directamente
`tracks.jsonl` reconstruyendo las detecciones — no depende de `core.tracking` en absoluto, así que
también sirve para volver a renderizar sin tener el pipeline de tracking cargado en memoria.

✅ **Verificado visualmente en ejecución real** (17 jul 2026, mismo clip) contra un frame extraído
del vídeo resultante: dos grupos de jugadores bien separados por color, los 4 árbitros del clip
correctamente en gris con su rol visible, y un jugador lejos del resto del grupo agrupado
correctamente con su equipo por apariencia (no por cercanía en el campo — confirma que usa el
embedding real).

✅ **Verificado en ejecución real** (17 jul 2026, `datasets/clips_analisis/corner/corner118.mp4`,
`--device cuda`) de punta a punta con `--embedder prtreid`. `prtreid`/`bpbreid` no están
vendorizados ni se distribuyen por PyPI — se instalan vía git sin versión fijada (ver
`requirements.txt` y § "Problemas conocidos" más abajo, que documenta los cinco incidentes reales
que hubo que resolver para llegar hasta aquí: dos imports rotos parcheados con stubs en
`sys.modules`, un dataset que había que registrar a mano, y un `torch.load` que había que forzar a
`weights_only=False`). Si tras reconstruir la imagen algo vuelve a fallar al cargar el modelo o
extraer embeddings, empieza por ahí — usa `--embedder stub` mientras tanto para no bloquear el
resto del pipeline.

Explícitamente fuera de alcance (ver roadmap): etiquetado izquierda/derecha de equipo (ya es
posible — `core/homography.py` da la posición en metros, solo falta implementarlo) y OCR de
dorsales (la otra mitad de este módulo en el roadmap).
