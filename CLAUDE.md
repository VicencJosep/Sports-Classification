# Sports-Classification

Análisis de partidos de fútbol a partir de vídeo. Resumen del pipeline y de las carpetas: `README.md` (corto, se puede leer entero).

## Reglas

- El código corre en el contenedor `Sports-Classification` (repo en `/app`), no en el host:
  `docker exec -w /app Sports-Classification python -m pytest tests -q`.
- Los commits y los push los hace el usuario.
- Documentación y comentarios en castellano.

## Gastar pocos tokens

- La salida de Bash es lo que más contexto consume (59 % medido): acótala con `tail`, `head` o `grep`, y manda los
  logs largos a un fichero dentro de `output/`.
- No leas enteros los documentos largos: busca con `grep` y lee la sección.
- No explores salvo que la tarea lo pida: `datasets/`, `models/`, `output/`, `runs/`, `temp/`, `Soccer_Master/`,
  `docs/archivo/`, `scripts/shot_benchmark/` y `scripts/ball_detector/` (experimentos de tiros, panorámica y balón).

## Qué leer según la tarea

| Tarea | Fichero |
|---|---|
| Qué toca ahora, decisiones y vías descartadas | `docs/hoja_de_ruta.md` §0, §2 y §4 (38 KB: por secciones) |
| Una cifra de benchmark | tabla de `docs/comparativa_benchmark.md` §7 y luego solo su fichero en `docs/comparativa_benchmark/` |
| Qué datos hay y con qué se evalúa | `docs/datasets.md` §3 y `docs/particion_datasets.md`; catálogo externo en `docs/datasets_catalogo/` |
| Usar o tocar un módulo | `docs/uso/<módulo>.md`; valores por defecto en `config.py` |
| Mejores jugadas | `docs/highlights_evaluacion.md`, `docs/highlights_soccerhigh.md` |
| Tracking | `docs/estado_arte_tracking.md` y la fila 7.10 de la comparativa |
| Fallos de dependencias o del build | `docs/dependencias_y_problemas.md` |
| Un artículo (T-DEED, AdaSpot, SoccerNet 2026) | `docs/papers/<nombre>.md` por secciones con `grep` (50-115 KB cada uno); no leas el PDF |
| Qué trasladar de PAVE (ganador de PCBAS 2026) a nuestros spotters | `docs/pave_valoracion.md` §0 y §6 |
| API | `backend/README.md` |

## Al documentar

- Resultado nuevo de benchmark: fichero `docs/comparativa_benchmark/7.NN_*.md` y una fila en la tabla de §7.
- Fin de sesión: entrada completa al final de `docs/archivo/hoja_de_ruta_bitacora.md` y una línea en el §7 de la hoja de ruta.
- Mantén cortos los documentos de `docs/`: el detalle va a su carpeta y el resumen se queda en una línea.

## Trampas

- `Soccer_Master/` sigue en uso: es el detector de personas por defecto (`--detector hybrid`) y sus pesos (5 GB) están
  en `Soccer_Master/sn-gamestate/pretrained_models`.
- Una carpeta llamada `datasets/` queda ignorada por git a cualquier profundidad (por eso existe `docs/datasets_catalogo/`).
