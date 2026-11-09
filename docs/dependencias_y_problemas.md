# Problemas conocidos: `Soccer_Master` y dependencias sin pin

> Sección sacada del README el 2026-11-09, sin cambios de contenido.

Sección viva — si vuelves a esto en otra sesión (de Claude Code o no), léela **antes**
de tocar `Soccer_Master/`. Documenta una ronda real de depuración (17 jul 2026) que costó
varias iteraciones porque los síntomas parecían no estar relacionados entre sí.

### Causa raíz común

`requirements.txt` solo fijaba mínimos (`torch>=2.1.0`, `transformers>=4.44.0`), nunca
versiones exactas. Cada `docker build` nuevo resuelve `pip install` contra lo que esté
disponible **ese día**, así que el entorno no es reproducible entre reconstrucciones: algo
que funcionaba hace semanas puede romperse hoy sin que nadie haya tocado una línea de este
repo. Vía web confirmé que hasta el propio "Quick Start" de `Soccer_Master/README.md`
(el repo de terceros vendorizado en `Soccer_Master/SoccerMaster/`) tiene el mismo problema:
instala `transformers` con `pip install git+https://github.com/huggingface/transformers`,
sin fijar commit. O sea, ni siquiera seguir esas instrucciones al pie de la letra habría
dado un entorno reproducible.

**Importante — ese Quick Start (conda, Python 3.10.16, `torch==2.4.1+cu121`,
`pip install -e .` de `sn-gamestate`/`tracklab`/`sam2`, `flash-attn`, `qwen-vl-utils`) es
para el pipeline GSR completo** (tracklab + SAM2 + Qwen2.5-VL: calibración de cámara, ReID,
OCR de dorsales vía VLM). **No es lo que usa este proyecto.** El propio
`Soccer_Master/SoccerMaster/infer_mp4.py` (script que ya funcionaba antes en este repo)
documenta en su docstring que ese pipeline completo está roto/incompleto en este checkout
(faltan `sn_gamestate/configs/dataset/soccernet_gs.yaml` y `tracklab/wrappers/datasets/`), y
por eso existe como alternativa ligera que solo usa `models/multi_task.py` + SigLIP2 + el
head DETR — exactamente lo que envuelve `core/tracking.py:SoccerMasterDetector` y
`core/events.py`. **No instales el stack conda/tracklab/SAM2/Qwen para arreglar esto** —
es una pista falsa, resuelve un problema distinto.

### Incidentes concretos (17 jul 2026) y su fix

| # | Síntoma | Fichero | Causa | Fix |
|---|---------|---------|-------|-----|
| 1 | `RuntimeError: The NVIDIA driver on your system is too old (found version 12080)` al primer `.to("cuda")` | — (dependencia) | `pip install torch` sin pin resolvió a torch 2.13.0, cuyo runtime CUDA por defecto es **13.0** (`requires_dist: cuda-toolkit==13.0.3`); el driver del host (570.211.01) solo soporta hasta CUDA 12.8 | Pin en `requirements.txt`: `torch==2.11.0+cu128` + `torchvision==0.26.0+cu128` vía `--extra-index-url https://download.pytorch.org/whl/cu128` |
| 2 | `AttributeError: 'SiglipVisionModel' object has no attribute 'vision_model'` | `Soccer_Master/SoccerMaster/models/soccer_master.py`, `VisionBackbone.__init__` | En `transformers` viejo, `SiglipVisionModel` anidaba el transformer en `.vision_model`. En `transformers` 5.x, `SiglipVisionModel` **es** el transformer directamente | `siglip_vision_model = getattr(model, "vision_model", model)` (compatible con ambas APIs) |
| 3 | `IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)` dentro de `SiglipAttention.forward` (`q_proj(...).view(...).transpose(1,2)`) | mismo fichero, `ResidualAttentionBlock.forward` | En `transformers` viejo, llamar a una capa del encoder devolvía `(hidden_states, attn_weights)`; el código hacía `[0]` para desempaquetar. En `transformers` 5.x, `SiglipEncoderLayer.forward` devuelve el tensor **directamente** — el `[0]` heredado ya no desempaqueta una tupla, **recorta la dimensión de batch del tensor**, y la forma va quedando corrompida capa a capa hasta que revienta unas capas más adelante (por eso el traceback apunta a un sitio que parece no tener relación con el cambio real) | Helper `_encoder_layer_hidden_states()` que solo indexa `[0]` si el resultado es efectivamente una tupla |

Ambos fixes de la tabla están **verificados línea a línea contra el código fuente real
instalado** de `transformers` (`site-packages/transformers/models/siglip/modeling_siglip.py`),
no son parches a ciegas.

### Extensión CUDA `MultiScaleDeformableAttention` (Deformable DETR) — ya blindada, no tocar

`Soccer_Master/SoccerMaster/models/deformable_detr/ops/` contiene una extensión CUDA que se
compila aparte (`ops/setup.py` / `ops/make.sh`), **no** vía `pip install -r requirements.txt`.
Ya existe:
- Un `.so` compilado (13 jul, para cpython-312): `ops/build/lib.linux-x86_64-cpython-312/MultiScaleDeformableAttention.cpython-312-x86_64-linux-gnu.so`.
- Un fallback defensivo a PyTorch puro (16 jul, ver `ops/functions/ms_deform_attn_func.py` y
  `ops/modules/ms_deform_attn.py`; los `.orig` junto a cada uno son las versiones previas al
  parche): si la extensión no importa o el tensor no está en CUDA, usa
  `ms_deform_attn_core_pytorch` en vez de fallar.

Si veis `ImportError: MultiScaleDeformableAttention` o un error de símbolos indefinidos tras
cambiar de versión de `torch` (el `.so` está enlazado contra el ABI de una versión concreta),
**no es grave**: cae solo al fallback (más lento en esa operación puntual, pero corre). Si
hace falta rendimiento y tocara recompilar:
```bash
cd Soccer_Master/SoccerMaster/models/deformable_detr/ops
python setup.py build install   # necesita nvcc; la imagen Docker (-devel) ya lo trae
```

### Inventario de cambios sobre el `Soccer_Master` original (para git / `.gitignore`)

`Soccer_Master/` es un repo de terceros vendorizado (copia local de
[haolinyang-hlyang/SoccerMaster](https://github.com/haolinyang-hlyang/SoccerMaster/), que a su
vez trae dentro copias de `sn-gamestate`, `tracklab` y `sam2`). Va ignorado en git salvo estos
ficheros — el `.gitignore` del proyecto ya tiene el bloque `Soccer_Master/*` + excepciones en
cascada para que solo estos suban:

| Fichero | Tipo de cambio | Qué es |
|---|---|---|
| `Soccer_Master/SoccerMaster/models/soccer_master.py` | **Modificado** (2 parches, esta sesión) | `VisionBackbone.__init__`: `model.vision_model` → `getattr(model, "vision_model", model)`. `ResidualAttentionBlock.forward`: `self.encoder(...)[0]` → helper `_encoder_layer_hidden_states(...)`. Ver tabla de incidentes #2 y #3 más arriba. |
| `Soccer_Master/SoccerMaster/models/deformable_detr/ops/functions/ms_deform_attn_func.py` | **Modificado** (sesión previa, 16 jul) | Import de `MultiScaleDeformableAttention` envuelto en try/except (`MSDA = None` si no compila). |
| `Soccer_Master/SoccerMaster/models/deformable_detr/ops/modules/ms_deform_attn.py` | **Modificado** (sesión previa, 16 jul) | Fallback a `ms_deform_attn_core_pytorch` si `MSDA is None` o el tensor no está en CUDA. |
| `Soccer_Master/SoccerMaster/infer_mp4.py` | **Añadido** y **borrado el 2026-11-09** (no se usaba; recuperable con `git show 41ed83a:<ruta>`) (sesión previa, 13 jul) | Script de inferencia standalone sobre un `.mp4`, alternativa al pipeline tracklab/SAM2/Qwen (roto en este checkout, ver docstring del propio fichero). No es upstream. |
| `Soccer_Master/SoccerMaster/eval_captions.py` | **Añadido** y **borrado el 2026-11-09** (no se usaba; recuperable con `git show 41ed83a:<ruta>`) (sesión previa, 13 jul) | Evalúa el head `VideoCaption` (retrieval SigLIP-style) contra `clips_analisis/`. No es upstream. |
| `Soccer_Master/SoccerMaster/eval_classification_head.py` | **Añadido** y **borrado el 2026-11-09** (no se usaba; recuperable con `git show 41ed83a:<ruta>`) (sesión previa, 13 jul) | Evalúa el head `CaptionClassification` (softmax, 23 clases fijas) contra `clips_analisis/`. No es upstream. |

Todo lo demás dentro de `Soccer_Master/` (incluyendo `sam2/`, `sn-gamestate/`, `tracklab/`,
checkpoints en `pretrained_models/`, y los artefactos compilados `ops/build/`, `ops/dist/`,
`*.egg-info/`) es upstream sin tocar o binario regenerable/descargable — no va a git. Los
`.orig` junto a los dos ficheros parcheados de `deformable_detr/ops/` (versiones previas al
parche) tampoco van a git; una vez esto esté en un commit, el propio `git diff`/`git blame`
cumple esa función y los `.orig` se pueden borrar si molestan.

Si en el futuro se parchea algo más dentro de `Soccer_Master/`, hay que añadir tanto la fila a
esta tabla como la línea `!ruta/al/fichero` correspondiente en `.gitignore` (respetando la
cascada: cada carpeta intermedia necesita su propia línea `!carpeta/` + `carpeta/*` antes de
poder excepcionar un fichero nieto — si no, la excepción no hace nada porque git no baja a mirar
dentro de una carpeta ya ignorada).

### Versiones fijadas actualmente (RTX 4090, driver 570.211.01, dentro del contenedor)

- `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128` — confirmado: soluciona el incidente #1 (se probó de verdad, el contenedor llegó más allá del punto donde crasheaba).
- `transformers==5.14.1` — versión contra la que se diagnosticaron y parchearon los incidentes #2 y #3. Ambos **confirmados solucionados en ejecución real** (`python -m core.tracking` completo, `--detector hybrid --device cuda`, sin errores).

### Dependencias solo-git sin pin: `prtreid`/`bpbreid` (`core/classification.py`)

Segunda instancia del mismo patrón de riesgo que `transformers`/`torch` sin pin (arriba): PRTReId
(el extractor de embeddings de apariencia que usa `core/classification.py` para asignar equipos)
depende de `prtreid`/`bpbreid`, dos paquetes que **no se distribuyen por PyPI** — la única
instalación documentada por sus propios autores es `pip install git+https://github.com/VlSomers/prtreid`
(sin commit fijado). Un `docker build` en un día distinto puede traer una versión distinta sin
ningún aviso. Si `core/classification.py` con `--embedder prtreid` falla al cargar el modelo o al
extraer embeddings tras reconstruir la imagen, sospecha esto primero — usa `--embedder stub`
mientras tanto para no bloquearte, y compara contra el `cfg`/versión que funcionó la última vez.

**Incidente concreto ya visto (17 jul 2026)**: `from prtreid.scripts.main import build_config`
importa en cascada `prtreid.utils.visualization`, que llama a `matplotlib.cm.get_cmap('hsv')` —
eliminada en `matplotlib` 3.9 (existía, deprecada, hasta 3.8.x). Como `prtreid`/`torchreid` no
pinean `matplotlib` en sus propios requisitos, `pip install` resolvía la última (3.11.1) y
`core/classification.py` reventaba con `AttributeError: module 'matplotlib.cm' has no attribute
'get_cmap'` **al importar `prtreid`, no al usar nada de plotting nuestro** — mismo patrón exacto
que los incidentes de `transformers`/Siglip: una dependencia transitiva sin pin, no algo que
hiciéramos mal. Fix: `matplotlib<3.9` fijado en `requirements.txt`.

Justo detrás, mismo `import prtreid` en cascada, apareció un segundo incidente de naturaleza
distinta: `prtreid.data.data_augmentation.random_occlusion` importa `skimage` a nivel de módulo
(aumento de datos de entrenamiento, que no usamos en inferencia) pero `scikit-image` no está en
el empaquetado de `prtreid` — no es un conflicto de versión, es una dependencia real que falta
del todo (`ModuleNotFoundError: No module named 'skimage'`). Fix: `scikit-image` añadido a
`requirements.txt`.

Un tercer incidente, más profundo, en el mismo fichero: tras resolver `skimage`, ese mismo
`random_occlusion.py` hace `from albumentations import (DualTransform, functional)`, y
`functional` como submódulo importable directamente desde la raíz de `albumentations` es una API
de versiones MUY anteriores a las reorganizaciones internas por tipo de efecto (blur/crops/
geometric/...) que tiene el paquete desde hace varias versiones mayores — **ninguna versión
razonablemente reciente de `albumentations` la expone así**, y bajar a una versión lo bastante
vieja para que exista arriesga romper otra cosa en la cadena de dependencias (numpy 2.x, compat
con Python 3.12, etc.). Confirmado en ejecución real (17 jul 2026):
`ImportError: cannot import name 'functional' from 'albumentations'`.

Esta vez el fix NO es un pin en `requirements.txt` — es código en
`core/classification.py::_stub_prtreid_random_occlusion()`: inyecta un módulo vacío (con un
`RandomOcclusion` dummy, porque otros módulos de `prtreid` sí necesitan que ese nombre exista) en
`sys.modules` **antes** de que `import prtreid` intente cargar el fichero real, así que el import
roto ni se ejecuta. `random_occlusion.py` es aumento de datos para ENTRENAMIENTO — no lo usamos,
solo llamamos a `FeatureExtractor` para inferencia — así que perdernos ese fichero no quita
funcionalidad real. La técnica (pre-poblar `sys.modules` con un stub antes del import que lo
dispara) está verificada de forma aislada, sin `prtreid` instalado, antes de aplicarla. Vive en
nuestro propio código (no en `site-packages`), así que sobrevive a cualquier reconstrucción de la
imagen — a diferencia de parchear el paquete instalado directamente.

Un cuarto incidente, ya no de import sino de **configuración**: una vez `import prtreid` funciona,
`prtreid.scripts.main.build_config(config=yacs_cfg)` llama internamente a
`compute_parts_num_and_names(cfg)`, que necesita ALGÚN dataset registrado bajo el nombre que
digan `cfg.data.sources`/`targets` (`"SoccerNet"` en nuestro `cfg`, copiado del `prtreid.yaml` de
`sn-gamestate`). El propio `prtreid` **no** registra ningún dataset por defecto — solo
`sn-gamestate` lo hace, con su propia `sn_gamestate.reid.prtreid_dataset.ReidDataset` (fuertemente
acoplada a `tracklab`: necesita un `TrackingDataset` completo con splits de train/query/gallery
de ground truth, inviable de replicar solo para inferencia). Confirmado en ejecución real
(17 jul 2026): `ValueError: Invalid dataset name. Received "SoccerNet", but expected to be one of
[...]`. Fix, en `core/classification.py::_register_prtreid_soccernet_dataset()`: `prtreid` SÍ trae
su propia clase `SoccerNet` (`prtreid.data.datasets.image.soccernet.SoccerNet`), importada pero no
registrada por defecto — la registramos nosotros bajo el nombre `"SoccerNet"`. No hace falta que
sea idéntica a la `ReidDataset` de `sn-gamestate`: con `masks.dir="pose_on_img_crops"` (nuestro
`cfg`), tanto una como la otra devuelven `None` en `get_masks_config` (ninguna reconoce esa
clave), así que el resultado final es el mismo da igual cuál se registre. Solo se registra la
clase (nunca se instancia), así que no dispara ningún escaneo de disco de un dataset que no
tenemos.

Un quinto incidente, ya sin relación con imports: una vez `build_config` puede resolver el
dataset, intenta leer el propio checkpoint (`load_checkpoint(cfg.model.load_weights)`, dentro de
`prtreid.utils.torchtools`) llamando a `torch.load(fpath, map_location=...)` sin especificar
`weights_only`. Desde PyTorch 2.6 el valor por defecto de ese argumento cambió de `False` a
`True`, y el checkpoint publicado de PRTReId (un pickle con objetos `numpy.core.multiarray.scalar`
de una versión de numpy antigua) no pasa esa comprobación de seguridad por defecto. Confirmado en
ejecución real (17 jul 2026): `_pickle.UnpicklingError: Weights only load failed ... Unsupported
global: GLOBAL numpy.core.multiarray.scalar`. Fix, en
`core/classification.py::_patch_torch_load_weights_only_false()`: parchea `torch.load` para que
use `weights_only=False` por defecto, con ámbito acotado al proceso (no toca `site-packages` ni
desactiva la comprobación de seguridad para el resto del proyecto) — aceptable porque el
checkpoint viene de una URL de Zenodo fija que ya verificamos por MD5 antes de usar, no es un
pickle arbitrario de origen desconocido.

✅ **Con estos cinco fixes, el pipeline completo de `core/classification.py --embedder prtreid`
corre de punta a punta** (verificado en ejecución real, 17 jul 2026, sobre
`datasets/clips_analisis/corner/corner118.mp4`: 23 tracklets → 11 vs 12 por equipo, 5 excluidos
como `never_player`, 778 crops embebidos, 0 degenerados). De paso se confirmó empíricamente algo
que en el plan original quedaba marcado como suposición sin verificar: PRTReId normaliza los
crops con estadísticas estándar de ImageNet (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
0.225]`) tras un resize a 256×128 — se ve en el log de `FeatureExtractor` al construirse.

Nota menor, no un bug: `build_config` crea al final un directorio numérico vacío en la raíz del
proyecto (`os.makedirs(cfg.data.save_dir)`, con `cfg.project.job_id` sin fijar explícitamente por
nosotros) — es un directorio de logs de `prtreid` que no usamos para nada; se puede borrar sin
miedo (`rmdir <número>`) y no debería ir a git.

**Patrón a esperar**: `import prtreid` arrastra un árbol de imports grande (data augmentation,
visualización, engines de entrenamiento...) que no necesitamos para inferencia pero que igualmente
tiene que resolver en el momento de importar. Es plausible que aparezcan más
`ModuleNotFoundError`/`ImportError`/`AttributeError` de este estilo la primera vez que se
reconstruya la imagen desde cero — no asumas que el próximo será alguno de los tres de arriba
otra vez; sigue el traceback hasta el import que falla y decide caso por caso si el fix es un pin
en `requirements.txt` (si es una API que existe en OTRA versión del paquete) o un stub en
`sys.modules` como este (si es un fichero de `prtreid` que no necesitamos para nada y cuya API
esperada ya no existe en ninguna versión razonable).

Tercera instancia del mismo patrón, ya resuelta: `core/homography.py` depende de
[PnLCalib](https://github.com/mguti97/PnLCalib), que ahora se clona en
`models/homography/PnLCalib/` (ignorado por
git, igual que `Soccer_Master/`) y está verificado end-to-end. Dio exactamente el fallo esperado
del patrón: `utils/utils_optimize.py` importa `shapely`, que no estaba instalado —
`ModuleNotFoundError` en el primer `import utils.utils_calib`. Añadido a `requirements.txt`.

Reutiliza `core.tracking._repo_import_context` sin cambios: PnLCalib define un paquete namespace
`utils` que choca con el `utils/` regular del proyecto, que es literalmente el mismo problema que
ese context manager ya resolvía para `Soccer_Master`.

Puntero cruzado para quien llegue a depurar cualquiera de las tres: `torch`/`transformers`
(pineados en `requirements.txt`, incidentes documentados arriba), `prtreid`/`bpbreid` (sin pin,
esta sección), `PnLCalib` (`shapely`, esta sección).

### A vigilar (no rompe todavía): `scikit-learn` sin techo y el `.joblib` de `core/events.py`

Visto en `analytics/engine.py` (17 jul 2026), solo como warning, no error: `InconsistentVersionWarning:
Trying to unpickle estimator ... from version 1.7.2 when using version 1.9.0`. El clasificador de
eventos (`svc_classifier_more_penalties.joblib`) se serializó con scikit-learn 1.7.2; `requirements.txt`
solo fija `scikit-learn>=1.4.0` (sin techo), así que un `docker build` en otro momento puede traer
una versión bastante más nueva. Por ahora sigue funcionando (mismo patrón de riesgo que
`transformers`/`prtreid` sin pin, todavía sin materializarse en rotura real) — si en algún momento
`core/events.py` empieza a fallar al cargar el `.joblib`, o las predicciones se vuelven raras sin
ningún error visible, fijar `scikit-learn==1.7.2` (o reentrenar/re-serializar el clasificador con
la versión que esté instalada) es el primer sitio a mirar.

### Si algo se rompe otra vez tras un `docker build`

1. Comprobar que `requirements.txt` sigue con las versiones pineadas de arriba — si alguien
   las soltó "para probar algo", ese suele ser el problema.
2. Si aparece un `AttributeError`/`IndexError` nuevo **dentro de `Soccer_Master/SoccerMaster/models/`**,
   sospechar el mismo patrón: `transformers` cambió otra API interna de Siglip/DETR entre la
   versión pineada y la recién instalada. Comparar el traceback contra el código fuente
   REAL instalado (`site-packages/transformers/models/siglip/modeling_siglip.py` o el módulo
   que corresponda) en vez de asumir que el código vendorizado de Soccer_Master tiene razón
   — casi siempre es el vendored code el que quedó desactualizado, no `transformers`.
3. No instalar el stack conda/Python 3.10/tracklab/SAM2/Qwen del Quick Start oficial — es
   para el pipeline GSR completo, que este proyecto no usa (ver más arriba).
4. Los parches viven directamente en los ficheros vendorizados (no hay un `.patch`/diff
   aparte), y esos ficheros son propiedad de `root` en el host aunque el contenedor corra
   como root — si una sesión de Claude Code sin acceso root al host necesita tocarlos, tiene
   que dárselos un comando para que lo ejecutéis vosotros dentro del contenedor (como se hizo
   aquí), no puede escribirlos directamente desde fuera.
