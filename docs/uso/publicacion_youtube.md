# Uso — publicación en YouTube Studio (`automation/molmo_publisher.py`)

> Sección sacada del README el 2026-11-09, sin cambios de contenido.

Último eslabón del pipeline: sube el `final.mp4` de `analytics/engine.py` a YouTube Studio
sin escribir un solo selector CSS. El bucle del agente es:

```
page.screenshot()  ->  MolmoWeb-4B ("point to the CREAR button")  ->  (x, y)  ->  page.mouse.click()
```

### Por qué un VLM y no selectores

YouTube Studio es una SPA de Polymer con Shadow DOM y clases ofuscadas que cambian entre
despliegues: cualquier XPath que escribamos hoy se rompe solo. MolmoWeb-4B, en cambio, es un
**agente web** (arquitectura Molmo 2: Qwen3-8B + SigLIP 2): se le da un objetivo en lenguaje
natural y responde con `{"thought": ..., "action": {"name": "click", "x": 34.6, "y": 8.7}}`,
en **porcentaje** del ancho/alto de la imagen. Cuando YouTube rediseñe la UI hay que retocar
objetivos en lenguaje natural (`StudioPrompts`), no selectores.

Ojo, que es fácil equivocarse aquí (yo lo hice en la primera versión): **no** es el Molmo de
2024 con *pointing* suelto (`point to X` → `<point x=… y=…>`). Cambia todo lo que lo rodea:

| | Molmo 2024 (`Molmo-7B-D-0924`) | MolmoWeb-4B (Molmo 2) |
|---|---|---|
| Carga | `AutoModelForCausalLM` | `AutoModelForImageTextToText` |
| Preproceso | `processor.process(...)` | `processor.apply_chat_template(...)` |
| Generación | `model.generate_from_batch(...)` | `model.generate(...)` |
| Prompt | `"point to the X"` | `"molmo_web_think: "` + plantilla Jinja (GOAL / PREVIOUS STEPS / CURRENTLY ACTIVE PAGE) |
| Respuesta | `<point x="52.3" y="14.7">` | `{"thought":…, "action":{"name":"click","x":…,"y":…}}` |
| Coordenadas | % de la imagen | **% de la imagen (igual)** |

`parse_molmo_response()` entiende los dos dialectos, así que `--model-id` puede apuntar a
cualquiera de las dos familias. Y hay dos detalles que sí cambian el diseño:

**Es un modelo de trayectoria**, no un localizador sin memoria: `_TrajectoryPointer` guarda
los pasos dados y los reinyecta como `PREVIOUS STEPS`, que es lo que le permite interpretar la
pantalla siguiente sabiendo que ya pulsó "CREAR".

**Es un agente completo, y a veces la respuesta correcta no es un clic.** Su espacio de
acciones incluye `goto`, `scroll`, `keyboard_type`, `send_msg_to_user`. Cuando devuelve una de
esas, no es que no haya encontrado el elemento: ha decidido que tocaba otra cosa.
`describe_non_click_action()` separa ambos casos en el log porque el arreglo es distinto
(contexto/prompt vs. timing). Pasó de verdad en el banco de pruebas: con la URL por defecto
(`studio.youtube.com`) sobre una página que era otra cosa, respondió
`goto(url='https://studio.youtube.com')` — decisión correcta para el estado que le habíamos
descrito. Por eso `predict_and_click` le pasa ahora el `page.url` y el `page.title()` reales:
**si le mientes sobre dónde está, razona sobre la mentira**.

El porcentaje está verificado contra su propio dataset de grounding
(`allenai/MolmoWeb-SyntheticGround`): en imagen de 1288×712, `bbox [257.4, 38.0, 693.4, 74.0]`
(centro 36.9% / 7.9%) ↔ respuesta del modelo `x=34.6, y=8.7`. Dentro de la caja, en las 5
muestras comprobadas. Y su agente oficial lo dice explícito en `convert_action_json_to_action_obj`:
*"Coordinates are expected as percentages (0-100) and converted to pixels using the screenshot
dimensions"*, con el recorte a `[1, dim-2]` que replica `Point.to_viewport_px`.

Dos detalles no obvios del mapeo de coordenadas, y ambos son fuente de clics desplazados:

- Las capturas se toman **siempre** con `full_page=False`. Una captura de página completa
  produce coordenadas fuera del viewport que `page.mouse.click` no puede alcanzar.
- El porcentaje se multiplica por el **viewport en píxeles CSS**, nunca por el tamaño en
  píxeles de la captura (que va multiplicado por el `deviceScaleFactor` de la pantalla).
  Normalizar a fracción [0, 1] es lo que desacopla ambas escalas (`Point.to_viewport_px`).

Las dos únicas excepciones a "todo por VLM" están justificadas: el selector de ficheros es
una ventana **nativa del SO** (no sale en la captura, se intercepta con `expect_file_chooser`),
y el chequeo de "¿se cerró el diálogo?" usa el tag del web component `ytcp-uploads-dialog`,
que es un nombre de elemento, no una clase generada por el bundler.

### Sesión de Google (una vez): display virtual + VNC

El login **no** se automatiza: es justo lo que Google protege con 2FA y CAPTCHAs. Se hace a
mano una vez y `launch_persistent_context` guarda las cookies en `automation/chrome_session/`
(en `.gitignore` — son credenciales de sesión) para todas las ejecuciones siguientes.

Pero este host es **headless**: `DISPLAY` vacío, sin sockets en `/tmp/.X11-unix`, sesión `tty`.
`headless=False` no tiene dónde dibujar. Por eso `automation/vnc_session.sh` levanta un display
virtual (Xvfb) + servidor VNC dentro del contenedor:

```bash
# dentro del contenedor
bash automation/vnc_session.sh start     # imprime la contraseña VNC generada
```

Desde tu portátil, túnel SSH contra la IP del contenedor en la red bridge (no hace falta
publicar puertos ni reiniciar nada) y cliente VNC a `localhost:5900`:

```bash
ssh -L 5900:172.17.0.3:5900 <usuario>@<host>
```

Y ya en el contenedor, con el display puesto:

```bash
export DISPLAY=:99
python -m automation.molmo_publisher --login-only
```

Requiere en el contenedor (ya instalado): `apt-get install -y xvfb x11vnc` y
`playwright install --with-deps chromium`.

Dos trampas que nos costaron un rato y están resueltas dentro del script, por si vuelves a
tocarlo:

- **`pkill -f "Xvfb :99"` se mata a sí mismo**: el patrón casa con la propia línea de comandos
  del shell que lo ejecuta. De ahí los patrones tipo `"[X]vfb :99"`.
- **Zombis**: el PID 1 del contenedor es `tail -f /dev/null`, que no recolecta hijos huérfanos.
  Cuando `x11vnc` muere deja un `<defunct>` que `pgrep` sigue listando, y el script se creía
  arrancado con el puerto 5900 sin escuchar a nadie. `alive()` filtra por estado, y `status`
  comprueba el puerto de verdad, no el proceso.

### Ejecución

```bash
# Ensayo en seco: pide coordenadas a Molmo y vuelca capturas con el punto marcado,
# SIN hacer un solo clic. Es la forma de validar prompts antes de tocar nada.
python -m automation.molmo_publisher --video output/miclip/final.mp4 \
    --dry-run --debug-dir output/miclip/publish_debug

# Subida real (privada por defecto)
python -m automation.molmo_publisher --video output/miclip/final.mp4 \
    --title "Test Clip Short" --visibility private --output output/miclip/publish.json
```

Integrado en el pipeline, encadenado al render del clip:

```bash
python -m main --source clip.mp4 --output-dir output/miclip --mode clip \
    --publish --publish-title "Test Clip Short"
```

`--publish` solo aplica a `--mode clip` (publica `final.mp4`) y escribe `publish.json` en
`--output-dir`. Si la publicación falla **no** tumba la ejecución: el análisis
(`telemetry.json` + `final.mp4`) ya está en disco y es el entregable de verdad.

Visibilidad limitada a `private`/`unlisted` a propósito: esto es un banco de pruebas, no
hay opción `public`. Todo lo configurable vive en `config.PublisherConfig` (viewport,
pausas, modelo, perfil de Chromium).

### Qué se rompe y cómo se depura

Las pausas antes de cada captura (`settle_ms`, `modal_settle_ms`, `upload_settle_ms`) no son
paranoia: Studio anima modales, y una captura tomada a mitad de la transición le da al VLM
una UI a medio pintar (opacidad/escala intermedias) con la que las coordenadas salen
desplazadas. `--debug-dir` guarda cada paso con una cruz roja en el punto predicho, que es
lo que distingue los dos modos de fallo: si la cruz apunta a otro elemento el problema es
el **prompt**; si apunta a una UI a medio dibujar, es el **timing**.

Un aviso sobre el **idioma de la UI**: Studio se muestra en el idioma de la cuenta, y los
prompts citan la etiqueta española con su traducción inglesa al lado (`"CREAR" (Create)`)
precisamente para no depender de eso. La pantalla de login sin sesión sale en inglés
(`hl=en`), así que hasta el primer login real no sabremos en qué idioma queda el asistente.

Requiere `playwright install --with-deps chromium` en el contenedor (ver § Instalación) y
GPU libre para el VLM: ~16 GB en bfloat16. Los autores **recomiendan float32**, que no cabe
en una 4090 de 24 GB (el modelo lleva Qwen3-8B de base); si las coordenadas salen erráticas,
`config.PublisherConfig.dtype` es la primera sospechosa.

### El VLM corre en OTRO intérprete (`/opt/venv-molmo`)

Esto es lo más importante que hay que saber antes de tocar nada aquí. El checkpoint se guardó
con **transformers 4.57.3** y su repo oficial pinea `transformers>=4.51,<5`. Este repo fija
`transformers==5.14.1` porque lo exige la API de Siglip de `Soccer_Master`. Son incompatibles
y ninguna de las dos versiones es negociable.

Con transformers 5.x el código remoto del checkpoint falla **en cadena** (los tres, medidos el
17 ago 2026): `KeyError: 'default'` en `ROPE_INIT_FUNCTIONS` → `TypeError: Unexpected keyword
argument image_use_col_tokens` en `ProcessorMixin` → `cache_position` a `None` en
`prepare_inputs_for_generation`. Se llegó a parchear los dos primeros y funcionaban, pero se
descartó esa vía: cada shim reimplementa a mano un trozo del delta 4.x→5.x, y el siguiente
fallo de ese tipo puede no ser una excepción ruidosa sino **coordenadas mal calculadas en
silencio**.

Solución: el VLM vive en `/opt/venv-molmo` y se le habla por tubería.

```
proceso principal  (/opt/venv, transformers 5.14.1)
├─ YouTubeStudioPublisher ──► Playwright ──► Chromium
└─ RemoteMolmoPointer ─┐          monta el prompt y parsea la respuesta AQUÍ
                       │  JSON por líneas (stdin/stdout), PNG en base64
                       ▼
   subproceso  (/opt/venv-molmo, transformers 4.57.3)
   └─ molmo_server.py ──► MolmoWebPointer ──► MolmoWeb-4B en GPU
```

Dos decisiones que mantienen esto barato de mantener:

- **El servidor importa nuestro propio `MolmoWebPointer`.** El repo está montado igual para
  ambos intérpretes y `molmo_publisher.py` solo importa stdlib a nivel de módulo (torch y
  transformers son imports perezosos dentro del constructor). Cero código duplicado entre venvs.
- **El prompt se monta y se parsea en el lado principal**; el subproceso solo hace
  imagen+texto → texto. Así la lógica que tiene tests (plantilla, historial, parser) vive en un
  único sitio y el otro lado no tiene nada que testear.

Crear el venv (una vez):

```bash
/opt/venv/bin/python -m venv /opt/venv-molmo
/opt/venv-molmo/bin/pip install "transformers==4.57.3" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" \
    accelerate einops safetensors pillow jinja2 numpy
```

Ojo: `python -m venv --system-site-packages` **no** sirve para reutilizar el torch del venv
principal — hereda del Python del sistema, no del venv padre, y te quedas sin torch. Hay que
instalarlo entero (~3 GB), que además es lo que queríamos: aislamiento de verdad.
