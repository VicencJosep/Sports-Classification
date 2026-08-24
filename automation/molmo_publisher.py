"""
automation/molmo_publisher.py
=============================
Publicación automática de clips en RRSS pilotada por un VLM: MolmoWeb-4B
"mira" la pantalla y decide DÓNDE hay que hacer clic; Playwright ejecuta el
ratón. Primer destino implementado: **YouTube Studio** (`studio.youtube.com`),
flujo de subida de un Short/clip corto ya renderizado por
``analytics/engine.py`` (``final.mp4``).

Filosofía de navegación (por qué no hay selectores CSS)
-------------------------------------------------------
YouTube Studio es una SPA de Polymer con web components, Shadow DOM y clases
ofuscadas que cambian entre despliegues: cualquier selector CSS/XPath que
escribamos hoy se rompe solo. En vez de eso, el bucle es:

    page.screenshot()  ->  MolmoWeb-4B (imagen + objetivo)  ->  (x, y)  ->  page.mouse.click()

MolmoWeb-4B es un **agente web** (arquitectura Molmo 2, Qwen3-8B + SigLIP 2):
se le da un objetivo en lenguaje natural y el estado de la página, y responde
con ``{"thought": ..., "action": {"name": "click", "x": 34.6, "y": 8.7}}``,
donde x/y son **porcentajes** (0-100) del ancho/alto de la imagen. Eso hace
que el mapeo a coordenadas de Playwright sea independiente del
`deviceScaleFactor`: basta multiplicar la fracción por el tamaño del
**viewport en píxeles CSS** (ver ``Point.to_viewport_px``), nunca por el
tamaño en píxeles de la captura.

Ojo al detalle de que es un modelo de TRAYECTORIA, no un localizador sin
memoria: ``MolmoWebPointer`` mantiene los pasos previos y los reinyecta como
PREVIOUS STEPS en cada prompt (ver la sección de MolmoWeb más abajo para el
protocolo exacto y las diferencias con la familia Molmo de 2024).

Corolario importante: las capturas SIEMPRE se toman con ``full_page=False``.
Una captura de página completa produciría coordenadas fuera del viewport que
``page.mouse.click`` no puede alcanzar.

Gestión de sesión
-----------------
``launch_persistent_context`` sobre un perfil local
(``./automation/chrome_session`` por defecto, ignorado por git) mantiene las
cookies de Google entre ejecuciones.
La PRIMERA vez hay que iniciar sesión a mano: lanza el script con
``--login-only``, haz login en la ventana que se abre (incluido 2FA) y ciérrala;
a partir de ahí el resto de ejecuciones reutilizan la sesión. No automatizamos
el login: es lo que Google protege activamente y lo que rompe con CAPTCHAs.

Alcance y prudencia
-------------------
Este módulo publica en la cuenta del propietario del perfil, así que por
defecto sube en visibilidad **privada** (``--visibility private``) y expone
``--dry-run``, un ensayo que recorre el flujo pidiendo coordenadas a Molmo y
volcando capturas anotadas en ``--debug-dir``. El ensayo SÍ hace clic en los
dos primeros pasos ("CREAR" -> "Subir vídeos" -> "Seleccionar archivos"),
porque son los únicos que no publican nada: abrir el asistente de subida es
reversible cerrando el diálogo. El selector de ficheros se dispara para
comprobar que el prompt acierta, pero el fichero NUNCA se entrega
(``set_files`` no se llama), así que no hay subida. Justo ahí el ratón se
congela y el ensayo termina ahí: sin fichero, el asistente no pasa de esa
pantalla (Detalles / Elementos / Comprobaciones / Visibilidad solo existen una
vez el vídeo está subiendo). Úsalo para validar los prompts de navegación
antes de dejar que suba nada.

Si lo que quieres es validar los prompts de ESAS pestañas, hace falta el modo
``--dry-run-draft``: entregar el fichero y subirlo es la única forma de que el
asistente las cree — no existe un "seleccionar sin subir", la transferencia
arranca en el mismo evento. Ese modo recorre el asistente entero y, en vez de
guardar, limpia: cierra el asistente y BORRA el vídeo desde su página de
edición (Studio lo autoguarda como borrador privado en cuanto termina la
subida, así que cerrar no basta). La limpieza se intenta también desde el
``finally``, por si algo revienta a mitad. El vídeo pasa por los servidores de
YouTube como borrador privado durante el ensayo; si el borrado fallara, el log
da la URL exacta para hacerlo a mano.

Uso rápido (desde la raíz del proyecto):
    # 1) sesión: login manual una única vez
    python -m automation.molmo_publisher --login-only

    # 2) ensayo: llega hasta el selector de ficheros y para (no sube nada)
    python -m automation.molmo_publisher --video output/miclip/final.mp4 --dry-run

    # 2b) ensayo completo: sube un borrador, recorre las pestañas y lo descarta
    python -m automation.molmo_publisher --video output/miclip/final.mp4 --dry-run-draft

    # 3) subida real, privada
    python -m automation.molmo_publisher --video output/miclip/final.mp4 \
        --title "Test Clip Short" --visibility private

Requiere ``playwright install --with-deps chromium`` dentro del contenedor
(ver README § instalación) y una GPU libre para el VLM (~16 GB en bfloat16;
los autores recomiendan float32, que no cabe en 24 GB).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from playwright.sync_api import BrowserContext, Page

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_ID: str = "allenai/MolmoWeb-4B"

# Intérprete con transformers <5, donde sí arranca el código remoto del
# checkpoint (ver automation/molmo_server.py para por qué hace falta otro venv).
DEFAULT_VENV_PYTHON: Path = Path("/opt/venv-molmo/bin/python")

DEFAULT_USER_DATA_DIR: Path = _PROJECT_ROOT / "automation" / "chrome_session"
DEFAULT_STUDIO_URL: str = "https://studio.youtube.com"

# Viewport fijo (píxeles CSS). Fijarlo importa más de lo que parece: el layout
# de Studio es responsive y los prompts describen posiciones ("arriba a la
# derecha"), así que un viewport distinto entre ejecuciones cambia lo que ve
# Molmo. 1440x900 deja ver el asistente de subida entero sin scroll.
DEFAULT_VIEWPORT: dict[str, int] = {"width": 1440, "height": 900}

# Pausas (ms) antes de capturar. YouTube Studio anima modales y diálogos; una
# captura tomada a mitad de la transición le da a Molmo una UI a medio pintar
# (opacidad/escala intermedias) y las coordenadas salen desplazadas.
DEFAULT_SETTLE_MS: int = 2500
MODAL_SETTLE_MS: int = 4000       # apertura del diálogo de subida
UPLOAD_SETTLE_MS: int = 8000      # tras soltar el fichero: parseo + formulario


# --------------------------------------------------------------------------- #
#  Prompts (lo único que hay que retocar cuando YouTube cambia su UI)
# --------------------------------------------------------------------------- #
#
# Cada prompt entra como GOAL de la plantilla de MolmoWeb, así que va en
# IMPERATIVO ("Click the ...") y no como "point to X": este modelo emite
# acciones (``{"action": {"name": "click", ...}}``), no puntos sueltos. El
# estilo imita el de sus datos de entrenamiento (MolmoWeb-SyntheticGround:
# "search for strawberry pretzel salad", "click the search icon").
#
# El modelo está entrenado en inglés y la UI está en español: se cita la
# etiqueta literal de pantalla entre comillas y se traduce al lado. Añadir la
# posición ("top right corner") sube el acierto cuando hay varios candidatos
# plausibles en la misma pantalla.

@dataclass(frozen=True)
class StudioPrompts:
    """Objetivos (GOAL) para cada paso del asistente de subida.

    Los textos van en **catalán primero** y con el equivalente castellano e
    inglés entre paréntesis. No es adorno: Studio se pinta en el idioma de la
    CUENTA de Google, no en el del navegador ni el del sistema, y la cuenta de
    este perfil lo tiene en catalán ("Crea", "Penja vídeos", "Desa"). Con los
    prompts solo en castellano el modelo acierta los botones icónicos y falla
    los que dependen de leer la etiqueta. Si cambias de cuenta/idioma, este es
    el sitio que hay que tocar.
    """

    create_button: str = (
        'Click the "Crea" (Catalan) / "CREAR" (Spanish) / "Create" button with '
        "the camera-plus icon in the top right corner of the page header."
    )
    upload_videos_item: str = (
        'Click the "Penja vídeos" (Catalan) / "Subir vídeos" (Spanish) / '
        '"Upload videos" option, the first one with an upward arrow icon, in '
        "the dropdown menu that just opened."
    )
    select_files_button: str = (
        'Click the black "Selecciona fitxers" (Catalan) / "SELECCIONAR '
        'ARCHIVOS" (Spanish) / "Select files" button in the center of the '
        "upload dialog."
    )
    title_field: str = (
        'Click the "Títol" (Catalan) / "Título" (Spanish) / "Title" text input '
        "box in the details form, the first text box under the Details heading."
    )
    kids_no_radio: str = (
        'Click the radio button next to "No, no és contingut creat per a nens" '
        '(Catalan) / "No, no es contenido creado para niños" (Spanish) / '
        '"No, it is not made for kids".'
    )
    next_button: str = (
        'Click the black "Següent" (Catalan) / "SIGUIENTE" (Spanish) / "Next" '
        "button in the bottom right corner of the dialog. It is the RIGHTMOST "
        'button. Do NOT click the pale "Enrere" / "Atrás" / "Back" button '
        "immediately to its left."
    )
    visibility_private: str = (
        'Click the round radio button immediately to the LEFT of the word '
        '"Privat" (Catalan) / "Privado" (Spanish) / "Private", the first '
        "option in the visibility list."
    )
    visibility_unlisted: str = (
        'Click the round radio button immediately to the LEFT of the word '
        '"Ocult" (Catalan) / "Oculto" (Spanish) / "Unlisted", the second '
        "option in the visibility list."
    )
    visibility_public: str = (
        'Click the round radio button immediately to the LEFT of the word '
        '"Públic" (Catalan) / "Público" (Spanish) / "Public", the third '
        "option in the visibility list."
    )
    save_button: str = (
        'Click the black "Desa" (Catalan) / "GUARDAR" (Spanish) / "Save" '
        "button in the bottom right corner of the dialog. It is the RIGHTMOST "
        'button. Do NOT click the pale "Enrere" / "Atrás" / "Back" button '
        "immediately to its left."
    )
    close_button: str = (
        "Click the X icon in the top right corner of the upload dialog, to the "
        'right of the "S\'ha desat com a privat" badge, to close the dialog.'
    )
    discard_option: str = (
        'Click the "Descarta el vídeo" (Catalan) / "Descartar vídeo" (Spanish) '
        '/ "Discard video" option. Do NOT click "Desa com a esborrany" / '
        '"Guardar como borrador" / "Save as draft", and do NOT click '
        '"Continua editant" / "Seguir editando" / "Continue editing".'
    )
    confirm_discard_button: str = (
        'Click the "Descarta" (Catalan) / "DESCARTAR" (Spanish) / "Discard" '
        "confirmation button in the small confirmation dialog."
    )
    options_menu: str = (
        "Click the vertical three-dots options menu icon in the top right area "
        "of the video edit page, next to the video title."
    )
    delete_forever_option: str = (
        'Click the "Suprimeix definitivament" (Catalan) / "Eliminar '
        'definitivamente" (Spanish) / "Delete forever" option in the menu that '
        "just opened."
    )
    delete_confirm_checkbox: str = (
        "Click the checkbox that acknowledges the video will be permanently "
        'deleted ("Entenc que..." / "Entiendo que..." / "I understand...").'
    )
    delete_confirm_button: str = (
        'Click the "Suprimeix definitivament" (Catalan) / "ELIMINAR '
        'DEFINITIVAMENTE" (Spanish) / "DELETE FOREVER" button at the bottom '
        "right of the confirmation dialog."
    )


PROMPTS = StudioPrompts()


# --------------------------------------------------------------------------- #
#  Punto predicho por el VLM
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Point:
    """Un punto devuelto por Molmo, en fracción [0, 1] del tamaño de la imagen.

    Se guarda normalizado a propósito: la captura viene en píxeles de
    dispositivo (viewport x deviceScaleFactor) mientras que ``page.mouse``
    trabaja en píxeles CSS. Normalizar es lo que desacopla ambas escalas.
    """

    x: float                    # 0.0 (izquierda) .. 1.0 (derecha)
    y: float                    # 0.0 (arriba)    .. 1.0 (abajo)
    label: str = ""             # "thought" del modelo (o alt= en Molmo clásico)
    raw: str = ""               # respuesta cruda del modelo (depuración)
    action: str = "click"       # nombre de la acción emitida (click/type/scroll/...)
    raw_action: str = ""        # acción reconstruida, p.ej. "click(x=34.6, y=8.7)";
                                # se reinyecta como PREVIOUS STEPS en el prompt siguiente

    def to_viewport_px(self, viewport: dict[str, int]) -> tuple[float, float]:
        """Fracción -> píxeles CSS del viewport (los que espera page.mouse).

        El recorte a ``[1, dim-2]`` es el mismo que aplica el agente oficial de
        MolmoWeb (``_pct_to_coord`` en allenai/molmoweb): una predicción que cae
        justo en el borde se sale del viewport y el clic se pierde.
        """
        w, h = viewport["width"], viewport["height"]
        return (
            max(1.0, min(self.x * w, w - 2.0)),
            max(1.0, min(self.y * h, h - 2.0)),
        )

    def to_telemetry(self) -> dict:
        return asdict(self)


class Pointer(Protocol):
    """Contrato mínimo que ``predict_and_click`` necesita de un VLM.

    Existe para poder inyectar un doble en tests (o un modelo remoto) sin
    cargar 4B de parámetros: cualquier objeto con este método vale.
    """

    def point(
        self,
        image_png: bytes,
        prompt: str,
        page_url: Optional[str] = None,
        page_title: Optional[str] = None,
    ) -> Optional[Point]:
        ...


# --------------------------------------------------------------------------- #
#  MolmoWeb-4B (arquitectura Molmo 2)
# --------------------------------------------------------------------------- #
#
# OJO: MolmoWeb-4B NO es el Molmo de 2024 con *pointing* suelto ("point to X"
# -> <point x=... y=...>). Es un AGENTE WEB con protocolo propio, y eso cambia
# tres cosas respecto a la familia Molmo clásica:
#
#   1. Se carga con AutoModelForImageTextToText (clase Molmo2ForConditionalGeneration),
#      no con AutoModelForCausalLM. Su procesador NO tiene .process() ni el
#      modelo .generate_from_batch(): son apply_chat_template() + generate().
#   2. El prompt es "molmo_web_think: " + una plantilla Jinja con la estructura
#      GOAL / PREVIOUS STEPS / CURRENTLY ACTIVE PAGE / NEXT STEP. Es un modelo
#      de TRAYECTORIA: se le pasan los pasos anteriores y decide el siguiente.
#   3. Responde con un JSON {"thought": ..., "action": {"name": "click", "x": ..,
#      "y": ..}}, no con etiquetas <point>.
#
# Lo que SÍ se mantiene (verificado contra sus datasets y su agente oficial):
# las coordenadas son PORCENTAJES 0-100 del ancho/alto de la imagen. Su
# repositorio allenai/molmoweb lo hace explícito en
# ``convert_action_json_to_action_obj``: "Coordinates are expected as
# percentages (0-100) and converted to pixels using the screenshot dimensions",
# con el recorte a [1, dim-2] que replica ``Point.to_viewport_px``.
#
# Comprobado numéricamente contra allenai/MolmoWeb-SyntheticGround (imagen
# 1288x712): bbox [257.4, 38.0, 693.4, 74.0] -> centro 36.9% / 7.9%, respuesta
# del modelo x=34.6 y=8.7. Dentro de la caja, en porcentaje. Idem en 4 muestras más.

# Plantilla oficial del checkpoint (README de allenai/MolmoWeb-4B). Se replica
# literal: el modelo está entrenado con estos encabezados exactos.
MOLMOWEB_THINK_TEMPLATE: str = """
# GOAL
{{ task_description }}

# PREVIOUS STEPS
{% for action in past_actions: -%}
## Step {{ action['index'] }}
THOUGHT: {{ action['thought'] }}
ACTION: {{ action['action'] }}
{% endfor %}
# CURRENTLY ACTIVE PAGE
Page {{ page_index }}: {{ page_title }} | {{ page_url }}

# NEXT STEP

"""

MOLMOWEB_SYSTEM_MESSAGE: str = "molmo_web_think"

# Fallback para la familia Molmo clásica (Molmo-7B-D-0924 y compañía), que sí
# emite <point x="52.3" y="14.7" alt="...">. Se mantiene para que --model-id
# pueda apuntar a esos checkpoints sin tocar código.
_POINT_RE = re.compile(r'x\s*(\d*)\s*=\s*"([\d.]+)"\s*y\s*\1\s*=\s*"([\d.]+)"')
_ALT_RE = re.compile(r'alt\s*=\s*"([^"]*)"')


class _TrajectoryPointer:
    """Parte común a los dos pointers: plantilla del prompt, historial y parseo.

    MolmoWeb es un modelo de TRAYECTORIA: cada predicción se le pide junto con
    los pasos ya dados (``PREVIOUS STEPS``), que es lo que le permite
    interpretar la pantalla actual sabiendo que ya pulsó "CREAR". Esa lógica es
    idéntica corra el modelo en este proceso (``MolmoWebPointer``) o al otro
    lado de una tubería (``RemoteMolmoPointer``), así que vive aquí una sola vez.

    Las subclases solo tienen que implementar ``describe_prompt``:
    imagen + prompt ya montado -> texto crudo del modelo.
    """

    track_history: bool = True

    def __init__(self, track_history: bool = True) -> None:
        self.track_history = track_history
        self._history: list[dict] = []

    # -- a implementar por la subclase ---------------------------------- #

    def describe_prompt(self, image_png: bytes, full_prompt: str) -> str:
        raise NotImplementedError

    # -- común ----------------------------------------------------------- #

    def point(
        self,
        image_png: bytes,
        prompt: str,
        page_url: Optional[str] = None,
        page_title: Optional[str] = None,
    ) -> Optional[Point]:
        """Captura PNG + objetivo en lenguaje natural -> punto normalizado.

        ``prompt`` va como GOAL de la plantilla, así que se redacta en
        imperativo ("Click the CREAR button..."), no como "point to X":
        este modelo emite acciones, no puntos sueltos.

        ``page_url``/``page_title`` rellenan la sección CURRENTLY ACTIVE PAGE.
        Importan más de lo que parece: si se le miente sobre dónde está, el
        modelo razona sobre esa mentira. Medido en el banco de pruebas — con la
        URL por defecto (``studio.youtube.com``) sobre una página que era otra
        cosa, respondió ``goto(url='https://studio.youtube.com')`` en vez de
        clicar, que es la decisión correcta para el estado que le habíamos
        descrito.
        """
        prompt_kwargs = {}
        if page_url is not None:
            prompt_kwargs["page_url"] = page_url
        if page_title is not None:
            prompt_kwargs["page_title"] = page_title

        raw = self.describe(image_png, prompt, **prompt_kwargs)
        point = parse_molmo_response(raw)

        if point is None:
            otra = describe_non_click_action(raw)
            if otra:
                # No es un fallo de localización: el agente propone otra acción.
                # Se registra aparte porque el arreglo es distinto (contexto o
                # prompt, no timing).
                logger.warning(
                    "MolmoWeb propuso '%s' en vez de un clic. Motivo: %s",
                    otra, _extract_thought(raw)[:200] or "(sin thought)",
                )
            else:
                logger.warning("MolmoWeb no devolvió coordenadas. Respuesta: %r", raw[:400])
            return None

        if self.track_history:
            self._history.append({
                "index": len(self._history) + 1,
                "thought": point.label,
                "action": point.raw_action or f"click(x={point.x * 100:.1f}, y={point.y * 100:.1f})",
            })
        return point

    def describe(self, image_png: bytes, prompt: str, **prompt_kwargs) -> str:
        """Monta el prompt y devuelve la respuesta cruda (sin parsear)."""
        return self.describe_prompt(image_png, self.build_prompt(prompt, **prompt_kwargs))

    def reset_history(self) -> None:
        """Olvida los pasos previos (nueva trayectoria, p.ej. otro vídeo)."""
        self._history.clear()

    def build_prompt(
        self,
        task_description: str,
        page_url: str = "https://studio.youtube.com",
        page_title: Optional[str] = None,
        page_index: int = 0,
    ) -> str:
        """Arma el prompt en el formato exacto con el que se entrenó el modelo."""
        from jinja2 import Template

        user_message = Template(MOLMOWEB_THINK_TEMPLATE).render(
            task_description=task_description,
            past_actions=self._history if self.track_history else [],
            page_index=page_index,
            page_title=page_title,
            page_url=page_url,
        )
        return f"{MOLMOWEB_SYSTEM_MESSAGE}: {user_message}"

    def close(self) -> None:
        """Libera recursos (no-op en el local; cierra el subproceso en el remoto)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class MolmoWebPointer(_TrajectoryPointer):
    """Carga MolmoWeb-4B EN ESTE PROCESO. Requiere transformers <5.

    OJO: el intérprete principal del repo tiene ``transformers==5.14.1`` fijado
    (lo exige la API de Siglip de Soccer_Master) y con esa versión el código
    remoto del checkpoint no arranca — ver ``automation/molmo_server.py`` para
    el detalle de los tres fallos en cadena. Por eso el camino normal es
    ``RemoteMolmoPointer``, que ejecuta ESTA MISMA clase bajo
    ``/opt/venv-molmo`` (transformers 4.57.3).

    Esta clase sigue siendo la que hace el trabajo: la usa el servidor.

    Args:
        model_id: checkpoint en Hugging Face.
        device: "cuda" | "cpu" (None -> cuda si está disponible). En CPU cada
            predicción tarda minutos: inviable para un flujo interactivo.
        dtype: los autores **recomiendan float32** ("we recommend using the
            default float32 precision"). En una 24 GB no cabe (Qwen3-8B de
            base), así que el defecto aquí es bfloat16; si ves coordenadas
            erráticas, esta es la primera sospechosa.
        max_new_tokens: el JSON de acción cabe de sobra en 200 tokens.
        track_history: si False, cada predicción es independiente (sin
            PREVIOUS STEPS). Útil para depurar un paso suelto.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = 200,
        track_history: bool = True,
    ) -> None:
        super().__init__(track_history=track_history)
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.model_id = model_id
        self.track_history = track_history
        self._history: list[dict] = []

        torch_dtype = getattr(torch, dtype)
        logger.info("Cargando %s en %s (%s) ...", model_id, self.device, torch_dtype)
        if self.device == "cuda":
            libre, total = (x / 2**30 for x in torch.cuda.mem_get_info())
            logger.info("VRAM antes de cargar: %.1f GiB libres de %.1f", libre, total)

        try:
            # trust_remote_code: Molmo 2 publica su clase de modelo y su
            # preprocesado en el propio repo del checkpoint.
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                attn_implementation="sdpa",
                # Todo en la GPU o nada. Con device_map="auto", si otro proceso
                # ocupa VRAM, accelerate reparte capas a la CPU sin fallar: el
                # error sale 40 s más tarde, en la primera predicción y con la
                # cara de "NotImplementedError: Cannot copy out of meta tensor".
                # La GPU de esta máquina es compartida, así que un OOM honesto
                # aquí ahorra media hora de diagnóstico.
                device_map={"": 0} if self.device == "cuda" else None,
            )
        except Exception as exc:
            if self.device == "cuda" and "out of memory" in str(exc).lower():
                libre, total = (x / 2**30 for x in torch.cuda.mem_get_info())
                raise RuntimeError(
                    f"{model_id} no cabe en la GPU: quedan {libre:.1f} GiB libres de "
                    f"{total:.1f} y en {dtype} hacen falta ~10. La GPU está compartida "
                    "con otros trabajos: mira quién la ocupa con 'nvidia-smi' y espera "
                    "a que termine (o usa --device cpu, con predicciones de minutos)."
                ) from exc
            raise

        # Cinturón y tirantes: si algún día se vuelve a "auto", que el reparto a
        # CPU se note AQUÍ y no en mitad de una publicación a medio hacer.
        en_meta = [n for n, p in self.model.named_parameters() if p.device.type == "meta"]
        if en_meta:
            raise RuntimeError(
                f"{len(en_meta)} parámetros quedaron en el dispositivo 'meta' (descargados "
                f"a CPU por falta de VRAM); el modelo no puede generar. Primero: {en_meta[0]}"
            )

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, padding_side="left"
        )
        self.model.eval()

    # ------------------------------------------------------------------ #

    def describe_prompt(self, image_png: bytes, full_prompt: str) -> str:
        """Imagen + prompt YA montado -> texto crudo del modelo."""
        import io

        from PIL import Image

        torch = self._torch
        image = Image.open(io.BytesIO(image_png)).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": full_prompt},
                {"type": "image", "image": image},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        )
        # token_type_ids fuera: HF lo usa para activar atención bidireccional en
        # los tokens de imagen, y MolmoWeb se entrenó solo con atención causal
        # (lo advierte su propio README).
        inputs = {
            k: v.to(self.model.device) for k, v in inputs.items() if k != "token_type_ids"
        }

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        generated = output[0, inputs["input_ids"].size(1):]
        return self.processor.decode(generated, skip_special_tokens=True).strip()


class RemoteMolmoPointer(_TrajectoryPointer):
    """Ejecuta el VLM en un subproceso con su propio intérprete. **Camino normal.**

    El checkpoint exige ``transformers>=4.51,<5`` (se guardó con la 4.57.3) y
    este repo fija la 5.14.1 por la API de Siglip de Soccer_Master. En vez de
    reimplementar a mano el delta 4.x->5.x a base de parches —con riesgo de
    fallar en silencio, devolviendo coordenadas mal calculadas en lugar de una
    excepción—, el modelo corre en ``/opt/venv-molmo`` y se le habla por
    tubería (ver ``automation/molmo_server.py``).

    Cumple el mismo ``Pointer`` protocol que el pointer local, así que
    ``YouTubeStudioPublisher`` no distingue uno de otro: el prompt se monta y
    la respuesta se parsea aquí (código compartido en ``_TrajectoryPointer``);
    al otro lado solo se hace imagen+texto -> texto.

    Args:
        venv_python: intérprete con transformers <5.
        startup_timeout_s: margen para cargar el modelo. Generoso a propósito:
            la PRIMERA vez descarga ~9 GB de pesos.
        request_timeout_s: margen por predicción.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = 200,
        track_history: bool = True,
        venv_python: Path = DEFAULT_VENV_PYTHON,
        startup_timeout_s: float = 1800.0,
        request_timeout_s: float = 300.0,
    ) -> None:
        super().__init__(track_history=track_history)
        import subprocess

        venv_python = Path(venv_python)
        if not venv_python.exists():
            raise FileNotFoundError(
                f"No existe el intérprete del VLM: {venv_python}. Créalo con:\n"
                f"  /opt/venv/bin/python -m venv /opt/venv-molmo\n"
                f"  /opt/venv-molmo/bin/pip install 'transformers==4.57.3' torch torchvision "
                f"accelerate einops safetensors pillow jinja2"
            )

        self.request_timeout_s = request_timeout_s
        cmd = [
            str(venv_python), "-m", "automation.molmo_server",
            "--model-id", model_id,
            "--dtype", dtype,
            "--max-new-tokens", str(max_new_tokens),
        ]
        if device:
            cmd += ["--device", device]

        logger.info("Lanzando servidor del VLM: %s", " ".join(cmd))
        # cwd = raíz del repo para que "automation.molmo_server" sea importable.
        # stderr se hereda: los logs de carga del modelo salen por consola.
        self._proc = subprocess.Popen(
            cmd, cwd=str(_PROJECT_ROOT),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )

        hello = self._read_response(startup_timeout_s, contexto="arranque")
        if not hello.get("ready"):
            raise RuntimeError(f"El servidor del VLM no arrancó: {hello}")
        self.device = hello.get("device", "?")
        logger.info("Servidor del VLM listo (device=%s)", self.device)

    # ------------------------------------------------------------------ #

    def describe_prompt(self, image_png: bytes, full_prompt: str) -> str:
        import base64

        self._send({
            "image_b64": base64.b64encode(image_png).decode("ascii"),
            "prompt": full_prompt,
        })
        resp = self._read_response(self.request_timeout_s, contexto="predicción")
        if not resp.get("ok"):
            raise RuntimeError(f"El VLM devolvió error: {resp.get('error')}")
        return resp.get("raw", "")

    def close(self) -> None:
        """Cierra el subproceso (idempotente)."""
        proc = getattr(self, "_proc", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            self._send({"cmd": "shutdown"})
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        finally:
            logger.info("Servidor del VLM cerrado")

    # ------------------------------------------------------------------ #

    def _send(self, payload: dict) -> None:
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"El servidor del VLM ha muerto (código {self._proc.returncode}); "
                "revisa su salida de error más arriba"
            )
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()

    def _read_response(self, timeout_s: float, contexto: str) -> dict:
        """Lee una línea JSON del subproceso, vigilando que siga vivo.

        El timeout se implementa con ``select`` sobre el pipe en vez de un
        ``readline()`` a pelo: si el servidor muere al cargar el modelo (falta
        de VRAM, por ejemplo), un readline bloqueante se quedaría colgado para
        siempre en vez de reportar el fallo.
        """
        import select
        import time

        limite = time.time() + timeout_s
        while True:
            restante = limite - time.time()
            if restante <= 0:
                self.close()
                raise TimeoutError(f"El VLM no respondió en {timeout_s:.0f}s ({contexto})")

            listo, _, _ = select.select([self._proc.stdout], [], [], min(restante, 5.0))
            if listo:
                linea = self._proc.stdout.readline()
                if not linea:  # EOF: el proceso se ha cerrado
                    raise RuntimeError(
                        f"El servidor del VLM cerró la salida durante el {contexto} "
                        f"(código {self._proc.poll()})"
                    )
                try:
                    return json.loads(linea)
                except json.JSONDecodeError:
                    # Ruido en stdout que no es del protocolo: lo registro y sigo.
                    logger.debug("Línea no-JSON del servidor: %r", linea[:200])
                    continue
            elif self._proc.poll() is not None:
                raise RuntimeError(
                    f"El servidor del VLM murió durante el {contexto} "
                    f"(código {self._proc.returncode})"
                )


def parse_molmo_response(raw: str) -> Optional[Point]:
    """Respuesta cruda -> ``Point``, aceptando los dos dialectos de la familia.

    1. MolmoWeb / Molmo 2: JSON ``{"thought": ..., "action": {"name": "click",
       "x": 34.6, "y": 8.7}}`` (el JSON puede venir envuelto en texto o en un
       bloque markdown, de ahí la búsqueda del primer objeto balanceado).
    2. Molmo clásico: etiqueta ``<point x="52.3" y="14.7" alt="...">``.

    En ambos, x/y son porcentajes 0-100 y se devuelven en fracción [0, 1], que
    es lo que consume ``Point.to_viewport_px``. Se descartan valores fuera de
    rango: el modelo alucina alguno cuando el elemento no está en pantalla, y
    un clic en el borde es peor que no hacer nada.
    """
    if not raw:
        return None

    point = _parse_action_json(raw) or _parse_point_tag(raw)
    if point is None:
        return None

    if not (0.0 <= point.x <= 1.0 and 0.0 <= point.y <= 1.0):
        logger.warning(
            "Coordenadas fuera de rango descartadas: (%.1f%%, %.1f%%)",
            point.x * 100, point.y * 100,
        )
        return None
    return point


def _parse_action_json(raw: str) -> Optional[Point]:
    """Dialecto MolmoWeb: primer objeto JSON con action.x / action.y."""
    for data in _iter_json_objects(raw):
        action = data.get("action")
        if not isinstance(action, dict) or "x" not in action or "y" not in action:
            continue
        name = str(action.get("name", "click"))
        args = ", ".join(f"{k}={v!r}" for k, v in action.items() if k != "name")
        return Point(
            x=float(action["x"]) / 100.0,
            y=float(action["y"]) / 100.0,
            label=str(data.get("thought", ""))[:200],
            raw=raw,
            action=name,
            raw_action=f"{name}({args})",
        )
    return None


def describe_non_click_action(raw: str) -> Optional[str]:
    """Si el modelo propuso una acción SIN coordenadas, la describe; si no, None.

    MolmoWeb es un agente completo: su espacio de acciones incluye ``goto``,
    ``scroll``, ``keyboard_type``, ``send_msg_to_user``... Cuando devuelve una
    de esas, no es que no haya localizado el elemento — es que ha decidido que
    tocaba otra cosa. Distinguir ambos casos importa porque el arreglo es
    distinto: contexto/prompt equivocado en un caso, timing o prompt ambiguo en
    el otro.
    """
    for accion in _iter_action_dicts(raw):
        if "x" in accion and "y" in accion:
            return None
        nombre = str(accion.get("name", "?"))
        args = ", ".join(f"{k}={v!r}" for k, v in accion.items() if k != "name")
        return f"{nombre}({args})"
    return None


def _extract_thought(raw: str) -> str:
    """Saca el campo ``thought`` de la respuesta (cadena vacía si no hay)."""
    for data in _iter_json_objects(raw):
        if "thought" in data:
            return str(data["thought"])
    return ""


def _iter_json_objects(raw: str):
    """Objetos JSON balanceados dentro de un texto (puede venir con markdown alrededor)."""
    for start in (i for i, c in enumerate(raw) if c == "{"):
        depth = 0
        for end in range(start, len(raw)):
            if raw[end] == "{":
                depth += 1
            elif raw[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        yield json.loads(raw[start:end + 1])
                    except json.JSONDecodeError:
                        pass
                    break


def _iter_action_dicts(raw: str):
    """Los sub-diccionarios ``action`` de los objetos JSON de la respuesta."""
    for data in _iter_json_objects(raw):
        accion = data.get("action")
        if isinstance(accion, dict):
            yield accion


# Los radios de visibilidad llevan un atributo ``name`` estable en el web
# component de Studio (PRIVATE/UNLISTED/PUBLIC). Es lo bastante contractual como
# para pinchar por DOM: el VLM falla justo aquí (objetivos parecidos, dianas
# pequeñas) y equivocarse de radio es el peor error posible de este módulo.
_VISIBILITY_RADIOS = {"private": "PRIVATE", "unlisted": "UNLISTED", "public": "PUBLIC"}

# Detalls -> Elements -> Comprovacions -> Visibilitat son 3 avances; el margen
# extra cubre un clic perdido (o uno que acabe en "Enrere") sin dar vueltas
# eternas si el asistente se atasca.
_MAX_WIZARD_ADVANCES = 6

_VIDEO_ID_RE = re.compile(r"(?:shorts/|watch\?v=|youtu\.be/|/video/)([A-Za-z0-9_-]{11})")


def _parse_point_tag(raw: str) -> Optional[Point]:
    """Dialecto Molmo clásico: ``<point x="..." y="..." alt="...">``."""
    match = _POINT_RE.search(raw)
    if match is None:
        return None
    alt = _ALT_RE.search(raw)
    return Point(
        x=float(match.group(2)) / 100.0,
        y=float(match.group(3)) / 100.0,
        label=alt.group(1) if alt else "",
        raw=raw,
        action="click",
    )


# --------------------------------------------------------------------------- #
#  Puente VLM <-> Playwright
# --------------------------------------------------------------------------- #

def capture(page: "Page", settle_ms: int = DEFAULT_SETTLE_MS) -> bytes:
    """Espera a que la UI se asiente y devuelve una captura del VIEWPORT.

    ``full_page=False`` no es un detalle cosmético: ver el docstring del módulo
    (coordenadas fuera del viewport = clic imposible).
    """
    page.wait_for_timeout(settle_ms)
    return page.screenshot(full_page=False)


def predict_and_click(
    page: "Page",
    prompt: str,
    pointer: Pointer,
    *,
    settle_ms: int = DEFAULT_SETTLE_MS,
    retries: int = 2,
    dry_run: bool = False,
    debug_dir: Optional[Path] = None,
    step: str = "step",
) -> Optional[Point]:
    """Captura -> Molmo -> ``page.mouse.click``. El corazón del agente.

    Args:
        page: página de Playwright ya navegada.
        prompt: objetivo en imperativo, en inglés (ver ``StudioPrompts``).
        pointer: cualquier objeto que cumpla el Protocol ``Pointer``.
        settle_ms: pausa antes de capturar (animaciones/modales).
        retries: reintentos si el modelo no devuelve coordenadas. Cada
            reintento vuelve a CAPTURAR: si el fallo era una animación aún en
            curso, la segunda captura ya la pilla terminada.
        dry_run: predice y registra, pero no hace clic.
        debug_dir: si se indica, vuelca la captura con el punto marcado.
        step: nombre del paso, solo para logs y nombres de fichero.

    Returns:
        El ``Point`` predicho, o None si tras los reintentos no hubo predicción
        (el llamador decide si eso es fatal o si tiene un plan B por DOM).
    """
    viewport = page.viewport_size or DEFAULT_VIEWPORT

    for attempt in range(1, retries + 1):
        image_png = capture(page, settle_ms)
        # Estado REAL de la página: el modelo razona sobre lo que se le diga que
        # hay en CURRENTLY ACTIVE PAGE, así que mentirle ahí sale caro (ver
        # _TrajectoryPointer.point).
        try:
            contexto = {"page_url": page.url, "page_title": page.title()}
        except Exception:
            contexto = {}
        point = pointer.point(image_png, prompt, **contexto)

        if point is None:
            logger.warning("[%s] sin coordenadas (intento %d/%d)", step, attempt, retries)
            continue

        x, y = point.to_viewport_px(viewport)
        logger.info(
            "[%s] Molmo -> (%.0f, %.0f) px%s%s",
            step, x, y,
            f" · {point.label}" if point.label else "",
            " · DRY-RUN (sin clic)" if dry_run else "",
        )

        if debug_dir is not None:
            _dump_debug(debug_dir, step, image_png, point)

        if not dry_run:
            # move + click: algunos controles de Studio solo se "despiertan"
            # con el hover previo (tooltips, botones de la barra superior).
            page.mouse.move(x, y)
            page.wait_for_timeout(200)
            page.mouse.click(x, y)

        return point

    logger.error("[%s] Molmo no localizó el elemento tras %d intentos", step, retries)
    return None


def _dump_debug(debug_dir: Path, step: str, image_png: bytes, point: Point) -> None:
    """Guarda la captura con una cruz roja en el punto predicho.

    Es la herramienta de depuración principal de este módulo: si un paso falla,
    la cruz dice si el problema fue el prompt (apunta a otra cosa) o el timing
    (la UI aún no estaba pintada).
    """
    try:
        import io

        from PIL import Image, ImageDraw

        debug_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(io.BytesIO(image_png)).convert("RGB")
        px, py = point.x * image.width, point.y * image.height

        draw = ImageDraw.Draw(image)
        draw.line([(px - 20, py), (px + 20, py)], fill=(255, 0, 0), width=3)
        draw.line([(px, py - 20), (px, py + 20)], fill=(255, 0, 0), width=3)
        draw.ellipse([px - 12, py - 12, px + 12, py + 12], outline=(255, 0, 0), width=3)

        image.save(debug_dir / f"{step}.png")
    except Exception:  # depuración: nunca debe tumbar la publicación
        logger.exception("No se pudo escribir la captura de depuración de '%s'", step)


# --------------------------------------------------------------------------- #
#  Resultado
# --------------------------------------------------------------------------- #

@dataclass
class PublishResult:
    """Resultado de un intento de publicación (serializable a telemetría)."""

    video_path: str
    title: str
    visibility: str
    published: bool
    dry_run: bool
    steps_completed: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_telemetry(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
#  Agente de YouTube Studio
# --------------------------------------------------------------------------- #

class YouTubeStudioPublisher:
    """Recorre el asistente de subida de YouTube Studio guiado por Molmo.

    Un método por fase del asistente, en el mismo orden que la UI:

        _step_open_upload_dialog   Crear -> Subir vídeos
        _step_select_file          diálogo de fichero del SO
        _step_fill_details         título + "no es contenido para niños"
        _step_advance_wizard       Detalles -> Elementos -> Comprobaciones -> Visibilidad
        _step_set_visibility       Privado/Oculto
        _step_save                 Guardar

    El objeto NO abre el navegador: recibe una ``Page`` ya viva. Así el mismo
    agente sirve para una sesión efímera (``publish_clip``) o para una ventana
    de larga duración que publique varios clips seguidos.

    En ``dry_run`` se recorren de verdad los dos primeros pasos (abrir el
    asistente y disparar el selector de ficheros: ambos reversibles y sin
    efecto público), el fichero NO se entrega y el flujo se detiene ahí, que es
    lo más lejos que se puede llegar sin subir el vídeo.

    Con ``dry_run_draft`` el ensayo sí entrega el fichero: el vídeo se sube a
    YouTube como BORRADOR privado (única forma de que existan las pestañas
    Detalles/Elementos/Comprobaciones/Visibilidad), se recorre el asistente
    entero y al final, en vez de guardar, se cierra el asistente y se BORRA el
    vídeo subido (``_step_discard_draft``). La limpieza se intenta también si
    algo falla a mitad (ver ``publish``).
    """

    def __init__(
        self,
        page: "Page",
        pointer: Pointer,
        *,
        prompts: StudioPrompts = PROMPTS,
        dry_run: bool = False,
        dry_run_draft: bool = False,
        debug_dir: Optional[Path] = None,
        settle_ms: int = DEFAULT_SETTLE_MS,
        modal_settle_ms: int = MODAL_SETTLE_MS,
        upload_settle_ms: int = UPLOAD_SETTLE_MS,
    ) -> None:
        self.page = page
        self.pointer = pointer
        self.prompts = prompts
        self.dry_run = dry_run
        # Variante del ensayo que SÍ sube (borrador privado) para poder recorrer
        # las pestañas del asistente, y lo descarta al final. Solo tiene sentido
        # dentro de un dry-run: en una publicación real no se descarta nada.
        self.dry_run_draft = dry_run and dry_run_draft
        self.debug_dir = debug_dir
        self.settle_ms = settle_ms
        self.modal_settle_ms = modal_settle_ms
        self.upload_settle_ms = upload_settle_ms
        self._steps: list[str] = []
        # En dry-run el ratón está vivo hasta que se dispara el selector de
        # ficheros; a partir de ahí se congela (ver _freeze_clicks).
        self._clicks_frozen = False
        # Contabilidad del borrador: si el fichero llegó a entregarse, hay algo
        # en el canal que HAY que descartar sí o sí antes de salir.
        self._file_delivered = False
        self._draft_discarded = False
        self._draft_discard_failed = False
        # Pasos que ya han volcado su captura (con la cruz en el punto
        # pinchado): _mark no debe pisarlas con una captura sin marcar.
        self._captured: set[str] = set()

    # ------------------------------------------------------------------ #
    #  API pública
    # ------------------------------------------------------------------ #

    def publish(
        self,
        video_path: Path | str,
        title: str,
        *,
        description: str = "",
        visibility: str = "private",
        studio_url: str = DEFAULT_STUDIO_URL,
    ) -> PublishResult:
        """Sube ``video_path`` y lo guarda con la visibilidad indicada.

        Args:
            video_path: .mp4 local (pensado para clips de 3-4 s / Shorts).
            title: título del vídeo (se escribe reemplazando el que YouTube
                autorrellena con el nombre del fichero).
            description: descripción opcional (p.ej. métricas de
                ``telemetry.json``). Vacío -> no se toca el campo.
            visibility: "private" | "unlisted" | "public". El defecto sigue
                siendo privado; "public" hay que pedirlo a mano.
            studio_url: por si se quiere apuntar a una marca/canal concreto
                (``.../channel/<id>``).
        """
        video_path = Path(video_path)
        result = PublishResult(
            video_path=str(video_path),
            title=title,
            visibility=visibility,
            published=False,
            dry_run=self.dry_run,
        )

        if not video_path.exists():
            result.error = f"No existe el vídeo: {video_path}"
            logger.error(result.error)
            return result
        if visibility not in _VISIBILITY_RADIOS:
            result.error = (
                f"Visibilidad no soportada: {visibility!r} "
                f"({'|'.join(_VISIBILITY_RADIOS)})"
            )
            logger.error(result.error)
            return result

        try:
            logger.info("Navegando a %s ...", studio_url)
            self.page.goto(studio_url, wait_until="domcontentloaded")
            self._settle(self.settle_ms)
            self._assert_logged_in()

            self._step_open_upload_dialog()
            self._step_select_file(video_path)

            if self.dry_run and not self.dry_run_draft:
                # Frontera del ensayo: lo siguiente sería entregar el fichero y
                # la subida arrancaría sola. Sin fichero, el asistente no pasa
                # de esta pantalla (Detalles/Elementos/Comprobaciones/
                # Visibilidad no existen todavía), así que seguir prediciendo
                # solo produciría capturas repetidas de este mismo diálogo.
                self._snapshot("10_parado_antes_de_subir")
                self._mark("10_parado_antes_de_subir")
                logger.info(
                    "DRY-RUN terminado en el último punto sin retorno: el vídeo NO se "
                    "ha entregado. Pasos recorridos: %s",
                    " -> ".join(self._steps) or "(ninguno)",
                )
                return result  # steps_completed lo rellena el finally

            self._step_fill_details(title, description)
            self._step_advance_wizard()
            self._step_set_visibility(visibility)

            if self.dry_run_draft:
                # Mismo asistente que una publicación real, pero la última
                # pulsación es "descartar", no "guardar".
                self._step_discard_draft()
            else:
                self._step_save()
                result.published = True
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Fallo publicando %s", video_path)
            self._snapshot("error")
        finally:
            # Red de seguridad del modo borrador: si se subió el fichero y el
            # flujo murió antes de descartarlo (o ni siquiera llegó al final),
            # se intenta el descarte igualmente. Un borrador olvidado en el
            # canal es exactamente lo que este modo promete no dejar.
            if self.dry_run_draft and self._file_delivered and not self._draft_discarded:
                try:
                    self._step_discard_draft()
                except Exception:
                    logger.exception("El descarte de emergencia del borrador también falló")
                    self._draft_discard_failed = True

            if self._draft_discard_failed and not result.error:
                result.error = (
                    "Ensayo con borrador: el vídeo se subió pero NO se pudo descartar. "
                    "Bórralo a mano en studio.youtube.com -> Contenido."
                )
            result.steps_completed = list(self._steps)

        return result

    # ------------------------------------------------------------------ #
    #  Fases del asistente
    # ------------------------------------------------------------------ #

    def _step_open_upload_dialog(self) -> None:
        """Paso 1: "CREAR" (cabecera) -> "Subir vídeos" (menú desplegable)."""
        self._click(self.prompts.create_button, step="01_crear", settle_ms=self.settle_ms,
                    required=True)
        # El desplegable entra con animación: sin esta pausa Molmo ve el menú
        # a medio desplegar y apunta a la posición final equivocada.
        self._click(self.prompts.upload_videos_item, step="02_subir_videos",
                    settle_ms=1500, required=True)

    def _step_select_file(self, video_path: Path) -> None:
        """Paso 2: entrega del .mp4 al diálogo de subida.

        Aquí se rompe deliberadamente la filosofía "solo VLM": el selector de
        ficheros es una ventana NATIVA del sistema operativo, que no sale en
        ``page.screenshot()`` y sobre la que Playwright no puede mover el ratón.
        Playwright intercepta ese diálogo con ``expect_file_chooser``, así que
        el clic guiado por Molmo sirve solo para DISPARARLO y el fichero se
        entrega por API.

        Plan B: escribir la ruta directamente en el ``<input type=file>`` del
        DOM. No es frágil como un selector de layout — el input existe por
        contrato del estándar HTML, no por decisión de diseño de YouTube.
        """
        page = self.page
        if self.dry_run and not self.dry_run_draft:
            self._dry_run_open_file_chooser()
            return

        try:
            with page.expect_file_chooser(timeout=20_000) as fc_info:
                self._click(self.prompts.select_files_button, step="03_seleccionar_archivos",
                            settle_ms=self.modal_settle_ms, required=True)
            file_chooser = fc_info.value
            file_chooser.set_files(str(video_path))
            logger.info("Fichero entregado vía file chooser: %s", video_path.name)
        except Exception as exc:
            logger.warning("file_chooser falló (%s); probando input[type=file] del DOM", exc)
            page.set_input_files("input[type=file]", str(video_path))
            logger.info("Fichero entregado vía input[type=file]: %s", video_path.name)

        # A partir de aquí hay bytes en YouTube: en modo borrador, el descarte
        # final deja de ser opcional.
        self._file_delivered = True
        self._mark("03_archivo_seleccionado")
        # La subida arranca sola y el formulario de detalles aparece encima.
        self._settle(self.upload_settle_ms)

    def _dry_run_open_file_chooser(self) -> None:
        """Paso 2 en dry-run: abre el selector de ficheros y lo descarta.

        Es el ÚLTIMO clic real del ensayo. Se dispara el diálogo nativo para
        comprobar que el prompt de "Seleccionar archivos" acierta, pero
        Playwright lo intercepta y, al no llamar a ``set_files``, no se entrega
        ningún vídeo: no hay subida que abortar después. Pase lo que pase, al
        salir el ratón queda congelado, porque a partir de aquí cualquier clic
        caería sobre una UI que en la ejecución real ya estaría subiendo.
        """
        try:
            with self.page.expect_file_chooser(timeout=20_000):
                self._click(self.prompts.select_files_button,
                            step="03_seleccionar_archivos",
                            settle_ms=self.modal_settle_ms, required=True)
            logger.info("DRY-RUN: selector de ficheros abierto y descartado (no se sube nada)")
            self._mark("03_selector_abierto_sin_fichero")
        except Exception as exc:
            logger.warning("DRY-RUN: el selector de ficheros no llegó a abrirse (%s)", exc)
        finally:
            self._freeze_clicks("fichero no entregado")

    def _step_fill_details(self, title: str, description: str) -> None:
        """Paso 3: título, descripción opcional y el radio obligatorio de "niños"."""
        # Título: YouTube lo autorrellena con el nombre del fichero, así que hay
        # que seleccionar todo antes de teclear o el texto queda concatenado.
        if self._click(self.prompts.title_field, step="04_titulo", required=True):
            self._press("Control+A")
            self._press("Delete")
            self._type(title)
            self._mark("04_titulo_escrito")

        if description:
            # La descripción está justo debajo del título: se llega con Tab en
            # vez de gastar una predicción del VLM (el orden de tabulación del
            # formulario es estable, el layout no).
            self._press("Tab")
            self._type(description)
            self._mark("05_descripcion_escrita")

        # Campo OBLIGATORIO: sin marcarlo, "SIGUIENTE" no avanza.
        # Está más abajo en el formulario -> hay que hacer scroll para que
        # entre en el viewport y Molmo pueda verlo.
        if not self._clicks_disabled:
            self.page.mouse.wheel(0, 600)
        self._click(self.prompts.kids_no_radio, step="06_no_para_ninos", required=True)

    def _step_advance_wizard(self) -> None:
        """Paso 3b: avanzar hasta Visibilitat (Detalls -> Elements -> Comprovacions).

        No se cuentan clics: se avanza HASTA que la pantalla de visibilidad
        aparece. Contar era frágil y se rompió en la primera publicación real —
        "Enrere" está pegado a "Següent", el VLM pinchó atrás en el tercer
        avance y a partir de ahí toda la secuencia iba una pantalla desfasada,
        con el paso de visibilidad pinchando "Següent". Comprobando el estado,
        un clic perdido solo cuesta una vuelta más del bucle.

        No se toca nada en Elements (pantallas finales/tarjetas) ni en
        Comprovacions (derechos de autor): los valores por defecto valen y cada
        pantalla extra es una predicción más que puede fallar.
        """
        for intento in range(1, _MAX_WIZARD_ADVANCES + 1):
            if self._on_visibility_step():
                logger.info("Asistente en la pantalla de visibilidad (%d avances)", intento - 1)
                self._mark("07_visibilidad_alcanzada")
                return

            step = f"07_siguiente_{intento}"
            # "next-button" es el id del web component, no una clase ofuscada.
            if self._dom_click("ytcp-uploads-dialog #next-button", "botón Següent",
                               step=step):
                self._mark(step)
            else:
                self._click(self.prompts.next_button, step=step, required=True)
            self._settle(self.modal_settle_ms)

        self._snapshot("07_asistente_atascado")
        raise RuntimeError(
            f"El asistente no llegó a la pantalla de visibilidad tras "
            f"{_MAX_WIZARD_ADVANCES} avances (mira 07_asistente_atascado.png)"
        )

    def _on_visibility_step(self) -> bool:
        """True si el asistente muestra ya la pantalla de visibilidad.

        Dos señales independientes porque ninguna es segura por sí sola: los
        radios (contrato del web component, pero Studio podría renombrarlos) y
        el texto del bloque "Desa o publica", que solo existe en esta pantalla
        —el rótulo "Visibilitat" del stepper de arriba sale en las cuatro, así
        que no vale para distinguirlas—.
        """
        visibles, _ = self._visibility_state()
        if visibles:
            return True
        try:
            texto = self.page.locator("ytcp-uploads-dialog").first.inner_text(timeout=3000)
        except Exception:
            return False
        return bool(re.search(r"desa o publica|guardar o publicar|save or publish", texto, re.I))

    def _step_set_visibility(self, visibility: str) -> None:
        """Paso 4: visibilidad. El único paso que va por DOM antes que por VLM.

        Dos razones para romper aquí la filosofía del módulo:

        * es donde el VLM falló sistemáticamente —tres radios idénticos, poco
          contraste, y el objetivo anterior ("Següent") tirando de él—, y
        * un fallo aquí no es visible: "Desa" se queda deshabilitado y la
          subida muere tres minutos después con un error que no menciona la
          visibilidad. Peor aún, equivocarse de radio significaría publicar en
          "Públic" un vídeo que se pidió privado.

        El atributo ``name`` del radio (PRIVATE/UNLISTED/PUBLIC) es del web
        component, no una clase ofuscada. Y después se COMPRUEBA qué quedó
        marcado: pinchar no es seleccionar.
        """
        prompt = {
            "private": self.prompts.visibility_private,
            "unlisted": self.prompts.visibility_unlisted,
            "public": self.prompts.visibility_public,
        }[visibility]
        step = f"08_visibilidad_{visibility}"
        selector = f"tp-yt-paper-radio-button[name='{_VISIBILITY_RADIOS[visibility]}']"

        if self._dom_click(selector, f"radio de visibilidad '{visibility}'", step=step):
            self._mark(step)
        else:
            self._click(prompt, step=step, required=True)
        self._settle(self.modal_settle_ms)

        visibles, marcada = self._visibility_state()
        if not visibles:
            # No se han encontrado los radios: puede que Studio haya cambiado el
            # tag. No es motivo para abortar —el clic pudo funcionar—, pero sí
            # para dejarlo dicho antes de que falle "Desa".
            logger.warning(
                "No se pudieron localizar los radios de visibilidad para verificar "
                "la selección; si 'Desa' no cierra el diálogo, empieza por aquí"
            )
            return
        if marcada == visibility:
            logger.info("Visibilidad '%s' confirmada", visibility)
            self._mark(f"08_visibilidad_confirmada_{visibility}")
            return
        if marcada is None:
            self._snapshot("08_visibilidad_sin_marcar")
            raise RuntimeError(
                f"Ningún radio de visibilidad quedó marcado (se pedía {visibility!r}). "
                "'Desa' seguirá deshabilitado hasta elegir una."
            )
        self._snapshot("08_visibilidad_equivocada")
        raise RuntimeError(
            f"Se marcó la visibilidad {marcada!r} y se había pedido {visibility!r}"
        )

    def _visibility_state(self) -> tuple[bool, Optional[str]]:
        """(¿se ven los radios de visibilidad?, cuál está marcado).

        Distinguir "no los encuentro" de "no hay ninguno marcado" importa: lo
        primero es un selector obsoleto (aviso), lo segundo un paso fallido
        (error).
        """
        visibles = False
        marcada = None
        for visibility, name in _VISIBILITY_RADIOS.items():
            try:
                radio = self.page.locator(f"tp-yt-paper-radio-button[name='{name}']")
                if radio.count() == 0:
                    continue
                visibles = True
                if radio.first.get_attribute("aria-checked", timeout=2000) == "true":
                    marcada = visibility
            except Exception:
                continue
        return visibles, marcada

    def _step_discard_draft(self) -> None:
        """Cierre del ensayo con subida: deja el canal como estaba.

        Lo que se aprendió del primer ensayo real: cuando el vídeo ya se ha
        subido, Studio lo AUTOGUARDA como borrador privado (badge "S\'ha desat
        com a privat") y la X cierra el asistente sin preguntar nada. O sea que
        "descartar" no es un botón: hay que borrar el vídeo después. Por eso el
        método hace tres cosas, en este orden:

          1. leer el ID del vídeo del enlace que el asistente muestra (es el
             único sitio donde aparece, y desaparece al cerrar);
          2. cerrar el asistente, descartando por el camino si Studio llega a
             preguntar (versiones antiguas sí lo hacen);
          3. ir a ``/video/<id>/edit`` y borrarlo definitivamente.

        Ningún clic es ``required``: este método corre también desde el
        ``finally`` de ``publish`` y una excepción aquí taparía el error
        original. Se marca ``_draft_discarded`` al entrar, así que el camino
        normal y la red de seguridad no pueden pisarse.
        """
        if self._draft_discarded:
            return
        self._draft_discarded = True
        logger.info("Ensayo con borrador: eliminando el vídeo subido ...")

        video_id = self._draft_video_id()
        self._close_upload_dialog()

        if video_id is None:
            self._draft_discard_failed = True
            self._snapshot("93_sin_id_de_video")
            logger.error(
                "No se pudo leer el ID del vídeo subido, así que no se puede borrar "
                "solo. Míralo a mano en studio.youtube.com -> Contingut."
            )
            return

        self._delete_uploaded_video(video_id)

    def _close_upload_dialog(self) -> None:
        """Cierra el asistente de subida (DOM primero, VLM de reserva)."""
        if self._upload_dialog_closed():
            return

        # El id "close-button" es del web component de Studio, no una clase
        # ofuscada por el bundler: es de lo poco estable que hay aquí.
        if not self._dom_click("ytcp-uploads-dialog #close-button", "X del asistente",
                               step="90_cerrar_dialogo"):
            self._click(self.prompts.close_button, step="90_cerrar_dialogo",
                        settle_ms=self.settle_ms)
        self._settle(self.modal_settle_ms)

        # Si Studio pregunta qué hacer con el vídeo, se descarta. Si ya lo había
        # autoguardado como borrador, no pregunta y esto no se ejecuta.
        if not self._upload_dialog_closed():
            if not self._dom_click_text(r"descarta|descartar|discard", "opción Descartar",
                                        step="91_descartar"):
                self._click(self.prompts.discard_option, step="91_descartar",
                            settle_ms=self.modal_settle_ms)
            self._settle(self.modal_settle_ms)

        if self._upload_dialog_closed():
            self._mark("90_dialogo_cerrado")
        else:
            self._snapshot("90_dialogo_sigue_abierto")
            logger.warning("El asistente sigue abierto tras intentar cerrarlo")

    def _delete_uploaded_video(self, video_id: str) -> None:
        """Borra definitivamente el vídeo del ensayo desde su página de edición.

        Ir directo a ``/video/<id>/edit`` es determinista —es una URL, no un
        layout— y evita tener que encontrar la fila del borrador en Contingut.
        A partir de ahí sí hay que pinchar: menú "⋮" -> "Suprimeix
        definitivament" -> casilla de confirmación -> botón final.
        """
        url = f"https://studio.youtube.com/video/{video_id}/edit"
        logger.info("Borrando el vídeo del ensayo: %s", url)

        try:
            self.page.goto(url, wait_until="domcontentloaded")
        except Exception:
            logger.exception("No se pudo abrir la página de edición del vídeo")
            self._draft_discard_failed = True
            return
        self._settle(self.settle_ms * 2)

        # (paso, selector DOM si lo hay, prompt del VLM de reserva)
        secuencia = (
            ("94_menu_opciones", "ytcp-icon-button#options-button, #options-button",
             self.prompts.options_menu),
            ("95_suprimeix_definitivament", None, self.prompts.delete_forever_option),
            ("96_casilla_confirmacion", None, self.prompts.delete_confirm_checkbox),
            ("97_confirmar_borrado", None, self.prompts.delete_confirm_button),
        )
        for step, selector, prompt in secuencia:
            if selector and self._dom_click(selector, step, step=step):
                self._mark(step)
            else:
                self._click(prompt, step=step, settle_ms=self.modal_settle_ms)
            self._settle(self.modal_settle_ms)

        self._settle(3000)
        if self._video_is_gone(video_id):
            logger.info("Vídeo del ensayo eliminado (%s): el canal queda como estaba.", video_id)
            self._mark("97_video_eliminado")
            return

        self._draft_discard_failed = True
        self._snapshot("97_borrado_fallido")
        logger.error(
            "NO se pudo borrar el vídeo del ensayo. Sigue subido como borrador PRIVADO. "
            "Bórralo a mano aquí: %s", url,
        )

    def _draft_video_id(self) -> Optional[str]:
        """ID del vídeo recién subido, leído del enlace que enseña el asistente."""
        for selector in ("ytcp-uploads-dialog a[href*='youtu']",
                         "a[href*='youtu.be'], a[href*='/shorts/'], a[href*='watch?v=']"):
            try:
                href = self.page.locator(selector).first.get_attribute("href", timeout=5000)
            except Exception:
                continue
            match = _VIDEO_ID_RE.search(href or "")
            if match:
                logger.info("Vídeo del ensayo: %s", match.group(1))
                return match.group(1)

        logger.warning("No se encontró el enlace del vídeo en el asistente")
        return None

    def _video_is_gone(self, video_id: str) -> bool:
        """True si la página de edición del vídeo ya no existe (Studio redirige)."""
        try:
            self.page.goto(f"https://studio.youtube.com/video/{video_id}/edit",
                           wait_until="domcontentloaded")
            self.page.wait_for_timeout(self.settle_ms)
            return video_id not in self.page.url
        except Exception:
            return False

    def _dom_click(
        self,
        selector: str,
        what: str,
        *,
        step: Optional[str] = None,
        timeout_ms: int = 4000,
    ) -> bool:
        """Clic por selector, para lo que sí tiene un contrato estable.

        Devuelve False en vez de lanzar: siempre es un atajo, nunca el único
        camino — el VLM queda de reserva. Con ``step`` deja la misma captura
        anotada que un paso guiado por el VLM (cruz roja incluida), para que
        ``--debug-dir`` no tenga huecos donde el DOM ha ido más rápido.
        """
        try:
            locator = self.page.locator(selector).first
            locator.wait_for(state="visible", timeout=timeout_ms)
            if step is not None:
                self._dump_dom_target(step, locator, what)
            locator.click(timeout=timeout_ms)
            logger.info("DOM: clic en %s", what)
            return True
        except Exception as exc:
            logger.info("DOM: %s no clicable (%s); lo intenta el VLM",
                        what, type(exc).__name__)
            return False

    def _dom_click_text(
        self,
        pattern: str,
        what: str,
        *,
        step: Optional[str] = None,
        timeout_ms: int = 4000,
    ) -> bool:
        """Clic por TEXTO dentro del último diálogo abierto.

        Acotado al diálogo a propósito: un ``get_by_text`` suelto sobre toda la
        página de Studio pillaría cualquier "Descarta..." de la interfaz.
        """
        try:
            scope = self.page.locator("tp-yt-paper-dialog, ytcp-confirmation-dialog").last
            objetivo = scope.get_by_text(re.compile(pattern, re.I)).first
            objetivo.wait_for(state="visible", timeout=timeout_ms)
            if step is not None:
                self._dump_dom_target(step, objetivo, what)
            objetivo.click(timeout=timeout_ms)
            logger.info("DOM: clic en %s", what)
            return True
        except Exception as exc:
            logger.info("DOM: %s no clicable (%s); lo intenta el VLM",
                        what, type(exc).__name__)
            return False

    def _dump_dom_target(self, step: str, locator, what: str) -> None:
        """Captura con la cruz sobre el elemento que se va a pinchar por DOM.

        El centro sale de ``bounding_box()`` (píxeles CSS) y se guarda como
        fracción, igual que hace el VLM en ``Point``: así las dos rutas
        producen imágenes idénticas y comparables.
        """
        if self.debug_dir is None:
            return
        try:
            box = locator.bounding_box()
            if box is None:
                return
            viewport = self.page.viewport_size or DEFAULT_VIEWPORT
            punto = Point(
                x=(box["x"] + box["width"] / 2) / viewport["width"],
                y=(box["y"] + box["height"] / 2) / viewport["height"],
                label=f"DOM: {what}",
                raw="",
                action="click",
            )
            _dump_debug(self.debug_dir, step, capture(self.page, 0), punto)
            self._captured.add(step)
        except Exception:
            logger.exception("No se pudo volcar la captura DOM de '%s'", step)

    def _reset_pointer_history(self) -> None:
        """Arranca una trayectoria nueva si el pointer la lleva (MolmoWeb sí)."""
        reset = getattr(self.pointer, "reset_history", None)
        if callable(reset):
            reset()
            logger.debug("Trayectoria del VLM reiniciada")

    def _step_save(self) -> None:
        """Paso 4b: "GUARDAR".

        Si el procesado del vídeo aún no ha terminado el botón sale
        deshabilitado y el clic no hace nada; por eso se reintenta con una
        espera larga en medio en vez de dar el paso por bueno a la primera.
        """
        for attempt in range(1, 4):
            step = f"09_guardar_intento_{attempt}"
            # Mismo motivo que en el avance: "Desa" tiene "Enrere" pegado a la
            # izquierda y aquí un clic torcido no es recuperable (se vuelve a
            # Comprovacions con el vídeo a medio publicar).
            if self._dom_click("ytcp-uploads-dialog #done-button", "botón Desa",
                               step=step):
                self._mark(step)
            else:
                self._click(self.prompts.save_button, step=step,
                            settle_ms=self.settle_ms, required=(attempt == 3))
            self._settle(5000)
            if self._upload_dialog_closed():
                logger.info("Diálogo cerrado: el vídeo se ha guardado.")
                self._mark("09_guardado")
                return
            logger.warning(
                "El diálogo sigue abierto tras 'Guardar' (intento %d/3); "
                "probablemente el vídeo aún se está procesando. Reintentando ...",
                attempt,
            )
            self._settle(15_000)

        raise RuntimeError(
            "'Guardar' no cerró el diálogo tras 3 intentos. Si la visibilidad no "
            "llegó a quedar marcada, el botón está deshabilitado y el clic no hace "
            "nada por muchas veces que se repita (mira 08_visibilidad_*.png)"
        )

    # ------------------------------------------------------------------ #
    #  Utilidades internas
    # ------------------------------------------------------------------ #

    @property
    def _clicks_disabled(self) -> bool:
        """True cuando hay que predecir sin tocar la página.

        Solo pasa en dry-run y solo después de congelar el ratón: los primeros
        pasos del ensayo son clics reales.
        """
        return self.dry_run and self._clicks_frozen

    def _freeze_clicks(self, reason: str) -> None:
        """Apaga los clics de lo que queda de ensayo (no-op fuera de dry-run).

        En modo borrador NO se congela nunca: ahí ya hay un vídeo subido y
        quedarse sin ratón significaría quedarse sin poder descartarlo, que es
        peor que seguir intentando pinchar.
        """
        if not self.dry_run or self.dry_run_draft or self._clicks_frozen:
            return
        self._clicks_frozen = True
        logger.info("DRY-RUN: ratón congelado (%s); a partir de aquí solo capturas", reason)

    def _click(
        self,
        prompt: str,
        *,
        step: str,
        settle_ms: Optional[int] = None,
        required: bool = False,
    ) -> Optional[Point]:
        """``predict_and_click`` + registro del paso + política de error."""
        # Una trayectoria por objetivo. MolmoWeb reinyecta los PREVIOUS STEPS y,
        # si se le dejan los del objetivo ANTERIOR, sigue contestando a ese: en
        # la primera publicación real el paso de visibilidad razonó "the goal is
        # to click 'Següent'" y volvió a pinchar el botón de siguiente, con lo
        # que "Desa" se quedó deshabilitado y la subida murió. Con el mismo
        # objetivo repetido tampoco ayuda: al segundo "Següent" respondió
        # "[ANSWER] Done" en vez de pinchar. Así que aquí cada paso arranca
        # limpio y el modelo solo mira la pantalla actual.
        self._reset_pointer_history()
        point = predict_and_click(
            self.page,
            prompt,
            self.pointer,
            settle_ms=self.settle_ms if settle_ms is None else settle_ms,
            dry_run=self._clicks_disabled,
            debug_dir=self.debug_dir,
            step=step,
        )
        if point is None:
            if required and not self.dry_run:
                raise RuntimeError(f"Paso '{step}': Molmo no localizó el elemento")
            # En dry-run seguimos adelante para recorrer el flujo entero y
            # recoger TODAS las capturas fallidas de una tacada; eso sí, si el
            # paso fallido era obligatorio y aún estábamos haciendo clics, la
            # página ya no está donde creemos: mejor congelar el ratón que
            # seguir pinchando a ciegas.
            if required:
                self._freeze_clicks(f"paso obligatorio '{step}' sin coordenadas")
            logger.warning("Paso '%s' sin coordenadas; continuando", step)
        else:
            if self.debug_dir is not None:
                self._captured.add(step)
            self._mark(step)
        return point

    def _press(self, key: str) -> None:
        """Pulsa una tecla — con el ratón congelado no se toca el teclado."""
        if self._clicks_disabled:
            logger.info("DRY-RUN: se pulsaría %s", key)
            return
        self.page.keyboard.press(key)

    def _type(self, text: str) -> None:
        """Teclea como una persona (delay) — con el ratón congelado no escribe."""
        if self._clicks_disabled:
            logger.info("DRY-RUN: se escribiría %r", text)
            return
        self.page.keyboard.type(text, delay=40)

    def _settle(self, ms: int) -> None:
        self.page.wait_for_timeout(ms)

    def _mark(self, step: str) -> None:
        """Registra el paso y, si nadie lo ha capturado ya, captura la pantalla.

        Así ``--debug-dir`` acaba con una imagen por CADA paso de
        ``steps_completed``, incluidos los que no son un clic (fichero
        entregado, título escrito, visibilidad confirmada, guardado). Los pasos
        con clic ya traen la suya con la cruz en el punto pinchado y no se
        pisan.
        """
        self._steps.append(step)
        if step not in self._captured:
            self._snapshot(step)
            self._captured.add(step)

    def _snapshot(self, step: str) -> None:
        """Captura sin predicción (para post-mortem de un fallo)."""
        if self.debug_dir is None:
            return
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=str(self.debug_dir / f"{step}.png"), full_page=False)
        except Exception:
            logger.exception("No se pudo capturar el estado de '%s'", step)

    def _assert_logged_in(self) -> None:
        """Comprueba que el perfil persistente sigue con sesión iniciada.

        Chequeo por URL (accounts.google.com) en vez de por DOM: es lo único
        que Google no puede cambiar sin cambiar su propio flujo de login.
        """
        url = self.page.url
        if "accounts.google.com" in url or "signin" in url:
            raise RuntimeError(
                "El perfil no tiene sesión de Google iniciada (redirigido a "
                f"{url}). Ejecuta primero: python -m automation.molmo_publisher --login-only"
            )

    def _upload_dialog_closed(self) -> bool:
        """True si el asistente de subida ya no está en pantalla.

        Se apoya en ``ytcp-uploads-dialog``, el nombre del web component de
        Studio: no es una clase CSS ofuscada por el bundler sino el tag del
        elemento, mucho más estable. Si algún día desaparece, el fallback
        conservador es "sigue abierto" y el flujo reintenta.
        """
        try:
            return self.page.locator("ytcp-uploads-dialog").count() == 0
        except Exception:
            return False


# --------------------------------------------------------------------------- #
#  Sesión de navegador (contexto persistente)
# --------------------------------------------------------------------------- #

def launch_studio_context(
    playwright,
    *,
    user_data_dir: Path = DEFAULT_USER_DATA_DIR,
    headless: bool = False,
    viewport: Optional[dict[str, int]] = None,
    slow_mo_ms: int = 0,
) -> "BrowserContext":
    """Abre Chromium con perfil persistente (cookies de Google reutilizables).

    ``headless=False`` por defecto: además de permitir depurar viendo el flujo,
    Studio se comporta peor en headless y el login manual inicial exige ventana.
    """
    user_data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Abriendo Chromium con perfil persistente en %s", user_data_dir)

    return playwright.chromium.launch_persistent_context(
        user_data_dir=str(user_data_dir),
        headless=headless,
        viewport=viewport or DEFAULT_VIEWPORT,
        slow_mo=slow_mo_ms,
        args=[
            "--disable-blink-features=AutomationControlled",  # menos fricción en el login
            "--no-first-run",
            "--no-default-browser-check",
        ],
    )


def publish_clip(
    video_path: Path | str,
    title: str,
    *,
    description: str = "",
    visibility: str = "private",
    pointer: Optional[Pointer] = None,
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
    dtype: str = "bfloat16",
    max_new_tokens: int = 200,
    venv_python: Path = DEFAULT_VENV_PYTHON,
    user_data_dir: Path = DEFAULT_USER_DATA_DIR,
    headless: bool = False,
    viewport: Optional[dict[str, int]] = None,
    settle_ms: int = DEFAULT_SETTLE_MS,
    modal_settle_ms: int = MODAL_SETTLE_MS,
    upload_settle_ms: int = UPLOAD_SETTLE_MS,
    dry_run: bool = False,
    dry_run_draft: bool = False,
    debug_dir: Optional[Path] = None,
    studio_url: str = DEFAULT_STUDIO_URL,
    keep_open_ms: int = 3000,
) -> PublishResult:
    """Flujo completo de un tirón: abre navegador, publica y cierra.

    Es el punto de entrada que consume el pipeline (``main.py --publish``, que
    rellena estos argumentos desde ``config.PublisherConfig``).
    ``pointer`` permite reutilizar uno ya cargado si se publican varios clips
    seguidos (y entonces lo cierra quien lo creó); si es None, se levanta aquí
    un ``RemoteMolmoPointer`` y se cierra al terminar.
    """
    propio_pointer = pointer is None
    if pointer is None:
        # Remoto por defecto: el checkpoint no arranca con el transformers del
        # venv principal (ver RemoteMolmoPointer / automation/molmo_server.py).
        pointer = RemoteMolmoPointer(
            model_id=model_id, device=device, dtype=dtype,
            max_new_tokens=max_new_tokens, venv_python=venv_python,
        )

    try:
        return _publish_with_pointer(
            pointer, video_path, title,
            description=description, visibility=visibility, studio_url=studio_url,
            user_data_dir=user_data_dir, headless=headless, viewport=viewport,
            settle_ms=settle_ms, modal_settle_ms=modal_settle_ms,
            upload_settle_ms=upload_settle_ms, dry_run=dry_run,
            dry_run_draft=dry_run_draft, debug_dir=debug_dir,
            keep_open_ms=keep_open_ms,
        )
    finally:
        # Si el pointer lo creamos aquí, aquí se cierra: si no, se queda un
        # subproceso con 9 GB de VLM en la GPU tras terminar la publicación.
        if propio_pointer:
            pointer.close()


def _publish_with_pointer(
    pointer: Pointer,
    video_path: Path | str,
    title: str,
    *,
    description: str,
    visibility: str,
    studio_url: str,
    user_data_dir: Path,
    headless: bool,
    viewport: Optional[dict[str, int]],
    settle_ms: int,
    modal_settle_ms: int,
    upload_settle_ms: int,
    dry_run: bool,
    dry_run_draft: bool,
    debug_dir: Optional[Path],
    keep_open_ms: int,
) -> PublishResult:
    """Abre el navegador y publica con un ``pointer`` ya construido."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        context = launch_studio_context(
            playwright, user_data_dir=user_data_dir, headless=headless, viewport=viewport
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            publisher = YouTubeStudioPublisher(
                page, pointer,
                dry_run=dry_run, dry_run_draft=dry_run_draft, debug_dir=debug_dir,
                settle_ms=settle_ms,
                modal_settle_ms=modal_settle_ms,
                upload_settle_ms=upload_settle_ms,
            )
            result = publisher.publish(
                video_path, title,
                description=description, visibility=visibility, studio_url=studio_url,
            )
            # Margen para ver el estado final antes de que se cierre la ventana.
            page.wait_for_timeout(keep_open_ms)
            return result
        finally:
            context.close()


def interactive_login(
    *,
    user_data_dir: Path = DEFAULT_USER_DATA_DIR,
    studio_url: str = DEFAULT_STUDIO_URL,
    timeout_min: int = 10,
) -> None:
    """Abre Studio y espera a que inicies sesión A MANO (una única vez).

    El login de Google es exactamente lo que no automatizamos: 2FA, CAPTCHAs y
    detección de automatización. Se hace a mano una vez y el perfil persistente
    guarda las cookies para todas las ejecuciones siguientes.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        context = launch_studio_context(playwright, user_data_dir=user_data_dir)
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(studio_url, wait_until="domcontentloaded")
            logger.info(
                "Inicia sesión en la ventana abierta. Esperando hasta %d min a que "
                "la URL sea studio.youtube.com ...", timeout_min,
            )
            page.wait_for_url("**/studio.youtube.com/**", timeout=timeout_min * 60_000)
            page.wait_for_timeout(3000)
            logger.info("Sesión guardada en %s", user_data_dir)
        finally:
            context.close()


# --------------------------------------------------------------------------- #
#  Ejecución standalone
# --------------------------------------------------------------------------- #

def main() -> None:
    import config

    cfg = config.PublisherConfig()

    parser = argparse.ArgumentParser(
        description="Publicación automática de clips en YouTube Studio (MolmoWeb-4B + Playwright)"
    )
    parser.add_argument("--video", type=Path, default=None,
                        help="Clip .mp4 a subir (p.ej. output/miclip/final.mp4)")
    parser.add_argument("--title", default="Test Clip Short", help="Título del vídeo")
    parser.add_argument("--description", default="", help="Descripción (opcional)")
    parser.add_argument("--visibility", choices=["private", "unlisted", "public"],
                        default=cfg.default_visibility,
                        help="Visibilidad al guardar (por defecto: private). "
                             "'public' publica de verdad, para todo el mundo")
    parser.add_argument("--login-only", action="store_true",
                        help="Solo abre el navegador para el login manual inicial y sale")
    parser.add_argument("--dry-run", action="store_true",
                        help="Ensayo: navega con clics reales hasta abrir el selector "
                             "de ficheros y para ahí, sin entregar el vídeo")
    parser.add_argument("--dry-run-draft", action="store_true",
                        help="Ensayo COMPLETO: implica --dry-run pero sí entrega el vídeo "
                             "(se sube como borrador privado) para recorrer todas las "
                             "pestañas del asistente, y al final lo BORRA en vez de "
                             "guardarlo. No publica nada, pero el vídeo pasa por YouTube")
    parser.add_argument("--user-data-dir", type=Path, default=cfg.user_data_dir,
                        help="Perfil persistente de Chromium (cookies de Google)")
    parser.add_argument("--debug-dir", type=Path, default=None,
                        help="Carpeta donde volcar las capturas anotadas de cada paso")
    parser.add_argument("--model-id", default=cfg.model_id)
    parser.add_argument("--device", default=cfg.device, help="cuda | cpu (por defecto: auto)")
    parser.add_argument("--headless", action="store_true",
                        help="Sin ventana (no recomendado: Studio se comporta peor)")
    parser.add_argument("--studio-url", default=cfg.studio_url)
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON de salida con el resultado (si no, solo stdout)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.login_only:
        interactive_login(user_data_dir=args.user_data_dir, studio_url=args.studio_url)
        return

    if args.video is None:
        parser.error("--video es obligatorio (o usa --login-only)")
    if not args.video.exists():
        parser.error(f"No existe: {args.video}")

    result = publish_clip(
        args.video,
        args.title,
        description=args.description,
        visibility=args.visibility,
        model_id=args.model_id,
        device=args.device,
        max_new_tokens=cfg.max_new_tokens,
        user_data_dir=args.user_data_dir,
        headless=args.headless or cfg.headless,
        viewport={"width": cfg.viewport_width, "height": cfg.viewport_height},
        settle_ms=cfg.settle_ms,
        modal_settle_ms=cfg.modal_settle_ms,
        upload_settle_ms=cfg.upload_settle_ms,
        dry_run=args.dry_run or args.dry_run_draft,
        dry_run_draft=args.dry_run_draft,
        debug_dir=args.debug_dir,
        studio_url=args.studio_url,
    )

    logger.info(
        "%s | pasos: %s",
        "PUBLICADO" if result.published else ("DRY-RUN" if result.dry_run else "FALLIDO"),
        " -> ".join(result.steps_completed) or "(ninguno)",
    )
    if result.error:
        logger.error(result.error)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result.to_telemetry(), indent=2, ensure_ascii=False))
        logger.info("Resultado escrito en %s", args.output)

    raise SystemExit(0 if (result.published or (result.dry_run and not result.error)) else 1)


if __name__ == "__main__":
    main()
