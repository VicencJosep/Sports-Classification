"""
core/events_vllm.py
====================
Clasificación de eventos de fútbol (gol, penalti, falta, córner...) vía
**VideoARM** (orquestador agéntico, https://github.com/MILVLG/videoarm)
hablando con **Qwen3.5-9B servido localmente con vLLM** por un endpoint
OpenAI-compatible — backend alternativo a ``core/events.py`` (V-JEPA2 + SVC).

Mismo contrato público que ``core.events.EventClassifier``
(``predict``/``predict_window`` -> ``EventPrediction``, reutilizado de
``core.events``, NO duplicado) para que ``analytics/engine.py``/``main.py``
solo cambien qué clase construyen, no cómo la usan (ambos consumen ese
contrato de forma duck-typed).

Taxonomía: los 9 eventos reales de ``core/events.py`` (card, corner, foul,
goal, kick-off, penalty, shot, substitution, throw-in) + un único
``"background"`` genérico. Colapsa los 7 subtipos de fondo de la SVC original
a propósito: ningún consumidor del contrato depende de ellos
(``analytics.engine._merge_event_windows`` solo mira
``is_background``/``label``) y promptear un VLM para distinguir "público" de
"banquillo" de "cámara principal" es frágil y de poco valor — mismo espíritu
que la propia SVC, que ya fusiona ``shots_on/off_target`` y ``yellow/red
card`` (ver docstring de ``core/events.py``).

vLLM sirve en un proceso APARTE (contenedor propio, lanzado por
``scripts/serve_events_vllm.sh`` — ver README § "Servir Qwen3.5-9B con
vLLM"): a diferencia de ``core.events.EventClassifier`` (carga el backbone
in-process), este módulo solo habla HTTP OpenAI-compatible con ese servidor
— no importa ``torch``/``transformers`` en ningún sitio. VideoARM (paquete
``videoarm``, instalado vía git — ver requirements.txt) es la única
dependencia pesada, y es un import perezoso (dentro de ``__init__``), igual
que el resto de módulos de ``core/`` con dependencias externas no siempre
instaladas.

Sobre el audio: VideoARM tiene un cuarto componente, ``audio_transcriber``
(default ``whisper-1``), que nuestro servidor vLLM no puede atender — detrás
no hay ningún ``/v1/audio/transcriptions``. No hace falta desactivarlo: el
propio ``videoarm.core.agent`` hace ``ffprobe`` del stream de audio ANTES de
llamar a la API y, si no hay, devuelve ``{"status": "no_audio"}`` sin tocar
la red y marca ``video_has_audio=False``, lo que inyecta un mensaje de
sistema ("Do not call audio_transcriber") en las iteraciones siguientes.
Por eso ``predict_window`` recorta SIN pista de audio (``-an``): además de
ahorrar trabajo a ffmpeg, garantiza ese camino y basta con una iteración
desperdiciada como mucho. El prompt lo refuerza pidiendo que no se
transcriba audio.

Uso rápido (standalone, desde la raíz del proyecto, con el servidor vLLM ya
arrancado — ver README):
    python -m core.events_vllm --source clip.mp4
    python -m core.events_vllm --source clips_folder/ --output events.jsonl
    python -m core.events_vllm --source partido.mp4 --start-s 120 --end-s 126
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

from core.events import EventPrediction  # reutilizado, no duplicado

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL: str = "http://localhost:8010/v1"  # ver scripts/serve_events_vllm.sh
DEFAULT_MODEL_ID: str = "Qwen/Qwen3.5-9B"
DEFAULT_API_KEY: str = "EMPTY"          # vLLM no valida la key; VideoARM exige que exista
DEFAULT_MAX_ITERATIONS: int = 2         # Antes tenía que ser 1 (ver _build_agent_class): de la
                                         # iteración 2 en adelante VideoARM deja de forzar
                                         # tool_choice="required" y Qwen3.5 escribe mal sus tool
                                         # calls sin esa restricción. _Qwen35CompatAgent lo fuerza
                                         # en TODAS las iteraciones, así que 2 ya es seguro -- y
                                         # es lo que deja a clip_analyzer (pregunta dirigida sobre
                                         # un rango de frames) entrar en juego, en vez de que el
                                         # agente responda solo con el caption genérico de
                                         # scene_snapper. Subir más no está medido todavía.
DEFAULT_TOTAL_FRAMES_LIMIT: int = 32
DEFAULT_FRAME_ANALYSIS_MAX_FRAMES: int = 16   # ver _configure_environment: clip_analyzer manda
                                               # frames SUELTOS (no mosaico, a diferencia de
                                               # scene_snapper), y el servidor limita
                                               # --limit-mm-per-prompt a 16 imágenes/prompt
                                               # (scripts/serve_events_vllm.sh). Con el default de
                                               # VideoARM (50) la llamada devolvía 400 en silencio
                                               # -- _execute_tool solo atrapa TypeError, así que
                                               # ni siquiera se veía como error, el resultado
                                               # llegaba vacío (frame_range None, answer "")

EVENT_LABELS: tuple[str, ...] = (
    "card", "corner", "foul", "goal", "kick-off",
    "penalty", "shot", "substitution", "throw-in",
)
BACKGROUND_LABEL: str = "background"
ALL_LABELS: tuple[str, ...] = EVENT_LABELS + (BACKGROUND_LABEL,)

_LABELS_JOINED = ", ".join(f'"{label}"' for label in ALL_LABELS)
_PROBABILITIES_TEMPLATE = ", ".join(f'"{label}": 0.0' for label in ALL_LABELS)

# Se probaron definiciones largas por etiqueta (insistiendo en que "shot" no
# hace falta que entre) para atacar el fallo dominante medido, shot ->
# background. Empeoró: 5/33 vs 8/33 en el eval de 33 clips centrado en
# shot/goal (ver README), con "goal" cayendo de 4/8 a 0/8 y shot->background
# subiendo de 7/16 a 11/16. Descartado -- revertido a este prompt corto, que
# es el que dio el mejor resultado medido hasta ahora.
_QUESTION = (
    "You are a football (soccer) event classifier. Watch this clip and decide "
    "which event, if any, is happening. Valid labels: "
    f"{_LABELS_JOINED}. If none of those events is occurring (crowd shots, "
    'bench, main broadcast camera between plays, replay with no new action...), '
    'use "background".\n'
    "There is no audio worth transcribing in this clip (no relevant "
    "commentary) — do not use any audio transcription tool, judge only from "
    "the visual content.\n"
    "Respond with ONLY a single-line JSON object, with your estimated "
    "probability for EVERY label listed above (they should sum to "
    "approximately 1.0), and nothing else:\n"
    f'{{"probabilities": {{{_PROBABILITIES_TEMPLATE}}}}}'
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# Pregunta fija para clip_analyzer (ver _build_agent_class -> _execute_tool):
# sustituye la que el propio agente improvisaría. Apunta directo al fallo
# dominante medido (shot -> background): pide un veredicto explícito sobre
# el balón, no una descripción genérica de "qué pasa".
#
# Enmarca esto como "is this a SHOT, and does it enter the net" -- shot va
# primero y goal queda como cláusula subordinada. Se probó (y descartó) dar
# a las tres opciones (goal/shot/ninguno) el mismo peso gramatical con un
# a/b/c explícito: empeoró TODO (shot 57%->36%, goal 33%->42% pero sigue muy
# por debajo del 58% sin forzar nada, ver README). Contraintuitivo pero
# medido dos veces: el marco "sesgado" hacia shot da el mejor resultado del
# HÍBRIDO (69% vs 59% con el a/b/c) -- lo que importa no es que VideoARM
# solo sea bueno en goal (el directo ya se queda los goles claros antes de
# escalar, ver HybridEventClassifier), sino que sea lo mejor posible en el
# residuo que SÍ le llega, mayoritariamente shots.
_CLIP_ANALYZER_QUESTION = (
    "Focus specifically on the ball. Is a player striking, heading, or "
    "kicking the ball with a clear goal-scoring attempt (a shot) visible in "
    "this frame range? If yes, does the ball visibly cross the goal line "
    "into the net (a goal), or does it NOT enter the net -- saved by the "
    "goalkeeper, blocked, hitting the post/bar, or missing the target "
    "(still a shot, just not a goal)? If there is no shot attempt at all in "
    "this range, say so explicitly. Be specific about what happens to the "
    "ball, not the players around it."
)


# --------------------------------------------------------------------------- #
#  Recorte de ventana temporal (predict_window)
# --------------------------------------------------------------------------- #

@contextmanager
def _extract_window_clip(video_path: Path, start_s: float, end_s: float) -> Iterator[Path]:
    """Recorta ``[start_s, end_s]`` de ``video_path`` a un .mp4 temporal.

    VideoARM.ask() toma una RUTA de vídeo, no frames — a diferencia de
    ``core.events.sample_frames_window`` (decodifica solo la ventana pedida
    en memoria), aquí hace falta un fichero real de la ventana.

    Reencodea (``-c:v libx264``, NO ``-c copy``): con solape 50%
    (``event_stride_s=3.0`` sobre ``event_window_s=6.0``, ver
    ``config.FullMatchConfig``) el corte tiene que caer exacto en
    ``start_s``/``end_s`` — cortar al keyframe más cercano desplazaría
    sistemáticamente el contenido visto por el agente. Sin audio (``-an``):
    ver aviso de ``audio_transcriber`` en el docstring del módulo.
    """
    duration = max(end_s - start_s, 0.04)
    with tempfile.TemporaryDirectory(prefix="events_vllm_") as tmp_dir:
        clip_path = Path(tmp_dir) / "window.mp4"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", str(video_path),
            "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "ultrafast",
            "-an", "-loglevel", "error", str(clip_path),
        ]
        subprocess.run(cmd, check=True, timeout=30)
        yield clip_path


# --------------------------------------------------------------------------- #
#  Parseo de la respuesta (JSON estricto + fallback en dos niveles)
# --------------------------------------------------------------------------- #

def _extract_json_object(text: str) -> Optional[dict]:
    """Primer objeto JSON en ``text``, tolerante a fences ```json``` o prosa
    alrededor. ``None`` si no hay nada parseable.

    También tolera dict de Python (comillas simples: ``{'label': 'goal'}``)
    en vez de JSON real -- observado en la práctica una vez que
    ``clip_analyzer`` entra en juego (``max_iterations=2``, ver
    ``DEFAULT_MAX_ITERATIONS``): el modelo a veces reproduce el formato de la
    evidencia que acaba de leer (la propia respuesta de ``clip_analyzer`` es
    un dict serializado con ``repr``, no con ``json.dumps``) en vez del JSON
    pedido en el prompt. Sin este fallback esa distribución -- perfectamente
    buena -- se tiraba entera a la heurística de nivel 3 de ``_parse_response``.
    """
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    candidate = match.group(0)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            import ast
            obj = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            return None
    return obj if isinstance(obj, dict) else None


def _uniform_probabilities(label: str, confidence: float) -> dict[str, float]:
    remainder = (1.0 - confidence) / (len(ALL_LABELS) - 1)
    return {l: (confidence if l == label else remainder) for l in ALL_LABELS}


def _parse_response(raw: str, video_path_label: str) -> EventPrediction:
    """``raw`` -> ``EventPrediction``, en tres niveles decrecientes de
    confianza:

    1. JSON con ``probabilities`` completo (las 10 etiquetas) — el formato
       pedido en ``_QUESTION``. Da una distribución real (no calibrada, pero
       real), no una repartición sintética.
    2. JSON con solo ``{"label": ..., "confidence": ...}`` — el modelo no
       siguió el formato pedido pero al menos da una etiqueta; se sintetiza
       una distribución uniforme sobre el resto SOLO para no romper el
       contrato ``dict[str, float]`` de ``EventPrediction.probabilities``.
    3. Ni siquiera eso parsea — heurística de buscar el nombre de una
       etiqueta en el texto crudo, con warning (degradación de última
       instancia, no se espera que ocurra a menudo).
    """
    obj = _extract_json_object(raw)

    if obj and isinstance(obj.get("probabilities"), dict) and obj["probabilities"]:
        raw_probs = obj["probabilities"]
        probabilities = {l: float(raw_probs.get(l, 0.0)) for l in ALL_LABELS}
        label = max(probabilities, key=probabilities.get)
        confidence = probabilities[label]
    elif obj and "label" in obj:
        label = str(obj["label"]).strip().lower()
        if label not in ALL_LABELS:
            logger.warning("Etiqueta fuera de vocabulario %r, degradando a background", label)
            label = BACKGROUND_LABEL
        confidence = float(obj.get("confidence", 0.6))
        probabilities = _uniform_probabilities(label, confidence)
    else:
        logger.warning("Respuesta de VideoARM no parseable como JSON, usando heurística: %r", raw[:200])
        lowered = raw.lower()
        label = next((l for l in ALL_LABELS if l in lowered), BACKGROUND_LABEL)
        confidence = 0.5
        probabilities = _uniform_probabilities(label, confidence)

    return EventPrediction(
        video_path=video_path_label,
        label=label,
        confidence=confidence,
        probabilities=probabilities,
        is_background=(label == BACKGROUND_LABEL),
    )


# --------------------------------------------------------------------------- #
#  Parche de compatibilidad VideoARM <-> plantilla de chat de Qwen3.5
# --------------------------------------------------------------------------- #

def _build_agent_class():
    """Devuelve una subclase de ``VideoARMAgent`` compatible con Qwen3.5.

    ``VideoARMAgent._force_final_answer`` (el camino que se toma al agotar el
    presupuesto de iteraciones, y el ÚNICO que produce una respuesta en prosa
    en vez de otra llamada a tool) añade su instrucción final como mensaje de
    rol ``system`` AL FINAL de la conversación. La plantilla de chat de Qwen3.5
    lo rechaza de plano:

        400 {"message": "System message must be at the beginning."}

    ...con lo que el forzado devuelve ``None`` y VideoARM contesta el literal
    "Maximum iterations reached without a conclusive answer.". Con
    ``max_iterations=1`` — nuestra configuración, ver ``DEFAULT_MAX_ITERATIONS``
    — ese camino es el que da SIEMPRE la respuesta, así que sin este parche el
    backend no produce ni una sola clasificación.

    El parche es mínimo a propósito: misma llamada, mismo prompt, solo cambia
    el rol ``system`` -> ``user``. Se hace por subclase y no forkeando el
    paquete para que ``pip install -U`` de videoarm siga siendo trivial; si
    upstream lo arregla, esta subclase se vuelve un no-op inofensivo.

    Es una función y no una clase de módulo porque ``videoarm`` es un import
    perezoso (ver docstring del módulo): la clase base no existe hasta que
    alguien instancia el clasificador.
    """
    from videoarm.api.client import call_openai_model_with_tools, handle_json_parsing_error
    from videoarm.core.agent import VideoARMAgent

    class _Qwen35CompatAgent(VideoARMAgent):
        def _call_controller(self, messages, params, api_key, base_url, tool_choice):
            """Fuerza qué tool se llama en cada iteración -- no deja que lo
            decida ni la gramática de upstream ni el propio agente.

            Iteración 1: ``tool_choice="required"`` (cualquier tool; en la
            práctica siempre sale ``scene_snapper``, la vista general).
            Upstream solo restringe la gramática AQUÍ (``tool_choice =
            "required" if iteration == 1 else "auto"``, ver
            ``_reasoning_loop``); de la 2 en adelante, sin restricción,
            Qwen3.5 escribe mal sus propias llamadas a tool
            (``<=clip_analyzer`` en vez de ``<function=clip_analyzer>``) y
            VideoARM se traga ese texto como respuesta final.

            Iteración 2+: se fuerza la función CONCRETA ``clip_analyzer``
            (``tool_choice={"type": "function", "function": {"name": ...}}``,
            soportado por vLLM -- probado a mano contra el servidor). No
            basta con "required" genérico: medido sobre 61 clips, con solo
            "required" el agente repetía ``scene_snapper`` el 54% de las
            veces en vez de pasar a ``clip_analyzer`` -- la tool que analiza
            a MÁS resolución (512px vs 256px) y un rango de frames más
            estrecho, la única con alguna chance de ver si el balón entra o
            no a la red. Forzar la función deja la elección de QUÉ rango
            mirar al agente (su punto fuerte real, ver README) pero
            garantiza que se ejecute -- ya no es una posibilidad, es cada vez.
            """
            forced = "required" if self.analysis_iterations_used <= 1 else {
                "type": "function", "function": {"name": "clip_analyzer"}
            }
            return super()._call_controller(messages, params, api_key, base_url, forced)

        def _execute_tool(self, video_path, tool_call):
            """Sustituye la pregunta de ``clip_analyzer`` por
            ``_CLIP_ANALYZER_QUESTION`` -- fija, no la que improvisa el
            agente.

            El agente sigue eligiendo QUÉ rango de frames analizar (eso no se
            toca); lo que se sustituye es SOLO la pregunta, porque medido, la
            que el agente propone por su cuenta suele ser genérica ("Describe
            the event from frames X to Y") en vez de apuntar específicamente
            a la disyuntiva que de verdad hace falta resolver -- shot vs.
            goal vs. ningún intento. El ``frame_range``/``reason`` del agente
            se dejan intactos.
            """
            if tool_call["function"]["name"] == "clip_analyzer":
                args = handle_json_parsing_error(
                    tool_call["function"].get("arguments", ""), "tool arguments"
                ) or {}
                args["question"] = _CLIP_ANALYZER_QUESTION
                tool_call = {
                    **tool_call,
                    "function": {**tool_call["function"], "arguments": json.dumps(args)},
                }
            return super()._execute_tool(video_path, tool_call)

        def _force_final_answer(self, messages, params, api_key, base_url):
            try:
                response = call_openai_model_with_tools(
                    messages=messages
                    + [
                        {
                            "role": "user",   # upstream usa "system" -> 400 en Qwen3.5
                            "content": (
                                "You have reached the step budget.  Based on all gathered "
                                "evidence, provide your best final answer.  Do not call any tools."
                            ),
                        }
                    ],
                    model_name=self.model_name,
                    tools=None,
                    tool_choice="none",
                    api_key=api_key,
                    endpoints=base_url,
                    **(params or {}),
                )
            except Exception:
                logger.exception("Falló el forzado de respuesta final de VideoARM")
                return None

            if response and response.get("content"):
                return response["content"]
            return None

    return _Qwen35CompatAgent


# --------------------------------------------------------------------------- #
#  Clasificador directo (sin VideoARM) -- experimental, ver README
# --------------------------------------------------------------------------- #

DEFAULT_DIRECT_NUM_FRAMES: int = 12


def _frame_to_data_url(frame) -> str:
    """``np.ndarray`` RGB (formato de ``core.events.sample_frames``) -> data
    URL JPEG en memoria. Sin fichero temporal intermedio -- a diferencia de
    ``videoarm.api.client.local_image_to_data_url``, que lee de disco (para
    lo que primero habría que volcar cada frame con ``cv2.imwrite``)."""
    import base64
    import cv2

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("No se pudo codificar el frame a JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


class DirectVLMEventClassifier:
    """Mismo modelo (Qwen3.5-9B/vLLM), sin VideoARM: un único mensaje con N
    frames + ``_QUESTION``, sin tool-calling, sin HM3, sin bucle de
    iteraciones.

    Existe para responder una pregunta concreta y medible: la capa de
    orquestación de VideoARM (navegación coarse-to-fine, ``scene_snapper``/
    ``clip_analyzer`` como tools, memoria HM3 reconstruida cada turno) está
    pensada para vídeo LARGO, donde decidir qué parte mirar es la mitad del
    problema. Nuestros clips ya vienen recortados a 4-6s -- ¿aporta algo esa
    orquestación aquí, o es puro overhead (latencia + tres incompatibilidades
    parcheadas a mano, ver ``_build_agent_class``) sobre lo que el VLM ya
    hace solo si se le dan los frames directamente? Comparación con números
    en README § "¿Aporta valor VideoARM?".

    Mismo contrato público que ``VideoARMEventClassifier``/``EventClassifier``
    (``predict``/``predict_window`` -> ``EventPrediction``).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model_id: str = DEFAULT_MODEL_ID,
        api_key: str = DEFAULT_API_KEY,
        num_frames: int = DEFAULT_DIRECT_NUM_FRAMES,
    ) -> None:
        from openai import OpenAI  # import perezoso, igual que el resto de core/ con deps externas

        self.model_id = model_id
        self.num_frames = num_frames
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def predict(self, video_path: Path | str) -> EventPrediction:
        from core.events import sample_frames

        video_path = Path(video_path)
        frames = sample_frames(video_path, self.num_frames)
        return self._ask(frames, str(video_path))

    def predict_window(
        self, video_path: Path | str, start_s: float, end_s: float
    ) -> EventPrediction:
        from core.events import sample_frames_window

        video_path = Path(video_path)
        frames = sample_frames_window(video_path, start_s, end_s, self.num_frames)
        return self._ask(frames, str(video_path))

    def predict_many(self, video_paths: Sequence[Path | str]) -> Iterator[EventPrediction]:
        for video_path in video_paths:
            yield self.predict(video_path)

    def _ask(self, frames, video_path_label: str) -> EventPrediction:
        content = [{"type": "text", "text": _QUESTION}]
        for frame in frames:
            content.append({"type": "image_url", "image_url": {"url": _frame_to_data_url(frame)}})

        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=512,
        )
        elapsed = time.perf_counter() - started
        raw = response.choices[0].message.content or ""
        logger.info("VLM directo respondió en %.1fs para %s", elapsed, Path(video_path_label).name)
        return _parse_response(raw, video_path_label)


# --------------------------------------------------------------------------- #
#  Clasificador (VideoARM)
# --------------------------------------------------------------------------- #

class VideoARMEventClassifier:
    """Clasifica clips de fútbol en eventos vía VideoARM + Qwen3.5-9B/vLLM.

    Mismo contrato público que ``core.events.EventClassifier``: ``predict``/
    ``predict_window``/``predict_many`` -> ``EventPrediction``. A diferencia
    de aquel, no hay estado de GPU que cargar aquí — el constructor solo fija
    la configuración de VideoARM (env vars) e instancia su agente, que a su
    vez habla HTTP con el servidor vLLM.

    Args:
        base_url: endpoint OpenAI-compatible del servidor vLLM
            (``vllm serve Qwen/Qwen3.5-9B ...``, ver README § vLLM).
        model_id: debe coincidir con el nombre servido por ese vLLM.
        api_key: no se valida contra vLLM; el SDK ``openai`` exige un string
            no vacío.
        max_iterations / total_frames_limit: overrides de
            ``videoarm.config.model_config.PIPELINE_CONFIG`` — VideoARM está
            pensado para vídeo largo con localización iterativa; nuestros
            clips ya vienen acotados (recortados por evento, o una ventana de
            pocos segundos vía ``predict_window``), así que no deberían
            necesitar muchas iteraciones. Empezar bajo y subir solo si las
            respuestas salen incompletas/truncadas.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model_id: str = DEFAULT_MODEL_ID,
        api_key: str = DEFAULT_API_KEY,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        total_frames_limit: int = DEFAULT_TOTAL_FRAMES_LIMIT,
        frame_analysis_max_frames: int = DEFAULT_FRAME_ANALYSIS_MAX_FRAMES,
    ) -> None:
        self.model_id = model_id
        self._configure_environment(
            base_url, model_id, api_key, max_iterations, total_frames_limit, frame_analysis_max_frames
        )

        # Import perezoso (dentro de _build_agent_class): solo si se instancia
        # esta clase. Ver el docstring de _build_agent_class para el parche.
        agent_cls = _build_agent_class()

        logger.info("Conectando a VideoARM (controller=%s, base_url=%s) ...", model_id, base_url)
        self._agent = agent_cls(model_name=model_id)

    @staticmethod
    def _configure_environment(
        base_url: str, model_id: str, api_key: str, max_iterations: int, total_frames_limit: int,
        frame_analysis_max_frames: int = DEFAULT_FRAME_ANALYSIS_MAX_FRAMES,
    ) -> None:
        """Fija las env vars que lee ``videoarm.config.model_config.ModelConfig``
        antes de instanciar el agente (que es quien llama a ``get_config()``).

        Ojo, ``get_config()`` es un singleton por proceso: **la configuración
        de la PRIMERA instancia gana**. Si se construyen dos
        ``VideoARMEventClassifier`` con distinto ``base_url``/``api_key`` en el
        mismo proceso, el segundo hablará con el servidor del primero. Peor
        aún con ``max_iterations``/``total_frames_limit``: viven en
        ``ModelConfig.PIPELINE_CONFIG``, que es un atributo de CLASE, así que
        el segundo sí los pisa — y también para el primero. No es un problema
        hoy (pipeline y CLI construyen uno solo), pero conviene saberlo antes
        de, por ejemplo, comparar dos configuraciones en un mismo script.

        Los tres componentes con impacto visual/de razonamiento (controller,
        clip_analyzer, scene_snapper) apuntan todos al mismo Qwen3.5-9B: a
        diferencia del uso de referencia de VideoARM (o3 + gpt-4.1 separados),
        aquí un único VLM local hace de LLM+VLM. ``audio_transcriber`` se deja
        sin configurar a propósito: no apunta a ningún sitio y no hace falta
        que lo haga, porque el agente detecta por ``ffprobe`` que el clip no
        tiene audio antes de llamar a la API (ver la nota sobre audio en el
        docstring del módulo).
        """
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_KEY"] = api_key
        for component in ("CONTROLLER", "CLIP_ANALYZER", "SCENE_SNAPPER"):
            os.environ[f"VIDEOARM_MODEL_{component}"] = model_id
        os.environ["VIDEOARM_MAX_ITERATIONS"] = str(max_iterations)
        os.environ["VIDEOARM_TOTAL_FRAMES_LIMIT"] = str(total_frames_limit)
        os.environ["VIDEOARM_FRAME_ANALYSIS_MAX_FRAMES"] = str(frame_analysis_max_frames)

    # ------------------------------------------------------------------ #

    def predict(self, video_path: Path | str) -> EventPrediction:
        """Clasifica un único clip .mp4 entero (ya recortado en torno a un evento)."""
        video_path = Path(video_path)
        return self._ask(video_path, str(video_path))

    def predict_window(
        self, video_path: Path | str, start_s: float, end_s: float
    ) -> EventPrediction:
        """Clasifica solo la ventana ``[start_s, end_s]`` de ``video_path``
        (partido completo, ventana deslizante — ver ``core/windowing.py``).

        ``EventPrediction.video_path`` sigue siendo la ruta del vídeo fuente
        (igual que ``core.events.EventClassifier.predict_window``); la
        ventana concreta la lleva el ``Window`` del llamador."""
        video_path = Path(video_path)
        with _extract_window_clip(video_path, start_s, end_s) as clip_path:
            return self._ask(clip_path, str(video_path))

    def predict_many(
        self, video_paths: Sequence[Path | str]
    ) -> Iterator[EventPrediction]:
        """Clasifica varios clips en secuencia (generador, apto para pipelines)."""
        for video_path in video_paths:
            yield self.predict(video_path)

    # ------------------------------------------------------------------ #

    def _ask(self, clip_path: Path, video_path_label: str) -> EventPrediction:
        started = time.perf_counter()
        raw = self._agent.ask(
            str(clip_path), _QUESTION, is_multiple_choice=False, save_result=False
        )
        elapsed = time.perf_counter() - started
        logger.info("VideoARM respondió en %.1fs para %s", elapsed, clip_path.name)

        # VideoARM devuelve None, no lanza, cuando agota sus reintentos contra
        # la API (``videoarm.api.client.retry_with_exponential_backoff``
        # captura la excepción y retorna None). Sin esto un único hipo del
        # servidor reventaría una pasada de partido completo, que son cientos
        # de ventanas: se degrada a "background" con confianza 0.0, que es lo
        # que ``analytics.engine`` ya sabe descartar.
        if not raw:
            logger.warning(
                "VideoARM no devolvió respuesta para %s (servidor caído o "
                "reintentos agotados), degradando a background", clip_path.name
            )
            return EventPrediction(
                video_path=video_path_label,
                label=BACKGROUND_LABEL,
                confidence=0.0,
                probabilities={l: 0.0 for l in ALL_LABELS},
                is_background=True,
            )

        return _parse_response(raw, video_path_label)


# --------------------------------------------------------------------------- #
#  Híbrido (directo + escalada a VideoARM) -- el mejor resultado medido
# --------------------------------------------------------------------------- #

class HybridEventClassifier:
    """``DirectVLMEventClassifier`` primero; si predice ``"goal"`` se confía
    en él, si no se re-consulta con ``VideoARMEventClassifier``.

    Nace de comparar los tres backends sobre el pool completo de
    ``clips_analisis/{goal,shots_on_target,shots_off_target}`` (61 clips, ver
    README § "¿Aporta valor VideoARM?"):

    ==============  =====  ======  ======
    backend         total  goal    shot
    ==============  =====  ======  ======
    SVC             44%    48%     39%
    VideoARM        49%    58%     39%
    Directo         51%    82%     14%
    Híbrido (v1)    66%    91%     36%
    **Híbrido (v2)** **69%** 85%   **50%**
    ==============  =====  ======  ======

    v1 -> v2: se forzó ``clip_analyzer`` (antes el agente lo elegía como
    segunda tool solo el 46% de las veces) y se le fijó una pregunta
    dirigida (``_CLIP_ANALYZER_QUESTION``) en vez de la genérica que
    improvisaba -- además de un bug real que dejaba esa tool fallando en
    silencio (ver README). Eso solo, sin tocar el modelo, subió ``shot`` de
    36% a 50% dentro del híbrido. Detalle contraintuitivo importante en el
    comentario de ``_CLIP_ANALYZER_QUESTION``: la pregunta que mejor
    funciona AQUÍ es la que peor deja a VideoARM en ``goal`` si se mide
    aislado -- no importa, porque en el híbrido casi nunca le llega un gol
    claro (el directo se lo queda antes).

    Ninguno de los backends individuales gana en las dos clases a la vez: el
    directo clava ``goal`` (evento visualmente inequívoco) pero falla
    ``shot`` sistemáticamente hacia ``background`` (68% de los tiros
    reales); VideoARM es el único que compite en ``shot`` (su
    ``clip_analyzer``, bien forzado, es lo que aporta ahí). Enrutar por la
    propia predicción del directo bastó para quedarse con lo mejor de los
    dos, sin ninguna llamada adicional al modelo respecto de tener los dos
    backends corriendo por separado.

    La precisión del directo prediciendo ``"goal"`` es 27/32 = 84% en ese
    mismo eval -- suficiente para fiarse como filtro de enrutado. Se
    comprobó (y descartó) añadir un umbral de confianza al enrutado: la
    confianza que reporta el propio modelo NO discrimina goles reales de
    falsos positivos (0.90-0.98 en ambos casos, sin separación) — está mal
    calibrada para esto, así que un umbral no ayudaría.

    Coste: los clips que NO enrutan a "goal" (29/61 = 48% en el eval) pagan
    la latencia de los dos backends en serie (~4s + ~9s), no solo uno --
    intermedio entre los ~4s del directo solo y los ~11s de VideoARM solo.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model_id: str = DEFAULT_MODEL_ID,
        api_key: str = DEFAULT_API_KEY,
        direct_num_frames: int = DEFAULT_DIRECT_NUM_FRAMES,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        total_frames_limit: int = DEFAULT_TOTAL_FRAMES_LIMIT,
    ) -> None:
        self._direct = DirectVLMEventClassifier(
            base_url=base_url, model_id=model_id, api_key=api_key, num_frames=direct_num_frames
        )
        self._videoarm = VideoARMEventClassifier(
            base_url=base_url, model_id=model_id, api_key=api_key,
            max_iterations=max_iterations, total_frames_limit=total_frames_limit,
        )

    def predict(self, video_path: Path | str) -> EventPrediction:
        first = self._direct.predict(video_path)
        if first.label == "goal":
            return first
        return self._videoarm.predict(video_path)

    def predict_window(
        self, video_path: Path | str, start_s: float, end_s: float
    ) -> EventPrediction:
        first = self._direct.predict_window(video_path, start_s, end_s)
        if first.label == "goal":
            return first
        return self._videoarm.predict_window(video_path, start_s, end_s)

    def predict_many(self, video_paths: Sequence[Path | str]) -> Iterator[EventPrediction]:
        for video_path in video_paths:
            yield self.predict(video_path)


# --------------------------------------------------------------------------- #
#  Verificador de tiros (cascada de partido completo)
# --------------------------------------------------------------------------- #

# Pregunta 1: binaria y corta. Las definiciones largas por etiqueta ya
# empeoraron dos veces al 9B (ver _QUESTION y _CLIP_ANALYZER_QUESTION); aquí
# solo se nombran los falsos positivos MEDIDOS de V-JEPA2 en partido
# completo (Hull–Sheffield W.: centros, balones fuera, bloqueos, cabezazos
# que no van a puerta), que son justo los candidatos que este verificador
# tiene que tumbar.
_SHOT_QUESTION = (
    "These frames are from a football (soccer) broadcast, in chronological "
    "order. Does a player take a shot at goal in them (striking or heading "
    "the ball towards the goal, trying to score)? Crosses, passes into the "
    "box, clearances and balls going out of play are not shots. "
    "Answer with one word: yes or no."
)
_SHOT_ANSWERS: tuple[str, ...] = ("yes", "no")

# Pregunta 2 (solo si hay tiro): el desenlace. "goal" emite también el tiro
# (convención gol ⊂ tiro de SoccerNet, ver core/spotting.py).
_OUTCOME_QUESTION = (
    "These frames are from a football (soccer) broadcast, in chronological "
    "order, and a player takes a shot at goal in them. What happens to the "
    "shot? goal: the ball ends up in the net. saved: the goalkeeper stops "
    "it. blocked: a defender blocks it. missed: it goes wide or over. "
    "Answer with one word: goal, saved, blocked or missed."
)
SHOT_OUTCOMES: tuple[str, ...] = ("goal", "saved", "blocked", "missed")

# Entra en el nombre de la caché de veredictos: cambiar cualquiera de las dos
# preguntas o el muestreo de frames exige subirla, o se mezclarían respuestas
# de prompts distintos.
SHOT_VERIFIER_PROMPT_VERSION: str = "v1"

# Por debajo de esta masa de probabilidad en respuestas válidas, el modelo no
# está siguiendo el formato "una palabra" y el P(sí) normalizado no es fiable.
_MIN_ANSWER_MASS: float = 0.5


@dataclass
class ShotVerdict:
    """Respuesta del verificador para un instante candidato."""

    t_s: float
    p_shot: float                    # P("yes") normalizada sobre yes/no
    outcome: Optional[str]           # None si p_shot < outcome_min_p_shot
    outcome_probs: dict[str, float]  # vacío si no se preguntó el desenlace
    answer_mass: float               # prob. total en respuestas válidas (1er token)

    def to_telemetry(self) -> dict:
        return asdict(self)


def _choice_probs(top_logprobs, choices: Sequence[str]) -> tuple[dict[str, float], float]:
    """Probabilidad de cada opción a partir del PRIMER token generado.

    Suma la probabilidad de todas las variantes de token que abren una sola
    opción ("Yes", " yes", "yes."...) y normaliza entre opciones. Devuelve
    también la masa total cubierta, para detectar cuándo el modelo no está
    respondiendo con una de las palabras pedidas.

    Se hace así, y no con salidas estructuradas de vLLM, porque el nombre de
    ese parámetro ha cambiado entre versiones (``guided_choice`` ->
    ``structured_outputs``); ``logprobs``/``top_logprobs`` son API OpenAI
    estándar.
    """
    mass = {c: 0.0 for c in choices}
    for entry in top_logprobs:
        tok = entry.token.strip().lower()
        if not tok:
            continue
        owners = [c for c in choices if c.startswith(tok) or tok.startswith(c)]
        if len(owners) == 1:
            mass[owners[0]] += math.exp(entry.logprob)
    total = sum(mass.values())
    if total == 0:
        return {c: 1 / len(choices) for c in choices}, 0.0
    return {c: m / total for c, m in mass.items()}, total


class ShotVerifier:
    """Segunda etapa de la cascada de partido completo: V-JEPA2 propone
    instantes candidatos a tiro (picos de ``core/spotting.py`` con umbral
    bajo) y este verificador los confirma o los tumba mirando el clip
    alrededor de cada uno.

    Mismo Qwen3.5-9B/vLLM que ``DirectVLMEventClassifier``, sin VideoARM, con
    tres diferencias pensadas para lo medido:

    * Pregunta binaria ("¿hay un tiro?") y, solo si la respuesta es sí, una
      segunda pregunta por el desenlace -- en vez de una clasificación entre
      10 etiquetas donde el fallo dominante era confundir desenlaces
      (shot_off_target -> goal).
    * Una sola palabra de respuesta (``max_tokens=1``) y su probabilidad del
      primer token como score continuo, en vez de la confianza que el modelo
      escribe él mismo (0,90-0,98 tanto en aciertos como en errores, ver
      ``HybridEventClassifier``). Con un score así se puede ajustar un
      umbral. Llamada directa a temperatura 0: sin agente que elija qué
      frames mirar, que es de donde venía la varianza entre pasadas.
    * Clip asimétrico ``[t - before_s, t + after_s]`` para que se vea el
      desenlace (parada, balón en la red, reacción), con la mitad de los
      frames concentrados en ``t ± dense_half_s`` (el golpeo) y la otra
      mitad repartida por todo el clip. 8 + 8 = 16, el límite de imágenes
      por prompt del servidor (``--limit-mm-per-prompt``).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model_id: str = DEFAULT_MODEL_ID,
        api_key: str = DEFAULT_API_KEY,
        before_s: float = 3.0,
        after_s: float = 5.0,
        dense_half_s: float = 1.0,
        n_dense: int = 8,
        n_context: int = 8,
        outcome_min_p_shot: float = 0.5,
    ) -> None:
        from openai import OpenAI

        self.model_id = model_id
        self.before_s = before_s
        self.after_s = after_s
        self.dense_half_s = dense_half_s
        self.n_dense = n_dense
        self.n_context = n_context
        self.outcome_min_p_shot = outcome_min_p_shot
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def verify(self, video_path: Path | str, t_s: float) -> ShotVerdict:
        """¿Hay un tiro en torno a ``t_s``? Y si lo hay, ¿cómo acaba?"""
        frames = self._frames_around(Path(video_path), t_s)
        probs, mass = self._ask_choice(frames, _SHOT_QUESTION, _SHOT_ANSWERS)
        p_shot = probs["yes"]

        outcome: Optional[str] = None
        outcome_probs: dict[str, float] = {}
        if p_shot >= self.outcome_min_p_shot:
            outcome_probs, _ = self._ask_choice(frames, _OUTCOME_QUESTION, SHOT_OUTCOMES)
            outcome = max(outcome_probs, key=outcome_probs.get)

        return ShotVerdict(
            t_s=t_s, p_shot=p_shot, outcome=outcome, outcome_probs=outcome_probs, answer_mass=mass
        )

    def p_shot_window(self, video_path: Path | str, start_s: float, end_s: float) -> float:
        """Solo la pregunta binaria sobre una ventana con frames uniformes.
        Para usar el VLM como ESCÁNER (``VLMShotWindowClassifier``) y medir
        qué tiros ve que V-JEPA2 no propone."""
        from core.events import sample_frames_window

        frames = sample_frames_window(
            Path(video_path), start_s, end_s, self.n_dense + self.n_context
        )
        probs, _ = self._ask_choice(frames, _SHOT_QUESTION, _SHOT_ANSWERS)
        return probs["yes"]

    def frame_times(self, t_s: float) -> list[float]:
        """Instantes muestreados alrededor de ``t_s``, en orden: ``n_dense``
        en ``t ± dense_half_s`` y ``n_context`` FUERA de ese tramo, repartidos
        antes/después en proporción a la duración de cada lado (sin
        solaparse con los densos: si no, algunos saldrían duplicados)."""
        start = max(0.0, t_s - self.before_s)
        end = t_s + self.after_s
        dense_start = max(start, t_s - self.dense_half_s)
        dense_end = min(end, t_s + self.dense_half_s)
        before_len, after_len = dense_start - start, end - dense_end
        n_before = round(self.n_context * before_len / (before_len + after_len)) if before_len + after_len else 0
        n_after = self.n_context - n_before
        before = np.linspace(start, dense_start, n_before + 1)[:-1] if n_before else []
        after = np.linspace(dense_end, end, n_after + 1)[1:] if n_after else []
        dense = np.linspace(dense_start, dense_end, self.n_dense)
        return [float(t) for t in (*before, *dense, *after)]

    def _frames_around(self, video_path: Path, t_s: float) -> list:
        from core.events import sample_frames_at

        return sample_frames_at(video_path, self.frame_times(t_s))

    def _ask_choice(self, frames, question: str, choices: Sequence[str]) -> tuple[dict[str, float], float]:
        # Imágenes primero y la pregunta al final: la instrucción "una palabra"
        # es lo último que ve el modelo antes de generar ese único token.
        content = [
            {"type": "image_url", "image_url": {"url": _frame_to_data_url(f)}} for f in frames
        ]
        content.append({"type": "text", "text": question})

        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )
        first = response.choices[0].logprobs.content[0]
        probs, mass = _choice_probs(first.top_logprobs, choices)
        if mass < _MIN_ANSWER_MASS:
            logger.warning(
                "Solo %.0f%% de la probabilidad en respuestas válidas %s (generó %r)",
                100 * mass, list(choices), first.token,
            )
        return probs, mass


class VLMShotWindowClassifier:
    """Adaptador de ``ShotVerifier`` al contrato ``predict_window`` ->
    ``EventPrediction``, para escanear un partido entero con el VLM
    (``scripts/scan_event_windows.py --backend vlm-shot``) y comparar su
    recall de tiros con el de V-JEPA2 sobre la misma evaluación. Solo emite
    "shot" y "background"."""

    def __init__(self, verifier: ShotVerifier) -> None:
        self._verifier = verifier

    def predict_window(self, video_path: Path | str, start_s: float, end_s: float) -> EventPrediction:
        p = self._verifier.p_shot_window(video_path, start_s, end_s)
        is_shot = p >= 0.5
        return EventPrediction(
            video_path=str(video_path),
            label="shot" if is_shot else BACKGROUND_LABEL,
            confidence=p if is_shot else 1 - p,
            probabilities={"shot": p, BACKGROUND_LABEL: 1 - p},
            is_background=not is_shot,
        )


# --------------------------------------------------------------------------- #
#  Ejecución standalone
# --------------------------------------------------------------------------- #

def _resolve_sources(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(source.glob("**/*.mp4"))
    return [source]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clasificación de eventos de fútbol (VideoARM + Qwen3.5-9B/vLLM)"
    )
    parser.add_argument("--source", required=True, help="Clip .mp4 o carpeta con .mp4")
    parser.add_argument("--output", default=None, help="JSONL de salida (si no, solo stdout)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Endpoint OpenAI-compatible del servidor vLLM")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--total-frames-limit", type=int, default=DEFAULT_TOTAL_FRAMES_LIMIT)
    parser.add_argument("--backend", choices=["hybrid", "videoarm", "direct"], default="hybrid",
                         help="'hybrid' (por defecto, mejor medido -- ver HybridEventClassifier), "
                              "'videoarm' (orquestador con tool-calling) o "
                              "'direct' (mismo modelo, sin VideoARM)")
    parser.add_argument("--direct-num-frames", type=int, default=DEFAULT_DIRECT_NUM_FRAMES,
                         help="Solo --backend direct: frames muestreados enviados en un único mensaje")
    parser.add_argument("--start-s", type=float, default=None, help="Si se indica junto a --end-s, prueba predict_window en vez de predict")
    parser.add_argument("--end-s", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    source = Path(args.source)
    if not source.exists():
        parser.error(f"No existe: {source}")
    if (args.start_s is None) != (args.end_s is None):
        parser.error("--start-s y --end-s deben indicarse juntos")

    video_paths = _resolve_sources(source)
    if not video_paths:
        parser.error(f"Ningún .mp4 encontrado en: {source}")

    if args.backend == "direct":
        classifier = DirectVLMEventClassifier(
            base_url=args.base_url,
            model_id=args.model_id,
            api_key=args.api_key,
            num_frames=args.direct_num_frames,
        )
    elif args.backend == "videoarm":
        classifier = VideoARMEventClassifier(
            base_url=args.base_url,
            model_id=args.model_id,
            api_key=args.api_key,
            max_iterations=args.max_iterations,
            total_frames_limit=args.total_frames_limit,
        )
    else:
        classifier = HybridEventClassifier(
            base_url=args.base_url,
            model_id=args.model_id,
            api_key=args.api_key,
            direct_num_frames=args.direct_num_frames,
            max_iterations=args.max_iterations,
            total_frames_limit=args.total_frames_limit,
        )

    out_f = open(args.output, "w") if args.output else None
    try:
        for video_path in video_paths:
            try:
                if args.start_s is not None:
                    prediction = classifier.predict_window(video_path, args.start_s, args.end_s)
                else:
                    prediction = classifier.predict(video_path)
            except Exception:
                logger.exception("Error procesando %s", video_path)
                continue

            tag = "·bg·" if prediction.is_background else "EVENT"
            logger.info(
                "[%s] %-40s -> %-30s (%.1f%%)",
                tag, video_path.name, prediction.label, prediction.confidence * 100,
            )
            if out_f:
                out_f.write(json.dumps(prediction.to_telemetry()) + "\n")
    finally:
        if out_f:
            out_f.close()

    if args.output:
        logger.info("Predicciones escritas en %s", args.output)


if __name__ == "__main__":
    main()
