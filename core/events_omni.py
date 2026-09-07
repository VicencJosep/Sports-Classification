"""
core/events_omni.py
====================
Clasificación de eventos de fútbol (gol, tiro, córner, falta...) sobre un
clip de vídeo ya recortado, vía **Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit**
servido localmente con vLLM por un endpoint OpenAI-compatible — backend
alternativo a ``core/events.py`` (V-JEPA2 + SVC).

Diferencia clave frente a ``core/events_vllm.py`` (rama ``videoARM``, no
presente en esta rama): aquel backend usa VideoARM, un orquestador agéntico
con tool-calling (``scene_snapper``/``clip_analyzer``) pensado para vídeo
LARGO donde decidir qué parte mirar es la mitad del problema, sobre un modelo
de solo texto+imágenes (Qwen3.5-9B) al que hay que darle frames sueltos.
Aquí el modelo es **nativamente audiovisual** (Qwen3-Omni acepta vídeo y
audio como modalidades directas, no solo imágenes) y los clips de
``clips_analisis/`` ya vienen recortados a ~5s en torno a un único evento, así
que no hay nada que "navegar": este módulo manda el clip ENTERO (vídeo +
audio, sin trocear en frames) en una única llamada, sin agente ni bucle de
iteraciones — mismo espíritu que ``DirectVLMEventClassifier`` de
``events_vllm.py``, pero con el modelo viendo la ventana temporal continua en
vez de N imágenes muestreadas.

Recorte tomado de ``native_video_understanding`` (``/media/dd1/Projects/Pol/
native_video_understanding/avlib.py``), que ya sirve y consulta este mismo
checkpoint contra este mismo host: el payload (``video_url`` con el .mp4 en
base64 + ``media_io_kwargs``/``mm_processor_kwargs`` con el fps declarado, y
el audio como ``input_audio`` SEPARADO — ``use_audio_in_video`` revienta esta
build de vLLM en ``merge_interleaved_embeddings``) y la imagen Docker
(``vllm-audio:local``) están reutilizados de ahí, no reinventados.

Sirve el modelo aparte (contenedor propio, ``scripts/serve_events_omni.sh``):
igual que ``events_vllm.py``, este módulo solo habla HTTP con ese servidor —
no importa ``torch``/``transformers``, la única dependencia es ``requests``.

Estado: EXPERIMENTAL / sin medir todavía. Los defaults de fps/max_tokens/
sampling están documentados con su porqué pero no validados contra
``clips_analisis/`` (ver docstring de cada constante) — el primer uso real
debería revisar truncamiento (``finish_reason == "length"``) y ajustar antes
de sacar ninguna conclusión de calidad, mismo proceso que documenta el README
para ``events_vllm.py``.

Uso rápido (standalone, desde la raíz del proyecto, con el servidor vLLM ya
arrancado — ver ``scripts/serve_events_omni.sh``):
    python -m core.events_omni --source clip.mp4
    python -m core.events_omni --source clips_folder/ --output events.jsonl
    python -m core.events_omni --source partido.mp4 --start-s 120 --end-s 126
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence

import requests

from core.events import EventPrediction  # reutilizado, no duplicado

logger = logging.getLogger(__name__)

# Mismo host/puerto que native_video_understanding/avlib.py (SERVERS["local"])
# -- 8002, no 8010 (ese lo usa events_vllm.py/Qwen3.5-9B en la rama videoARM)
# ni 8000/8001 (repscan_takedown_agent). Puerto libre verificado 2026-09-07.
DEFAULT_URL: str = "http://localhost:8002/v1/chat/completions"
DEFAULT_MODEL_ID: str = "cyankiwi/Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit"
DEFAULT_API_KEY: str = "EMPTY"  # vLLM no la valida; se manda por si el proxy la exige

# fps con el que el procesador multimodal de vLLM muestrea el vídeo (se manda
# DOS veces, en media_io_kwargs Y mm_processor_kwargs -- ver avlib.py: vLLM
# usa el fps DECLARADO para fechar los frames, no el real del contenedor, así
# que un valor equivocado no rompe la llamada pero desincroniza cualquier
# timestamp que el modelo intente razonar). 4.0 (~20-24 frames para los
# clips de 5-6s de clips_analisis/) es un punto de partida conservador -- sin
# medir todavía cuántos tokens de vídeo cuesta por frame en este checkpoint
# ni si hace falta más resolución temporal para separar shot_on/off_target.
DEFAULT_FPS: float = 4.0

# Alto (px) al que se reescala el vídeo antes de mandarlo -- ver
# ``_extract_video`` para por qué esto NO es opcional (CUDA OOM medido en el
# encoder de visión al mandar clips_analisis/ a su resolución nativa,
# 1280x720). 224 es el valor medido y en uso en
# ``native_video_understanding/avlib.py`` contra este mismo checkpoint.
DEFAULT_HEIGHT: int = 224

# Presupuesto de generación, incluye la traza de razonamiento (el modelo es
# la variante "Thinking": --reasoning-parser qwen3 en el servidor separa esa
# traza a reasoning_content, pero SIGUE consumiendo tokens/tiempo de
# max_tokens). max_model_len del servidor local es 8192 (ver
# scripts/serve_events_omni.sh) -- 3072 deja margen de sobra para un prompt
# de vídeo corto + audio + JSON de respuesta, pero no se ha medido si el
# razonamiento real de este checkpoint sobre un clip de fútbol se queda corto
# o largo frente a esto. Si sale finish_reason == "length", subir esto antes
# que tocar nada más.
DEFAULT_MAX_TOKENS: int = 3072

# Qwen recomienda para el modo "thinking" de la familia Qwen3 (de la que
# Qwen3-Omni-Thinking hereda el comportamiento): temperature=0.6, top_p=0.95,
# top_k=20, y explícitamente NO greedy decoding (temperature=0 puede producir
# bucles de repetición en la traza de razonamiento). A diferencia de
# DirectVLMEventClassifier (events_vllm.py), que usa temperature=0.0 sobre un
# modelo SIN razonamiento forzado -- aquí sí importa seguir la recomendación
# del propio modelo.
DEFAULT_TEMPERATURE: float = 0.6
DEFAULT_TOP_P: float = 0.95
DEFAULT_TOP_K: int = 20

DEFAULT_ENABLE_THINKING: bool = True  # es el checkpoint "-Thinking-"; no tiene
                                       # sentido apagarlo por defecto (al
                                       # contrario que Qwen3.5-9B en
                                       # events_vllm.py, que es un modelo base
                                       # con thinking activable, no un
                                       # checkpoint entrenado para pensar)

DEFAULT_INCLUDE_AUDIO: bool = True  # los clips de clips_analisis/ llevan
                                     # pista de audio (AAC) -- silbato/rugido
                                     # de grada son señal real para
                                     # goal/foul/corner que un backend
                                     # solo-vídeo (events.py, events_vllm.py)
                                     # no puede usar. Se degrada solo (log +
                                     # sigue sin audio) si el clip no tiene
                                     # pista o ffmpeg falla.

DEFAULT_TIMEOUT_S: float = 300.0
DEFAULT_RETRIES: int = 1
DEFAULT_RETRY_WAIT_S: float = 5.0

# --------------------------------------------------------------------------- #
#  Taxonomía
# --------------------------------------------------------------------------- #

# Superset de las 8 carpetas de datasets/clips_analisis/ (corner, foul, goal,
# kick-off, shots_off_target, shots_on_target, substitution, yellow_card):
# se añaden penalty/red_card/throw-in porque un partido real los produce
# aunque clips_analisis/ no tenga clips dedicados para evaluarlos todavía (se
# probará igualmente contra los que sí hay). shot_on/off_target se dejan
# SEPARADOS -- a diferencia de la SVC de core/events.py, que los fusiona
# porque la frontera le resultaba difusa (ver su docstring) -- para no
# renunciar de entrada a esa distinción con un modelo mucho más capaz;
# fusionarlos es trivial más tarde si se mide que tampoco aquí se sostiene.
EVENT_LABELS: tuple[str, ...] = (
    "corner", "foul", "goal", "kick-off", "penalty", "red_card",
    "yellow_card", "shot_on_target", "shot_off_target", "substitution",
    "throw-in",
)
BACKGROUND_LABEL: str = "background"
ALL_LABELS: tuple[str, ...] = EVENT_LABELS + (BACKGROUND_LABEL,)

_LABELS_JOINED = ", ".join(f'"{label}"' for label in ALL_LABELS)
_PROBABILITIES_TEMPLATE = ", ".join(f'"{label}": 0.0' for label in ALL_LABELS)

_QUESTION = (
    "You are a football (soccer) match event classifier. You are given a "
    "short clip from a live TV broadcast, with its original video AND "
    "audio. Watch and listen to the whole clip and decide which single "
    "event, if any, is its main action.\n"
    f"Valid labels: {_LABELS_JOINED}.\n"
    '- "shot_on_target": a shot that stays on target -- it scores, or is '
    "saved/stopped by the goalkeeper or on the line. Does not include a "
    "shot that clearly misses the goal.\n"
    '- "shot_off_target": a shot that misses the goal (wide, over the bar), '
    "is blocked before reaching the target, or hits the post/bar and "
    "bounces away without scoring.\n"
    '- "goal": the ball fully crosses the goal line into the net and the '
    "goal is given (celebration, restart at the center circle...).\n"
    '- "background": none of the above -- crowd shots, bench, the main '
    "broadcast camera between plays, a replay with no new action, etc.\n"
    "Use everything you can see AND hear (crowd reaction, whistle, "
    "commentary if present) as evidence.\n"
    "Respond with ONLY a single-line JSON object, with your estimated "
    "probability for EVERY label listed above (they should sum to "
    "approximately 1.0), and nothing else:\n"
    f'{{"probabilities": {{{_PROBABILITIES_TEMPLATE}}}}}'
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


# --------------------------------------------------------------------------- #
#  Recorte de ventana temporal (predict_window) y extracción de audio
# --------------------------------------------------------------------------- #

@contextmanager
def _extract_video(
    video_path: Path,
    height: int,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
) -> Iterator[Path]:
    """``video_path`` (opcionalmente recortado a ``[start_s, end_s]``) ->
    .mp4 temporal reescalado a ``height`` px de alto, SIN audio (``-an`` -- el
    audio se extrae aparte, ver ``_extract_audio_wav``, porque el servidor no
    soporta pasar vídeo con audio muxed, ver docstring del módulo).

    El reescalado (``-vf scale=-2:{height}``) es OBLIGATORIO, no cosmético:
    mandar el vídeo a su resolución nativa (1280x720 en clips_analisis/)
    revienta el encoder de visión con CUDA OOM en la primera petición real
    (medido 2026-09-07 -- el profiling de arranque de vLLM no estima bien el
    pico de activación de un vídeo de resolución completa, y el engine muere
    para SIEMPRE, no solo esa petición: hay que reiniciar el contenedor).
    224p es el valor que ``native_video_understanding/avlib.py`` ya tiene
    medido y en producción contra este mismo checkpoint/servidor (~72 tokens
    de vídeo por frame ahí, ver su ``TOK_PER_FRAME_224P``) -- reutilizado, no
    inventado.

    Reencodea siempre (``-c:v libx264``, NO ``-c copy``): además de que el
    recorte por tiempo necesita caer exacto en ``start_s``/``end_s`` (mismo
    motivo que ``core/events_vllm.py::_extract_window_clip``), el propio
    escalado ya obliga a reencodear.
    """
    with tempfile.TemporaryDirectory(prefix="events_omni_") as tmp_dir:
        clip_path = Path(tmp_dir) / "clip.mp4"
        cmd = ["ffmpeg", "-y"]
        if start_s is not None:
            cmd += ["-ss", f"{start_s:.3f}"]
        cmd += ["-i", str(video_path)]
        if start_s is not None and end_s is not None:
            cmd += ["-t", f"{max(end_s - start_s, 0.04):.3f}"]
        cmd += [
            "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-preset", "ultrafast",
            "-an", "-loglevel", "error", str(clip_path),
        ]
        subprocess.run(cmd, check=True, timeout=30)
        yield clip_path


def _extract_audio_wav(
    video_path: Path, start_s: Optional[float] = None, end_s: Optional[float] = None
) -> Optional[bytes]:
    """Pista de audio de ``video_path`` (o de ``[start_s, end_s]`` si se dan)
    como WAV mono 16kHz en memoria, vía ``ffmpeg ... pipe:1`` -- sin fichero
    temporal.

    Devuelve ``None`` (no lanza) si el clip no tiene pista de audio o ffmpeg
    falla: el audio es evidencia complementaria, no crítica, y un clip mudo
    no debería tirar la clasificación entera (ver ``DEFAULT_INCLUDE_AUDIO``).
    """
    cmd = ["ffmpeg", "-y"]
    if start_s is not None:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(video_path)]
    if start_s is not None and end_s is not None:
        cmd += ["-t", f"{max(end_s - start_s, 0.04):.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-f", "wav", "-loglevel", "error", "pipe:1"]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("No se pudo extraer audio de %s: %s", video_path.name, exc)
        return None
    if not proc.stdout:
        logger.info("%s no tiene pista de audio, clasificando solo con vídeo", video_path.name)
        return None
    return proc.stdout


# --------------------------------------------------------------------------- #
#  Parseo de la respuesta (mismo esquema de 3 niveles que events_vllm.py)
# --------------------------------------------------------------------------- #

def _extract_json_object(text: str) -> Optional[dict]:
    """Primer objeto JSON en ``text``, tolerante a fences ```json``` o prosa
    alrededor (p.ej. restos de razonamiento que se cuelen en ``content``) y a
    dict de Python con comillas simples."""
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
    confianza (idéntico esquema a ``core/events_vllm.py::_parse_response``,
    ver ese docstring para el razonamiento completo):
    1. JSON con ``probabilities`` completo (las ``len(ALL_LABELS)`` etiquetas).
    2. JSON con solo ``{"label": ..., "confidence": ...}``.
    3. Heurística de buscar el nombre de una etiqueta en el texto crudo.
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
        logger.warning("Respuesta no parseable como JSON, usando heurística: %r", raw[:300])
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
#  Clasificador
# --------------------------------------------------------------------------- #

class OmniEventClassifier:
    """Clasifica clips de fútbol en eventos vía Qwen3-Omni-30B-A3B-Thinking-
    AWQ-4bit servido con vLLM, mandando el clip ENTERO (vídeo + audio) en una
    única llamada nativa -- sin muestreo de frames a imágenes sueltas
    (a diferencia de ``core/events.py``) ni orquestación agéntica (a
    diferencia de ``core/events_vllm.py`` en la rama ``videoARM``).

    Mismo contrato público que ``core.events.EventClassifier``: ``predict``/
    ``predict_window``/``predict_many`` -> ``EventPrediction``. No hay estado
    de GPU aquí -- el constructor solo guarda la config de la petición HTTP;
    habla con el servidor vLLM (proceso aparte, ver
    ``scripts/serve_events_omni.sh``).

    Args:
        url: endpoint de chat completions OpenAI-compatible del servidor
            vLLM (NO solo el ``base_url`` -- incluye ``/v1/chat/completions``,
            igual que ``native_video_understanding/avlib.py``).
        fps: fps declarado al procesador multimodal para muestrear el vídeo
            (ver ``DEFAULT_FPS`` para el porqué de no ser más alto todavía).
        include_audio: si se manda también la pista de audio como
            ``input_audio`` (se degrada solo a solo-vídeo si el clip es mudo).
        enable_thinking: si la plantilla de chat abre razonamiento
            (``<think>...</think>``, separado por el servidor con
            ``--reasoning-parser qwen3`` a ``reasoning_content`` -- ``content``
            debería llegar ya limpio).
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        model_id: str = DEFAULT_MODEL_ID,
        api_key: str = DEFAULT_API_KEY,
        fps: float = DEFAULT_FPS,
        height: int = DEFAULT_HEIGHT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        enable_thinking: bool = DEFAULT_ENABLE_THINKING,
        include_audio: bool = DEFAULT_INCLUDE_AUDIO,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        retries: int = DEFAULT_RETRIES,
        retry_wait_s: float = DEFAULT_RETRY_WAIT_S,
    ) -> None:
        self.url = url
        self.model_id = model_id
        self.api_key = api_key
        self.fps = fps
        self.height = height
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.enable_thinking = enable_thinking
        self.include_audio = include_audio
        self.timeout_s = timeout_s
        self.retries = retries
        self.retry_wait_s = retry_wait_s

    # ------------------------------------------------------------------ #

    def predict(self, video_path: Path | str) -> EventPrediction:
        """Clasifica un único clip .mp4 entero (ya recortado en torno a un evento)."""
        video_path = Path(video_path)
        audio_bytes = _extract_audio_wav(video_path) if self.include_audio else None
        with _extract_video(video_path, self.height) as clip_path:
            video_bytes = clip_path.read_bytes()
            return self._ask(video_bytes, audio_bytes, str(video_path))

    def predict_window(
        self, video_path: Path | str, start_s: float, end_s: float
    ) -> EventPrediction:
        """Clasifica solo la ventana ``[start_s, end_s]`` de ``video_path``
        (partido completo, ventana deslizante). ``EventPrediction.video_path``
        sigue siendo la ruta del vídeo fuente, igual que en
        ``core.events.EventClassifier.predict_window``."""
        video_path = Path(video_path)
        audio_bytes = (
            _extract_audio_wav(video_path, start_s, end_s) if self.include_audio else None
        )
        with _extract_video(video_path, self.height, start_s, end_s) as clip_path:
            video_bytes = clip_path.read_bytes()
            return self._ask(video_bytes, audio_bytes, str(video_path))

    def predict_many(
        self, video_paths: Sequence[Path | str]
    ) -> Iterator[EventPrediction]:
        """Clasifica varios clips en secuencia (generador, apto para pipelines).

        Secuencial a propósito, no concurrente: mismo motivo que
        ``native_video_understanding/avlib.py`` documenta para este servidor
        -- varios encodes multimodales en vuelo pueden tumbar el engine con
        ``--gpu-memory-utilization`` ajustado, y un engine caído responde
        HTTP 500 en milisegundos a todo lo siguiente (pasada "rápida" que en
        realidad no clasificó nada)."""
        for video_path in video_paths:
            yield self.predict(video_path)

    # ------------------------------------------------------------------ #

    def _ask(
        self, video_bytes: bytes, audio_bytes: Optional[bytes], video_path_label: str
    ) -> EventPrediction:
        content = [{"type": "text", "text": _QUESTION}]
        content.append({
            "type": "video_url",
            "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(video_bytes).decode()},
        })
        if audio_bytes is not None:
            content.append({
                "type": "input_audio",
                "input_audio": {"data": base64.b64encode(audio_bytes).decode(), "format": "wav"},
            })

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            "media_io_kwargs": {"video": {"fps": self.fps}},
            "mm_processor_kwargs": {"fps": self.fps},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        started = time.perf_counter()
        last_exc: Optional[Exception] = None
        response = None
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout_s)
                resp.raise_for_status()
                response = resp.json()
                break
            except Exception as exc:  # servidor caído, timeout, 500 por engine muerto...
                last_exc = exc
                if attempt < self.retries:
                    logger.warning(
                        "Fallo llamando al servidor Omni (intento %d/%d): %s",
                        attempt + 1, self.retries + 1, exc,
                    )
                    time.sleep(self.retry_wait_s)
        elapsed = time.perf_counter() - started

        if response is None:
            logger.warning(
                "Servidor Omni no respondió tras %d intento(s) para %s (%s), degradando a background",
                self.retries + 1, Path(video_path_label).name, last_exc,
            )
            return EventPrediction(
                video_path=video_path_label,
                label=BACKGROUND_LABEL,
                confidence=0.0,
                probabilities={l: 0.0 for l in ALL_LABELS},
                is_background=True,
            )

        choice = response["choices"][0]
        finish_reason = choice.get("finish_reason")
        raw = (choice["message"].get("content") or "").strip()
        usage = response.get("usage") or {}
        logger.info(
            "Omni respondió en %.1fs para %s (finish_reason=%s, completion_tokens=%s)",
            elapsed, Path(video_path_label).name, finish_reason, usage.get("completion_tokens"),
        )
        if finish_reason == "length":
            logger.warning(
                "Respuesta truncada (max_tokens=%d) para %s -- subir --max-tokens si esto es "
                "frecuente, el JSON final puede no haber cerrado", self.max_tokens, video_path_label,
            )

        return _parse_response(raw, video_path_label)


# --------------------------------------------------------------------------- #
#  Ejecución standalone
# --------------------------------------------------------------------------- #

def _resolve_sources(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(source.glob("**/*.mp4"))
    return [source]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clasificación de eventos de fútbol (Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit / vLLM)"
    )
    parser.add_argument("--source", required=True, help="Clip .mp4 o carpeta con .mp4")
    parser.add_argument("--output", default=None, help="JSONL de salida (si no, solo stdout)")
    parser.add_argument("--url", default=DEFAULT_URL, help="Endpoint chat/completions del servidor vLLM")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Alto (px) al reescalar el vídeo -- ver DEFAULT_HEIGHT, no bajar de ~180 sin remedir el CUDA OOM que resuelve")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--no-thinking", action="store_true", help="Desactiva enable_thinking (más rápido, sin razonamiento)")
    parser.add_argument("--no-audio", action="store_true", help="No manda la pista de audio, solo vídeo")
    parser.add_argument("--start-s", type=float, default=None, help="Si se indica junto a --end-s, usa predict_window en vez de predict")
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

    classifier = OmniEventClassifier(
        url=args.url,
        model_id=args.model_id,
        api_key=args.api_key,
        fps=args.fps,
        height=args.height,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_thinking=not args.no_thinking,
        include_audio=not args.no_audio,
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
