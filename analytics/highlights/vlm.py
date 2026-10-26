"""El VLM decide sobre los candidatos: qué pasa, si merece highlight, dónde
empieza la jugada y cuándo vuelve el partido tras la celebración.

Dos APIs, mismo modelo (Qwen3.5 con visión):
- ``ollama``: API nativa ``/api/chat`` (``think: false`` es la única forma
  fiable de apagar el razonamiento ahí; por su API OpenAI-compatible
  ``chat_template_kwargs`` se ignora y la respuesta llega vacía);
- ``openai``: vLLM u otro servidor compatible (``scripts/serve_events_vllm.sh``
  de la rama feat/T-DEED, puerto 8010).

Los frames van como imágenes JPEG sueltas con la hora del vídeo
sobreimpresa, para que el modelo conteste "en qué frame" con un timestamp.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Sube cuando cambien los prompts o el muestreo de frames: invalida la caché
# de veredictos (``vlm_cache.json``) de las generaciones anteriores.
PROMPT_VERSION = "3"


def hms(t_s: float) -> str:
    t = int(round(t_s))
    return f"{t // 3600}:{(t % 3600) // 60:02d}:{t % 60:02d}"


def parse_hms(text) -> Optional[float]:
    """'1:46:07' | '46:07' | 6367 | '6367.5' -> segundos; None si no se entiende."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip().lstrip("t=").strip()
    m = re.search(r"(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)", s)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"(\d+):(\d{1,2}(?:\.\d+)?)", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def extract_json(text: str) -> dict:
    """Primer objeto JSON de la respuesta (los modelos envuelven en ``` o
    añaden prosa). {} si no hay nada parseable."""
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    chunk = text[start:end + 1]
    for attempt in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
        try:
            obj = json.loads(attempt)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            continue
    return {}


# --------------------------------------------------------------------------- #
#  Frames
# --------------------------------------------------------------------------- #

def extract_frames(video: Path, times_s: Sequence[float], width: int = 640, label: bool = True) -> list[tuple[float, bytes]]:
    """``[(t_s, jpeg)]`` para cada instante (los que no se pueden leer se
    omiten). Con ``label`` se sobreimprime ``t=H:MM:SS`` arriba a la izquierda."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    out: list[tuple[float, bytes]] = []
    try:
        for t in times_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            if w > width:
                frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
            if label:
                text = f"t={hms(t)}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (4, 4), (tw + 12, th + 14), (0, 0, 0), -1)
                cv2.putText(frame, text, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                out.append((t, buf.tobytes()))
    finally:
        cap.release()
    return out


def spaced_times(t0: float, t1: float, n: int) -> list[float]:
    if n <= 1 or t1 <= t0:
        return [t0]
    return [t0 + (t1 - t0) * i / (n - 1) for i in range(n)]


# --------------------------------------------------------------------------- #
#  Cliente
# --------------------------------------------------------------------------- #

class VLMClient:
    def __init__(
        self, base_url: str = "http://localhost:11434", model: str = "qwen3.5:9b-q8_0", api: str = "ollama",
        api_key: str = "EMPTY", timeout_s: float = 900.0, retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api = api
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.retries = retries
        self.last_latency_s: float = 0.0

    def chat(self, prompt: str, images: Sequence[bytes], max_tokens: int = 700) -> str:
        b64 = [base64.b64encode(im).decode() for im in images]
        if self.api == "ollama":
            url = f"{self.base_url}/api/chat"
            message = {"role": "user", "content": prompt}
            if b64:
                message["images"] = b64
            body = {
                "model": self.model, "stream": False, "think": False, "keep_alive": "5m",
                "messages": [message],
                "options": {"temperature": 0, "num_predict": max_tokens},
            }
            headers = {"Content-Type": "application/json"}
        else:
            url = f"{self.base_url}/chat/completions" if self.base_url.endswith("/v1") else f"{self.base_url}/v1/chat/completions"
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}"}} for x in b64
            ]
            body = {
                "model": self.model, "messages": [{"role": "user", "content": content}],
                "temperature": 0, "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        data = json.dumps(body).encode()
        last_err: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    payload = json.load(resp)
                self.last_latency_s = time.time() - t0
                if self.api == "ollama":
                    return payload["message"]["content"]
                return payload["choices"][0]["message"]["content"] or ""
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_err = exc
                logger.warning("VLM (%s) fallo intento %d: %r", self.model, attempt + 1, exc)
                time.sleep(2.0)
        raise RuntimeError(f"VLM sin respuesta tras {self.retries + 1} intentos: {last_err!r}")

    def ping(self) -> bool:
        try:
            url = f"{self.base_url}/api/tags" if self.api == "ollama" else (
                f"{self.base_url}/models" if self.base_url.endswith("/v1") else f"{self.base_url}/v1/models")
            with urllib.request.urlopen(url, timeout=10) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError):
            return False


class CacheOnlyClient(VLMClient):
    """Cliente que no pregunta nada: sirve para regenerar highlights con los veredictos ya cacheados
    (``vlm_cache.json``, ``scoreboard.json``) sin modelo ni GPU, p.ej. al medir un cambio de la política de corte.
    Lo que no esté en caché falla como fallaría un VLM caído y el pipeline sigue sin ese veredicto."""

    def __init__(self, model: str = "qwen3.5:9b-q8_0") -> None:
        super().__init__(model=model, api="cache", retries=0)

    def chat(self, prompt: str, images: Sequence[bytes], max_tokens: int = 700) -> str:
        raise RuntimeError("VLM en modo solo caché: veredicto no cacheado")

    def ping(self) -> bool:
        return True


# --------------------------------------------------------------------------- #
#  Preguntas
# --------------------------------------------------------------------------- #

_JUDGE_PROMPT = """You are selecting highlights from a football (soccer) match video.
These {n} frames are in chronological order, each labelled with its video timestamp (top-left, "t=H:MM:SS").
They cover one candidate play: its build-up and its key moment, which the automatic analysis places around t={anchor}. The last {n_dense} frames are 1 second apart around that moment: look at them closely to tell a goal (ball in the net, players celebrating) from a save, a block or a miss.

Judge the play as a highlights editor would. A goal always deserves a highlight. A good play can also be a dribbling run, a long solo carry, a fast combination that ends in a shot, a great save, or a flurry of shots. Routine passes, throw-ins, fouls or stoppages are not highlights.

Answer with JSON only, no prose, with exactly these keys:
{{
  "what_happens": "<one or two sentences>",
  "is_shot": true or false,
  "is_goal": true or false,
  "goal_confidence": <0.0-1.0>,
  "skill": <1-10, technical quality: dribbles, long run, combination play>,
  "excitement": <1-10, how much a fan would want to see this>,
  "worth_highlight": true or false,
  "play_start": "<timestamp of the frame where the build-up of THIS play begins; use a label from the frames>",
  "title_es": "<short title in Spanish from Spain, max 8 words>",
  "description_es": "<one sentence in Spanish from Spain>"
}}"""

_END_PROMPT = """A key moment of a football (soccer) play (a shot or a goal) happened at about t={anchor}.
These {n} frames come AFTER it, in chronological order, about {step:.0f} seconds apart, each labelled with its video timestamp (top-left, "t=H:MM:SS").

Classify EVERY frame with one of these kinds:
- "action": the ball is in play and the play (or a follow-up, e.g. a rebound) is still going on
- "celebration": players celebrating a goal, hugging, running to the fans, the bench celebrating
- "crowd": the camera shows the crowd, fans or the stands
- "replay": a slow-motion or repeated view of the key moment, or a close-up of a player's reaction
- "walking_back": players walking or jogging back into position, waiting, the goalkeeper picking up the ball, nothing happening
- "play": the game has restarted (kick-off from the centre spot, goal kick, throw-in, corner...) and normal play goes on

Answer with compact JSON only, one pair [timestamp, kind] per frame, in order:
{{"frames": [["<timestamp label>", "<kind>"], ...], "notes": "<a few words>"}}"""

# Frames de estos tipos son la reacción al momento clave: el clip acaba tras el último de ellos. Las
# repeticiones ("replay") solo cuentan tras un gol (clip_end_from_kinds): la realización de TV también las
# encadena tras una parada y alargaban clips de ocasión hasta el tope.
_END_REACTION_KINDS: frozenset[str] = frozenset({"celebration", "crowd"})
_REPLAY_KIND = "replay"

_RANK_PROMPT = """You are the editor of the highlights reel of one football (soccer) match.
Below are candidate plays found automatically. Each has a short description from a visual check and some measurements: shot_prob and goal_prob (0-1, from a video event detector), crowd (crowd/commentary loudness jump in dB right after the moment), passes and carry_m (metres one player carried the ball) in the build-up, and "after": what the detector saw next. The visual check sometimes confuses a goal with a save; "after" is more reliable: a kick-off from the centre spot means a goal was scored, a corner or throw-in means it was not.

Pick the plays that deserve to be in the highlights, ordered from best to worst. For a whole match that is typically 3 to 6 plays. Goals ALWAYS go in. After goals, prefer: skilful dribbling runs or long solo carries ending in a shot, fast team moves ending in a shot, spectacular shots or saves, sequences with several shots in a row. Reject routine plays, duplicates of the same moment and anything that is not really a chance.

Candidates:
{candidates}

Answer with JSON only:
{{
  "ranking": [{{"id": "<candidate id>", "rank": 1, "reason": "<a few words>"}}, ...],
  "rejected": ["<candidate id>", ...]
}}"""


def judge_candidate(client: VLMClient, video: Path, start_s: float, anchor_s: float, n_frames: int = 14, width: int = 640) -> dict:
    """Valoración de la jugada: frames espaciados desde un poco antes del
    inicio por reglas hasta el momento clave, y densos (1/s) alrededor de
    este: si el balón entra o lo saca el portero se ve en 1-2 s."""
    n_dense = max(4, n_frames // 2)
    dense = [anchor_s - 2.0 + i for i in range(n_dense)]
    sparse = spaced_times(max(0.0, start_s - 3.0), anchor_s - 3.0, n_frames - n_dense)
    frames = extract_frames(video, sparse + dense, width=width)
    if not frames:
        return {}
    prompt = _JUDGE_PROMPT.format(n=len(frames), anchor=hms(anchor_s), n_dense=n_dense)
    text = client.chat(prompt, [jpg for _, jpg in frames], max_tokens=700)
    out = extract_json(text)
    out["_raw"] = text
    out["_latency_s"] = client.last_latency_s
    out["_frame_times"] = [t for t, _ in frames]
    ps = parse_hms(out.get("play_start"))
    out["play_start_s"] = ps if ps is not None and start_s - 10.0 <= ps <= anchor_s else None
    return out


def refine_end(client: VLMClient, video: Path, anchor_s: float, max_post_s: float, step_s: float = 4.0,
               n_frames: int = 16, width: int = 640, default_post_s: float = 12.0) -> dict:
    """¿Cuándo vuelve el partido? Frames cada ``step_s`` desde poco después
    del ancla hasta ``max_post_s``."""
    t0 = anchor_s + 3.0
    times = [t0 + i * step_s for i in range(n_frames) if t0 + i * step_s <= anchor_s + max_post_s]
    frames = extract_frames(video, times, width=width)
    if not frames:
        return {}
    prompt = _END_PROMPT.format(anchor=hms(anchor_s), n=len(frames), step=step_s)
    text = client.chat(prompt, [jpg for _, jpg in frames], max_tokens=900)
    out = extract_json(text)
    out["_raw"] = text
    out["_latency_s"] = client.last_latency_s
    out["_frame_times"] = [t for t, _ in frames]
    kinds: list[tuple[float, str]] = []
    for row in out.get("frames") or []:
        if isinstance(row, dict):
            t, kind = parse_hms(row.get("t")), row.get("kind")
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            t, kind = parse_hms(row[0]), row[1]
        else:
            continue
        kind = str(kind or "").strip().lower()
        if t is not None and kind:
            kinds.append((t, kind))
    kinds.sort()
    out["kinds"] = kinds
    out["clip_end_s"] = clip_end_from_kinds(kinds, anchor_s, step_s, max_post_s, default_post_s)
    out["celebration_seen"] = any(k in _END_REACTION_KINDS for _, k in kinds)
    out["replay_seen"] = any(k == _REPLAY_KIND for _, k in kinds)
    return out


def _last_reaction(kinds: list[tuple[float, str]], reaction: frozenset[str],
                   transparent: frozenset[str] = frozenset()) -> Optional[float]:
    """Último frame de reacción, admitiendo un frame suelto de otro tipo en medio; los de tipo
    ``transparent`` ni alargan ni cortan la racha (repeticiones entre dos tramos de celebración)."""
    reaction_end = None
    gap = 0
    for t, kind in kinds:
        if kind in reaction:
            reaction_end, gap = t, 0
        elif kind in transparent:
            continue
        elif reaction_end is not None:
            gap += 1
            if gap >= 2:
                break
    return reaction_end


def first_celebration(kinds: list[tuple[float, str]]) -> Optional[float]:
    """Primer frame de celebración o grada: en un gol, el gol real está justo antes (el ancla de T-DEED o del
    audio puede ir 20-30 s por delante en cámara única)."""
    return next((t for t, kind in kinds if kind in _END_REACTION_KINDS), None)


def clip_end_from_kinds(
    kinds: list[tuple[float, str]], anchor_s: float, step_s: float, max_post_s: float, default_post_s: float,
    replay_as_reaction: bool = True, replay_tail_max_s: Optional[float] = None,
) -> Optional[float]:
    """Fin del clip a partir de la clasificación por frame:

    - con reacción (celebración, grada; repetición solo si ``replay_as_reaction``, es decir, tras un gol): tras
      el último frame de reacción, admitiendo un frame suelto de otro tipo en medio. Las repeticiones que van
      después de la última celebración alargan como mucho ``replay_tail_max_s``;
    - sin reacción pero con la jugada aún viva al principio (rechace, segundo
      remate): tras ese tramo inicial de acción, con tope corto;
    - si no, ``None`` (se queda el fin provisional de las reglas)."""
    if not kinds:
        return None
    if replay_as_reaction:
        # Tras un gol las repeticiones no cortan la celebración (van entre dos tramos de celebración) y, si
        # cierran el clip, lo alargan como mucho replay_tail_max_s tras la última celebración.
        real_end = _last_reaction(kinds, _END_REACTION_KINDS, transparent=frozenset({_REPLAY_KIND}))
        with_replay = _last_reaction(kinds, _END_REACTION_KINDS | {_REPLAY_KIND})
        if with_replay is not None and replay_tail_max_s is not None:
            cap = (real_end if real_end is not None else anchor_s + default_post_s - step_s) + replay_tail_max_s
            with_replay = min(with_replay, cap)
        reaction_end = max(with_replay, real_end) if real_end is not None else with_replay
    else:
        reaction_end = _last_reaction(kinds, _END_REACTION_KINDS)
    if reaction_end is not None:
        return min(reaction_end + step_s, anchor_s + max_post_s)
    leading = 0
    for _, kind in kinds:
        if kind != "action":
            break
        leading += 1
    if leading:
        return min(kinds[leading - 1][0] + step_s, anchor_s + default_post_s + 6.0)
    return anchor_s + default_post_s


def rank_candidates(client: VLMClient, items: list[dict]) -> dict:
    """Ranking relativo, solo texto: ``items`` = ``[{id, t_s, what_happens,
    is_goal, goal_confidence, shot, goal, audio, passes, carry_m}]``.
    Devuelve ``{"rank": {id: posición}, "rejected": [...], "reasons": {id: str}}``."""
    lines = []
    for it in items:
        lines.append(
            f"- {it['id']} (t={hms(it['t_s'])}): {it.get('what_happens') or '(no description)'} "
            f"[visual check: goal={'yes' if it.get('is_goal') else 'no'} ({float(it.get('goal_confidence') or 0):.1f}); "
            f"shot_prob={it['shot']:.2f}, goal_prob={it['goal']:.2f}, crowd=+{it['audio']:.0f} dB, "
            f"passes={it['passes']:.0f}, carry_m={it['carry_m']:.0f}; after: {it.get('after', '')}]"
        )
    text = client.chat(_RANK_PROMPT.format(candidates="\n".join(lines)), [], max_tokens=900)
    out = extract_json(text)
    rank: dict[str, int] = {}
    reasons: dict[str, str] = {}
    for k, row in enumerate(out.get("ranking") or [], start=1):
        if isinstance(row, dict) and row.get("id"):
            rank.setdefault(str(row["id"]), int(row.get("rank") or k))
            reasons[str(row["id"])] = str(row.get("reason") or "")
        elif isinstance(row, str):
            rank.setdefault(row, k)
    rejected = [str(x) for x in (out.get("rejected") or []) if isinstance(x, (str, int))]
    return {"rank": rank, "rejected": rejected, "reasons": reasons, "_raw": text, "_latency_s": client.last_latency_s}
