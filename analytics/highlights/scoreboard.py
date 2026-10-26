"""Marcador de la retransmisión: un gol es un cambio de marcador.

En TV el rótulo (equipos, goles y reloj) está en una esquina durante casi
todo el partido, también en primeros planos y celebraciones. El audio de TV no
sirve para detectar goles (ver ``pipeline.audio_informative``) y el saque de
centro cuesta de ver con la cámara panorámica móvil, pero el marcador cambia
justo en el gol. Se lee con el mismo VLM sobre un recorte de la franja
superior (sin dependencias de OCR) y se compara antes y después de cada
candidato. Si el vídeo no tiene marcador (cámara única) no hay señal y el
resto de la cascada decide como hasta ahora.
"""

from __future__ import annotations

import bisect
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from .vlm import VLMClient, extract_json

# Sube si cambian el prompt o el recorte: invalida las lecturas cacheadas.
SCOREBOARD_VERSION = "1"

logger = logging.getLogger(__name__)

PROMPT = (
    "This is the top strip of a football TV broadcast frame. If a scoreboard overlay is visible (team names or "
    "abbreviations, the two goal counts and the match clock), read it. Answer ONLY with JSON: "
    '{"scoreboard": true, "home": "<abbr>", "away": "<abbr>", "home_score": <int>, "away_score": <int>, '
    '"clock": "<mm:ss>"} or {"scoreboard": false} if there is no scoreboard in the image (close-up, replay '
    "or advertisement). Do not guess: digits you cannot read clearly mean no scoreboard."
)


# Esquinas donde las retransmisiones ponen el rótulo, por orden de probabilidad:
# (y0, y1, x0, x1) en fracciones del frame.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "top-left": (0.0, 0.16, 0.0, 0.55),
    "top-right": (0.0, 0.16, 0.45, 1.0),
    "bottom-left": (0.84, 1.0, 0.0, 0.55),
    "bottom-right": (0.84, 1.0, 0.45, 1.0),
}


def extract_strip(video: Path, t_s: float, region: str = "top-left", width: int = 1100) -> Optional[bytes]:
    """JPEG de la esquina ``region`` del frame en ``t_s`` donde suele ir el
    rótulo; ``None`` si no se puede leer."""
    import cv2

    y0, y1, x0, x1 = REGIONS[region]

    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            return None
        h, w = frame.shape[:2]
        crop = frame[int(h * y0): int(h * y1), int(w * x0): int(w * x1)]
        if crop.shape[1] > width:
            scale = width / crop.shape[1]
            crop = cv2.resize(crop, (width, int(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return buf.tobytes() if ok else None
    finally:
        cap.release()


def read_score(client: VLMClient, video: Path, t_s: float, region: str = "top-left") -> Optional[dict]:
    """``{"home": 1, "away": 0, "clock": "27:56"}`` o ``None`` si no se ve el marcador."""
    strip = extract_strip(video, t_s, region)
    if strip is None:
        return None
    reply = extract_json(client.chat(PROMPT, [strip], max_tokens=160))
    if not reply.get("scoreboard"):
        return None
    try:
        home, away = int(reply["home_score"]), int(reply["away_score"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= home <= 20 and 0 <= away <= 20):
        return None
    clock = reply.get("clock")
    return {"home": home, "away": away, "clock": clock if isinstance(clock, str) and re.fullmatch(r"\d{1,3}:\d{2}", clock) else None}


# --------------------------------------------------------------------------- #
#  Línea temporal del marcador
# --------------------------------------------------------------------------- #

Score = tuple[int, int]


@dataclass
class ScoreChange:
    """El marcador pasó de ``before`` a ``after`` entre ``t_lo`` y ``t_hi``
    (el rótulo se actualiza unos segundos después del gol)."""

    t_lo: float
    t_hi: float
    before: Score
    after: Score

    def to_dict(self) -> dict:
        return {"t_lo": self.t_lo, "t_hi": self.t_hi, "before": list(self.before), "after": list(self.after)}


@dataclass
class ScoreTimeline:
    available: bool                              # el vídeo lleva marcador legible
    changes: list[ScoreChange] = field(default_factory=list)
    read_times: list[float] = field(default_factory=list)   # instantes con marcador legible
    n_samples: int = 0

    def brackets(self, t_s: float, before_s: float = 90.0, after_from_s: float = 20.0, after_to_s: float = 110.0) -> bool:
        """¿Hay lectura válida antes y después de ``t_s`` (esta última pasado
        el retraso del rótulo)? Solo entonces "el marcador no cambió" prueba
        que no hubo gol."""
        def any_in(lo: float, hi: float) -> bool:
            i = bisect.bisect_left(self.read_times, lo)
            return i < len(self.read_times) and self.read_times[i] <= hi

        return any_in(t_s - before_s, t_s - 1.0) and any_in(t_s + after_from_s, t_s + after_to_s)

    def to_dict(self) -> dict:
        return {"available": self.available, "n_samples": self.n_samples, "n_valid": len(self.read_times),
                "changes": [c.to_dict() for c in self.changes]}


def _confirmed_changes(valid: list[tuple[float, Score]]) -> list[tuple[int, int, Score, Score]]:
    """Cambios aceptados como ``(índice de la última lectura con el marcador
    anterior, índice de la primera con el nuevo, antes, después)``. Una
    lectura suelta que no se repite en la siguiente, que baja algún contador
    o que sube más de 3 goles de golpe es un error de lectura."""
    out: list[tuple[int, int, Score, Score]] = []
    state: Optional[Score] = None
    last_state_i = 0
    for i, (_, s) in enumerate(valid):
        if state is None:
            state = s
            last_state_i = i
            continue
        if s == state:
            last_state_i = i
            continue
        gained = (s[0] - state[0]) + (s[1] - state[1])
        monotonic = s[0] >= state[0] and s[1] >= state[1]
        confirmed = i + 1 < len(valid) and valid[i + 1][1] == s
        if monotonic and 1 <= gained <= 3 and confirmed:
            out.append((last_state_i, i, state, s))
            state = s
            last_state_i = i
    return out


def find_region(client: VLMClient, video: Path, duration_s: float, n_probe: int = 8, min_valid: int = 4) -> Optional[str]:
    """Esquina donde está el rótulo: la primera de ``REGIONS`` con al menos
    ``min_valid`` lecturas legibles en ``n_probe`` instantes repartidos por el
    partido; ``None`` si ninguna (vídeo sin marcador)."""
    times = [duration_s * (0.08 + 0.84 * i / max(n_probe - 1, 1)) for i in range(n_probe)]
    for region in REGIONS:
        valid = 0
        for k, t in enumerate(times):
            valid += read_score(client, video, t, region) is not None
            if valid >= min_valid or valid + (len(times) - k - 1) < min_valid:
                break
        if valid >= min_valid:
            return region
    return None


def build_timeline(
    client: VLMClient, video: Path, duration_s: float, step_s: float = 30.0, min_valid: float = 0.25,
    cache_path: Optional[Path] = None, model: str = "", progress: Optional[Callable[[int, int], None]] = None,
) -> ScoreTimeline:
    """Lee el marcador cada ``step_s`` y localiza cada cambio por bisección
    (±4 s). Las lecturas y la esquina del rótulo se cachean en
    ``cache_path``."""
    cache: dict[str, Optional[dict]] = {}
    region: Optional[str] = None
    probed = False
    if cache_path is not None and cache_path.exists():
        with open(cache_path) as f:
            blob = json.load(f)
        if blob.get("version") == SCOREBOARD_VERSION and blob.get("model") == model:
            cache = blob.get("reads", {})
            if "region" in blob:
                region, probed = blob["region"], True
    dirty = 0

    def save() -> None:
        if cache_path is None:
            return
        tmp = cache_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"version": SCOREBOARD_VERSION, "model": model, "region": region, "reads": cache}, f)
        tmp.replace(cache_path)

    if not probed:
        region = find_region(client, video, duration_s)
        save()
    if region is None:
        logger.info("Sin marcador legible en ninguna esquina: no hay señal de goles por marcador")
        return ScoreTimeline(False, n_samples=0)

    def read(t: float) -> Optional[Score]:
        nonlocal dirty
        key = f"{t:.1f}"
        if key not in cache:
            cache[key] = read_score(client, video, t, region)
            dirty += 1
            if dirty % 25 == 0:
                save()
        r = cache[key]
        return (r["home"], r["away"]) if r else None

    times = [float(t) for t in range(0, max(int(duration_s) - 2, 1), max(int(step_s), 1))] + [max(duration_s - 3.0, 0.0)]
    valid: list[tuple[float, Score]] = []
    for k, t in enumerate(times):
        s = read(t)
        if s is not None:
            valid.append((t, s))
        if progress:
            progress(k, len(times))

    # Un gol casi al final de la parte deja una sola lectura con el marcador
    # nuevo (luego el rótulo se esconde): se confirma con lecturas vecinas.
    if len(valid) >= 2 and valid[-1][1] != valid[-2][1]:
        t_last, s_last = valid[-1]
        for off in (2.0, -2.0, 4.0, -4.0):
            t2 = t_last + off
            if valid[-2][0] < t2 < duration_s - 0.5 and read(t2) == s_last:
                valid = sorted(valid + [(t2, s_last)])
                break

    available = len(valid) >= max(8, min_valid * len(times))
    changes: list[ScoreChange] = []
    if available:
        for lo_i, hi_i, before, after in _confirmed_changes(valid):
            lo, hi = valid[lo_i][0], valid[hi_i][0]
            while hi - lo > 4.0:
                mid = (lo + hi) / 2.0
                r, t_read = None, mid
                for off in (0.0, 1.0, -1.0, 2.0, -2.0):
                    if lo < mid + off < hi:
                        t_read = mid + off
                        r = read(t_read)
                        if r is not None:
                            break
                if r == before:
                    lo = t_read
                elif r == after:
                    hi = t_read
                else:
                    break
            changes.append(ScoreChange(lo, hi, before, after))
    save()
    return ScoreTimeline(available, changes, [t for t, _ in valid], len(times))


def main() -> None:
    """Mide solo el lector de marcador contra goles etiquetados (SoccerNet
    ``Labels-ball.json``: etiqueta GOAL, ``position`` en ms de vídeo)::

        python -m analytics.highlights.scoreboard --video 720p.mp4 --labels Labels-ball.json --cache /tmp/sb.json
    """
    import argparse

    import cv2

    from config import HighlightsConfig

    from .vlm import hms

    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="Labels-ball.json / Labels-v2.json con los goles de referencia")
    parser.add_argument("--half", type=int, help="solo los goles de esa parte (Labels-v2: gameTime '1 - 12:30'; vídeo = una parte)")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--step", type=float, default=HighlightsConfig().scoreboard_step_s)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = HighlightsConfig()
    client = VLMClient(cfg.vlm_base_url, cfg.vlm_model, cfg.vlm_api, cfg.vlm_api_key, cfg.vlm_timeout_s)
    cap = cv2.VideoCapture(str(args.video))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1.0)
    cap.release()
    timeline = build_timeline(client, args.video, duration, args.step, cfg.scoreboard_min_valid, args.cache, cfg.vlm_model)
    print(f"{args.video}: marcador {'legible' if timeline.available else 'NO legible'} "
          f"({len(timeline.read_times)}/{timeline.n_samples} lecturas)")
    for ch in timeline.changes:
        print(f"  cambio {ch.before} -> {ch.after} entre {hms(ch.t_lo)} y {hms(ch.t_hi)}")
    if args.labels:
        with open(args.labels) as f:
            goals = sorted(float(a["position"]) / 1000.0 for a in json.load(f)["annotations"]
                           if a["label"].lower() == "goal" and (args.half is None or a["gameTime"].startswith(f"{args.half} -")))
        used: set[int] = set()
        for g in goals:
            hit = next((i for i, ch in enumerate(timeline.changes)
                        if i not in used and ch.t_lo - cfg.scoreboard_goal_before_s <= g <= ch.t_hi + 3.0), None)
            if hit is not None:
                used.add(hit)
            print(f"  gol {hms(g)}: {'detectado' if hit is not None else 'NO detectado'}")
        print(f"  RESUMEN goles {len(used)}/{len(goals)}, cambios sin gol de referencia {len(timeline.changes) - len(used)}")


if __name__ == "__main__":
    main()
