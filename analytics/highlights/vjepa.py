"""V-JEPA2 + SVC como segundo juez de gol sobre los candidatos (sin marcador).

Ni el VLM ni V-JEPA2 distinguen bien un gol de una parada por separado (AP
0,27 y 0,19 sobre los 7 EFL, 13 goles, sin rótulo), pero se equivocan en
jugadas distintas y sumados doblan la AP; con el saque de centro que V-JEPA2
ve después del gol (clase ``kick-off``) y la ausencia de córner/banda llega
a ~0,6 (``docs/highlights_evaluacion.md``). Es el camino cuando no hay
marcador legible; el marcador, si existe, manda (``pipeline.goal_decision``).

Las ventanas de 6 s van en una rejilla fija del partido (ventana k =
[6k, 6k+6)) para que candidatos vecinos compartan lecturas; se cachean con
las 16 probabilidades en ``vjepa_cache.json``.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from config import HighlightsConfig

    from .candidates import Candidate

logger = logging.getLogger(__name__)

# Sube si cambia el muestreo o el significado de las ventanas: invalida la caché.
VJEPA_VERSION = "1"

VJ_FEATURES = ("vj_goal", "vj_shot", "vj_kickoff_after", "vj_deadball_after")


class VJEPAWindows:
    """Probabilidades de V-JEPA2+SVC por ventana de la rejilla, con caché y
    carga perezosa del modelo (solo si hace falta calcular alguna ventana)."""

    def __init__(
        self, video: Path, duration_s: float, cache_path: Optional[Path], window_s: float = 6.0,
        classifier_path: Optional[str] = None, model_id: Optional[str] = None, device: Optional[str] = None,
    ) -> None:
        self.video = Path(video)
        self.duration_s = float(duration_s)
        self.cache_path = cache_path
        self.window_s = float(window_s)
        self.classifier_path = classifier_path
        self.model_id = model_id
        self.device = device
        self._clf = None
        self.windows: dict[str, dict[str, float]] = {}
        self.n_computed = 0
        if cache_path is not None and Path(cache_path).exists():
            with open(cache_path) as f:
                blob = json.load(f)
            if blob.get("version") == VJEPA_VERSION and blob.get("window_s") == self.window_s \
                    and blob.get("model_id") == self._meta()["model_id"] and blob.get("classifier") == self._meta()["classifier"]:
                self.windows = blob.get("windows", {})

    def _meta(self) -> dict:
        from core.events import DEFAULT_CLASSIFIER_PATH, DEFAULT_MODEL_ID

        return {"model_id": self.model_id or DEFAULT_MODEL_ID,
                "classifier": Path(self.classifier_path or DEFAULT_CLASSIFIER_PATH).name}

    def _classifier(self):
        if self._clf is None:
            from core.events import DEFAULT_CLASSIFIER_PATH, DEFAULT_MODEL_ID, EventClassifier

            self._clf = EventClassifier(
                classifier_path=self.classifier_path or DEFAULT_CLASSIFIER_PATH,
                model_id=self.model_id or DEFAULT_MODEL_ID, device=self.device,
            )
        return self._clf

    def save(self) -> None:
        if self.cache_path is None:
            return
        tmp = Path(self.cache_path).with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"version": VJEPA_VERSION, "window_s": self.window_s, **self._meta(), "windows": self.windows}, f)
        tmp.replace(self.cache_path)

    # --- rejilla ------------------------------------------------------- #

    def keys_overlapping(self, t0: float, t1: float) -> list[int]:
        """Ventanas de la rejilla que tocan [t0, t1] y caben enteras en el vídeo."""
        if t1 <= t0:
            return []
        k0 = max(int(math.floor(max(t0, 0.0) / self.window_s)), 0)
        k1 = int(math.floor(max(t1 - 1e-6, 0.0) / self.window_s))
        k_max = int(math.floor(self.duration_s / self.window_s)) - 1
        return [k for k in range(k0, k1 + 1) if k <= k_max]

    def missing(self, keys: list[int]) -> list[int]:
        return [k for k in keys if str(k) not in self.windows]

    def ensure(self, keys: list[int], progress: Optional[Callable[[int, int], None]] = None) -> None:
        todo = self.missing(keys)
        for i, k in enumerate(todo):
            if progress:
                progress(i, len(todo))
            t0 = k * self.window_s
            pred = self._classifier().predict_window(self.video, t0, t0 + self.window_s)
            self.windows[str(k)] = {c: round(float(p), 4) for c, p in pred.probabilities.items()}
            self.n_computed += 1
            if self.n_computed % 20 == 0:
                self.save()
        if todo:
            self.save()

    def pmax(self, cls: str, t0: float, t1: float) -> float:
        keys = self.keys_overlapping(t0, t1)
        return max((self.windows[str(k)].get(cls, 0.0) for k in keys if str(k) in self.windows), default=0.0)


# --------------------------------------------------------------------------- #
#  Puntuación de candidatos
# --------------------------------------------------------------------------- #

def goal_keys(vw: VJEPAWindows, anchor_s: float, cfg: "HighlightsConfig") -> list[int]:
    return vw.keys_overlapping(anchor_s - cfg.vjepa_goal_pre_s, anchor_s + cfg.vjepa_goal_post_s)


def after_keys(vw: VJEPAWindows, anchor_s: float, next_anchor_s: float, cfg: "HighlightsConfig") -> list[int]:
    """Ventanas donde buscar el saque de centro (y el balón parado) que sigue
    a la jugada; no pasan del siguiente candidato para no atribuirle el saque
    del gol de la jugada siguiente."""
    return vw.keys_overlapping(anchor_s + 3.0, min(anchor_s + cfg.vjepa_after_to_s, next_anchor_s))


def score_candidate(vw: VJEPAWindows, anchor_s: float, next_anchor_s: float, cfg: "HighlightsConfig", with_after: bool) -> dict[str, float]:
    f = {
        "vj_goal": vw.pmax("goal", anchor_s - cfg.vjepa_goal_pre_s, anchor_s + cfg.vjepa_goal_post_s),
        "vj_shot": vw.pmax("shot", anchor_s - cfg.vjepa_goal_pre_s, anchor_s + 3.0),
    }
    if with_after:
        f["vj_kickoff_after"] = vw.pmax("kick-off", anchor_s + cfg.vjepa_after_from_s, min(anchor_s + cfg.vjepa_after_to_s, next_anchor_s))
        f["vj_deadball_after"] = max(vw.pmax("corner", anchor_s + 3.0, min(anchor_s + cfg.vjepa_deadball_to_s, next_anchor_s)),
                                     vw.pmax("throw-in", anchor_s + 3.0, min(anchor_s + cfg.vjepa_deadball_to_s, next_anchor_s)))
    return f


def goal_fusion(verdict: dict, f: dict) -> Optional[float]:
    """Votos sumados para "gol" sin marcador: V-JEPA2 (gol en el golpeo,
    saque de centro después, sin córner/banda después), el VLM y el tiro.
    ``None`` si el candidato no pasó por V-JEPA2. Pesos y umbral medidos en
    los 7 EFL (``docs/highlights_evaluacion.md``)."""
    if "vj_goal" not in f:
        return None
    try:
        vlm = float(verdict.get("goal_confidence", 1.0) or 0.0) if verdict.get("is_goal") else 0.0
    except (TypeError, ValueError):
        vlm = 1.0 if verdict.get("is_goal") else 0.0
    shot = max(float(f.get("shot", 0.0)), float(f.get("vj_shot", 0.0)))
    return (float(f["vj_goal"]) + vlm + float(f.get("vj_kickoff_after", 0.0))
            - 0.5 * float(f.get("vj_deadball_after", 0.0)) + 0.5 * shot)


def apply_vjepa(
    vw: VJEPAWindows, cands: list["Candidate"], cfg: "HighlightsConfig",
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Puntúa los candidatos con V-JEPA2 y los reordena. Dos fases: el
    golpeo para TODOS (un gol puede ir muy abajo por reglas) y las ventanas
    posteriores solo para los mejores por reglas o con indicio de gol.
    Devuelve el número de ventanas calculadas."""
    before = vw.n_computed
    by_time = sorted(cands, key=lambda c: c.anchor_s)
    next_anchor = {c.id: (by_time[i + 1].anchor_s - 8.0 if i + 1 < len(by_time) else vw.duration_s)
                   for i, c in enumerate(by_time)}
    keys: list[int] = []
    for c in cands:
        keys += goal_keys(vw, c.anchor_s, cfg)
    vw.ensure(sorted(set(keys)), progress)
    for c in cands:
        c.features.update(score_candidate(vw, c.anchor_s, next_anchor[c.id], cfg, with_after=False))

    top = {c.id for c in sorted(cands, key=lambda c: -c.rule_score)[: cfg.vjepa_max_candidates]}
    with_after = [c for c in cands if c.id in top or c.features["vj_goal"] >= cfg.vjepa_after_min_goal
                  or c.features.get("score_goal", 0.0) >= 0.5]
    keys = []
    for c in with_after:
        keys += after_keys(vw, c.anchor_s, next_anchor[c.id], cfg)
    vw.ensure(sorted(set(keys)), progress)
    for c in with_after:
        c.features.update(score_candidate(vw, c.anchor_s, next_anchor[c.id], cfg, with_after=True))
        c.rule_score += cfg.w_vj_goal * c.features["vj_goal"] + cfg.w_vj_kickoff * c.features["vj_kickoff_after"]
        if "vjepa" not in c.sources:
            c.sources = sorted(set(c.sources) | {"vjepa"})
    cands.sort(key=lambda c: -c.rule_score)
    return vw.n_computed - before
