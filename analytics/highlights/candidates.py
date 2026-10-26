"""De las señales a candidatos a highlight, por reglas y sin mirar el vídeo.

Un candidato nace de un *ancla* (el momento clave: pico de tiro/gol de
T-DEED aunque esté por debajo del umbral de la telemetría, pico sostenido de
audio con el balón en juego, o evento emitido) y crece hacia atrás por la
cadena de acciones del mismo equipo hasta el balón parado o la pérdida
anterior: eso es "la jugada". La puntuación por reglas solo ordena: el VLM
decide después sobre los mejores.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from .signals import CENTER_XY, DEAD_BALL_EVENTS, GOAL_CENTERS, GRID_S, MatchSignals

if TYPE_CHECKING:
    from config import HighlightsConfig

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    id: str
    anchor_s: float            # momento clave (tiro/gol)
    start_s: float             # inicio de la jugada (reglas; el VLM puede afinarlo)
    end_s: float               # fin provisional (reglas; el VLM lo afina)
    team: int                  # equipo que ataca (-1 desconocido)
    sources: list[str]         # qué anclas lo formaron: shot, goal, audio, event:shot...
    features: dict[str, float]
    rule_score: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        return cls(**d)


# --------------------------------------------------------------------------- #
#  Anclas
# --------------------------------------------------------------------------- #

def _peaks(values: np.ndarray, min_value: float, nms_s: float) -> list[int]:
    """Índices de máximos locales >= min_value, suprimiendo los que caen a
    menos de nms_s de uno mayor."""
    idx = np.where(values >= min_value)[0]
    if idx.size == 0:
        return []
    order = idx[np.argsort(-values[idx], kind="stable")]
    radius = int(round(nms_s / GRID_S))
    taken: list[int] = []
    for i in order:
        if all(abs(int(i) - j) > radius for j in taken):
            taken.append(int(i))
    return sorted(taken)


def _n_actions_near(sig: MatchSignals, t_s: float, radius_s: float) -> int:
    return sum(1 for a in sig.actions if abs(a["t_s"] - t_s) <= radius_s)


def find_anchors(sig: MatchSignals, cfg: "HighlightsConfig") -> list[tuple[float, str, float]]:
    """``[(t_s, fuente, fuerza)]`` sin fusionar."""
    anchors: list[tuple[float, str, float]] = []
    shot = np.maximum(sig.curve("shot"), sig.curve("sn_shot"))
    goal = np.maximum(sig.curve("goal"), sig.curve("sn_goal"))
    for i in _peaks(shot, cfg.shot_min_score, 8.0):
        anchors.append((float(sig.t[i]), "shot", float(shot[i])))
    for i in _peaks(goal, cfg.goal_min_score, 8.0):
        anchors.append((float(sig.t[i]), "goal", float(goal[i])))
    if sig.has_audio:
        for i in _peaks(sig.audio_sustained, cfg.audio_min_sustained_db, 30.0):
            t_peak = float(sig.t[i])
            if _n_actions_near(sig, t_peak, 30.0) < cfg.audio_min_actions:
                continue        # calentamiento, descanso, gente junto al micro
            # El grito empieza justo después del momento clave: ancla en el
            # arranque del pico instantáneo, un segundo antes.
            w = sig.window(t_peak - 1.0, t_peak + 6.0)
            j = w.start + int(np.argmax(sig.audio_rel[w]))
            anchors.append((max(0.0, float(sig.t[j]) - 1.0), "audio", float(sig.audio_sustained[i])))
    for e in sig.events:
        if e["label"] in ("shot", "goal"):
            anchors.append((float(e["t_s"]), f"event:{e['label']}", float(e["score"])))
    anchors.sort()
    return anchors


def merge_anchors(sig: MatchSignals, anchors: list[tuple[float, str, float]], merge_s: float) -> list[dict]:
    """Agrupa anclas a menos de ``merge_s`` entre sí (en cadena). El ancla del
    grupo es el pico de tiro si lo hay; si no, la más temprana."""
    shot = np.maximum(sig.curve("shot"), sig.curve("sn_shot"))
    groups: list[list[tuple[float, str, float]]] = []
    for a in anchors:
        if groups and a[0] - groups[-1][-1][0] <= merge_s:
            groups[-1].append(a)
        else:
            groups.append([a])
    out = []
    for g in groups:
        t0, t1 = g[0][0], g[-1][0]
        w = sig.window(t0 - 2.0, t1 + 2.0)
        j = w.start + int(np.argmax(shot[w]))
        anchor_s = float(sig.t[j]) if shot[j] >= 0.05 else t0
        out.append({"anchor_s": anchor_s, "sources": sorted({src for _, src, _ in g}),
                    "strength": {src: max(s for _, s2, s in g if s2 == src) for src in {src for _, src, _ in g}}})
    return out


# --------------------------------------------------------------------------- #
#  Jugada (hacia atrás) y fin provisional (hacia delante)
# --------------------------------------------------------------------------- #

def team_at(sig: MatchSignals, t0: float, t1: float) -> int:
    teams = sig.team[sig.window(t0, t1)]
    teams = teams[teams >= 0]
    if teams.size == 0:
        return -1
    vals, counts = np.unique(teams, return_counts=True)
    return int(vals[np.argmax(counts)])


def _dead_ball_between(sig: MatchSignals, t0: float, t1: float, cfg: "HighlightsConfig") -> bool:
    if t1 <= t0:
        return False
    if sig.curve("out")[sig.window(t0, t1)].max(initial=0.0) >= cfg.dead_ball_out_score:
        return True
    return any(e["label"] in DEAD_BALL_EVENTS and t0 < e["t_s"] < t1 for e in sig.events)


def play_start(sig: MatchSignals, anchor_s: float, team: int, cfg: "HighlightsConfig") -> tuple[float, list[dict]]:
    """Primera acción de la cadena del equipo atacante que lleva al ancla.
    Devuelve ``(start_s, acciones de la jugada)``."""
    acts = [a for a in sig.actions if anchor_s - cfg.play_max_back_s <= a["t_s"] <= anchor_s + 0.5]
    chain: list[dict] = []
    prev_t = anchor_s + 0.5
    for a in reversed(acts):
        other_team = team >= 0 and a["team"] >= 0 and a["team"] != team
        if other_team and chain:
            break
        if prev_t - a["t_s"] > cfg.play_gap_s and chain:
            break
        if _dead_ball_between(sig, a["t_s"] + 0.3, prev_t - 0.3, cfg) and chain:
            break
        if other_team:
            # Un rival toca justo antes del ancla (rechace, pérdida): la
            # jugada es nuestra desde ahí.
            prev_t = a["t_s"]
            continue
        chain.append(a)
        prev_t = a["t_s"]
        if a["label"] == "throw_in":
            break           # el saque de banda abre la jugada
    chain.reverse()
    if chain:
        start = chain[0]["t_s"] - cfg.play_pre_s
    else:
        start = anchor_s - cfg.play_default_back_s
    start = min(max(start, anchor_s - cfg.play_max_back_s), anchor_s - cfg.play_min_back_s)
    return max(0.0, start), chain


def provisional_end(sig: MatchSignals, anchor_s: float, cfg: "HighlightsConfig") -> float:
    end = anchor_s + cfg.post_default_s
    if sig.has_audio:
        # Alargar mientras dure la celebración (audio sostenido por encima
        # del fondo), con tope.
        w = sig.window(anchor_s + 3.0, anchor_s + cfg.post_max_s)
        sus = sig.audio_sustained[w]
        above = np.where(sus > cfg.audio_tail_db)[0]
        if above.size and above[0] <= 4:
            # primer hueco de la racha que arranca al principio
            gaps = np.where(np.diff(above) > 2)[0]
            last = above[gaps[0]] if gaps.size else above[-1]
            end = max(end, float(sig.t[w.start + last]) + 2.0)
    return min(max(end, anchor_s + cfg.post_min_s), anchor_s + cfg.post_max_s, sig.duration_s)


# --------------------------------------------------------------------------- #
#  Características y puntuación
# --------------------------------------------------------------------------- #

def _xy_at(sig: MatchSignals, b: int) -> Optional[np.ndarray]:
    for arr in (sig.ball_xy, sig.possessor_xy):
        if np.isfinite(arr[b]).all():
            return arr[b]
    return None


def _dist_to_goal(xy: np.ndarray) -> float:
    return min(math.hypot(xy[0] - gx, xy[1] - gy) for gx, gy in GOAL_CENTERS)


def _drive_metres(sig: MatchSignals, t0: float, t1: float) -> tuple[float, float]:
    """Mayor desplazamiento (m) de un mismo poseedor con el balón en [t0, t1]
    y su duración (s). Tramos de poseedor constante (admite 1 bin de hueco)."""
    w = sig.window(t0, t1)
    pids = sig.possessor[w]
    xys = sig.possessor_xy[w]
    best_m, best_s = 0.0, 0.0
    n, i = len(pids), 0
    while i < n:
        if pids[i] < 0:
            i += 1
            continue
        pid, j = pids[i], i
        while j + 1 < n and (pids[j + 1] == pid or (pids[j + 1] < 0 and j + 2 < n and pids[j + 2] == pid)):
            j += 1
        seg = xys[i:j + 1]
        ok = np.isfinite(seg).all(axis=1)
        if ok.sum() >= 2:
            first, last = seg[ok][0], seg[ok][-1]
            m = float(math.hypot(float(last[0] - first[0]), float(last[1] - first[1])))
            if m > best_m:
                best_m, best_s = m, (j - i + 1) * GRID_S
        i = j + 1
    return best_m, best_s


def restart_after(sig: MatchSignals, anchor_s: float, until_s: float) -> tuple[float, float]:
    """Evidencia de saque de centro tras el ancla (= gol): ``(curva Kick-off
    máx., 1 si balón/poseedor en el círculo central con los equipos
    repartidos por mitades y sin acciones en los 6 s previos)``. Se mira
    hasta ``until_s`` (el siguiente ancla) para no atribuir a un tiro el
    saque que sigue al gol de la jugada siguiente."""
    w = sig.window(anchor_s + 8.0, min(anchor_s + 150.0, until_s))
    kick = float(sig.curve("sn_kickoff")[w].max(initial=0.0))
    center = 0.0
    for b in range(w.start, w.stop):
        xy = _xy_at(sig, b)
        if xy is None or math.hypot(xy[0] - CENTER_XY[0], xy[1] - CENTER_XY[1]) > 5.0:
            continue
        if sig.n_players_xy[b] < 6 or not np.isfinite(sig.players_left_frac[b]):
            continue
        if abs(sig.players_left_frac[b] - 0.5) > 0.2:
            continue
        t_b = float(sig.t[b])
        if _n_actions_near(sig, t_b - 3.0, 3.0) == 0:
            center = 1.0
            break
    return kick, center


def compute_features(sig: MatchSignals, anchor_s: float, start_s: float, chain: list[dict], team: int,
                     next_anchor_s: float) -> dict[str, float]:
    shot = np.maximum(sig.curve("shot"), sig.curve("sn_shot"))
    goal = np.maximum(sig.curve("goal"), sig.curve("sn_goal"))
    w_shot = sig.window(anchor_s - 2.0, anchor_s + 2.0)
    w_goal = sig.window(anchor_s - 2.0, anchor_s + 12.0)
    w_audio = sig.window(anchor_s - 1.0, anchor_s + 6.0)
    f: dict[str, float] = {
        "shot": float(shot[w_shot].max(initial=0.0)),
        "goal": float(goal[w_goal].max(initial=0.0)),
        "cross": float(sig.curve("cross")[sig.window(start_s, anchor_s)].max(initial=0.0)),
        "audio_sustained": float(sig.audio_sustained[w_audio].max(initial=0.0)) if sig.has_audio else 0.0,
        "audio_peak": float(sig.audio_rel[sig.window(anchor_s - 1.0, anchor_s + 10.0)].max(initial=0.0)) if sig.has_audio else 0.0,
        "play_len_s": float(anchor_s - start_s),
        "n_actions": float(len(chain)),
        "n_passes": float(sum(1 for a in chain if a["label"] in ("pass", "high_pass", "header"))),
        "n_drives": float(sum(1 for a in chain if a["label"] == "drive")),
        "n_players": float(len({a["tracker_id"] for a in chain})),
    }
    f["drive_m"], f["drive_s"] = _drive_metres(sig, start_s, anchor_s + 1.0)
    dists = [_dist_to_goal(xy) for b in range(*sig.window(anchor_s - 1.5, anchor_s + 0.5).indices(sig.n_bins))
             if (xy := _xy_at(sig, b)) is not None]
    f["dist_goal_m"] = float(min(dists)) if dists else float("nan")
    f["kickoff_curve"], f["center_restart"] = restart_after(sig, anchor_s, next_anchor_s)
    # Primer balón parado emitido tras el ancla (córner, banda, falta...): un
    # gol no va seguido de un córner, va seguido de un saque de centro.
    after = [e["t_s"] - anchor_s for e in sig.events
             if e["label"] in DEAD_BALL_EVENTS and anchor_s + 3.0 < e["t_s"] < next_anchor_s + 8.0]
    f["next_deadball_s"] = float(min(after)) if after else 999.0
    # Juego en marcha antes de la jugada (en el descanso o el calentamiento
    # también hay tiros a puerta y luego un saque de centro, pero sin acciones
    # de balón en los minutos previos).
    f["n_actions_before_90s"] = float(sum(1 for a in sig.actions if anchor_s - 90.0 <= a["t_s"] <= anchor_s - 3.0))
    return f


def rule_score(f: dict[str, float], n_shots_60: int, cfg: "HighlightsConfig") -> float:
    near_goal = 0.0 if not math.isfinite(f["dist_goal_m"]) else min(max((40.0 - f["dist_goal_m"]) / 30.0, 0.0), 1.0)
    restart = 1.0 if (f["center_restart"] >= 1.0 or f["kickoff_curve"] >= 0.05) else 0.0
    return (
        cfg.w_goal * f["goal"]
        + cfg.w_shot * f["shot"]
        + cfg.w_audio * min(f["audio_sustained"], 15.0)
        + cfg.w_restart * restart
        + cfg.w_passes * min(f["n_passes"], 10.0)
        + cfg.w_drive * min(f["drive_m"], 40.0)
        + cfg.w_burst * min(max(n_shots_60 - 1, 0), 3)
        + cfg.w_near_goal * near_goal
    )


# --------------------------------------------------------------------------- #
#  Todo junto
# --------------------------------------------------------------------------- #

def build_candidates(sig: MatchSignals, cfg: "HighlightsConfig") -> list[Candidate]:
    anchors = find_anchors(sig, cfg)
    groups = merge_anchors(sig, anchors, cfg.anchor_merge_s)
    logger.info("Anclas: %d (%d grupos)", len(anchors), len(groups))
    shot_times = [g["anchor_s"] for g in groups if "shot" in g["sources"] or "event:shot" in g["sources"]]
    out: list[Candidate] = []
    for k, g in enumerate(groups):
        a = g["anchor_s"]
        team = team_at(sig, a - 2.0, a + 0.5)
        start, chain = play_start(sig, a, team, cfg)
        if team < 0 and chain:
            teams = [c["team"] for c in chain if c["team"] >= 0]
            team = max(set(teams), key=teams.count) if teams else -1
        next_anchor = groups[k + 1]["anchor_s"] - 8.0 if k + 1 < len(groups) else sig.duration_s
        feats = compute_features(sig, a, start, chain, team, next_anchor)
        n_shots_60 = sum(1 for s in shot_times if abs(s - a) <= 60.0)
        feats["n_shots_60"] = float(n_shots_60)
        for src, strength in g["strength"].items():
            feats[f"src_{src.replace(':', '_')}"] = float(strength)
        out.append(Candidate(
            id=f"cand-{k:03d}", anchor_s=a, start_s=start, end_s=provisional_end(sig, a, cfg), team=team,
            sources=g["sources"], features=feats, rule_score=rule_score(feats, n_shots_60, cfg),
        ))
    out.sort(key=lambda c: -c.rule_score)
    return out


def make_goal_candidate(sig: MatchSignals, anchor_s: float, cand_id: str, cfg: "HighlightsConfig") -> Candidate:
    """Candidato para un gol que el marcador confirma pero ningún detector
    marcó: el ancla es el momento en que cambia el rótulo (menos unos
    segundos) y la jugada se reconstruye hacia atrás como de costumbre."""
    team = team_at(sig, anchor_s - 2.0, anchor_s + 0.5)
    start, chain = play_start(sig, anchor_s, team, cfg)
    feats = compute_features(sig, anchor_s, start, chain, team, anchor_s + 60.0)
    feats["n_shots_60"] = 1.0
    return Candidate(
        id=cand_id, anchor_s=anchor_s, start_s=start, end_s=provisional_end(sig, anchor_s, cfg), team=team,
        sources=["scoreboard"], features=feats, rule_score=rule_score(feats, 1, cfg),
    )
