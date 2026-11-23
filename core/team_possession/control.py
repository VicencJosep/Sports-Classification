"""
core/team_possession/control.py
===============================
Evidencia de posesión de equipo y de balón muerto por fotograma, por reglas.

Dos salidas a partir de los rasgos de ``features``:

* :func:`rule_labels`: máquina de estados hacia delante con la semántica del
  reto (Link & Hoernig 2017 / Vidal-Codina 2022): el equipo en posesión es el
  **último control válido**; el rival solo gana cuando uno de sus jugadores
  está en la zona de posesión y el balón muestra un cambio cinemático
  atribuible (o permanece junto a él ``touch_frames``), y nunca durante un
  duelo (jugadores de ambos equipos a menos de ``duel_radius_m``). Balón suelto
  o sin observar = sigue siendo del último poseedor.
* :func:`emission_scores`: evidencias blandas ``[T, 3]`` (H, A, D) para el
  Viterbi de ``decode``: combinan cercanía por equipo, duelo, balón fuera,
  balón parado sin nadie cerca y actividad colectiva baja.

También :func:`nearest_labels`, la línea base del reto (equipo del jugador más
cercano, D si no hay balón), para comparar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FrameFeatures, run_lengths, window_max
from .frames import AWAY, HOME
from .labels import A, D, H, X


@dataclass
class ControlConfig:
    pz_radius_m: float = 1.5  # zona de posesión (databallpy: 1,5 m; FIFA: 0,5-1 m)
    duel_radius_m: float = 1.5
    touch_frames: int = 3  # rival pegado al balón este tiempo = control aunque no haya cambio cinemático
    kin_window: int = 3  # ±fotogramas donde buscar el cambio cinemático
    accel_mps2: float = 4.0  # Link & Hoernig: pico de aceleración >= 4 m/s²
    dir_change_deg: float = 10.0  # Vidal-Codina: cambio de dirección > 10°
    speed_change_mps: float = 5.0  # o cambio de velocidad > 5 m/s (en kin_window)
    # balón muerto
    static_far_s: float = 1.5  # balón parado y nadie a < static_far_m durante este tiempo
    static_far_m: float = 2.0
    low_activity_mps: float = 1.2  # media de velocidad de los jugadores por debajo = parados
    low_activity_s: float = 2.0
    missing_long_s: float = 4.0  # balón sin observar más de esto: evidencia débil de D
    # pesos de las evidencias blandas
    w_near: float = 3.0
    w_duel: float = 1.0
    w_out: float = 6.0
    w_static: float = 3.0
    w_activity: float = 2.0
    w_missing: float = 0.5
    w_dead_prior: float = -0.5  # sesgo de D cuando no hay evidencia (negativo = favorece juego)
    # puente en los huecos del balón: evidencia del equipo más cercano en la última y la
    # siguiente observación, con peso que decae con la distancia temporal al hueco
    bridge: bool = True
    w_bridge: float = 2.0
    bridge_tau_s: float = 3.0


def kinematic_change(f: FrameFeatures, cfg: ControlConfig) -> np.ndarray:
    """[T] bool: en ``[t-k, t+k]`` el balón acelera, gira o cambia de velocidad lo bastante."""
    k = cfg.kin_window
    acc = window_max(f.ball_accel, k, f.period_runs)
    turn = window_max(f.ball_dir_change, k, f.period_runs)
    sp_max = window_max(f.ball_speed, k, f.period_runs)
    sp_min = -window_max(-f.ball_speed, k, f.period_runs)
    change = np.zeros(len(acc), dtype=bool)
    change |= np.isfinite(acc) & (acc >= cfg.accel_mps2)
    change |= np.isfinite(turn) & (turn >= cfg.dir_change_deg)
    change |= np.isfinite(sp_max) & np.isfinite(sp_min) & ((sp_max - sp_min) >= cfg.speed_change_mps)
    return change


def dead_ball_evidence(f: FrameFeatures, cfg: ControlConfig) -> dict[str, np.ndarray]:
    """Señales booleanas de balón muerto por fotograma."""
    fps = f.fps
    far = np.minimum(f.dist_home, f.dist_away) > cfg.static_far_m
    static_far = f.ball_static & far
    static_far_run = run_lengths(static_far, f.period_runs) >= int(cfg.static_far_s * fps)
    low_act = np.isfinite(f.all_mean_speed) & (f.all_mean_speed < cfg.low_activity_mps)
    low_act_run = run_lengths(low_act, f.period_runs) >= int(cfg.low_activity_s * fps)
    missing_long = (f.since_obs > int(cfg.missing_long_s * fps))
    return {"out": f.ball_out, "static_far": static_far_run, "low_activity": low_act_run,
            "missing_long": missing_long}


def nearest_labels(f: FrameFeatures) -> np.ndarray:
    """Línea base del reto: equipo del jugador más cercano al balón; D si no hay balón."""
    out = np.full(len(f.ball_obs), D, dtype=np.int8)
    out[(f.nearest_team == HOME) & f.ball_obs] = H
    out[(f.nearest_team == AWAY) & f.ball_obs] = A
    return out


def rule_labels(f: FrameFeatures, cfg: ControlConfig | None = None,
                dead: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Máquina de estados hacia delante (ver docstring del módulo)."""
    cfg = cfg or ControlConfig()
    dead = dead or dead_ball_evidence(f, cfg)
    is_dead = dead["out"] | dead["static_far"] | dead["low_activity"]
    kin = kinematic_change(f, cfg)
    T = len(f.ball_obs)
    out = np.full(T, D, dtype=np.int8)
    code_of = {HOME: H, AWAY: A}
    for s, e in f.period_runs:
        team = -1
        rival_streak = 0
        for t in range(s, e):
            if is_dead[t]:
                out[t] = D
                rival_streak = 0
                continue
            if f.ball_obs[t] and f.nearest[t] >= 0 and f.nearest_dist[t] <= cfg.pz_radius_m:
                cand = int(f.nearest_team[t])
                duel = f.n_home_near[t] > 0 and f.n_away_near[t] > 0
                if cand in (HOME, AWAY):
                    if team == -1:
                        team = cand
                        rival_streak = 0
                    elif cand == team:
                        rival_streak = 0
                    else:
                        rival_streak += 1
                        if not duel and (kin[t] or rival_streak >= cfg.touch_frames):
                            team = cand
                            rival_streak = 0
                else:
                    rival_streak = 0
            else:
                rival_streak = 0
            out[t] = code_of.get(team, D)
    return out


def emission_scores(f: FrameFeatures, cfg: ControlConfig | None = None,
                    dead: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Evidencias blandas ``[T, 3]`` (log-verosimilitudes no normalizadas) de H, A, D."""
    cfg = cfg or ControlConfig()
    dead = dead or dead_ball_evidence(f, cfg)
    T = len(f.ball_obs)
    s = np.zeros((T, 3), dtype=np.float64)
    r = cfg.pz_radius_m
    near_h = np.clip(1.0 - f.dist_home / r, 0.0, 1.0)
    near_a = np.clip(1.0 - f.dist_away / r, 0.0, 1.0)
    near_h[~np.isfinite(near_h)] = 0.0
    near_a[~np.isfinite(near_a)] = 0.0
    duel = (f.n_home_near > 0) & (f.n_away_near > 0)
    # cercanía por equipo: solo si el balón se observa; en duelo se reparte
    s[:, H] += cfg.w_near * np.where(duel, 0.5 * (near_h + near_a), near_h)
    s[:, A] += cfg.w_near * np.where(duel, 0.5 * (near_h + near_a), near_a)
    s[:, D] += cfg.w_dead_prior
    s[dead["out"], D] += cfg.w_out
    s[dead["static_far"], D] += cfg.w_static
    s[dead["low_activity"], D] += cfg.w_activity
    s[dead["missing_long"], D] += cfg.w_missing
    if cfg.bridge:
        from .features import _prev_next_observed
        prev, nxt = _prev_next_observed(f.ball_obs, f.period_runs)
        idx = np.arange(T)
        gap = ~f.ball_obs
        for ref, dt in ((prev, idx - prev), (nxt, nxt - idx)):
            ok = gap & (ref >= 0)
            w = cfg.w_bridge * np.exp(-dt[ok] / (cfg.bridge_tau_s * f.fps))
            team_ref = f.nearest_team[ref[ok]]
            dist_ref = f.nearest_dist[ref[ok]]
            inside = np.isfinite(dist_ref) & (dist_ref <= 2 * r)
            w = w * inside
            tgt = np.flatnonzero(ok)
            s[tgt[team_ref == HOME], H] += w[team_ref == HOME]
            s[tgt[team_ref == AWAY], A] += w[team_ref == AWAY]
    # con el balón en juego y observado cerca de alguien, D es improbable
    alive_evidence = f.ball_obs & (np.minimum(f.dist_home, f.dist_away) <= r) & ~dead["out"]
    s[alive_evidence, D] -= cfg.w_near
    return s


def log_softmax(scores: np.ndarray) -> np.ndarray:
    m = scores.max(axis=1, keepdims=True)
    z = scores - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))
