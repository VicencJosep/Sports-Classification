"""
core/team_possession/features.py
================================
Rasgos por fotograma para decidir posesión de equipo y balón muerto a partir
de un :class:`PositionalData`. Todo son arrays alineados con los fotogramas y
calculados **dentro de cada periodo** (nunca a través del descanso).

Los rasgos siguen la literatura (Link & Hoernig 2017, Vidal-Codina et al. 2022,
Lang et al. 2022): distancia balón-jugador por equipo, cinemática del balón
(velocidad, aceleración, cambio de dirección) para detectar toques, zona de
duelo, actividad colectiva de los jugadores para detectar paradas, balón fuera
del campo y longitud de los huecos del balón.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import AWAY, HOME, PositionalData


@dataclass
class FeatureConfig:
    vel_step: int = 2  # diferencias centradas a ±step fotogramas (0,08 s a 25 fps)
    smooth_window: int = 5  # media móvil del balón (fotogramas) antes de derivar; 1 = sin suavizar
    duel_radius_m: float = 1.5  # jugadores de ambos equipos a menos de esto = duelo
    out_margin_m: float = 0.3  # margen fuera de las líneas para "balón fuera"
    static_speed_mps: float = 0.5  # balón parado
    min_dir_speed_mps: float = 1.0  # velocidad mínima para medir cambio de dirección


@dataclass
class FrameFeatures:
    fps: float
    ball_obs: np.ndarray  # [T] bool
    ball_vel: np.ndarray  # [T, 2] m/s (NaN si no calculable)
    ball_speed: np.ndarray  # [T]
    ball_accel: np.ndarray  # [T] |Δv|/Δt en m/s² (NaN si no calculable)
    ball_dir_change: np.ndarray  # [T] grados entre v(t-s) y v(t+s) (NaN)
    dist: np.ndarray  # [T, N] distancia balón-slot (NaN)
    nearest: np.ndarray  # [T] slot más cercano o -1
    nearest_dist: np.ndarray  # [T] (inf si nadie)
    nearest_team: np.ndarray  # [T] HOME/AWAY/-1
    dist_home: np.ndarray  # [T] distancia mínima de un local al balón (inf)
    dist_away: np.ndarray  # [T]
    n_home_near: np.ndarray  # [T] locales dentro de duel_radius
    n_away_near: np.ndarray  # [T]
    player_speed: np.ndarray  # [T, N] m/s (NaN)
    team_mean_speed: np.ndarray  # [T, 2] media de velocidad de los visibles por equipo (NaN)
    all_mean_speed: np.ndarray  # [T]
    n_visible: np.ndarray  # [T]
    ball_out: np.ndarray  # [T] bool (False si no se observa)
    ball_static: np.ndarray  # [T] bool
    since_obs: np.ndarray  # [T] fotogramas desde la última observación del balón (0 si observado)
    until_obs: np.ndarray  # [T] fotogramas hasta la siguiente observación (0 si observado)
    period_runs: list[tuple[int, int]]


def _central_velocity(xy: np.ndarray, fps: float, step: int, runs: list[tuple[int, int]]) -> np.ndarray:
    """Velocidad por diferencias centradas a ±step, dentro de cada tramo; NaN si falta."""
    T = xy.shape[0]
    vel = np.full(xy.shape, np.nan, dtype=np.float32)
    dt = 2 * step / fps
    for s, e in runs:
        seg = xy[s:e]
        n = e - s
        if n <= 2 * step:
            continue
        v = (seg[2 * step:] - seg[:-2 * step]) / dt
        vel[s + step:e - step] = v
        # bordes: diferencia hacia delante / hacia atrás a 1 paso
        if n > 1:
            vel[s:s + step] = (seg[1:step + 1] - seg[:step]) * fps
            vel[e - step:e] = (seg[-step:] - seg[-step - 1:-1]) * fps
    return vel


def _smooth_runs(xy: np.ndarray, w: int, runs: list[tuple[int, int]]) -> np.ndarray:
    """Media móvil de ventana ``w`` sobre valores observados, por tramo; NaN sigue NaN."""
    if w <= 1:
        return xy
    out = xy.copy()
    half = w // 2
    for s, e in runs:
        seg = xy[s:e]
        valid = np.isfinite(seg).all(axis=1)
        filled = np.where(valid[:, None], seg, 0.0)
        csum = np.concatenate([np.zeros((1, seg.shape[1])), np.cumsum(filled, axis=0)])
        ccnt = np.concatenate([[0], np.cumsum(valid)])
        n = e - s
        lo = np.clip(np.arange(n) - half, 0, n); hi = np.clip(np.arange(n) + half + 1, 0, n)
        cnt = (ccnt[hi] - ccnt[lo])[:, None]
        mean = np.where(cnt > 0, (csum[hi] - csum[lo]) / np.maximum(cnt, 1), np.nan)
        out[s:e] = np.where(valid[:, None], mean, np.nan)
    return out


def _runs_since_until(obs: np.ndarray, runs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    T = len(obs)
    since = np.zeros(T, dtype=np.int32)
    until = np.zeros(T, dtype=np.int32)
    for s, e in runs:
        c = 0
        for t in range(s, e):
            c = 0 if obs[t] else c + 1
            since[t] = c
        c = 0
        for t in range(e - 1, s - 1, -1):
            c = 0 if obs[t] else c + 1
            until[t] = c
    return since, until


def compute_features(data: PositionalData, cfg: FeatureConfig | None = None) -> FrameFeatures:
    cfg = cfg or FeatureConfig()
    runs = data.period_runs()
    T, N = data.n_frames, data.n_slots
    fps = data.fps
    ball = data.ball_xy.astype(np.float32)
    ball_obs = np.isfinite(ball).all(axis=1)

    # cinemática del balón: solo entre observaciones consecutivas dentro del tramo
    ball_s = _smooth_runs(ball, cfg.smooth_window, runs)
    ball_vel = _central_velocity(ball_s, fps, cfg.vel_step, runs)
    ball_speed = np.linalg.norm(ball_vel, axis=1)
    accel_vec = _central_velocity(ball_vel, fps, cfg.vel_step, runs)
    ball_accel = np.linalg.norm(accel_vec, axis=1)
    s = cfg.vel_step
    dir_change = np.full(T, np.nan, dtype=np.float32)
    for a, b in runs:
        if b - a <= 2 * s:
            continue
        v0 = ball_vel[a:b - 2 * s]; v1 = ball_vel[a + 2 * s:b]
        n0 = np.linalg.norm(v0, axis=1); n1 = np.linalg.norm(v1, axis=1)
        ok = (n0 > cfg.min_dir_speed_mps) & (n1 > cfg.min_dir_speed_mps)
        cosang = np.full(len(n0), np.nan, dtype=np.float32)
        cosang[ok] = np.clip((v0[ok] * v1[ok]).sum(1) / (n0[ok] * n1[ok]), -1, 1)
        dir_change[a + s:b - s] = np.degrees(np.arccos(cosang))

    # distancias balón-jugador
    diff = data.players_xy - ball[:, None, :]
    dist = np.linalg.norm(diff, axis=2)  # NaN donde falte alguno
    dist_f = np.where(np.isfinite(dist), dist, np.inf)
    nearest = dist_f.argmin(axis=1)
    nearest_dist = dist_f[np.arange(T), nearest]
    nearest = np.where(np.isfinite(nearest_dist), nearest, -1)
    nearest_team = np.where(nearest >= 0, data.team[np.clip(nearest, 0, N - 1)], -1).astype(np.int8)
    home_mask = data.team == HOME
    away_mask = data.team == AWAY
    dist_home = dist_f[:, home_mask].min(axis=1) if home_mask.any() else np.full(T, np.inf)
    dist_away = dist_f[:, away_mask].min(axis=1) if away_mask.any() else np.full(T, np.inf)
    n_home_near = (dist_f[:, home_mask] <= cfg.duel_radius_m).sum(axis=1)
    n_away_near = (dist_f[:, away_mask] <= cfg.duel_radius_m).sum(axis=1)

    # actividad de los jugadores
    pvel = _central_velocity(data.players_xy.reshape(T, -1), fps, cfg.vel_step, runs).reshape(T, N, 2)
    player_speed = np.linalg.norm(pvel, axis=2)
    visible = np.isfinite(player_speed)
    n_visible = np.isfinite(data.players_xy).all(axis=2).sum(axis=1)
    with np.errstate(invalid="ignore"):
        team_mean_speed = np.stack([
            np.nanmean(np.where(visible & home_mask[None, :], player_speed, np.nan), axis=1),
            np.nanmean(np.where(visible & away_mask[None, :], player_speed, np.nan), axis=1),
        ], axis=1)
        all_mean_speed = np.nanmean(np.where(visible, player_speed, np.nan), axis=1)

    m = cfg.out_margin_m
    ball_out = ball_obs & ((ball[:, 0] < -m) | (ball[:, 0] > data.pitch_length + m)
                           | (ball[:, 1] < -m) | (ball[:, 1] > data.pitch_width + m))
    ball_static = ball_obs & np.isfinite(ball_speed) & (ball_speed < cfg.static_speed_mps)
    since_obs, until_obs = _runs_since_until(ball_obs, runs)

    return FrameFeatures(
        fps=fps, ball_obs=ball_obs, ball_vel=ball_vel, ball_speed=ball_speed, ball_accel=ball_accel,
        ball_dir_change=dir_change, dist=dist, nearest=nearest, nearest_dist=nearest_dist,
        nearest_team=nearest_team, dist_home=dist_home, dist_away=dist_away,
        n_home_near=n_home_near, n_away_near=n_away_near, player_speed=player_speed,
        team_mean_speed=team_mean_speed, all_mean_speed=all_mean_speed, n_visible=n_visible,
        ball_out=ball_out, ball_static=ball_static, since_obs=since_obs, until_obs=until_obs,
        period_runs=runs,
    )


def run_lengths(mask: np.ndarray, runs: list[tuple[int, int]]) -> np.ndarray:
    """[T] longitud acumulada de la racha de ``True`` que llega hasta t (0 si False)."""
    out = np.zeros(len(mask), dtype=np.int32)
    for s, e in runs:
        c = 0
        for t in range(s, e):
            c = c + 1 if mask[t] else 0
            out[t] = c
    return out


def window_max(x: np.ndarray, k: int, runs: list[tuple[int, int]]) -> np.ndarray:
    """Máximo deslizante en ``[t-k, t+k]`` dentro de cada tramo (NaN se ignora)."""
    out = np.full(len(x), np.nan, dtype=np.float32)
    for s, e in runs:
        seg = np.where(np.isfinite(x[s:e]), x[s:e], -np.inf)
        n = e - s
        if n == 0:
            continue
        res = np.full(n, -np.inf)
        for d in range(-k, k + 1):
            lo, hi = max(0, -d), min(n, n - d)
            if hi > lo:
                res[lo:hi] = np.maximum(res[lo:hi], seg[lo + d:hi + d])
        out[s:e] = np.where(np.isfinite(res), res, np.nan)
    return out


# ------------------------------------------------------- matriz para aprender
def _prev_next_observed(obs: np.ndarray, runs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """[T] índice de la última / siguiente observación del balón dentro del tramo (-1 si no hay)."""
    T = len(obs)
    prev = np.full(T, -1, dtype=np.int64)
    nxt = np.full(T, -1, dtype=np.int64)
    idx = np.arange(T)
    for s, e in runs:
        o = obs[s:e]
        p = np.where(o, idx[s:e], -1)
        prev[s:e] = np.maximum.accumulate(p)
        n = np.where(o, idx[s:e], T + 1)
        nxt[s:e] = np.minimum.accumulate(n[::-1])[::-1]
    nxt[nxt > T] = -1
    return prev, nxt


FEATURE_NAMES = [
    "ball_obs", "since_obs_s", "until_obs_s", "gap_s",
    "dist_home", "dist_away", "dist_diff", "n_home_near", "n_away_near",
    "ball_speed", "ball_out", "ball_static", "ball_x_rel", "ball_y_rel",
    "prev_dist_home", "prev_dist_away", "prev_ago_s", "next_dist_home", "next_dist_away", "next_in_s",
    "home_mean_speed", "away_mean_speed", "all_mean_speed", "n_visible",
    "home_cx_rel", "away_cx_rel", "home_spread", "away_spread",
    "low_act_run_s", "static_far_run_s", "t_from_start_s", "t_to_end_s",
    "n_home_5m", "n_away_5m", "gap_jump_m",
]


CONTEXT_COLS = ["ball_obs", "dist_home", "dist_away", "n_home_near", "n_away_near", "ball_speed",
                "all_mean_speed", "ball_out", "ball_static"]
CONTEXT_OFFSETS_S = (0.5, 1.0, 2.0)


def feature_names(context_offsets_s=CONTEXT_OFFSETS_S) -> list[str]:
    names = list(FEATURE_NAMES)
    for off in context_offsets_s:
        for sign in ("-", "+"):
            names += [f"{c}@{sign}{off:g}s" for c in CONTEXT_COLS]
    return names


def _shift_within_runs(X: np.ndarray, k: int, runs: list[tuple[int, int]]) -> np.ndarray:
    """Copia de X desplazada k fotogramas (k>0: valor de t-k; k<0: de t+k), por tramo, repitiendo el borde."""
    out = np.empty_like(X)
    for s, e in runs:
        n = e - s
        idx = np.clip(np.arange(n) - k, 0, n - 1)
        out[s:e] = X[s:e][idx]
    return out


def feature_matrix(f: FrameFeatures, data: PositionalData, low_activity_mps: float = 1.2,
                   static_far_m: float = 2.0, cap_s: float = 10.0, cap_m: float = 30.0,
                   context_offsets_s=CONTEXT_OFFSETS_S) -> np.ndarray:
    """[T, len(feature_names())] float32 para un clasificador por fotograma.

    Incluye rasgos "hacia atrás" y "hacia delante" (última y siguiente
    observación del balón, tiempo hasta el final del periodo): el problema es
    offline, así que mirar al futuro es legítimo y es lo que puentea los huecos.
    ``context_offsets_s`` añade copias de ``CONTEXT_COLS`` desplazadas ±esos
    segundos (contexto temporal para el clasificador).
    """
    T = data.n_frames
    fps = data.fps
    runs = f.period_runs
    prev, nxt = _prev_next_observed(f.ball_obs, runs)
    idx = np.arange(T)
    dh = np.minimum(np.where(np.isfinite(f.dist_home), f.dist_home, cap_m), cap_m)
    da = np.minimum(np.where(np.isfinite(f.dist_away), f.dist_away, cap_m), cap_m)
    has_prev = prev >= 0
    has_next = nxt >= 0
    pi = np.where(has_prev, prev, 0)
    ni = np.where(has_next, nxt, 0)
    # posición del balón: la última observada (o la siguiente al inicio del tramo)
    src = np.where(has_prev, pi, ni)
    bx = data.ball_xy[src, 0] / data.pitch_length
    by = data.ball_xy[src, 1] / data.pitch_width
    bx = np.where(np.isfinite(bx), bx, 0.5); by = np.where(np.isfinite(by), by, 0.5)
    home = data.team == 0
    away = data.team == 1
    with np.errstate(invalid="ignore"):
        hx = np.nanmean(data.players_xy[:, home, 0], axis=1) / data.pitch_length
        ax = np.nanmean(data.players_xy[:, away, 0], axis=1) / data.pitch_length
        hs = np.nanmean(np.nanstd(data.players_xy[:, home, :], axis=1), axis=1)
        as_ = np.nanmean(np.nanstd(data.players_xy[:, away, :], axis=1), axis=1)
    low_act = np.isfinite(f.all_mean_speed) & (f.all_mean_speed < low_activity_mps)
    static_far = f.ball_static & (np.minimum(f.dist_home, f.dist_away) > static_far_m)
    t_from = np.zeros(T); t_to = np.zeros(T)
    for s, e in runs:
        t_from[s:e] = (idx[s:e] - s) / fps
        t_to[s:e] = (e - 1 - idx[s:e]) / fps
    # jugadores de cada equipo a menos de 5 m de la última posición conocida del balón, y cuánto
    # se desplaza el balón entre la última y la siguiente observación (huecos largos = pase o despeje)
    ref_xy = data.ball_xy[src]
    dref = np.linalg.norm(data.players_xy - ref_xy[:, None, :], axis=2)
    dref = np.where(np.isfinite(dref), dref, np.inf)
    n_home_5 = (dref[:, home] <= 5.0).sum(axis=1).astype(np.float32)
    n_away_5 = (dref[:, away] <= 5.0).sum(axis=1).astype(np.float32)
    jump = np.where(has_prev & has_next, np.linalg.norm(data.ball_xy[ni] - data.ball_xy[pi], axis=1), 0.0)
    jump = np.minimum(np.nan_to_num(jump, nan=0.0), cap_m)
    cols = [
        f.ball_obs.astype(np.float32), np.minimum(f.since_obs / fps, cap_s), np.minimum(f.until_obs / fps, cap_s),
        np.minimum(np.where(f.ball_obs, 0, f.since_obs + f.until_obs) / fps, 2 * cap_s),
        dh, da, dh - da, f.n_home_near, f.n_away_near,
        np.where(np.isfinite(f.ball_speed), f.ball_speed, -1.0), f.ball_out.astype(np.float32),
        f.ball_static.astype(np.float32), bx, by,
        np.where(has_prev, dh[pi], cap_m), np.where(has_prev, da[pi], cap_m),
        np.where(has_prev, np.minimum((idx - pi) / fps, cap_s), cap_s),
        np.where(has_next, dh[ni], cap_m), np.where(has_next, da[ni], cap_m),
        np.where(has_next, np.minimum((ni - idx) / fps, cap_s), cap_s),
        np.nan_to_num(f.team_mean_speed[:, 0], nan=-1.0), np.nan_to_num(f.team_mean_speed[:, 1], nan=-1.0),
        np.nan_to_num(f.all_mean_speed, nan=-1.0), f.n_visible.astype(np.float32),
        np.nan_to_num(hx, nan=0.5), np.nan_to_num(ax, nan=0.5), np.nan_to_num(hs, nan=0.0), np.nan_to_num(as_, nan=0.0),
        run_lengths(low_act, runs) / fps, run_lengths(static_far, runs) / fps,
        np.minimum(t_from, 10 * cap_s), np.minimum(t_to, 10 * cap_s),
        n_home_5, n_away_5, jump,
    ]
    X = np.stack([np.asarray(c, dtype=np.float32) for c in cols], axis=1)
    assert X.shape[1] == len(FEATURE_NAMES), (X.shape, len(FEATURE_NAMES))
    if context_offsets_s:
        ci = [FEATURE_NAMES.index(c) for c in CONTEXT_COLS]
        base = X[:, ci]
        extra = []
        for off in context_offsets_s:
            k = int(round(off * fps))
            extra.append(_shift_within_runs(base, k, runs))
            extra.append(_shift_within_runs(base, -k, runs))
        X = np.concatenate([X] + extra, axis=1)
    return X
