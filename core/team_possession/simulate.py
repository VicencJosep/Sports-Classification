"""
core/team_possession/simulate.py
================================
Degradación controlada de un :class:`PositionalData` limpio (IDSSE, con balón
siempre visible) para parecerse a la entrada del reto y a nuestro tracking:

* :func:`drop_ball`: borra el balón en una fracción de fotogramas, en rachas y
  preferentemente donde el reto dice que se pierde (balón alto, zonas pobladas).
* :func:`anonymize_slots`: reparte cada jugador en varios tracklets anónimos
  (cortes al azar), conservando el equipo, como los 46 slots del ATD.

Con la misma semilla la degradación es reproducible, así que las variantes se
comparan sobre la misma entrada.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .frames import PositionalData


def drop_ball(data: PositionalData, frac: float = 0.41, seed: int = 0,
              mean_run_s: float = 1.2, high_z_m: float = 1.5, crowd_radius_m: float = 3.0,
              crowd_n: int = 3, bias: float = 3.0) -> PositionalData:
    """Devuelve una copia con el balón ausente en ~``frac`` de los fotogramas.

    Se siembran rachas de longitud geométrica (media ``mean_run_s``) en
    fotogramas elegidos con probabilidad ``bias`` veces mayor cuando el balón
    está alto (``ball_z`` > ``high_z_m``) o hay ``crowd_n`` o más jugadores a
    menos de ``crowd_radius_m``.
    """
    rng = np.random.default_rng(seed)
    T = data.n_frames
    fps = data.fps
    obs = data.ball_observed()
    weight = np.ones(T, dtype=np.float64)
    if data.ball_z is not None:
        weight[np.isfinite(data.ball_z) & (data.ball_z > high_z_m)] *= bias
    diff = data.players_xy - data.ball_xy[:, None, :]
    dist = np.linalg.norm(diff, axis=2)
    crowded = (np.where(np.isfinite(dist), dist, np.inf) <= crowd_radius_m).sum(axis=1) >= crowd_n
    weight[crowded] *= bias
    weight[~obs] = 0.0
    weight /= weight.sum()
    target = int(frac * obs.sum())
    missing = ~obs.copy()
    mean_run = max(1, int(mean_run_s * fps))
    n_seeds = 0
    while missing.sum() < target + (~obs).sum() and n_seeds < 10 * T:
        start = int(rng.choice(T, p=weight))
        length = int(rng.geometric(1.0 / mean_run))
        missing[start:start + length] = True
        n_seeds += 1
    ball = data.ball_xy.copy()
    ball[missing] = np.nan
    ball_z = None if data.ball_z is None else np.where(missing, np.nan, data.ball_z).astype(np.float32)
    return replace(data, ball_xy=ball, ball_z=ball_z, name=f"{data.name}_drop{int(frac * 100)}")


def anonymize_slots(data: PositionalData, cuts_per_minute: float = 1.5, seed: int = 0,
                    n_slots: int | None = None) -> PositionalData:
    """Corta cada trayectoria en tracklets anónimos y los reparte en slots.

    Cada jugador recibe cortes de Poisson a ``cuts_per_minute``; cada trozo va a
    un slot libre del mismo equipo (el equipo se conserva). ``n_slots`` limita
    el número de slots (por defecto 2 x jugadores, como el ATD del reto).
    """
    rng = np.random.default_rng(seed)
    T, N = data.n_frames, data.n_slots
    n_out = n_slots or 2 * N
    out_xy = np.full((T, n_out, 2), np.nan, dtype=np.float32)
    out_team = np.full(n_out, -1, dtype=np.int8)
    free_until = np.zeros(n_out, dtype=np.int64)  # fotograma hasta el que el slot está ocupado
    for i in range(N):
        present = np.flatnonzero(np.isfinite(data.players_xy[:, i]).all(axis=1))
        if len(present) == 0:
            continue
        n_cuts = rng.poisson(cuts_per_minute * len(present) / (60 * data.fps))
        cuts = np.sort(rng.choice(present, size=min(n_cuts, len(present) - 1), replace=False)) if n_cuts else []
        bounds = [present[0]] + [int(c) for c in cuts] + [present[-1] + 1]
        for a, b in zip(bounds[:-1], bounds[1:]):
            cands = [s for s in range(n_out) if free_until[s] <= a and out_team[s] in (-1, data.team[i])]
            s = cands[0] if cands else int(rng.integers(n_out))
            out_team[s] = data.team[i]
            out_xy[a:b, s] = data.players_xy[a:b, i]
            free_until[s] = b
    return replace(data, players_xy=out_xy, team=out_team,
                   player_ids=[f"slot{k}" for k in range(n_out)], name=f"{data.name}_anon")
