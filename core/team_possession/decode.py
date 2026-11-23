"""
core/team_possession/decode.py
==============================
Suavizado y decodificación de secuencias de etiquetas H/A/D por fotograma.

La métrica del reto (``spell_f1_mean``) cuenta posesiones como objetos: partir
una posesión larga en dos cuesta lo mismo que fallar una corta, y un cambio
adelantado o retrasado un segundo cuesta ~0,35. Por eso el decodificador pesa
más que el acierto por fotograma (ver ``docs/investigacion/reports/``):

* :func:`majority_smooth`: voto mayoritario en ventana (la línea base del reto).
* :func:`min_segment_filter`: elimina tramos más cortos que ``k`` sustituyéndolos
  por la moda de su entorno (Li & Link 2026).
* :func:`viterbi`: decodificación con costes de transición sobre log-evidencias
  por fotograma; con :func:`expand_min_duration` impone duración mínima por
  estado (HSMM aproximado por expansión de estados).

Todo se aplica **dentro de cada tramo** (``runs``: periodos) y nunca a través
del descanso.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .labels import A, D, H, X

Runs = Sequence[tuple[int, int]]


def _runs_or_all(runs: Optional[Runs], n: int) -> list[tuple[int, int]]:
    return list(runs) if runs else [(0, n)]


def apply_per_run(fn: Callable[[np.ndarray], np.ndarray], codes: np.ndarray,
                  runs: Optional[Runs] = None) -> np.ndarray:
    out = np.asarray(codes).copy()
    for start, stop in _runs_or_all(runs, len(out)):
        if stop > start:
            out[start:stop] = fn(out[start:stop])
    return out


# ------------------------------------------------------------------ suavizados
def majority_smooth(codes: np.ndarray, window: int, runs: Optional[Runs] = None,
                    dead_votes: bool = False) -> np.ndarray:
    """Voto mayoritario en ``[i - window, i + window]``.

    Con ``dead_votes=False`` reproduce la línea base del reto: solo votan H y A,
    empate para H, y D únicamente si nadie vota. Con ``dead_votes=True`` D vota
    como una clase más. X nunca vota.
    """
    if window <= 0:
        return np.asarray(codes).copy()

    def _one(c: np.ndarray) -> np.ndarray:
        n = len(c)
        classes = (H, A, D) if dead_votes else (H, A)
        cums = [np.concatenate(([0], np.cumsum(c == k))) for k in classes]
        lo = np.clip(np.arange(n) - window, 0, n)
        hi = np.clip(np.arange(n) + window + 1, 0, n)
        counts = np.stack([cm[hi] - cm[lo] for cm in cums], axis=1)  # [n, K]
        best = counts.argmax(axis=1)  # empate: primera clase (H)
        out = np.asarray(classes, dtype=np.int8)[best]
        out[counts.max(axis=1) == 0] = D
        return out

    return apply_per_run(_one, codes, runs)


def min_segment_filter(codes: np.ndarray, min_len: int, runs: Optional[Runs] = None,
                       max_passes: int = 5, classes: Sequence[int] = (H, A, D)) -> np.ndarray:
    """Sustituye los tramos más cortos que ``min_len`` por la moda de su entorno.

    Entorno = ``min_len`` fotogramas a cada lado del tramo, sin contar el propio
    tramo. Se repite hasta que no cambie nada o ``max_passes`` veces.
    """
    if min_len <= 1:
        return np.asarray(codes).copy()

    def _one(c: np.ndarray) -> np.ndarray:
        c = c.copy()
        n = len(c)
        for _ in range(max_passes):
            changed = False
            breaks = np.flatnonzero(c[1:] != c[:-1]) + 1
            bounds = np.concatenate(([0], breaks, [n]))
            for i in range(len(bounds) - 1):
                s, e = int(bounds[i]), int(bounds[i + 1])
                if e - s >= min_len:
                    continue
                ctx = np.concatenate([c[max(0, s - min_len):s], c[e:min(n, e + min_len)]])
                ctx = ctx[ctx != X]
                if len(ctx) == 0:
                    continue
                counts = [(int((ctx == k).sum()), -idx) for idx, k in enumerate(classes)]
                best = classes[-max(counts)[1]]
                if best != c[s]:
                    c[s:e] = best
                    changed = True
            if not changed:
                break
        return c

    return apply_per_run(_one, codes, runs)


# ------------------------------------------------------------------- viterbi
def viterbi(log_emit: np.ndarray, log_trans: np.ndarray,
            log_init: Optional[np.ndarray] = None) -> np.ndarray:
    """Secuencia de estados más probable. ``log_emit`` [T, K], ``log_trans`` [K, K]."""
    log_emit = np.asarray(log_emit, dtype=np.float64)
    T, K = log_emit.shape
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    log_trans = np.asarray(log_trans, dtype=np.float64)
    score = (np.zeros(K) if log_init is None else np.asarray(log_init, dtype=np.float64)) + log_emit[0]
    back = np.zeros((T, K), dtype=np.int32)
    ar = np.arange(K)
    for t in range(1, T):
        cand = score[:, None] + log_trans  # [K_prev, K]
        bt = cand.argmax(axis=0)
        back[t] = bt
        score = cand[bt, ar] + log_emit[t]
    states = np.zeros(T, dtype=np.int64)
    states[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states


def expand_min_duration(log_emit: np.ndarray, log_trans: np.ndarray, min_dur: int
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expande K estados en K·min_dur sub-estados para imponer duración mínima.

    Dentro de un estado solo se avanza de sub-estado en sub-estado (coste 0); el
    último sub-estado puede quedarse (coste 0) o saltar al primer sub-estado de
    otro estado con ``log_trans[k, j]``. Devuelve ``(log_emit_x, log_trans_x,
    to_state)`` donde ``to_state`` mapea sub-estado -> estado original.
    """
    T, K = log_emit.shape
    m = max(1, int(min_dur))
    S = K * m
    neg = -1e18
    trans = np.full((S, S), neg)
    for k in range(K):
        base = k * m
        for i in range(m - 1):
            trans[base + i, base + i + 1] = 0.0
        last = base + m - 1
        trans[last, last] = 0.0
        for j in range(K):
            if j != k:
                trans[last, j * m] = log_trans[k, j]
    to_state = np.repeat(np.arange(K), m)
    return log_emit[:, to_state], trans, to_state


def viterbi_decode(log_emit: np.ndarray, switch_cost: float | np.ndarray = 5.0,
                   min_dur: int = 1, runs: Optional[Runs] = None,
                   allowed: Optional[np.ndarray] = None) -> np.ndarray:
    """Viterbi de 3 estados (H, A, D) por tramo con coste de cambio y duración mínima.

    ``switch_cost`` es un escalar (mismo coste para todo cambio) o una matriz
    [3, 3] de costes positivos; ``allowed`` [3, 3] bool prohíbe transiciones.
    """
    log_emit = np.asarray(log_emit, dtype=np.float64)
    K = log_emit.shape[1]
    if np.isscalar(switch_cost):
        cost = np.full((K, K), float(switch_cost))
        np.fill_diagonal(cost, 0.0)
    else:
        cost = np.asarray(switch_cost, dtype=np.float64)
    log_trans = -cost
    if allowed is not None:
        log_trans = np.where(np.asarray(allowed, dtype=bool), log_trans, -1e18)
    out = np.full(len(log_emit), X, dtype=np.int8)
    for start, stop in _runs_or_all(runs, len(log_emit)):
        if stop <= start:
            continue
        emit = log_emit[start:stop]
        if min_dur > 1:
            emit_x, trans_x, to_state = expand_min_duration(emit, log_trans, min_dur)
            init = np.full(emit_x.shape[1], -1e18)
            init[np.arange(K) * min_dur] = 0.0  # se empieza en el primer sub-estado
            states = to_state[viterbi(emit_x, trans_x, init)]
        else:
            states = viterbi(emit, log_trans)
        out[start:stop] = states.astype(np.int8)
    return out
