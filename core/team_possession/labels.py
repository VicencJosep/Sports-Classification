"""
core/team_possession/labels.py
==============================
Códigos de etiqueta de posesión de equipo y utilidades de segmentos.

Los códigos son los del scorer oficial del reto (``spell_metrics``):
H=0 (local con balón en juego), A=1 (visitante), D=2 (balón muerto),
X=3 (sin respuesta / sin verdad).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

H, A, D, X = 0, 1, 2, 3
LABELS = ("H", "A", "D", "X")
_CODE_OF = {s: i for i, s in enumerate(LABELS)}


def codes_from_strings(labels) -> np.ndarray:
    arr = np.asarray(labels, dtype=object)
    out = np.full(len(arr), X, dtype=np.int8)
    for s, code in _CODE_OF.items():
        out[arr == s] = code
    return out


def strings_from_codes(codes: np.ndarray) -> np.ndarray:
    return np.asarray(LABELS, dtype=object)[np.asarray(codes, dtype=np.int64)]


def write_submission_csv(path: Path | str, codes: np.ndarray, n_frames: int | None = None) -> Path:
    """Escribe ``frame,label`` desde el fotograma 0, el formato de entrega del reto.

    Si ``n_frames`` supera la longitud de ``codes``, rellena con X.
    """
    codes = np.asarray(codes, dtype=np.int8)
    if n_frames is not None and n_frames > len(codes):
        codes = np.concatenate([codes, np.full(n_frames - len(codes), X, dtype=np.int8)])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("frame,label\n")
        for i, s in enumerate(strings_from_codes(codes)):
            f.write(f"{i},{s}\n")
    return path


def segments(codes: np.ndarray) -> list[tuple[int, int, int]]:
    """``[(start, stop, code)]`` tramos de etiqueta constante, ``stop`` exclusivo."""
    codes = np.asarray(codes)
    if len(codes) == 0:
        return []
    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(codes)]))
    return [(int(bounds[i]), int(bounds[i + 1]), int(codes[bounds[i]])) for i in range(len(bounds) - 1)]


def spells(codes: np.ndarray) -> list[tuple[int, int, int]]:
    """Posesiones: tramos maximales de H o de A (los D son paradas, no posesiones)."""
    return [s for s in segments(codes) if s[2] in (H, A)]


def describe(codes: np.ndarray, fps: float = 25.0) -> str:
    sp = spells(codes)
    durs = [(b - a) / fps for a, b, _ in sp]
    counts = {LABELS[k]: int((codes == k).sum()) for k in range(4)}
    mean_d = float(np.mean(durs)) if durs else 0.0
    return f"{len(sp)} posesiones, duración media {mean_d:.1f} s, fotogramas {counts}"


def merge_short_spells(codes: np.ndarray, min_len: int, max_passes: int = 10) -> np.ndarray:
    """Funde en la posesión anterior las posesiones H/A más cortas que ``min_len``.

    Sirve para pasar una verdad "por toque" (IDSSE: mediana 3 s) a la semántica
    de control del reto (14 s de media en la verdad pública): un toque del
    rival que no retiene el balón no es cambio de posesión. Solo se funde si el
    tramo anterior inmediato es una posesión (no D); los tramos cortos tras una
    parada se quedan. Se repite hasta que no cambie nada.
    """
    out = np.asarray(codes).copy()
    for _ in range(max_passes):
        changed = False
        segs = segments(out)
        for i, (a, b, c) in enumerate(segs):
            if c not in (H, A) or b - a >= min_len or i == 0:
                continue
            pa, pb, pc = segs[i - 1]
            if pc in (H, A) and pc != c:
                out[a:b] = pc
                changed = True
        if not changed:
            break
    return out
