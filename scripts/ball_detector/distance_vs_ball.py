"""¿Falla el balón cuando los jugadores están lejos de la cámara?

Cruza, frame a frame, la altura mediana de las cajas de jugador (proxy de distancia/zoom del plano) con si
hay balón detectado y poseedor, leyendo un ``possession.jsonl`` del pipeline (``core/possession.py``). Saca
dos tablas: por tamaño de jugador y por tramo de 5 min. Con ``--compare`` (otro possession.jsonl del MISMO
vídeo, p.ej. el tracking con el balón a 1280 px) añade las columnas del segundo fichero para ver dónde mejora.

Uso (contenedor, cwd /app):
    python scripts/ball_detector/distance_vs_ball.py --possession <video>.possession.jsonl --fps 30 \\
        [--compare <otro>.possession.jsonl] [--stride 5] --out output/ball_experiments/distance_vs_ball.txt
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

BUCKETS = [(0, 25), (25, 35), (35, 45), (45, 60), (60, 80), (80, 10_000)]
PLAYER_ROLES = {"player", "goalkeeper"}


def read_frames(path: Path, stride: int) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            idx = rec["frame_idx"]
            if idx % stride:
                continue
            heights = [t["xyxy"][3] - t["xyxy"][1] for t in rec.get("tracks", []) if t.get("role") in PLAYER_ROLES]
            out[idx] = {
                "n": len(heights),
                "h": statistics.median(heights) if heights else None,
                "ball": rec.get("ball") is not None,
                "poss": rec.get("possessor_tracker_id") is not None,
            }
    return out


def bucket_of(h: float | None) -> str:
    if h is None:
        return "sin jugadores"
    for lo, hi in BUCKETS:
        if lo <= h < hi:
            return f"{lo}-{hi} px" if hi < 10_000 else f">= {lo} px"
    return "?"


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:5.1f}%" if den else "   - "


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--possession", required=True, type=Path)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    A = read_frames(a.possession, a.stride)
    B = read_frames(a.compare, a.stride) if a.compare else None
    lines: list[str] = []
    lines.append(f"A = {a.possession}")
    if B is not None:
        lines.append(f"B = {a.compare}")
    lines.append(f"fps={a.fps:g}, 1 de cada {a.stride} frames ({len(A)} muestreados)")

    def summarize(frames: dict[int, dict], keys) -> tuple[int, int, int]:
        n = ball = poss = 0
        for k in keys:
            r = frames.get(k)
            if r is None:
                continue
            n += 1
            ball += r["ball"]
            poss += r["poss"]
        return n, ball, poss

    # --- por tamaño de jugador (altura mediana de las cajas, en A) ---
    by_bucket: dict[str, list[int]] = defaultdict(list)
    for idx, r in A.items():
        by_bucket[bucket_of(r["h"])].append(idx)
    order = ["sin jugadores"] + [bucket_of((lo + hi) / 2 if hi < 10_000 else lo) for lo, hi in BUCKETS]
    lines.append("")
    lines.append("Por tamaño de jugador (altura mediana de las cajas en A; cuanto más baja, más lejos/más abierto el plano)")
    head = f"{'tamaño':16s} {'frames':>7s} {'% total':>8s} {'balón A':>8s} {'poseedor A':>11s}"
    if B is not None:
        head += f" {'balón B':>8s} {'poseedor B':>11s}"
    lines.append(head)
    total = len(A)
    for name in order:
        keys = by_bucket.get(name, [])
        if not keys:
            continue
        n, ball, poss = summarize(A, keys)
        row = f"{name:16s} {n:7d} {pct(n, total):>8s} {pct(ball, n):>8s} {pct(poss, n):>11s}"
        if B is not None:
            nb, ballb, possb = summarize(B, keys)
            row += f" {pct(ballb, nb):>8s} {pct(possb, nb):>11s}"
        lines.append(row)

    # --- por tramo de 5 min ---
    seg_len = int(300 * a.fps)
    by_seg: dict[int, list[int]] = defaultdict(list)
    for idx in A:
        by_seg[idx // seg_len].append(idx)
    lines.append("")
    lines.append("Por tramo de 5 min (altura mediana de jugador en A, % de frames con balón / con poseedor)")
    head = f"{'tramo':>11s} {'frames':>7s} {'con jug.':>8s} {'alt. med.':>9s} {'balón A':>8s} {'poseedor A':>11s}"
    if B is not None:
        head += f" {'balón B':>8s} {'poseedor B':>11s}"
    lines.append(head)
    for seg in sorted(by_seg):
        keys = by_seg[seg]
        hs = [A[k]["h"] for k in keys if A[k]["h"] is not None]
        n, ball, poss = summarize(A, keys)
        row = (f"{seg * 5:4d}-{seg * 5 + 5:3d} min {n:7d} {pct(len(hs), n):>8s} "
               f"{(statistics.median(hs) if hs else 0):8.1f}px {pct(ball, n):>8s} {pct(poss, n):>11s}")
        if B is not None:
            nb, ballb, possb = summarize(B, keys)
            row += f" {pct(ballb, nb):>8s} {pct(possb, nb):>11s}"
        lines.append(row)

    # --- resumen ---
    keys_all = list(A)
    keys_play = [k for k in keys_all if A[k]["n"] >= 6]
    lines.append("")
    for label, keys in (("todos los frames", keys_all), ("frames con >= 6 jugadores (juego)", keys_play)):
        n, ball, poss = summarize(A, keys)
        row = f"{label}: {n} frames, balón A {pct(ball, n).strip()}, poseedor A {pct(poss, n).strip()}"
        if B is not None:
            nb, ballb, possb = summarize(B, keys)
            row += f" | balón B {pct(ballb, nb).strip()}, poseedor B {pct(possb, nb).strip()}"
        lines.append(row)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
