"""Experimento §3.6, parte juvenil/TV: ¿ve el T-DEED afinado en panorámica (SNB_pano1) los goles que el de
producción (challenge2) no ve en cámara única?

Para cada partido compara, cabeza de acciones de balón y cabeza SoccerNet, base frente a afinado:
* en cada gol de referencia (segundos de vídeo), el máximo de SHOT, GOAL y SHOT+GOAL en ±5 s;
* número de picos de SHOT >= 0,3 y GOAL >= 0,3 (suavizado 0,5 s, NMS 4 s) por 90 min de vídeo, y de las clases
  "Goal" / "Shots on/off target" de la cabeza SoccerNet.

Uso (dentro del contenedor, cwd /app, tras run_pano1_tv.sh):
    PYTHONPATH=/app python scripts/shot_benchmark/pano1_tv_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app")
from core.ball_spotting import read_scores  # noqa: E402
from core.spotting import find_peaks  # noqa: E402

FINE = "SNB_pano1"
BASE = "SoccerNetBall_challenge2"
SCAN = Path(f"output/shot_benchmark/tdeed/{FINE}_heads_tv")
VIDEOS = Path("backend/storage/videos")
# Goles en segundos de vídeo (ground_truth/highlights_goals.json; Clásico ±5 s leídos del marcador).
MATCHES = {
    "Mallorca (juvenil)": ("8e759934bde2156e", [4445.4, 4757.5, 5682.0]),
    "Espanyol (juvenil, 30 fps)": ("f0926c19277b28db", []),
    "Clásico (TV, in-sample)": ("573d9c1395b8f1c8", [2025.0, 2289.0, 5185.0, 5958.0, 6363.0]),
}
SMOOTH = 6
NMS_S = 4.0


def peaks_per_90(scores, label, min_score: float) -> float:
    curve = scores.curve(label)
    curve = np.convolve(curve, np.ones(SMOOTH) / SMOOTH, mode="same")
    times = (scores.frame_indices() / scores.fps).tolist()
    n = len(find_peaks(times, curve.tolist(), min_score, NMS_S))
    return n * 90.0 / max(scores.duration_s() / 60.0, 1e-6)


def win_max(scores, label, t: float, w: float = 5.0) -> float:
    rows = scores.frame_indices() / scores.fps
    m = (rows >= t - w) & (rows <= t + w)
    return float(scores.curve(label)[m].max()) if m.any() else 0.0


def main() -> None:
    for name, (vid, goals) in MATCHES.items():
        fps = float(json.loads((VIDEOS / vid / "video.json").read_text())["fps"])
        base = read_scores(VIDEOS / vid / f"source.tdeed.{BASE}.npz", fps=fps)
        fine_path = SCAN / f"{vid} [source].npz"
        if not fine_path.exists():
            print(f"== {name}: falta {fine_path}")
            continue
        fine = read_scores(fine_path, fps=fps)
        print(f"== {name}: {base.duration_s() / 60:.0f} min, fps {fps:g}")
        print(f"   {'':<28}{'base':>12}{'afinado':>12}")
        for label, th in (("SHOT", 0.3), ("GOAL", 0.3), ("Goal", 0.3), (("Shots on target", "Shots off target"), 0.3)):
            lname = label if isinstance(label, str) else "SN shots"
            try:
                b, f = peaks_per_90(base, label, th), peaks_per_90(fine, label, th)
            except KeyError as exc:
                print(f"   picos {lname} >= {th}: {exc}")
                continue
            print(f"   picos {lname + ' >= ' + str(th):<22}{b:>12.1f}{f:>12.1f}   (por 90 min)")
        for t in goals:
            h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
            print(f"   gol {h}:{m:02d}:{s:04.1f}  máx en ±5 s:")
            for label in ("SHOT", "GOAL", ("SHOT", "GOAL"), "Goal"):
                lname = label if isinstance(label, str) else "SHOT+GOAL"
                try:
                    b, f = win_max(base, label, t), win_max(fine, label, t)
                except KeyError:
                    continue
                print(f"      {lname:<24}{b:>12.3f}{f:>12.3f}")


if __name__ == "__main__":
    main()
