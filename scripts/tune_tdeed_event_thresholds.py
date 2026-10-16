"""Umbrales de los eventos de la cabeza SoccerNet de T-DEED (``core.ball_spotting.EVENT_SOURCES``) y
referencia de su calibración automática, sobre partidos del split de ENTRENAMIENTO de SoccerNet-v2
(``docs/particiones/soccernet_v2_entrenamiento_svc.txt``; los de evaluación no se tocan).

Entrada: escaneos de ``scripts/shot_benchmark/tdeed_scan_heads.py`` (``<partido> [1_720p].npz``,
las dos cabezas) y los ``Labels-v2.json`` de ``datasets/SoccerNet``. Para cada evento elige, con
todas las mitades juntas, el par (NMS, umbral) de máximo F1 a ±``--tol`` s, con el mismo suavizado
que el pipeline. Además mide, para ``config.BallSpottingConfig``:

* ``auto_sn_rate_per_min``: anotaciones de SoccerNet-v2 (todas las clases) por minuto de juego;
* ``auto_sn_reference_score``: mediana, por mitad, de la puntuación de "algún evento"
  (``analytics.engine.sn_event_peak_scores``) que da tantos picos como anotaciones tiene la mitad.

T-DEED se preentrenó con SoccerNet-v2, así que en estos partidos puntúa más alto que en partidos
nuevos: los umbrales salen prudentes, y la calibración automática los baja fuera de ese dominio.

Uso (dentro del contenedor, cwd /app):
    python -m scripts.tune_tdeed_event_thresholds \\
        --scans output/shot_benchmark/tdeed/SoccerNetBall_challenge2_heads_sntrain \\
        --output output/sn_event_tuning/thresholds.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

import config
from analytics.engine import active_minutes, sn_event_peak_scores
from core.ball_spotting import EVENT_SOURCES, read_scores, spot

logger = logging.getLogger(__name__)

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
NMS_GRID_S = (2.0, 4.0, 8.0)


def _labels_dir(game: str, root: Path) -> Path:
    hits = [p for p in root.glob(f"*/*/{game}") if (p / "Labels-v2.json").exists()]
    if len(hits) != 1:
        raise SystemExit(f"{game}: {len(hits)} carpetas con Labels-v2.json bajo {root}")
    return hits[0]


def _match(pred_t: np.ndarray, pred_s: np.ndarray, gt_t: np.ndarray, tol: float) -> np.ndarray:
    """Para cada predicción, ¿es acierto? Emparejado 1 a 1, de mayor a menor puntuación."""
    hit = np.zeros(len(pred_t), bool)
    used = np.zeros(len(gt_t), bool)
    for i in np.argsort(-pred_s):
        if not len(gt_t):
            break
        d = np.abs(gt_t - pred_t[i])
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = hit[i] = True
    return hit


def main() -> None:
    spot_cfg = config.PipelineConfig().ball_spotting
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scans", required=True, type=Path)
    parser.add_argument("--labels-root", type=Path, default=Path("datasets/SoccerNet"))
    parser.add_argument("--tol", type=float, default=3.0, help="tolerancia (s) para contar un acierto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sn_events = [label for label, srcs in EVENT_SOURCES.items() if not srcs[0].isupper()]
    halves = []
    for npz in sorted(args.scans.glob("*.npz")):
        game, half = npz.stem.rsplit(" [", 1)
        half = half.split("_")[0]
        anns = json.loads((_labels_dir(game, args.labels_root) / "Labels-v2.json").read_text())["annotations"]
        anns = [a for a in anns if a["gameTime"].startswith(f"{half} -")]
        scores = read_scores(npz, fps=25.0)
        halves.append((npz.stem, scores, anns))
    if not halves:
        raise SystemExit(f"Sin escaneos en {args.scans}")
    logger.info("%d mitades de %d partidos", len(halves), len({h[0].rsplit(' [', 1)[0] for h in halves}))

    # Referencia de la calibración de la cabeza SoccerNet.
    total_anns = total_minutes = 0
    refs = []
    for name, scores, anns in halves:
        minutes = active_minutes(scores, spot_cfg.action_nms_s)
        ranked = sn_event_peak_scores(scores, spot_cfg.event_smooth_frames)
        k = min(len(anns), len(ranked))
        refs.append(float(ranked[k - 1]) if k else np.nan)
        total_anns += len(anns)
        total_minutes += minutes
    rate = total_anns / max(total_minutes, 1)
    reference = float(np.nanmedian(refs))
    logger.info("Cabeza SoccerNet: %.2f anotaciones/min de juego; umbral de referencia %.3f (mitades: %s)",
                rate, reference, " ".join(f"{r:.2f}" for r in refs))

    result = {"auto_sn_rate_per_min": round(rate, 3), "auto_sn_reference_score": round(reference, 3),
              "tol_s": args.tol, "halves": [h[0] for h in halves], "events": {}}
    print(f"{'evento':14s} {'GT':>5s} {'NMS':>4s} {'umbral':>6s} {'P':>5s} {'R':>5s} {'F1':>5s}   F1 a 0,3/0,5/0,7")
    for label in sn_events:
        srcs = EVENT_SOURCES[label]
        n_gt = sum(sum(a["label"] in srcs for a in anns) for _, _, anns in halves)
        best = None
        curves = {}
        for nms in NMS_GRID_S:
            pooled_s, pooled_hit = [], []
            for _, scores, anns in halves:
                peaks = spot(scores, srcs, 0.02, nms, spot_cfg.event_smooth_frames)
                t = np.array([f / scores.fps for f, _ in peaks])
                s = np.array([v for _, v in peaks])
                gt = np.array([int(a["position"]) / 1000 for a in anns if a["label"] in srcs])
                pooled_s.append(s)
                pooled_hit.append(_match(t, s, gt, args.tol))
            s = np.concatenate(pooled_s)
            hit = np.concatenate(pooled_hit)
            f1s = []
            for th in THRESHOLDS:
                keep = s >= th
                tp = int(hit[keep].sum())
                p = tp / max(int(keep.sum()), 1)
                r = tp / max(n_gt, 1)
                f1 = 2 * p * r / max(p + r, 1e-9)
                f1s.append(f1)
                if best is None or f1 > best[0]:
                    best = (f1, nms, float(th), p, r)
            curves[nms] = f1s
        f1, nms, th, p, r = best
        at = [curves[nms][int(np.argmin(np.abs(THRESHOLDS - x)))] for x in (0.3, 0.5, 0.7)]
        print(f"{label:14s} {n_gt:5d} {nms:4.0f} {th:6.2f} {p:5.2f} {r:5.2f} {f1:5.2f}   "
              + " / ".join(f"{x:.2f}" for x in at))
        result["events"][label] = {"threshold": th, "nms_s": nms, "precision": round(p, 3),
                                   "recall": round(r, 3), "f1": round(f1, 3), "n_gt": n_gt}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        logger.info("Escrito %s", args.output)


if __name__ == "__main__":
    main()
