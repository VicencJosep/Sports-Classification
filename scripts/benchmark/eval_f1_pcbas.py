"""Métrica del reto SoccerNet 2026 Player-Centric Ball Action Spotting (PCBAS /
FOOTPASS, arXiv 2511.16183) aplicada a las predicciones de ``eval_bas.py``:
Macro-F1 por clase con umbral de confianza 0,15 y tolerancia ±12 frames a
25 fps (±0,48 s), emparejamiento 1 a 1.

Solo la parte de CLASE: la métrica oficial exige además equipo y dorsal del
ejecutor, que nuestro GT de BAS no tiene (dorsal) o no se evalúa aquí
(equipo). Es por tanto una cota SUPERIOR de lo que daría PCBAS, y sobre otros
vídeos: sirve para poner las cifras publicadas de PCBAS (PAVE 58,94) en la
misma escala, no para compararlas de forma directa.

Clases de FOOTPASS -> BAS: drive=DRIVE, pass=PASS (FOOTPASS no separa HIGH
PASS; aquí no se fusiona), cross=CROSS, shot=SHOT, header=HEADER,
throw-in=THROW IN, tackle=PLAYER SUCCESSFUL TACKLE, block=BALL PLAYER BLOCK.

Uso (contenedor):
    python -m scripts.benchmark.eval_f1_pcbas --model AdaSpot_big --split test
    python -m scripts.benchmark.eval_f1_pcbas --model SNB_pano1 --dataset st2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.benchmark.eval_bas import LABELS_ROOT, OUT_ROOT, OUT_ROOT_ST2, TDEED_DIR

CLASS_MAP = {"drive": "DRIVE", "pass": "PASS", "cross": "CROSS", "shot": "SHOT", "header": "HEADER",
             "throw-in": "THROW IN", "tackle": "PLAYER SUCCESSFUL TACKLE", "block": "BALL PLAYER BLOCK"}
THRESHOLD = 0.15
TOL_MS = 12 / 25 * 1000


def match(gt_ms: list[float], pred: list[tuple[float, float]]) -> tuple[int, int, int]:
    """(TP, FP, FN) con emparejamiento 1 a 1 greedy por confianza y GT más cercano."""
    gt = np.array(sorted(gt_ms))
    used = np.zeros(len(gt), bool)
    tp = 0
    for t, _ in sorted(pred, key=lambda p: -p[1]):
        if not len(gt):
            break
        d = np.abs(gt - t)
        d[used] = np.inf
        i = int(np.argmin(d))
        if d[i] <= TOL_MS:
            used[i] = True
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="efl", choices=["efl", "st2"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--view", default="vcam796vl1")
    args = ap.parse_args()

    if args.dataset == "efl":
        games = [g["video"] for g in json.loads((TDEED_DIR / "data" / "soccernetball" / f"{args.split}.json").read_text())]
        label_root, out_root = LABELS_ROOT, OUT_ROOT / args.model / args.split
    else:
        summary = json.loads((OUT_ROOT_ST2 / args.model / args.view / "summary.json").read_text())
        games, label_root, out_root = summary["games"], OUT_ROOT_ST2 / "labels", OUT_ROOT_ST2 / args.model / args.view
    preds_dir = next(out_root.glob("snms*")) / "preds"

    counts = {c: [0, 0, 0] for c in CLASS_MAP}
    for game in games:
        gt = json.loads((label_root / game / "Labels-ball.json").read_text())["annotations"]
        pr = json.loads((preds_dir / game / "results_spotting.json").read_text())["predictions"]
        for name, label in CLASS_MAP.items():
            g = [float(a["position"]) for a in gt if a["label"] == label]
            p = [(float(a["position"]), float(a["confidence"])) for a in pr
                 if a["label"] == label and a["confidence"] >= THRESHOLD]
            for k, v in enumerate(match(g, p)):
                counts[name][k] += v
    per_class = {}
    for name, (tp, fp, fn) in counts.items():
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = {"P": round(100 * prec, 1), "R": round(100 * rec, 1),
                           "F1": round(100 * (2 * prec * rec / (prec + rec) if prec + rec else 0.0), 1), "n_gt": tp + fn}
    macro = float(np.mean([v["F1"] for v in per_class.values()]))
    tag = args.split if args.dataset == "efl" else args.view
    print(f"{args.model} [{tag}] Macro-F1@0.15 ±12f (solo clase, 8 clases FOOTPASS): {macro:.2f}  "
          + "  ".join(f"{c} {v['F1']:.1f}" for c, v in per_class.items()))
    (out_root / "f1_pcbas.json").write_text(json.dumps({"model": args.model, "games": games, "macro_f1": round(macro, 2),
                                                        "per_class": per_class}, indent=1))


if __name__ == "__main__":
    main()
