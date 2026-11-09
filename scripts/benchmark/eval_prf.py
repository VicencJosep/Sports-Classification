"""Precisión / recall / F1 y recuentos (TP, FP, FN) por clase a un umbral de trabajo, a partir de las predicciones ya
post-procesadas por ``eval_bas.py`` (``<out>/snms12/preds/<partido>/results_spotting.json``). Complementa al AP (mAP@1),
que integra sobre todos los umbrales y no dice cuántos falsos positivos emite un modelo en uso.

Emparejamiento igual que el mAP@1 oficial: tolerancia ±0,5 s, 1 a 1, greedy por confianza. Para cada modelo se barre un
umbral GLOBAL (el mismo para todas las clases) y se elige el que maximiza la F1 micro sobre las clases comunes (sin FREE KICK
ni GOAL, que AdaSpot no predice); se informan también los umbrales fijos 0,15 (convención PCBAS) y 0,5.

Uso (contenedor):
    python -m scripts.benchmark.eval_prf --model SoccerNetBall_test --dataset efl
    python -m scripts.benchmark.eval_prf --model AdaSpot_big --dataset st2 --view vcam796vl1
    python -m scripts.benchmark.eval_prf --model AdaSpot_big --dataset pcbas --tag val_merged
    python -m scripts.benchmark.eval_prf --model SoccerNetBall_test --dataset snv2 --tag pilot20 --tol-ms 1000

Con ``--dataset snv2`` se evalúan las 17 clases de SoccerNet-v2 a partir de las predicciones de ``eval_snv2.py``
(``output/benchmark/snv2/<modelo>/<tag>/preds/<partido>/results_spotting.json``) contra ``Labels-v2.json``; la F1 global
cubre las 17 clases y la tolerancia por defecto sigue siendo ±0,5 s (las etiquetas de SN-v2 van al segundo: usar ``--tol-ms 1000``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.benchmark.eval_bas import (LABELS_ROOT, OUT_ROOT, OUT_ROOT_PCBAS, OUT_ROOT_ST2, PCBAS_CLASSES, TDEED_DIR,
                                        tdeed_eval_classes)
from scripts.benchmark.eval_snv2 import OUT_ROOT as OUT_ROOT_SNV2

TOL_MS = 500.0
EXCLUDED_FROM_OVERALL = {"FREE KICK", "GOAL"}


def match(gt_ms: np.ndarray, pred: list[tuple[float, float]], tol_ms: float = TOL_MS) -> tuple[int, int, int]:
    """(TP, FP, FN): 1 a 1, greedy por confianza, GT más cercano dentro de ±tol_ms."""
    used = np.zeros(len(gt_ms), bool)
    tp = 0
    for t, _ in sorted(pred, key=lambda p: -p[1]):
        if not len(gt_ms):
            break
        d = np.abs(gt_ms - t)
        d[used] = np.inf
        i = int(np.argmin(d))
        if d[i] <= tol_ms:
            used[i] = True
            tp += 1
    return tp, len(pred) - tp, len(gt_ms) - tp


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="efl", choices=["efl", "st2", "pcbas", "snv2"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--view", default="vcam796vl1")
    ap.add_argument("--tag", default="val", help="pcbas: val | val_merged; snv2: pilot20")
    ap.add_argument("--tol-ms", type=float, default=TOL_MS, help="Tolerancia de emparejamiento (±ms)")
    ap.add_argument("--grid", type=float, nargs="*", default=[round(x, 2) for x in np.arange(0.05, 0.96, 0.05)])
    args = ap.parse_args()

    if args.dataset == "efl":
        games = [g["video"] for g in json.loads((TDEED_DIR / "data" / "soccernetball" / f"{args.split}.json").read_text())]
        label_root, out_root, tag = LABELS_ROOT, OUT_ROOT / args.model / args.split, args.split
    elif args.dataset == "st2":
        out_root, tag = OUT_ROOT_ST2 / args.model / args.view, args.view
        games, label_root = json.loads((out_root / "summary.json").read_text())["games"], OUT_ROOT_ST2 / "labels"
    elif args.dataset == "pcbas":
        out_root, tag = OUT_ROOT_PCBAS / args.model / args.tag, args.tag
        games, label_root = json.loads((out_root / "summary.json").read_text())["games"], OUT_ROOT_PCBAS / "labels"
    else:
        out_root, tag = OUT_ROOT_SNV2 / args.model / args.tag, args.tag
        games, label_root = json.loads((out_root / "summary.json").read_text())["games"], LABELS_ROOT
    if args.dataset == "snv2":
        from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2
        preds_dir, classes, labels_name, excluded = out_root / "preds", list(EVENT_DICTIONARY_V2), "Labels-v2.json", set()
    else:
        preds_dir = (out_root / "snms12" / "preds") if (out_root / "snms12").exists() else next(out_root.glob("snms*")) / "preds"
        classes = list(PCBAS_CLASSES.values()) if args.dataset == "pcbas" else tdeed_eval_classes()
        labels_name, excluded = "Labels-ball.json", EXCLUDED_FROM_OVERALL

    def half_of(a: dict) -> int:
        return int(a["half"]) if "half" in a else int(str(a["gameTime"])[0])

    gt = {c: [] for c in classes}
    pr = {c: [] for c in classes}
    for game in games:
        ann = json.loads((label_root / game / labels_name).read_text())["annotations"]
        pred = json.loads((preds_dir / game / "results_spotting.json").read_text())["predictions"]
        for c in classes:  # un partido = una lista; cada partido se desplaza 10 h y cada parte 5 h para no emparejar entre ellos
            off = games.index(game) * 36_000_000
            gt[c] += [float(a["position"]) + off + (half_of(a) - 1) * 18_000_000 for a in ann if a["label"] == c]
            pr[c] += [(float(a["position"]) + off + (half_of(a) - 1) * 18_000_000, float(a["confidence"]))
                      for a in pred if a["label"] == c]
    gt = {c: np.array(sorted(v)) for c, v in gt.items()}

    def evaluate(thr: float) -> dict:
        per = {}
        for c in classes:
            tp, fp, fn = match(gt[c], [p for p in pr[c] if p[1] >= thr], args.tol_ms)
            p, r, f = prf(tp, fp, fn)
            per[c] = {"P": round(100 * p, 1), "R": round(100 * r, 1), "F1": round(100 * f, 1), "TP": tp, "FP": fp, "FN": fn,
                      "n_gt": int(len(gt[c]))}
        common = [c for c in classes if c not in excluded and len(gt[c])]
        tp, fp, fn = (sum(per[c][k] for c in common) for k in ("TP", "FP", "FN"))
        p, r, f = prf(tp, fp, fn)
        return {"threshold": thr, "overall": {"P": round(100 * p, 1), "R": round(100 * r, 1), "F1": round(100 * f, 1),
                                               "TP": tp, "FP": fp, "FN": fn, "classes": common},
                "macro_F1": round(float(np.mean([per[c]["F1"] for c in common])), 1), "per_class": per}

    sweep = {t: evaluate(t) for t in args.grid}
    best_t = max(sweep, key=lambda t: sweep[t]["overall"]["F1"])
    # Umbral óptimo POR CLASE (como los umbrales por evento de producción): el que maximiza la F1 de esa clase.
    per_class_best = {}
    for c in classes:
        tc = max(sweep, key=lambda t: (sweep[t]["per_class"][c]["F1"], -t))
        per_class_best[c] = {"threshold": tc, **sweep[tc]["per_class"][c]}
    res = {"model": args.model, "dataset": args.dataset, "tag": tag, "games": games, "tolerance_ms": args.tol_ms,
           "best": sweep[best_t], "at_0.15": sweep.get(0.15) or evaluate(0.15), "at_0.5": sweep.get(0.5) or evaluate(0.5),
           "per_class_best": per_class_best,
           "sweep": {str(t): {"F1": v["overall"]["F1"], "P": v["overall"]["P"], "R": v["overall"]["R"], "FP": v["overall"]["FP"]}
                     for t, v in sweep.items()}}
    suffix = "" if args.tol_ms == TOL_MS else f"_tol{int(args.tol_ms)}"
    (out_root / f"prf{suffix}.json").write_text(json.dumps(res, indent=1))
    b = res["best"]
    print(f"{args.model} [{args.dataset}/{tag}] mejor umbral {best_t:.2f}: P {b['overall']['P']} R {b['overall']['R']} "
          f"F1 {b['overall']['F1']} (TP {b['overall']['TP']} FP {b['overall']['FP']} FN {b['overall']['FN']}; macro-F1 {b['macro_F1']})  |  "
          + "  ".join(f"{c}: P{v['P']:.0f}/R{v['R']:.0f} FP{v['FP']}" for c, v in b["per_class"].items()))
    print("  umbral óptimo por clase: " + "  ".join(f"{c} @{v['threshold']:.2f} P{v['P']:.0f}/R{v['R']:.0f} F1 {v['F1']:.0f} FP{v['FP']}"
                                                  for c, v in per_class_best.items() if v["n_gt"]))


if __name__ == "__main__":
    main()
