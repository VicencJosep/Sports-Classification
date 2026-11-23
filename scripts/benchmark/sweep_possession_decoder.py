"""
Barre el decodificador (coste de cambio × duración mínima × filtro de tramos)
sobre las log-probabilidades por partido que guarda ``train_possession_model.py``
(``output/posesion/lomo_proba/*.npz``), sin reentrenar.

    python scripts/benchmark/sweep_possession_decoder.py --costs 16 24 32 48 64 --min-durs 1 25 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.team_possession import decode, spell_metrics as pm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--proba", type=Path, default=ROOT / "output" / "posesion" / "lomo_proba")
    ap.add_argument("--costs", type=float, nargs="+", default=[16, 24, 32, 48, 64])
    ap.add_argument("--min-durs", type=int, nargs="+", default=[1, 25, 50])
    ap.add_argument("--msf", type=int, nargs="+", default=[0], help="filtro de tramos < k fotogramas tras el Viterbi (0 = no)")
    args = ap.parse_args()
    folds = []
    for p in sorted(args.proba.glob("*.npz")):
        z = np.load(p)
        folds.append((p.stem, z["logp"].astype(np.float64), z["gt"], [tuple(r) for r in z["runs"]]))
    print(f"{len(folds)} partidos")
    best = (0.0, None)
    for c in args.costs:
        for md in args.min_durs:
            for k in args.msf:
                sp, ma = [], []
                for name, logp, gt, runs in folds:
                    pred = decode.viterbi_decode(logp, switch_cost=c, min_dur=md, runs=runs)
                    if k > 0:
                        pred = decode.min_segment_filter(pred, k, runs=runs)
                    r = pm.evaluate(gt, pred)
                    sp.append(r["spell_f1_mean"]); ma.append(r["macro_f1"])
                print(f"coste {c:>5g}  min_dur {md:>3}  msf {k:>3}  spell {np.mean(sp):.3f} (min {np.min(sp):.3f})  macro {np.mean(ma):.3f}", flush=True)
                if np.mean(sp) > best[0]:
                    best = (float(np.mean(sp)), (c, md, k))
    print("mejor:", best)


if __name__ == "__main__":
    main()
