"""
Entrena y evalúa (dejando un partido fuera) un clasificador por fotograma
H/A/D sobre IDSSE degradado (balón borrado al 41 %), y decodifica con Viterbi.

    python scripts/benchmark/train_possession_model.py                 # LOMO + modelo final
    python scripts/benchmark/train_possession_model.py --drop-ball 0.41 --merge-short 2 --seeds 2

Salida: tabla por partido (spellF1 y macroF1 del argmax y del Viterbi con
varios costes de cambio), y el modelo final entrenado con todos los partidos
en ``output/posesion/models/hgb_possession.joblib`` (sklearn
HistGradientBoosting, CPU, ~1 M filas, < 1 min).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.team_possession import decode, simulate, spell_metrics as pm  # noqa: E402
from core.team_possession.features import CONTEXT_OFFSETS_S, compute_features, feature_matrix, feature_names  # noqa: E402
from core.team_possession.io_sportec import load_npz  # noqa: E402
from core.team_possession.labels import X, merge_short_spells  # noqa: E402

CACHE = ROOT / "output" / "posesion" / "cache"
MODELS = ROOT / "output" / "posesion" / "models"


def build(path: Path, drop: float, seed: int, merge_s: float, anon: bool, context=None):
    data = load_npz(path)
    gt = merge_short_spells(data.labels, int(merge_s * data.fps)) if merge_s > 0 else data.labels
    if drop > 0:
        data = simulate.drop_ball(data, drop, seed=seed)
    if anon:
        data = simulate.anonymize_slots(data, seed=seed)
    feats = compute_features(data)
    Xm = feature_matrix(feats, data, context_offsets_s=context)
    return data.name, Xm, gt, feats.period_runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dataset", default="idsse")
    ap.add_argument("--drop-ball", type=float, default=0.41)
    ap.add_argument("--merge-short", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=1, help="degradaciones distintas por partido (aumenta datos)")
    ap.add_argument("--anon", action="store_true")
    ap.add_argument("--switch-costs", type=float, nargs="+", default=[2.0, 4.0, 6.0, 8.0])
    ap.add_argument("--min-dur", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--context", type=float, nargs="*", default=None, help="desplazamientos de contexto en segundos (por defecto los de features.CONTEXT_OFFSETS_S)")
    ap.add_argument("--no-lomo", action="store_true", help="solo entrena el modelo final")
    ap.add_argument("--out", type=Path, default=MODELS / "hgb_possession.joblib")
    ap.add_argument("--save-proba", type=Path, default=ROOT / "output" / "posesion" / "lomo_proba", help="carpeta donde guardar las log-probabilidades por partido (para barrer el decodificador sin reentrenar)")
    args = ap.parse_args()

    context = tuple(args.context) if args.context is not None else CONTEXT_OFFSETS_S
    files = sorted(CACHE.glob(f"{args.dataset}_*.npz"))
    t0 = time.time()
    sets = []
    for i, p in enumerate(files):
        for sd in range(args.seeds):
            sets.append(build(p, args.drop_ball, seed=100 * i + sd, merge_s=args.merge_short, anon=args.anon, context=context))
    print(f"{len(sets)} secuencias, {sum(len(s[2]) for s in sets)} fotogramas, rasgos en {time.time() - t0:.0f} s", flush=True)

    def fit(train):
        Xtr = np.concatenate([s[1] for s in train]); ytr = np.concatenate([s[2] for s in train])
        ok = ytr != X
        clf = HistGradientBoostingClassifier(max_iter=args.max_iter, learning_rate=0.08, max_leaf_nodes=63,
                                             l2_regularization=1.0, random_state=0)
        clf.fit(Xtr[ok], ytr[ok])
        return clf

    if not args.no_lomo:
        names = sorted({s[0].split("_drop")[0] for s in sets})
        rows = []
        for held in names:
            train = [s for s in sets if not s[0].startswith(held)]
            test = [s for s in sets if s[0].startswith(held)][0]
            t1 = time.time()
            clf = fit(train)
            proba = clf.predict_proba(test[1])  # columnas en clf.classes_ (0,1,2)
            logp = np.log(np.clip(proba, 1e-6, 1.0))
            gt = test[2]
            if args.save_proba:
                args.save_proba.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.save_proba / f"{held}.npz", logp=logp.astype(np.float32), gt=gt,
                                    runs=np.asarray(test[3], dtype=np.int64))
            res = {"match": held}
            res["argmax"] = pm.evaluate(gt, logp.argmax(1).astype(np.int8))
            for c in args.switch_costs:
                pred = decode.viterbi_decode(logp, switch_cost=c, min_dur=args.min_dur, runs=test[3])
                res[f"vit{c:g}"] = pm.evaluate(gt, pred)
            rows.append(res)
            line = f"{held:<13} " + "  ".join(f"{k} spell {v['spell_f1_mean']:.3f} macro {v['macro_f1']:.3f}"
                                             for k, v in res.items() if k != "match")
            print(line + f"  ({time.time() - t1:.0f} s)", flush=True)
        print("\nMedia LOMO:")
        for k in rows[0]:
            if k == "match":
                continue
            print(f"  {k:<8} spell {np.mean([r[k]['spell_f1_mean'] for r in rows]):.3f}  "
                  f"macro {np.mean([r[k]['macro_f1'] for r in rows]):.3f}  "
                  f"vivo/muerto {np.mean([r[k]['alive_dead_f1'] for r in rows]):.3f}")

    clf = fit(sets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": feature_names(context), "context": context, "drop_ball": args.drop_ball,
                 "merge_short": args.merge_short, "dataset": args.dataset}, args.out)
    imp = None
    print("modelo final guardado en", args.out, f"({time.time() - t0:.0f} s en total)")


if __name__ == "__main__":
    main()
