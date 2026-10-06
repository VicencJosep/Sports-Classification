"""Común a los experimentos de re-puntuación de candidatos a tiro (7 EFL).
Candidatos = picos de la curva shot+goal con θ>=0.02 y NMS 4 s. Etiqueta =
emparejado 1 a 1 con GT de tiro a ±3 s.
Se concatena delante de cada experimento: cat common.py X.py | docker exec -i ... python -"""
import glob
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from analytics.engine import read_event_windows
from core.spotting import find_peaks
from scripts.evaluate_spotting import load_ground_truth, Prediction, match_1to1, count, Counts, average_precision

THETA, NMS, TOL, EPS = 0.02, 4.0, 3.0, 1e-3
CACHES = sorted(glob.glob("datasets/SoccerNet/england_efl/2019-2020/*/720p.event_windows.vjepa2.w6s3.jsonl"))
EMB_DIR = Path("output/shot_benchmark/candidate_embeddings")


def load_matches(with_emb=False):
    matches = []
    for c in CACHES:
        meta, rows = read_event_windows(Path(c))
        classes = sorted(rows[0]["probabilities"])
        P = np.array([[r["probabilities"][k] for k in classes] for r in rows])
        times = np.array([(r["start_s"] + r["end_s"]) / 2 for r in rows])
        curve = P[:, classes.index("shot")] + P[:, classes.index("goal")]
        peaks = find_peaks(times.tolist(), curve.tolist(), THETA, NMS)
        idx = np.array([int(np.argmin(np.abs(times - t))) for t, _ in peaks])
        gt = load_ground_truth(Path(meta["source_video"]))[0].get("shot", [])
        pos = {pi for _, pi in match_1to1(gt, [Prediction(t, t, t, s) for t, s in peaks], TOL)}
        m = dict(name=Path(c).parent.name[13:40], L=np.log(P + EPS), curve=curve, idx=idx, peaks=peaks, gt=gt,
                 y=np.array([i in pos for i in range(len(peaks))], dtype=int))
        if with_emb:
            z = np.load(EMB_DIR / f"{Path(c).parent.name}.npz")
            assert np.array_equal(z["idx"], idx), f"candidatos distintos en {m['name']}"
            m["emb"] = z["emb"].astype(np.float32)
        matches.append(m)
    return matches


def ctx_feats(m, K):
    """Log-probabilidades del SVC: ventana central, media antes, media después,
    máx. después (±K ventanas), más pico/prominencia/media de la curva."""
    L, curve, n = m["L"], m["curve"], len(m["curve"])
    out = []
    for i in m["idx"]:
        js = np.clip(np.arange(i - K, i + K + 1), 0, n - 1)
        ctx, c = L[js], curve[js]
        out.append(np.concatenate([L[i], ctx[:K].mean(0), ctx[K + 1:].mean(0), ctx[K + 1:].max(0),
                                   [curve[i], curve[i] - c.min(), c.mean()]]))
    return np.array(out)


def LR(C=0.1):
    return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def lomo(matches, featfn, make_model):
    """Leave-one-match-out: score de cada candidato con un modelo que no vio su partido."""
    X = [featfn(m) for m in matches]
    out = []
    for k in range(len(matches)):
        tr = [j for j in range(len(matches)) if j != k]
        model = make_model().fit(np.vstack([X[j] for j in tr]), np.concatenate([matches[j]["y"] for j in tr]))
        out.append(model.predict_proba(X[k])[:, 1])
    return out


def ap_from(scores, y, n_gt):
    """AP como evaluate_spotting.average_precision, con etiquetas fijas por candidato
    (picos a >= 6 s y tolerancia ±3 s: el emparejado no cambia al filtrar)."""
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(y[order])
    prec = np.maximum.accumulate((tp / np.arange(1, len(y) + 1))[::-1])[::-1]
    return float(np.sum(prec * np.diff(np.concatenate([[0.0], tp / n_gt]))))


def evaluate(matches, per_match_scores):
    n_gt = sum(len(m["gt"]) for m in matches)
    s = np.concatenate(per_match_scores)
    y = np.concatenate([m["y"] for m in matches])
    order = np.argsort(-s, kind="stable")
    tp = np.cumsum(y[order])
    prec, rec = tp / np.arange(1, len(y) + 1), tp / n_gt
    at = {}
    for r in (0.5, 0.6, 0.7, 0.8):
        ok = np.where(rec >= r)[0]
        at[r] = (prec[ok].max(), (ok[prec[ok].argmax()] + 1) / len(matches)) if len(ok) else (np.nan, np.nan)
    return (ap_from(s, y, n_gt), [ap_from(sc, m["y"], len(m["gt"])) for m, sc in zip(matches, per_match_scores)], at)


def check_against_official(matches):
    base = [m["curve"][m["idx"]] for m in matches]
    pm = [(m["gt"], [Prediction(t, t, t, float(x)) for (t, _), x in zip(m["peaks"], sc)]) for m, sc in zip(matches, base)]
    print(f"check AP V-JEPA2: oficial {100 * average_precision(pm, TOL):.2f}%  vectorizado {100 * evaluate(matches, base)[0]:.2f}%")


def report(matches, methods, baseline="V-JEPA2 (pico)"):
    res = {name: evaluate(matches, s) for name, s in methods.items()}
    base = res[baseline][1]
    print(f"\n{'método':<22}{'AP±3s':>7}{'mejor que base':>16}   P@R50        P@R60        P@R70        P@R80   (cand/partido)")
    for name, (ap, apm, at) in res.items():
        better = "-" if name == baseline else f"{sum(a > b for a, b in zip(apm, base))}/7"
        cells = "  ".join(f"{100 * p:4.1f}% ({c:4.0f})" for p, c in at.values())
        print(f"{name:<22}{100 * ap:6.1f}%{better:>16}   {cells}")
    print("\nAP por partido:")
    print(f"{'':<28}" + "".join(f"{n[:15]:>17}" for n in methods))
    for k, m in enumerate(matches):
        print(f"{m['name']:<28}" + "".join(f"{100 * res[n][1][k]:16.1f}%" for n in methods))
    return res
