"""Generalización de T-DEED (SoccerNetBall_challenge2) a 6 partidos SoccerNet-v2 de otras ligas.
GT: Labels-v2.json, "Shots on target" + "Shots off target" (y, aparte, contando también "Goal")."""
import json, re, statistics
from pathlib import Path
import numpy as np
from core.spotting import find_peaks
from scripts.evaluate_spotting import Prediction, match_1to1, count, Counts, average_precision

TD = Path("output/shot_benchmark/tdeed/SoccerNetBall_challenge2_v2"); FPS = 25.0
videos = [l.strip() for l in open("output/shot_benchmark/experiments/v2_videos.txt") if l.strip()]
data = []
for v in videos:
    rel = Path(v).relative_to("/mnt/datasets/SoccerNet").parent; half = Path(v).name[0]
    z = np.load(TD / f"{rel.name} [{half}_720p].npz")
    names = list(z["classes"]); P = z["probs"].astype(np.float32)
    t = (np.arange(len(P)) * int(z["stride"]) / FPS).tolist()
    k = 6; smooth = lambda x: np.convolve(x, np.ones(k) / k, mode="same")
    shot = smooth(P[:, names.index("SHOT")])
    ann = [a for a in json.load(open(Path("datasets/SoccerNet") / rel / "Labels-v2.json"))["annotations"]
           if a["gameTime"].split(" - ")[0] == half]
    gt = sorted(int(a["position"]) / 1000 for a in ann if a["label"].startswith("Shots"))
    others = [(int(a["position"]) / 1000, a["label"]) for a in ann]
    preds = [Prediction(tt, tt, tt, float(s)) for tt, s in find_peaks(t, shot.tolist(), 0.01, 4.0)]
    data.append(dict(name=f"{rel.parts[0]} | {rel.name[19:]} [{half}]", gt=gt, preds=preds, others=others))

n_gt = sum(len(d["gt"]) for d in data); n_match = len(data) / 2
print(f"{len(data)} mitades ({n_match:.0f} partidos), {n_gt} tiros anotados")
for tol in (1.0, 3.0, 5.0):
    pm = [(d["gt"], d["preds"]) for d in data]
    ap = average_precision(pm, tol)
    ceil = sum(count(g, p, tol).tp for g, p in pm) / n_gt
    line = f"±{tol:g}s  AP {100*ap:5.1f}%  techo recall {100*ceil:4.1f}% |"
    for th in (0.2, 0.3, 0.5):
        tot = Counts()
        for g, p in pm: tot.add(count(g, p, tol, th))
        line += f"  θ={th}: P {100*tot.precision:4.1f}% R {100*tot.recall:4.1f}% ({tot.n_pred/n_match:.0f}/part)"
    print(line)

# offset y qué son las falsas alarmas (θ=0.3, ±5 s)
diffs, fp_near = [], {}
for d in data:
    kept = [p for p in d["preds"] if p.score >= 0.3]
    pairs = match_1to1(d["gt"], kept, 5.0)
    diffs += [d["gt"][gi] - kept[pi].t_s for gi, pi in pairs]
    hit = {pi for _, pi in pairs}
    for pi, p in enumerate(kept):
        if pi in hit: continue
        near = sorted((abs(tt - p.t_s), l) for tt, l in d["others"] if abs(tt - p.t_s) <= 10)
        key = near[0][1] if near else "nada anotado a ±10s"
        fp_near[key] = fp_near.get(key, 0) + 1
q = statistics.quantiles(diffs, n=4)
print(f"offset GT - pico (aciertos ±5s, n={len(diffs)}): mediana {statistics.median(diffs):+.2f}s (p25 {q[0]:+.2f}, p75 {q[2]:+.2f})")
print("falsas alarmas a θ=0.3 (±5s), por anotación v2 más cercana:", dict(sorted(fp_near.items(), key=lambda kv: -kv[1])))
print("\nPor partido (θ=0.3, ±5s):")
for i in range(0, len(data), 2):
    tot = Counts()
    for d in data[i:i + 2]: tot.add(count(d["gt"], d["preds"], 5.0, 0.3))
    print(f"  {data[i]['name'][:-4]:<62} GT {tot.n_gt:3d} pred {tot.n_pred:3d} TP {tot.tp:3d}  P {100*tot.precision:4.0f}%  R {100*tot.recall:4.0f}%")
