
# --- T-DEED como detector de tiros: evaluación en partido completo (se concatena tras common.py) ---
# Uso: cat common.py tdeed_eval.py | python - MODELO
import sys, json
MODEL = sys.argv[1] if len(sys.argv) > 1 else "SoccerNetBall_test"
CLEAN = ("Middlesbrough", "Stoke", "Reading") if MODEL == "SoccerNetBall_test" else ("Middlesbrough",)
TD = Path(f"output/shot_benchmark/tdeed/{MODEL}")
FPS = 25.0

def tdeed_peaks(z, cls="SHOT", nms_s=4.0, min_score=0.01):
    names = list(z["classes"]); p = z["probs"][:, names.index(cls)].astype(np.float32)
    t = np.arange(len(p)) * int(z["stride"]) / FPS
    # máximos locales + NMS (mismo criterio que core.spotting.find_peaks), sobre la curva suavizada 0,5 s
    k = 6; ps = np.convolve(p, np.ones(k) / k, mode="same")
    return find_peaks(t.tolist(), ps.tolist(), min_score, nms_s), (t, p)

matches = load_matches()
rows = []
for m, c in zip(matches, CACHES):
    f = TD / f"{Path(c).parent.name}.npz"
    if not f.exists():
        continue
    z = np.load(f)
    peaks, (t, p) = tdeed_peaks(z)
    names = list(z["classes"]); P = z["probs"].astype(np.float32)
    def win_max(cls, tc, w=3.0):
        lo, hi = int(max(0, (tc - w) * FPS / 2)), int((tc + w) * FPS / 2) + 1
        return float(P[lo:hi, names.index(cls)].max()) if hi > lo else 0.0
    m["td_peaks"] = peaks
    m["td_shot_at_cand"] = np.array([win_max("SHOT", tc) for tc, _ in m["peaks"]])
    m["td_cross_at_cand"] = np.array([win_max("CROSS", tc) for tc, _ in m["peaks"]])
    m["clean"] = any(k in m["name"] for k in CLEAN)
    rows.append(m)

def pr_table(gts_preds, label):
    """gts_preds: [(gt, [Prediction])]. AP y precisión a distintos recall, emparejado 1 a 1 ±3 s y ±1 s."""
    for tol in (3.0, 1.0):
        ap = average_precision(gts_preds, tol)
        scores = sorted({round(p.score, 3) for _, ps in gts_preds for p in ps}, reverse=True)
        best = {}
        for th in scores:
            tot = Counts()
            for g, ps in gts_preds:
                tot.add(count(g, ps, tol, th))
            for r in (0.5, 0.6, 0.7, 0.8):
                if tot.recall >= r and tot.precision > best.get(r, (0, 0, 0))[0]:
                    best[r] = (tot.precision, tot.n_pred / len(gts_preds), th)
        cells = "  ".join(f"R{int(100*r)}: P {100*best[r][0]:4.1f}% ({best[r][1]:3.0f}/part, θ={best[r][2]:.2f})" if r in best else f"R{int(100*r)}: --" for r in (0.5, 0.6, 0.7, 0.8))
        n_gt = sum(len(g) for g, _ in gts_preds)
        max_rec = sum(count(g, ps, tol, 0.0).tp for g, ps in gts_preds) / n_gt
        print(f"  {label:<26} ±{tol:g}s  AP {100*ap:5.1f}%  techo recall {100*max_rec:4.1f}%  {cells}")

for title, sel in (("PARTIDOS LIMPIOS (no vistos al entrenar T-DEED)", [m for m in rows if m["clean"]]),
                   ("partidos de ENTRENAMIENTO de T-DEED (solo referencia, optimista)", [m for m in rows if not m["clean"]])):
    if not sel:
        continue
    print(f"\n=== {title}: {[m['name'].strip() for m in sel]} — {sum(len(m['gt']) for m in sel)} tiros ===")
    pr_table([(m["gt"], [Prediction(t, t, t, float(s)) for t, s in m["peaks"]]) for m in sel], "V-JEPA2 (picos)")
    pr_table([(m["gt"], [Prediction(t, t, t, float(s)) for t, s in m["td_peaks"]]) for m in sel], "T-DEED SHOT (picos)")
    pr_table([(m["gt"], [Prediction(t, t, t, float(s)) for (t, _), s in zip(m["peaks"], m["td_shot_at_cand"])]) for m in sel],
             "V-JEPA2 cand × T-DEED")
    # tiro vs centro sobre los candidatos de V-JEPA2
    cross = []
    for m, c in zip(matches, CACHES):
        if m not in sel: continue
        ann = [(int(a["position"]) / 1000, a["label"]) for a in json.load(open(Path(c).parent / "Labels-ball.json"))["annotations"]]
        for (tc, _), y, sh in zip(m["peaks"], m["y"], m["td_shot_at_cand"]):
            near = {l for at, l in ann if abs(at - tc) <= 3}
            kind = "tiro" if y else "centro" if "CROSS" in near and "SHOT" not in near else "sin anotar" if not near else None
            if kind: cross.append((kind, sh))
    for th in (0.1, 0.2, 0.3, 0.5):
        d = {k: (np.mean([s >= th for kk, s in cross if kk == k]), sum(kk == k for kk, _ in cross)) for k in ("tiro", "centro", "sin anotar")}
        print(f"  candidatos V-JEPA2 con T-DEED SHOT ≥ {th}: " + "  ".join(f"{k} {100*v:4.0f}% (n={n})" for k, (v, n) in d.items()))
    pos = [s for k, s in cross if k == "tiro"]; neg = [s for k, s in cross if k == "centro"]
    print(f"  AUC tiro/centro (T-DEED sobre candidatos): {np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]):.2f}")
