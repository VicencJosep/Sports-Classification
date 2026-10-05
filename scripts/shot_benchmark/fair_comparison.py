
# --- Comparativa JUSTA: mismos candidatos y mismos partidos limpios para todos los métodos ---
import json
CLEAN = ("Middlesbrough", "Stoke", "Reading")
matches = load_matches(with_vlm=True)
TD = Path("output/shot_benchmark/tdeed/SoccerNetBall_test")
for m, c in zip(matches, CACHES):
    z = np.load(TD / f"{Path(c).parent.name}.npz"); names = list(z["classes"]); P = z["probs"].astype(np.float32)
    def wmax(tc, w=3.0):
        lo, hi = int(max(0, (tc - w) * 25 / 2)), int((tc + w) * 25 / 2) + 1
        return float(P[lo:hi, names.index("SHOT")].max())
    m["td"] = np.array([wmax(tc) for tc, _ in m["peaks"]])
    m["clean"] = any(k in m["name"] for k in CLEAN)
    t = np.arange(len(P)) * 2 / 25.0; p = np.convolve(P[:, names.index("SHOT")], np.ones(6) / 6, mode="same")
    m["td_peaks"] = find_peaks(t.tolist(), p.tolist(), 0.01, 4.0)

# 1) banco de 150 candidatos difíciles, restringido a los partidos limpios
bench = json.loads(Path("output/shot_benchmark/vlm_bench/candidates.json").read_text())
key = {(r["match"], r["t_s"]): r for r in bench}
res = {}
for p in sorted(Path("output/shot_benchmark/vlm_bench").glob("*.jsonl")):
    res[p.stem] = {(d["match"], d["t_s"]): d for d in map(json.loads, p.read_text().splitlines()) if d.get("shot") is not None}
sub = {k: r for k, r in key.items() if any(c in r["match"] for c in CLEAN)}
tdb = {}
for m, c in zip(matches, CACHES):
    for (tc, _), s in zip(m["peaks"], m["td"]):
        k = (Path(c).parent.name, round(tc, 3))
        if k in sub: tdb[k] = s
print(f"=== Banco de candidatos difíciles, solo partidos limpios ({len(sub)} candidatos: "
      f"{sum(r['kind']=='shot' for r in sub.values())} tiros, {sum(r['kind']=='cross' for r in sub.values())} centros, "
      f"{sum(r['kind']=='unannotated' for r in sub.values())} repeticiones/sin anotar) ===")
def auc(pos, neg): return np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
rows = [("V-JEPA2 (score)", {k: (r["vjepa"], r["vjepa"] > 0.3) for k, r in sub.items()}),
        ("Qwen3.5-9B sí/no v1", {k: (r["vlm_v1"], r["vlm_v1"] > 0.5) for k, r in sub.items()})]
for name, lab in (("frames_json", "Qwen3.5-9B análisis+JSON"), ("video_think", "Qwen3.5-9B vídeo+razonamiento"),
                  ("videoarm", "VideoARM + Qwen3.5-9B"), ("omni_video", "Qwen3-Omni-30B vídeo+razon.")):
    rows.append((lab, {k: (res[name][k]["score"], res[name][k]["shot"]) for k in sub if k in res.get(name, {})}))
rows.append(("T-DEED (test) SHOT", {k: (tdb[k], tdb[k] >= 0.3) for k in sub if k in tdb}))
print(f"{'método':<32}{'n':>4} {'dice tiro: tiros':>17} {'centros':>9} {'repeticiones':>13} {'AUC tiro/centro':>16}")
for name, d in rows:
    by = lambda kind: [d[k] for k in d if sub[k]["kind"] == kind]
    s, c, u = by("shot"), by("cross"), by("unannotated")
    print(f"{name:<32}{len(d):>4} {100*np.mean([x[1] for x in s]):>16.0f}% {100*np.mean([x[1] for x in c]):>8.0f}% {100*np.mean([x[1] for x in u]):>12.0f}% {auc([x[0] for x in s], [x[0] for x in c]):>16.2f}")

# 2) partido completo, 3 partidos limpios
sel = [m for m in matches if m["clean"]]
n_gt = sum(len(m["gt"]) for m in sel)
print(f"\n=== Partido completo, 3 partidos EFL limpios ({n_gt} tiros) ===")
def full(label, per):
    for tol in (3.0,):
        ap = average_precision(per, tol)
        best = {}
        for th in sorted({round(p.score, 3) for _, ps in per for p in ps}, reverse=True):
            tot = Counts()
            for g, ps in per: tot.add(count(g, ps, tol, th))
            for r in (0.5, 0.7):
                if tot.recall >= r and tot.precision > best.get(r, (0, 0))[0]: best[r] = (tot.precision, tot.n_pred / len(per))
        cells = "  ".join(f"P@R{int(r*100)} {100*best[r][0]:4.0f}% ({best[r][1]:3.0f} alarmas/partido)" if r in best else f"P@R{int(r*100)} --" for r in (0.5, 0.7))
        print(f"  {label:<40} AP ±3s {100*ap:5.1f}%   {cells}")
full("V-JEPA2 + SVC (picos)", [(m["gt"], [Prediction(t, t, t, float(s)) for t, s in m["peaks"]]) for m in sel])
full("V-JEPA2 → verificador Qwen3.5-9B (v1)", [(m["gt"], [Prediction(t, t, t, float(s)) for (t, _), s in zip(m["peaks"], m["p_vlm"])]) for m in sel])
full("V-JEPA2 → T-DEED como verificador", [(m["gt"], [Prediction(t, t, t, float(s)) for (t, _), s in zip(m["peaks"], m["td"])]) for m in sel])
full("T-DEED (test) solo, picos SHOT", [(m["gt"], [Prediction(t, t, t, float(s)) for t, s in m["td_peaks"]]) for m in sel])
