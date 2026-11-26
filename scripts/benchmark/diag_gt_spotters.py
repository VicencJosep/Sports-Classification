"""Diagnóstico de los spotters contra una verdad propia (sin GPU): de dónde salen los fallos.

Complementa a ``eval_gt_prf.py`` (misma verdad, mismas cachés ``<vídeo>.tdeed.<modelo>.npz``, mismo post-proceso y mismo
emparejamiento ±0,5 s) con lo que la tabla de precisión / recall no dice:

* intervalo de confianza de la F1 (bootstrap por bloques de un minuto) al umbral óptimo y a un umbral fijo, y la
  diferencia de cada modelo con el de referencia (el primero de ``--models``) sobre los mismos bloques;
* presencia sin etiqueta (¿hay una acción en ese instante?) a ±0,5, ±1 y ±2 s: separa instante de detección;
* confianza: probabilidad máxima a ±0,5 s de cada fila de la verdad frente a instantes sin evento (ceguera o confianza);
* punto de producto: umbral automático del pipeline (``analytics.engine.calibrate_spot_thresholds``) y lo que emite;
* del modelo de referencia: desfase de ancla por clase, etiqueta en las filas emparejadas, a qué distancia de la verdad
  caen los falsos positivos y si otro modelo dispara también en ellos;
* parones: picos por minuto dentro y fuera de los huecos de más de 20 s sin ninguna fila anotada.

No ajusta nada: describe. Las clases son las BAS presentes en la verdad sin FREE KICK ni GOAL (como ``eval_gt_prf``).

Uso (contenedor):
    python -m scripts.benchmark.diag_gt_spotters --gt backend/storage/outputs/<id>/full/review/ground_truth_*.csv \
        --video output/la_roca_inferencia/la_roca_56m.mp4 --fps 30 \
        --models SoccerNetBall_challenge2 SNB_pano1 AdaSpot_small_adabn --json-out output/x.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.benchmark import eval_gt_prf as E

MERGE = {"PASS": "PASE", "HIGH PASS": "PASE"}


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return p, r, f


def pct(x) -> list[float]:
    return [round(100 * float(v), 1) for v in x]


def match(gt: np.ndarray, pt: np.ndarray, ps: np.ndarray, tol: float):
    """Emparejamiento 1 a 1 greedy por confianza: (verdad emparejada, predicción acertada, pareja de cada verdad)."""
    used = np.zeros(len(gt), bool)
    tp = np.zeros(len(pt), bool)
    pair = -np.ones(len(gt), int)
    for j in np.argsort(-ps):
        d = np.abs(gt - pt[j])
        d[used] = np.inf
        if len(d) and d.min() <= tol:
            i = int(d.argmin())
            used[i], tp[j], pair[i] = True, True, j
    return used, tp, pair


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True, help="el primero es el de referencia (producción)")
    ap.add_argument("--fps", type=float)
    ap.add_argument("--span", type=float, nargs=2)
    ap.add_argument("--fixed", type=float, default=0.25, help="umbral fijo común a todos los modelos")
    ap.add_argument("--title", default="")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    all_rows, span = E.load_gt(args.gt, tuple(args.span) if args.span else None)
    present = {r["cls"] for r in all_rows}
    act = [c for c in E.BAS if c in present and c not in E.NOT_TUNED]
    rows = [r for r in all_rows if r["cls"] in act]
    gt_t = np.array([r["t"] for r in rows])
    gt_c = np.array([r["cls"] for r in rows])
    all_t = np.sort(np.array([r["t"] for r in all_rows]))
    gt = {c: gt_t[gt_c == c] for c in act}
    mins = (span[1] - span[0]) / 60
    nblk = int(mins) + 1
    boot = rng.integers(0, nblk, (2000, nblk))
    gaps = [(a, b) for a, b in zip(all_t[:-1], all_t[1:]) if b - a > 20]
    gap_in = sum(b - a - 4 for a, b in gaps) / 60          # 2 s de margen a cada lado
    gap_out = mins - sum(b - a for a, b in gaps) / 60
    out: dict = {"title": args.title, "span": list(span), "classes": act,
                 "gt": {"rows": len(all_rows), "actions": len(rows), "minutes": round(mins, 1),
                        "per_min": round(len(rows) / mins, 1), "gaps20": len(gaps),
                        "gaps20_min": round(sum(b - a for a, b in gaps) / 60, 1),
                        "per_min_play": round(len(all_rows) / gap_out, 1) if gap_out > 0 else None}, "models": {}}
    print(f"== {args.title}: {len(rows)} acciones ({len(all_rows)} filas) en {mins:.1f} min = {len(rows) / mins:.1f}/min; "
          f"{len(gaps)} huecos >20 s = {out['gt']['gaps20_min']} min; en juego {out['gt']['per_min_play']} filas/min")

    C = {m: E.load_cache(args.video.with_name(f"{args.video.stem}.tdeed.{m}.npz"), args.fps) for m in args.models}

    def any_probs(c):
        m = c.probs[:, [c.classes.index(k) for k in act]].max(1)
        return np.stack([1 - m, m], 1)

    def any_peaks(c):
        a = np.array(E.peaks(any_probs(c), ["background", "ANY"], c.stride, c.fps, span)["ANY"])
        return a[:, 0], a[:, 1]

    def win_max(col, t, c, tol=0.5):
        i0 = max(0, int((t - tol) * c.fps / c.stride))
        return float(col[i0:int((t + tol) * c.fps / c.stride) + 1].max())

    def blocks(ts):
        return np.bincount(((np.asarray(ts) - span[0]) // 60).astype(int), minlength=nblk)[:nblk]

    def f1_boot(b):
        tp, fp, fn = (x[boot].sum(-1) for x in b)
        return 2 * tp / np.maximum(2 * tp + fp + fn, 1)

    PK = {m: E.peaks(C[m].probs, C[m].classes, C[m].stride, C[m].fps, span) for m in args.models}
    ANY = {m: any_peaks(C[m]) for m in args.models}

    def events(m, thr, merge):
        ev = sorted(((s, t, merge.get(k, k)) for k in act for t, s in PK[m].get(k, []) if s >= thr), reverse=True)
        kept: list = []
        for s, t, k in ev:      # con clases fundidas, un solo pico por golpeo y clase final
            if not merge or not any(k2 == k and abs(t2 - t) <= 0.4 for _, t2, k2 in kept):
                kept.append((s, t, k))
        gc = np.array([merge.get(k, k) for k in gt_c])
        tp_t, fp_t, fn_t = [], [], []
        for k in sorted(set(gc)):
            p = [(t, s) for s, t, kk in kept if kk == k]
            ptk, psk = np.array([x[0] for x in p]), np.array([x[1] for x in p])
            u, tpk, _ = match(gt_t[gc == k], ptk, psk, 0.5)
            tp_t += list(ptk[tpk])
            fp_t += list(ptk[~tpk])
            fn_t += list(gt_t[gc == k][~u])
        return [blocks(x) for x in (tp_t, fp_t, fn_t)], prf(len(tp_t), len(fp_t), len(fn_t))

    cand = rng.uniform(span[0] + 2, span[1] - 2, 8000)
    bg = cand[np.abs(cand[:, None] - all_t[None, :]).min(1) > 1.5][:1500]

    # ------------------------------------------------------------ por modelo
    B_fixed = {}
    for m in args.models:
        c = C[m]
        r: dict = {}
        thr = E.best_threshold(gt, PK[m], act, 0.5, E.GRID)
        b_opt, p_opt = events(m, thr, {})
        b_fix, p_fix = events(m, args.fixed, {})
        _, p_mer = events(m, args.fixed, MERGE)
        B_fixed[m] = b_fix
        d_opt, d_fix = f1_boot(b_opt), f1_boot(b_fix)
        r["thr_opt"] = thr
        r["opt"], r["opt_ci"] = pct(p_opt), pct(np.percentile(d_opt, [2.5, 97.5]))
        r["fixed"], r["fixed_ci"] = pct(p_fix), pct(np.percentile(d_fix, [2.5, 97.5]))
        r["fixed_merged_pass"] = pct(p_mer)
        pt, ps = ANY[m]
        r["any"] = {}
        for t_any in (0.2, 0.3):
            k = ps >= t_any
            r["any"][str(t_any)] = {"per_min": round(float(k.sum() / mins), 1)}
            for tol in (0.5, 1.0, 2.0):
                u, tp, _ = match(gt_t, pt[k], ps[k], tol)
                r["any"][str(t_any)][f"tol{tol:g}"] = pct((tp.mean() if k.any() else 0, u.mean()))
        anyc = any_probs(c)[:, 1]
        own = np.array([win_max(c.probs[:, c.classes.index(k)], t, c) for t, k in zip(gt_t, gt_c)])
        anyv = np.array([win_max(anyc, t, c) for t in gt_t])
        bgv = np.array([win_max(anyc, t, c) for t in bg])
        r["conf"] = {"own_med": round(float(np.median(own)), 2), "own_ge10": pct([(own >= .1).mean()])[0],
                     "own_ge20": pct([(own >= .2).mean()])[0], "any_med": round(float(np.median(anyv)), 2),
                     "any_ge10": pct([(anyv >= .1).mean()])[0], "any_ge20": pct([(anyv >= .2).mean()])[0],
                     "bg_med": round(float(np.median(bgv)), 2), "bg_ge10": pct([(bgv >= .1).mean()])[0],
                     "bg_ge20": pct([(bgv >= .2).mean()])[0], "bg_n": len(bg)}
        k = ps >= args.fixed
        inside = np.array([any(a + 2 < v < b - 2 for a, b in gaps) for v in pt[k]], bool)
        outside = ~np.array([any(a < v < b for a, b in gaps) for v in pt[k]], bool)
        r["gaps"] = {"in_per_min": round(float(inside.sum() / gap_in), 1) if gap_in > 0 else None,
                     "out_per_min": round(float(outside.sum() / gap_out), 1) if gap_out > 0 else None}
        out["models"][m] = r
        print(f"{m:26s} óptimo {thr:.2f}: {r['opt'][0]:.0f}/{r['opt'][1]:.0f} F1 {r['opt'][2]:.0f} IC {r['opt_ci']} | "
              f"fijo {args.fixed}: {r['fixed'][0]:.0f}/{r['fixed'][1]:.0f} F1 {r['fixed'][2]:.0f} IC {r['fixed_ci']} | "
              f"pase fundido F1 {r['fixed_merged_pass'][2]:.0f} | sin etiqueta@0.2 {r['any']['0.2']} | conf {r['conf']} | "
              f"parones {r['gaps']}")
    ref = args.models[0]
    for m in args.models[1:]:
        d = f1_boot(B_fixed[m]) - f1_boot(B_fixed[ref])
        out["models"][m]["diff_vs_ref_fixed"] = pct([d.mean(), *np.percentile(d, [2.5, 97.5])])
        print(f"   {m} − {ref} al umbral fijo: {out['models'][m]['diff_vs_ref_fixed']}")

    # ------------------------------------------------------------ punto de producto (umbral automático)
    c = C[ref]
    thr_ref = args.fixed
    try:
        import config
        from analytics import engine
        from core.ball_spotting import read_scores

        cfg = config.BallSpottingConfig()
        sc = read_scores(args.video.with_name(f"{args.video.stem}.tdeed.{ref}.npz"), args.fps or c.fps)
        th = engine.calibrate_spot_thresholds(sc, cfg, [lab for lab, _, _ in cfg.event_params])
        thr_ref = float(th.action_min_score)
        five = [k for k in ("PASS", "DRIVE", "HIGH PASS", "HEADER", "THROW IN") if k in act]
        g5 = np.isin(gt_c, five)
        out["auto"] = {"action": thr_ref, "rate_threshold": th.rate_threshold, "active_minutes": th.active_minutes,
                       "note": "minutos activos y ritmo calculados sobre la caché entera, no solo el tramo", "points": {}}
        for thr in sorted({thr_ref, 0.2, 0.25, 0.3, 0.5}):
            acts = [a for a in engine.spotted_ball_actions(sc, thr, cfg.action_nms_s)
                    if span[0] <= a.frame_idx / sc.fps <= span[1]]
            pt = np.array([a.frame_idx / sc.fps for a in acts])
            ps = np.array([a.score for a in acts])
            pl = np.array([E.ACTION_TO_BAS[a.label] for a in acts])
            u, tp, _ = match(gt_t[g5], pt, ps, 0.5)
            cnt = np.zeros(3, int)
            for k in five:
                uk, tk, _ = match(gt_t[gt_c == k], pt[pl == k], ps[pl == k], 0.5)
                cnt += (tk.sum(), (~tk).sum(), (~uk).sum())
            out["auto"]["points"][f"{thr:.2f}"] = {"per_min": round(len(acts) / mins, 1), "label": pct(prf(*cnt)),
                                                    "any": pct(prf(tp.sum(), (~tp).sum(), (~u).sum()))}
        print(f"producto: umbral automático {thr_ref} (pico k-ésimo {th.rate_threshold}, {th.active_minutes} min activos); "
              + "; ".join(f"@{k} {v['per_min']}/min etiqueta {v['label']} sin {v['any']}" for k, v in out["auto"]["points"].items()))
    except Exception as exc:  # noqa: BLE001  (caché sin cabeza SN-v2, modelo que no es T-DEED...)
        out["auto"] = {"error": repr(exc)}
        print("producto: no disponible:", repr(exc))

    # ------------------------------------------------------------ modelo de referencia a su umbral de producto
    R: dict = {"model": ref, "threshold": thr_ref}
    offs = {}
    for k in act:
        p = np.array([t for t, s in PK[ref].get(k, []) if s >= 0.10])
        d = [float(p[np.abs(p - t).argmin()] - t) for t in gt[k] if len(p) and np.abs(p - t).min() <= 1.5]
        offs[k] = {"n": len(d), "of": int(len(gt[k])), "median": round(float(np.median(d)), 2) if d else None,
                   "p25": round(float(np.percentile(d, 25)), 2) if d else None,
                   "p75": round(float(np.percentile(d, 75)), 2) if d else None}
    R["offset_s"] = offs
    pt, ps = ANY[ref]
    k = ps >= thr_ref
    pt, ps = pt[k], ps[k]
    used, tp, pair = match(gt_t, pt, ps, 0.5)
    fp_t = pt[~tp]
    cols = [c.classes.index(x) for x in act]

    def lab_of(t):
        return act[int(c.probs[min(len(c.probs) - 1, int(round(t * c.fps / c.stride))), cols].argmax())]

    conf = {x: Counter() for x in act}
    for i in np.flatnonzero(used):
        conf[gt_c[i]][lab_of(pt[pair[i]])] += 1
    ok = sum(conf[x][x] for x in act)
    R["matched"] = int(used.sum())
    R["label_ok_pct"] = round(100 * ok / max(1, int(used.sum())), 1)
    R["confusion"] = {x: dict(v.most_common(4)) for x, v in conf.items() if v}
    if len(fp_t):
        dist = np.abs(fp_t[:, None] - all_t[None, :]).min(1)
        R["fp"] = {"n": int(len(fp_t)), "peaks": int(len(pt)), "labels": dict(Counter(lab_of(t) for t in fp_t).most_common()),
                   "tp_labels": dict(Counter(lab_of(t) for t in pt[tp]).most_common()),
                   "dist_pct": {"0.5-1s": pct([((dist > .5) & (dist <= 1)).mean()])[0], "1-2s": pct([((dist > 1) & (dist <= 2)).mean()])[0],
                                "2-5s": pct([((dist > 2) & (dist <= 5)).mean()])[0], ">5s": pct([(dist > 5).mean()])[0]},
                   "also_fires": {}}
        for other in args.models[1:]:
            oc = C[other]
            oany = any_probs(oc)[:, 1]

            def share(ts, oany=oany, oc=oc):
                return round(100 * float(np.mean([win_max(oany, t, oc) >= 0.2 for t in ts])), 1) if len(ts) else None

            ot, os_ = ANY[other]
            uo, _, _ = match(gt_t, ot[os_ >= args.fixed], os_[os_ >= args.fixed], 0.5)
            R["fp"]["also_fires"][other] = {"tp": share(pt[tp]), "fp": share(fp_t), "bg": share(bg),
                                            "union_recall": pct([(used | uo).mean()])[0],
                                            "picks_ref_misses": pct([uo[~used].mean()])[0] if (~used).any() else None}
    R["recall_any"] = pct([used.mean()])[0]
    out["ref"] = R
    medians = ", ".join(f"{k}: {v['median']}" for k, v in offs.items())
    print(f"referencia {ref} @{thr_ref}: etiqueta correcta {R['label_ok_pct']} % de {R['matched']} emparejadas; "
          f"desfase mediano (s) {medians}; FP {R.get('fp')}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
