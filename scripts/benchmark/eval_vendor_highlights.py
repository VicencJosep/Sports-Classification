"""Evalúa lo que devuelve un servicio de highlights (STREET u otro) sobre un
partido COMPLETO de SoccerTrack v2 (CC BY 4.0): goles de su página del partido
y segmentos de su vídeo de highlights, contra la verdad de terreno de 12 clases
del dataset, y nuestro T-DEED (caché de la cámara virtual) con la misma regla.

GT: ``output/benchmark/vendor/st2_<partido>_full.gt.json`` (se genera desde
``datasets/soccertrack-v2/bas/<partido>/<partido>_12_class_events.json``;
tiempo relativo al vídeo de cada parte, como el dataset).

Entradas del servicio (CSV transcritos a mano; ``#`` comenta):
* ``--goals``: ``parte,mm:ss[,equipo]`` con parte 1|2 y equipo left|right
  (lado en pantalla de la panorámica, como el GT) o home|away si no se sabe.
* ``--segments``: ``parte,inicio,fin[,etiqueta]`` con tiempos ``mm:ss`` del
  vídeo original de esa parte (no del reel). Si el reel no deja situar el
  segmento, se anota el instante central y se usa ``--seg-pad`` a cada lado.

Métricas:
* Goles: P/R/F1 a ±10 s y ±30 s (1 a 1 por parte), marcador por lado.
* Segmentos: cobertura de goles y de tiros del GT (el instante del GT cae
  dentro del segmento ±2 s), precisión (segmentos con al menos un gol o tiro
  del GT), número y duración total.
* Nuestro T-DEED (``--ours SNB_pano1``): picos de la clase GOAL y picos de
  SHOT+GOAL (umbral 0,3, NMS 1 s) como «candidatos a highlight» de 15 s
  ([t-10, t+5]), con las mismas métricas.

Uso (contenedor):
    python -m scripts.benchmark.eval_vendor_highlights --match 128057 --ours SNB_pano1
    python -m scripts.benchmark.eval_vendor_highlights --match 128057 --vendor-name street \\
        --goals output/benchmark/vendor/street_128057_goals.csv --segments output/benchmark/vendor/street_128057_segments.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.benchmark.eval_vendor import parse_time, prf

ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = ROOT / "output" / "benchmark" / "vendor"
ST2 = ROOT / "datasets" / "soccertrack-v2"
HALVES = {"1": "1st", "2": "2nd"}
GOAL_TOLS = (10.0, 30.0)
SEG_MARGIN = 2.0
OURS_BEFORE, OURS_AFTER = 10.0, 5.0


def build_gt(match: str) -> dict:
    src = ST2 / "bas" / match / f"{match}_12_class_events.json"
    d = json.loads(src.read_text())
    events = [{"half": HALVES[a["gameTime"].split(" - ")[0]], "t_s": round(int(a["position"]) / 1000, 3),
               "label": a["label"], "team": a.get("team"), "player_id": a.get("player_id")} for a in d["actions"]]
    return {"match": match, "source": f"SoccerTrack v2 (CC BY 4.0), {src.name}; tiempo relativo al vídeo de cada parte",
            "events": events}


def read_rows(path: Path) -> list[list[str]]:
    with open(path, newline="") as f:
        return [[c.strip() for c in r] for r in csv.reader(f)
                if r and r[0].strip() and not r[0].startswith("#") and r[0].strip().lower() not in ("parte", "half")]


def read_goals(path: Path) -> list[tuple[str, float, str | None]]:
    return [(HALVES[r[0]], parse_time(r[1]), r[2].lower() if len(r) > 2 and r[2] else None) for r in read_rows(path)]


def read_segments(path: Path, pad: float) -> list[tuple[str, float, float, str]]:
    out = []
    for r in read_rows(path):
        a, b = parse_time(r[1]), parse_time(r[2]) if len(r) > 2 and r[2] else None
        if b is None:
            a, b = a - pad, a + pad
        out.append((HALVES[r[0]], a, b, r[3] if len(r) > 3 else ""))
    return out


def ours(match: str, model: str, view: str, min_score: float, nms_s: float):
    """(goles por picos de GOAL, segmentos por picos de SHOT+GOAL) de nuestra caché."""
    from core.spotting import find_peaks

    goals, segs = [], []
    for half in HALVES.values():
        z = np.load(ST2 / "views" / match / f"{half}_{view}.tdeed.{model}.npz")
        probs, classes, stride = z["probs"].astype(np.float32), list(z["classes"]), int(z["stride"])
        t = (np.arange(len(probs)) * stride / 25.0).tolist()
        for ft, _ in find_peaks(t, probs[:, classes.index("GOAL")].tolist(), min_score, nms_s):
            goals.append((half, ft, None))
        curve = probs[:, classes.index("SHOT")] + probs[:, classes.index("GOAL")]
        for ft, _ in find_peaks(t, curve.tolist(), min_score, nms_s):
            segs.append((half, ft - OURS_BEFORE, ft + OURS_AFTER, "shot"))
    return goals, segs


def eval_goals(name: str, goals: list[tuple[str, float, str | None]], gt: list[dict]) -> dict:
    gt_goals = [e for e in gt if e["label"] == "GOAL"]
    res = {"n_gt": len(gt_goals), "n_pred": len(goals)}
    print(f"\n== {name}: goles  (GT {len(gt_goals)}, servicio {len(goals)})")
    for tol in GOAL_TOLS:
        tp = fp = fn = 0
        for half in HALVES.values():
            a, b, c = prf([e["t_s"] for e in gt_goals if e["half"] == half], [t for h, t, _ in goals if h == half], tol)
            tp, fp, fn = tp + a, fp + b, fn + c
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        res[f"tol{tol:g}"] = {"TP": tp, "FP": fp, "FN": fn, "P": round(100 * P, 1), "R": round(100 * R, 1), "F1": round(100 * F, 1)}
        print(f"   ±{tol:g} s: TP {tp} FP {fp} FN {fn}  P {100 * P:.1f} R {100 * R:.1f} F1 {100 * F:.1f}")
    score_gt = {s: sum(e["team"] == s for e in gt_goals) for s in ("left", "right")}
    res["score_gt"] = score_gt
    if any(team for _, _, team in goals):
        res["score_pred"] = {s: sum(team == s for _, _, team in goals) for s in ("left", "right")}
        print(f"   marcador GT left-right {score_gt['left']}-{score_gt['right']}, servicio "
              f"{res['score_pred']['left']}-{res['score_pred']['right']}")
    return res


def eval_segments(name: str, segs: list[tuple[str, float, float, str]], gt: list[dict]) -> dict:
    def covered(e: dict) -> bool:
        return any(h == e["half"] and a - SEG_MARGIN <= e["t_s"] <= b + SEG_MARGIN for h, a, b, _ in segs)

    def contains(h: str, a: float, b: float, label: str) -> bool:
        return any(e["half"] == h and e["label"] == label and a - SEG_MARGIN <= e["t_s"] <= b + SEG_MARGIN for e in gt)

    goals = [e for e in gt if e["label"] == "GOAL"]
    shots = [e for e in gt if e["label"] == "SHOT"]
    n_goal_cov, n_shot_cov = sum(map(covered, goals)), sum(map(covered, shots))
    with_goal = sum(contains(h, a, b, "GOAL") for h, a, b, _ in segs)
    with_shot_or_goal = sum(contains(h, a, b, "GOAL") or contains(h, a, b, "SHOT") for h, a, b, _ in segs)
    total = sum(b - a for _, a, b, _ in segs)
    res = {"n_segments": len(segs), "total_s": round(total, 1),
           "goals_covered": f"{n_goal_cov}/{len(goals)}", "shots_covered": f"{n_shot_cov}/{len(shots)}",
           "segments_with_goal": with_goal, "segments_with_shot_or_goal": with_shot_or_goal,
           "precision_shot_or_goal": round(100 * with_shot_or_goal / len(segs), 1) if segs else 0.0}
    print(f"== {name}: {len(segs)} segmentos, {total:.0f} s en total · cubren goles {res['goals_covered']}, tiros "
          f"{res['shots_covered']} · segmentos con gol {with_goal}, con tiro o gol {with_shot_or_goal} "
          f"({res['precision_shot_or_goal']} %)")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", default="128057")
    ap.add_argument("--vendor-name", default="servicio")
    ap.add_argument("--goals", type=Path, default=None, help="CSV parte,mm:ss[,equipo] del servicio")
    ap.add_argument("--segments", type=Path, default=None, help="CSV parte,inicio,fin[,etiqueta] del servicio")
    ap.add_argument("--seg-pad", type=float, default=7.5, help="Medio ancho si el CSV solo da el instante central")
    ap.add_argument("--ours", default=None, help="Modelo T-DEED de la caché de la cámara virtual (p.ej. SNB_pano1)")
    ap.add_argument("--view", default="vcam796vl1")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--nms-s", type=float, default=1.0)
    args = ap.parse_args()

    gt_path = VENDOR_DIR / f"st2_{args.match}_full.gt.json"
    if not gt_path.exists():
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.write_text(json.dumps(build_gt(args.match), indent=1))
    gt = json.loads(gt_path.read_text())["events"]
    goals_gt = [e for e in gt if e["label"] == "GOAL"]
    print(f"Partido {args.match}: {len(gt)} acciones GT, {len(goals_gt)} goles "
          + ", ".join(f"{e['half']} {int(e['t_s'] // 60)}:{int(e['t_s'] % 60):02d} {e['team']}" for e in goals_gt))
    results = {}
    if args.goals or args.segments:
        r = {}
        if args.goals:
            r["goals"] = eval_goals(args.vendor_name, read_goals(args.goals), gt)
        if args.segments:
            r["segments"] = eval_segments(args.vendor_name, read_segments(args.segments, args.seg_pad), gt)
        results[args.vendor_name] = r
    if args.ours:
        g, s = ours(args.match, args.ours, args.view, args.min_score, args.nms_s)
        name = f"T-DEED {args.ours} @{args.min_score:g}"
        results[name] = {"goals": eval_goals(f"{name} (picos GOAL, umbral {args.min_score})", g, gt),
                         "segments": eval_segments(f"{name} (picos SHOT+GOAL, [t-{OURS_BEFORE:g}, t+{OURS_AFTER:g}] s)", s, gt)}
    out = VENDOR_DIR / f"st2_{args.match}_full.eval.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(results)
    out.write_text(json.dumps(prev, indent=1))


if __name__ == "__main__":
    main()
