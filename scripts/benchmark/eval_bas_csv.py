"""Acierto de las cachés de los spotters (T-DEED, AdaSpot: ``<vídeo>.tdeed.<modelo>.npz``) contra la hoja de
anotación manual de un partido propio (``scripts/annotation_csv.py``), sin pasar por el pipeline.

Tres bloques:

1. **Acciones de balón, mAP@1** (±0,5 s) y a ±1 s, con el post-proceso (NMS suave) y el evaluador del repo de
   T-DEED, como ``eval_bas.py``: las cifras van en la misma escala que la tabla de docs/comparativa_benchmark.md.
2. **P/R/F1** por clase a un umbral global: el que maximiza la F1 micro, 0,15 (convención PCBAS) y 0,5; emparejado
   1 a 1, greedy por confianza (``eval_prf.match``), a ±0,5 s y ±1 s.
3. **Eventos** con los parámetros de producción (``config.BallSpottingConfig``: umbral, NMS y suavizado), emparejados
   1 a 1 a ±3 s como ``scripts/evaluate_match.py``.

Convenciones:

* Solo se puntúa el tramo anotado (de la primera a la última fila del CSV, más la tolerancia): lo que el CSV no
  cubre no cuenta ni como acierto ni como falso positivo.
* Un ``goal`` del CSV cuenta también como ``shot`` (gol ⊂ tiro, igual que ``evaluate_match.py``).
* Las medias "comunes" y la F1 micro solo cubren las clases que tienen anotación en el CSV y que predicen todos los
  modelos (sin FREE KICK ni GOAL, que AdaSpot no tiene). Las clases que la hoja no anota (p. ej. CROSS u OUT si el
  anotador no las usó) no puntúan: no se sabe si sus predicciones son falsas alarmas.
* Tiempos SIEMPRE con ``--fps`` (el real del vídeo). Las cachés de AdaSpot guardan ``fps=25`` fijo y
  ``util.io.store_json_snb`` de T-DEED convierte con 25 fps fijos; con un vídeo a 30 fps los dos desplazarían los
  instantes un 20 %, así que aquí se escriben las predicciones con el fps real y la ventana del NMS se escala a él.
* ``--offset-scan``: F1 micro de un modelo frente a un desfase añadido a sus predicciones, para comprobar que el
  vídeo de las cachés y el de la anotación están alineados (pico en 0).

Uso (contenedor, raíz del proyecto):
    python -m scripts.benchmark.eval_bas_csv --csv ground_truth/ground_truth_00m00-107m26.csv \\
        --video output/la_roca_inferencia/la_roca_56m.mp4 --fps 30 --tag la_roca --offset-scan
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import os
from pathlib import Path

import numpy as np

import config
from core.ball_spotting import EVENT_SOURCES, read_scores, spot_events
from scripts.annotation_csv import read_annotation_csv
from scripts.benchmark.eval_bas import _tdeed_eval, candidates, tdeed_eval_classes
from scripts.benchmark.eval_prf import match, prf
from scripts.evaluate_pipeline import _match_greedy

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "output" / "benchmark" / "bas_csv"

# acción de balón de annotation_csv -> clase de SN-BAS
TO_BAS = {
    "pass": "PASS", "drive": "DRIVE", "high_pass": "HIGH PASS", "header": "HEADER", "throw_in": "THROW IN",
    "cross": "CROSS", "shot": "SHOT", "goal": "GOAL", "out": "OUT", "free_kick": "FREE KICK",
    "player_successful_tackle": "PLAYER SUCCESSFUL TACKLE", "ball_player_block": "BALL PLAYER BLOCK",
}
NOT_IN_ADASPOT = {"FREE KICK", "GOAL"}
ROWS_PER_S_SNB = 12.5            # ventanas de NMS de T-DEED definidas a 25 fps con paso 2
ACTION_TOLS_MS = (500.0, 1000.0)
EVENT_TOL_S = 3.0
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def gt_actions(rows: list[dict]) -> list[tuple[str, float]]:
    """(clase BAS, segundos); un gol también como tiro."""
    out = []
    for r in rows:
        label = TO_BAS.get(r.get("ball_action_label") or "")
        if label:
            out.append((label, r["t_s"]))
            if label == "GOAL":
                out.append(("SHOT", r["t_s"]))
    return out


def write_labels(path: Path, actions: list[tuple[str, float]]) -> None:
    anns = [{"gameTime": f"1 - {int(t // 60):02d}:{int(t % 60):02d}", "label": c, "position": str(int(round(t * 1000))),
             "team": "", "visibility": "visible"} for c, t in actions]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"annotations": anns}, indent=1))


def write_preds(path: Path, game: str, events: list[dict], ms_per_row: float) -> list[dict]:
    """Igual que ``util.io.store_json_snb`` pero con el fps real; devuelve las predicciones escritas."""
    preds = []
    for e in events:
        pos = int(e["frame"] * ms_per_row)
        preds.append({"gameTime": f"1 - {pos // 60000}:{(pos % 60000) // 1000}", "label": e["label"], "position": pos,
                      "confidence": e["score"], "half": 1})
    path.mkdir(parents=True, exist_ok=True)
    (path / "results_spotting.json").write_text(json.dumps({"UrlLocal": game, "predictions": preds}))
    return preds


def prf_table(gt: dict[str, np.ndarray], preds: list[dict], classes: list[str], tol_ms: float) -> dict:
    """F1 micro sobre ``classes`` frente al umbral global; por clase al mejor umbral, a 0,15 y a 0,5."""
    by_class = {c: [(p["position"], p["confidence"]) for p in preds if p["label"] == c] for c in classes}

    def at(thr: float) -> dict:
        per = {}
        for c in classes:
            per[c] = match(gt[c], [p for p in by_class[c] if p[1] >= thr], tol_ms)
        tp, fp, fn = (sum(v[i] for v in per.values()) for i in range(3))
        p, r, f1 = prf(tp, fp, fn)
        return {"thr": float(thr), "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1,
                "per_class": {c: dict(zip(("tp", "fp", "fn"), v), f1=prf(*v)[2]) for c, v in per.items()}}

    sweep = [at(t) for t in THRESHOLDS]
    return {"best": max(sweep, key=lambda s: s["f1"]), "0.15": at(0.15), "0.5": at(0.5)}


def offset_scan(gt: dict[str, np.ndarray], preds: list[dict], classes: list[str], thr: float) -> list[tuple[float, float]]:
    """F1 micro (±0,5 s) con las predicciones desplazadas: grueso ±120 s y fino ±3 s alrededor del mejor."""
    def f1_at(shift_ms: float) -> float:
        tp = fp = fn = 0
        for c in classes:
            a, b, d = match(gt[c], [(p["position"] + shift_ms, p["confidence"]) for p in preds
                                    if p["label"] == c and p["confidence"] >= thr], 500.0)
            tp, fp, fn = tp + a, fp + b, fn + d
        return prf(tp, fp, fn)[2]

    coarse = [(s, f1_at(s * 1000)) for s in range(-120, 121)]
    best = max(coarse, key=lambda x: x[1])[0]
    fine = [(round(best + s / 10, 1), f1_at((best + s / 10) * 1000)) for s in range(-30, 31)]
    return sorted(coarse + fine)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--video", required=True, type=Path, help="vídeo de las cachés (<vídeo>.tdeed.<modelo>.npz)")
    ap.add_argument("--fps", required=True, type=float, help="fps REAL del vídeo")
    ap.add_argument("--tag", required=True, help="nombre del partido en la salida")
    ap.add_argument("--models", nargs="*", help="por defecto, todas las cachés junto al vídeo")
    ap.add_argument("--offset-scan", action="store_true")
    ap.add_argument("--offset-model", default="SoccerNetBall_challenge2")
    args = ap.parse_args()

    rows, warnings = read_annotation_csv(args.csv)
    for w in warnings:
        print("aviso CSV:", w)
    t0, t1 = min(r["t_s"] for r in rows), max(r["t_s"] for r in rows)
    actions = gt_actions(rows)
    out_root = OUT_ROOT / args.tag
    label_root = out_root / "labels"
    write_labels(label_root / args.tag / "Labels-ball.json", actions)
    gt_counts = {c: sum(1 for a, _ in actions if a == c) for c in tdeed_eval_classes()}
    annotated = [c for c in tdeed_eval_classes() if gt_counts[c]]
    common = [c for c in annotated if c not in NOT_IN_ADASPOT]
    gt_ms = {c: np.array([t * 1000 for a, t in actions if a == c]) for c in tdeed_eval_classes()}

    caches = sorted(args.video.parent.glob(f"{args.video.stem}.tdeed.*.npz"))
    models = args.models or [p.name.split(".tdeed.")[1][:-4] for p in caches]
    bs_cfg = config.BallSpottingConfig()
    event_params = {lab: (thr, nms) for lab, thr, nms in bs_cfg.event_params}
    gt_events: dict[str, list[float]] = {}
    for r in rows:
        if r.get("event_label"):
            gt_events.setdefault(r["event_label"], []).append(r["t_s"])
            if r["event_label"] == "goal":
                gt_events.setdefault("shot", []).append(r["t_s"])

    tdeed_eval = _tdeed_eval()
    summary = {"csv": str(args.csv), "video": str(args.video), "fps": args.fps, "span_s": [t0, t1],
               "gt_counts": gt_counts, "gt_events": {k: len(v) for k, v in gt_events.items()},
               "classes_annotated": annotated, "classes_common": common, "models": {}}
    for model in models:
        sc = read_scores(args.video.with_name(f"{args.video.stem}.tdeed.{model}.npz"))
        sc = dataclasses.replace(sc, fps=args.fps)
        row_s = sc.stride / args.fps
        lo, hi = int((t0 - 1.0) / row_s), int(np.ceil((t1 + 1.0) / row_s))
        probs = sc.probs.copy()
        probs[:max(lo, 0)] = 0.0
        probs[hi + 1:] = 0.0
        w_snms = max(1, round(tdeed_eval.WINDOWS_SNB[1] / ROWS_PER_S_SNB / row_s))
        pred = tdeed_eval.soft_non_maximum_supression(
            [{"video": args.tag, "events": candidates(probs, sc.classes), "fps": args.fps / sc.stride}],
            window=w_snms, threshold=0.01)
        preds = write_preds(out_root / model / "preds" / args.tag, args.tag, pred[0]["events"], row_s * 1000)

        res_m: dict = {"snms_window_rows": w_snms, "mAP": {}, "prf": {}}
        for metric in ("at1", "at2"):
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                res = tdeed_eval.aux_evaluate(str(label_root), str(out_root / model / "preds"), list_games=[args.tag],
                                              prediction_file="results_spotting.json", version=2, metric=metric,
                                              num_classes=12, label_files="Labels-ball.json", dataset="Ball",
                                              framerate=25)
            per_class = {c: round(100 * float(v), 2) for c, v in zip(tdeed_eval_classes(), res["a_mAP_per_class"])}
            has = [c for c in annotated if c in sc.classes]
            res_m["mAP"][metric] = {
                "annotated": round(float(np.mean([per_class[c] for c in has])), 2),
                "common": round(float(np.mean([per_class[c] for c in common])), 2),
                "per_class": {c: per_class[c] for c in annotated},
            }
        for tol in ACTION_TOLS_MS:
            res_m["prf"][f"{int(tol)}ms"] = prf_table(gt_ms, preds, common, tol)
        if args.offset_scan and model == args.offset_model:
            thr = res_m["prf"]["500ms"]["best"]["thr"]
            summary["offset_scan"] = {"model": model, "thr": thr, "f1_by_shift_s": offset_scan(gt_ms, preds, common, thr)}

        spotted = [e for e in spot_events(sc, event_params, bs_cfg.event_smooth_frames)
                   if t0 - EVENT_TOL_S <= e.t_s <= t1 + EVENT_TOL_S]
        ev = {}
        for label, gts in sorted(gt_events.items()):
            if not any(s in sc.classes + sc.sn_classes for s in EVENT_SOURCES[label]):
                ev[label] = None          # el modelo no tiene esta clase
                continue
            p = sorted((e for e in spotted if e.label == label or (label == "shot" and e.label == "goal")),
                       key=lambda e: e.t_s)
            m = _match_greedy(sorted(gts), [e.t_s for e in p], EVENT_TOL_S)
            tp = len(m)
            ev[label] = {"n_gt": len(gts), "n_pred": len(p), "tp": tp, "precision": tp / len(p) if p else None,
                         "recall": tp / len(gts), "emitted_in_production": label in bs_cfg.tdeed_event_labels,
                         "params": event_params[label], "pred": [(round(e.t_s, 1), round(e.score, 2)) for e in p]}
        res_m["events"] = ev
        summary["models"][model] = res_m
        best = res_m["prf"]["500ms"]["best"]
        print(f"{model:26s} mAP@1 comunes {res_m['mAP']['at1']['common']:5.1f}  anotadas "
              f"{res_m['mAP']['at1']['annotated']:5.1f}  | ±1 s {res_m['mAP']['at2']['common']:5.1f}  | "
              f"F1 micro {100 * best['f1']:4.1f} (P {100 * best['precision']:4.1f} R {100 * best['recall']:4.1f} "
              f"@{best['thr']:.2f})")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("->", out_root / "summary.json")


if __name__ == "__main__":
    os.environ.setdefault("TQDM_DISABLE", "1")
    main()
