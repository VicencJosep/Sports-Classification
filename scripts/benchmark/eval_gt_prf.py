"""Precisión / recall por clase de los spotters contra una verdad propia (sin GPU).

Verdad: ``ground_truth/<partido>.gt.json`` (``build_ground_truth.py``; Middlesbrough 0-20 min) o la hoja
``tiempo,etiqueta,dorsal,equipo`` que exporta el visor (``review/ground_truth_*.csv``, leída con
``scripts/annotation_csv.py``). Predicciones: cachés ``<vídeo>.tdeed.<modelo>.npz`` de ``core.ball_spotting``,
``adaspot_scores.py`` o ``fuse_scores.py``, con el post-proceso del mAP oficial (candidatos ≥ 0,01 y NMS suave de
12 filas, como ``eval_bas.py``) y el emparejamiento de ``eval_prf.py`` (±0,5 s, 1 a 1, greedy por confianza, por
clase). Para cada modelo se barre un umbral global y se elige el que maximiza la F1 micro sobre las clases BAS
presentes en la verdad (sin FREE KICK ni GOAL, que AdaSpot no predice); se informan además umbrales fijos.

Convenciones: un GOAL de la verdad cuenta también como SHOT si no hay un SHOT anotado a ±1,5 s (gol ⊂ tiro, como
``evaluate_match.py``). Córner, falta y saque de centro no son clases BAS: se puntúan con la cabeza SoccerNet-v2
(``probs_sn``, solo T-DEED de producción) a su mejor umbral por clase. ``--fuse A B`` añade la media 0,5 / 0,5 de dos
cachés (``fuse_scores.fuse``) como modelo más.

Además de ``eval_prf.py``: la fila «cualquier acción» (máximo por fila sobre las clases BAS, sin mirar la etiqueta,
a su propio mejor umbral) y, con ``--edits review/edits.jsonl``, el reparto de la verdad del visor en filas que el
pipeline ya tenía (sin tocar o corregidas) y filas añadidas a mano, más cuántas de las borradas sigue disparando
cada modelo: mide cuánto del acierto es sesgo de origen (la hoja nace de las detecciones del pipeline).

Uso (contenedor):
    python -m scripts.benchmark.eval_gt_prf --gt ground_truth/2019-10-01_Middlesbrough-Preston_North_End.gt.json \
        --video "datasets/SoccerNet/england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End/720p.mp4" \
        --models SoccerNetBall_test SoccerNetBall_challenge2 AdaSpot_big --fuse SoccerNetBall_challenge2 AdaSpot_big
    python -m scripts.benchmark.eval_gt_prf --gt backend/storage/outputs/<id>/full/review/ground_truth_00m00-99m35.csv \
        --video backend/storage/videos/<id>/source.mp4 --models SoccerNetBall_challenge2 AdaSpot_big \
        --span 2865 5975 --edits backend/storage/outputs/<id>/full/review/edits.jsonl --md-out output/juveniles/x.md
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.annotation_csv import normalize_label, read_annotation_csv
from scripts.benchmark.eval_bas import candidates, tdeed_eval_classes
from scripts.benchmark.eval_prf import prf

logger = logging.getLogger("eval_gt_prf")

BAS = tuple(tdeed_eval_classes())
ACTION_TO_BAS = {"pass": "PASS", "drive": "DRIVE", "header": "HEADER", "high_pass": "HIGH PASS", "out": "OUT",
                 "cross": "CROSS", "throw_in": "THROW IN", "shot": "SHOT", "ball_player_block": "BALL PLAYER BLOCK",
                 "player_successful_tackle": "PLAYER SUCCESSFUL TACKLE", "free_kick": "FREE KICK", "goal": "GOAL"}
EVENT_TO_SN = {"corner": "Corner", "foul": "Foul", "kick-off": "Kick-off", "penalty": "Penalty"}
RAW_TO_SN = {"FOUL": "Foul", "KICK-OFF": "Kick-off", "CORNER": "Corner", "PENALTY": "Penalty"}
NOT_TUNED = {"FREE KICK", "GOAL"}
NAMES = {"SoccerNetBall_test": "T-DEED 200MF", "SoccerNetBall_challenge2": "T-DEED producción",
         "AdaSpot_big": "AdaSpot-b", "AdaSpot_small": "AdaSpot-s", "Fusion_T200_AdaB": "Fusión 200MF+AdaSpot-b",
         "SNB_pano1": "T-DEED SNB_pano1"}
ES = {"PASS": "Pase", "DRIVE": "Conducción", "HEADER": "Cabezazo", "HIGH PASS": "Pase alto", "OUT": "Fuera",
      "CROSS": "Centro", "THROW IN": "Banda", "SHOT": "Tiro", "BALL PLAYER BLOCK": "Bloqueo",
      "PLAYER SUCCESSFUL TACKLE": "Entrada", "FREE KICK": "Libre directo", "GOAL": "Gol", "Corner": "Córner",
      "Foul": "Falta", "Kick-off": "Saque de centro", "Penalty": "Penalti"}
GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]


# ----------------------------------------------------------------------------- verdad
def csv_label_class(action: str | None, event: str | None) -> str | None:
    if action:
        return ACTION_TO_BAS[action]
    return EVENT_TO_SN.get(event or "")


def load_gt(path: Path, span: tuple[float, float] | None) -> tuple[list[dict], tuple[float, float]]:
    """Filas ``{"t": s, "cls": clase BAS o SN, "raw": etiqueta original}`` dentro del tramo."""
    rows = []
    if path.suffix == ".json":
        g = json.loads(path.read_text())
        for a in g["annotations"]:
            raw = a["label_raw"]
            cls = raw if raw in BAS else RAW_TO_SN.get(raw)
            if cls:
                rows.append({"t": float(a["timestamp_s"]), "cls": cls, "raw": raw})
        if span is None:
            sl = g.get("slice") or {}
            span = (float(sl.get("start_s") or 0.0), float(sl.get("end_s") or max(r["t"] for r in rows)))
    else:
        raw_rows, warnings = read_annotation_csv(path)
        for w in warnings:
            logger.warning(w)
        for r in raw_rows:
            cls = csv_label_class(r["ball_action_label"], r["event_label"])
            if cls:
                rows.append({"t": r["t_s"], "cls": cls, "raw": r["label_raw"]})
        if span is None:
            span = (min(r["t"] for r in rows), max(r["t"] for r in rows))
    rows = [r for r in rows if span[0] <= r["t"] <= span[1]]
    shots = np.array([r["t"] for r in rows if r["cls"] == "SHOT"])
    for r in [r for r in rows if r["cls"] == "GOAL"]:
        if not len(shots) or np.abs(shots - r["t"]).min() > 1.5:
            rows.append({"t": r["t"], "cls": "SHOT", "raw": r["raw"] + " (gol)"})
    rows.sort(key=lambda r: r["t"])
    return rows, span


# ----------------------------------------------------------------------------- predicciones
def load_cache(path: Path) -> SimpleNamespace:
    z = np.load(path, allow_pickle=True)
    fps = float(z["fps"]) if "fps" in z.files else 25.0
    out = SimpleNamespace(probs=z["probs"].astype(np.float32), classes=[str(c) for c in z["classes"]],
                          stride=int(z["stride"]), fps=fps, probs_sn=None, classes_sn=None)
    if "probs_sn" in z.files:
        out.probs_sn, out.classes_sn = z["probs_sn"].astype(np.float32), [str(c) for c in z["classes_sn"]]
    return out


def soft_nms(ev: list[dict], window: int, threshold: float = 0.01) -> list[dict]:
    """Misma salida que ``util.eval.soft_non_maximum_supression`` de T-DEED (``eval_bas`` la usa tal cual) pero en
    O(n log n): por clase se toma el candidato de mayor confianza (empate: el más temprano), se atenúan los de
    ±window filas con (Δ/window)² y se repite hasta bajar del umbral. La de T-DEED es cuadrática y sobre una parte
    de un partido amateur (muchos candidatos ≥ 0,01) tarda más de 10 min por modelo."""
    by_label: dict[str, dict[int, float]] = defaultdict(dict)
    for e in ev:
        by_label[e["label"]][e["frame"]] = e["score"]
    out = []
    for label, cur in by_label.items():
        heap = [(-s, f) for f, s in cur.items()]
        heapq.heapify(heap)
        while heap:
            s, f = heapq.heappop(heap)
            s = -s
            if cur.get(f) != s:                      # entrada obsoleta (ya atenuada o elegida)
                continue
            if s < threshold:
                break
            out.append({"label": label, "frame": f, "score": s})
            del cur[f]
            for g in range(f - window, f + window + 1):
                if g in cur:
                    cur[g] = cur[g] * abs(f - g) ** 2 / window ** 2
                    heapq.heappush(heap, (-cur[g], g))
    out.sort(key=lambda e: e["frame"])
    return out


def peaks(probs: np.ndarray, classes: list[str], stride: int, fps: float,
          span: tuple[float, float]) -> dict[str, list[tuple[float, float]]]:
    """NMS suave de 12 filas a stride 2 (``eval_bas``) -> por clase, lista de (segundos, confianza) en el tramo."""
    w = max(1, round(12 * 2 / stride))
    lo = max(0, int((span[0] - 5.0) * fps / stride))           # solo las filas del tramo, con 5 s de margen
    hi = min(len(probs), int((span[1] + 5.0) * fps / stride) + 1)
    ev = soft_nms(candidates(probs[lo:hi], tuple(classes), 0.01), w, 0.01)
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for e in ev:
        t = (lo + e["frame"]) * stride / fps
        if span[0] <= t <= span[1]:
            out[e["label"]].append((t, e["score"]))
    return out


def any_action_probs(c: SimpleNamespace) -> np.ndarray:
    cols = [c.classes.index(k) for k in BAS if k in c.classes]
    m = c.probs[:, cols].max(1)
    return np.stack([1 - m, m], 1)


def match_flags(gt_s: np.ndarray, pred: list[tuple[float, float]], tol_s: float) -> tuple[np.ndarray, int]:
    """Como ``eval_prf.match`` pero devuelve qué GT quedó emparejado (y los falsos positivos)."""
    used = np.zeros(len(gt_s), bool)
    fp = 0
    for t, _ in sorted(pred, key=lambda p: -p[1]):
        if not len(gt_s):
            fp += 1
            continue
        d = np.abs(gt_s - t)
        d[used] = np.inf
        i = int(np.argmin(d))
        if d[i] <= tol_s:
            used[i] = True
        else:
            fp += 1
    return used, fp


def counts_at(gt: dict[str, np.ndarray], pk: dict[str, list], classes: list[str], thr: float,
              tol_s: float) -> dict[str, tuple[int, int, int]]:
    out = {}
    for c in classes:
        pred = [p for p in pk.get(c, []) if p[1] >= thr]
        used, fp = match_flags(gt.get(c, np.zeros(0)), pred, tol_s)
        out[c] = (int(used.sum()), fp, int((~used).sum()))
    return out


def micro(cnt: dict[str, tuple[int, int, int]], classes) -> tuple[float, float, float]:
    tp, fp, fn = (sum(cnt[c][i] for c in classes if c in cnt) for i in range(3))
    return prf(tp, fp, fn)


def best_threshold(gt, pk, classes, tol_s, grid) -> float:
    return max(grid, key=lambda t: micro(counts_at(gt, pk, classes, t, tol_s), classes)[2])


# ----------------------------------------------------------------------------- ediciones del visor
def load_edits(path: Path) -> tuple[list[dict], list[dict]]:
    """(añadidas a mano, borradas) como ``{"t", "cls"}``; las etiquetas del visor usan ``_`` (``high_pass``)."""
    added, deleted = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        for op, key, dst in (("add", "after", added), ("delete", "before", deleted)):
            if d.get("op") == op and d.get(key):
                item = d[key]
                lab = normalize_label(item.get("label", ""))
                cls = ACTION_TO_BAS.get(lab.replace(" ", "_")) or EVENT_TO_SN.get(lab.replace(" ", "-"))
                t = item.get("timestamp_s")
                if t is None and item.get("start_s") is not None:
                    t = float(item["start_s"]) + 1.0          # evento T-DEED: pico = inicio + 1 s (review_service)
                if cls and t is not None:
                    dst.append({"t": float(t), "cls": cls})
    return added, deleted


def flag_added(rows: list[dict], added: list[dict]) -> np.ndarray:
    flags = np.zeros(len(rows), bool)
    by_cls = defaultdict(list)
    for a in added:
        by_cls[a["cls"]].append(a["t"])
    for i, r in enumerate(rows):
        ts = by_cls.get(r["cls"])
        if ts and min(abs(t - r["t"]) for t in ts) <= 0.06:
            flags[i] = True
    return flags


# ----------------------------------------------------------------------------- informe
def fmt_pr(p: float, r: float) -> str:
    return f"{100 * p:.0f} / {100 * r:.0f}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True, help=".gt.json o CSV del visor")
    ap.add_argument("--video", type=Path, required=True, help="vídeo junto al que están las cachés .tdeed.<modelo>.npz")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--fuse", nargs=2, metavar="MODELO", help="añade la media 0,5/0,5 de estas dos cachés")
    ap.add_argument("--span", type=float, nargs=2, metavar=("INICIO_S", "FIN_S"))
    ap.add_argument("--tol-s", type=float, default=0.5)
    ap.add_argument("--fixed", type=float, nargs="*", default=[0.2, 0.3, 0.5], help="umbrales fijos a informar")
    ap.add_argument("--edits", type=Path, help="review/edits.jsonl del visor (reparto por origen de la verdad)")
    ap.add_argument("--title", default="")
    ap.add_argument("--md-out", type=Path)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    rows, span = load_gt(args.gt, tuple(args.span) if args.span else None)
    gt: dict[str, np.ndarray] = {}
    for c in sorted({r["cls"] for r in rows}):
        gt[c] = np.array([r["t"] for r in rows if r["cls"] == c])
    n_gt = {c: len(v) for c, v in gt.items()}
    bas_in_gt = [c for c in BAS if n_gt.get(c)]
    tuned = [c for c in bas_in_gt if c not in NOT_TUNED]
    sn_in_gt = [c for c in n_gt if c not in BAS]
    logger.info("verdad %s: %d filas en %.0f-%.0f s; %s", args.gt.name, len(rows), *span,
                ", ".join(f"{c} {n}" for c, n in sorted(n_gt.items(), key=lambda kv: -kv[1])))

    caches = {m: load_cache(args.video.with_name(f"{args.video.stem}.tdeed.{m}.npz")) for m in args.models}
    if args.fuse:
        from scripts.benchmark.fuse_scores import fuse
        a, b = (caches[m] for m in args.fuse)
        probs, classes = fuse([a, b], np.full((2, len(BAS)), 0.5), "mean")
        caches["Fusion"] = SimpleNamespace(probs=probs, classes=list(classes), stride=a.stride, fps=a.fps,
                                           probs_sn=None, classes_sn=None)
        NAMES["Fusion"] = f"Fusión {NAMES.get(args.fuse[0], args.fuse[0])} + {NAMES.get(args.fuse[1], args.fuse[1])}"

    added = deleted = None
    if args.edits:
        added, deleted = load_edits(args.edits)
        added = [a for a in added if span[0] <= a["t"] <= span[1]]
        deleted = [d for d in deleted if span[0] <= d["t"] <= span[1]]
        is_added = flag_added(rows, added)
        logger.info("ediciones en el tramo: %d añadidas (%d filas de la verdad casan), %d borradas",
                    len(added), int(is_added.sum()), len(deleted))

    report: dict[str, dict] = {}
    for m, c in caches.items():
        pk = peaks(c.probs, c.classes, c.stride, c.fps, span)
        thr = best_threshold(gt, pk, tuned, args.tol_s, GRID)
        cnt = counts_at(gt, pk, bas_in_gt, thr, args.tol_s)
        r: dict = {"threshold": thr, "per_class": {k: {"tp": v[0], "fp": v[1], "fn": v[2], "prf": prf(*v)}
                                                    for k, v in cnt.items()},
                   "total": micro(cnt, tuned), "total_tol1s": micro(counts_at(gt, pk, bas_in_gt, thr, 1.0), tuned),
                   "fixed": {str(t): micro(counts_at(gt, pk, bas_in_gt, t, args.tol_s), tuned) for t in args.fixed}}
        # cualquier acción (sin etiqueta), a su propio umbral
        pk_any = peaks(any_action_probs(c), ["background", "ANY"], c.stride, c.fps, span)
        gt_any = {"ANY": np.sort(np.concatenate([gt[k] for k in bas_in_gt]))}
        thr_any = best_threshold(gt_any, pk_any, ["ANY"], args.tol_s, GRID)
        r["any"] = {"threshold": thr_any, "prf": micro(counts_at(gt_any, pk_any, ["ANY"], thr_any, args.tol_s), ["ANY"])}
        # cabeza SoccerNet-v2 (córner, falta, saque de centro) a su mejor umbral por clase
        if c.probs_sn is not None and sn_in_gt:
            pk_sn = peaks(c.probs_sn, c.classes_sn, c.stride, c.fps, span)
            r["sn"] = {}
            for k in sn_in_gt:
                t_k = best_threshold(gt, pk_sn, [k], args.tol_s, GRID)
                r["sn"][k] = {"threshold": t_k, "prf": micro(counts_at(gt, pk_sn, [k], t_k, args.tol_s), [k])}
        # reparto por origen (solo hoja del visor con edits.jsonl)
        if added is not None:
            tp_add = tp_kept = 0
            for k in tuned:
                idx = [i for i, rr in enumerate(rows) if rr["cls"] == k]
                used, _ = match_flags(gt[k], [p for p in pk.get(k, []) if p[1] >= thr], args.tol_s)
                flags = is_added[idx]
                tp_add += int(used[flags].sum())
                tp_kept += int(used[~flags].sum())
            n_add = int(sum(is_added[i] for i, rr in enumerate(rows) if rr["cls"] in tuned))
            n_kept = sum(n_gt[k] for k in tuned) - n_add
            fired = 0
            for d in deleted:
                fired += any(abs(t - d["t"]) <= args.tol_s and s >= thr for t, s in pk.get(d["cls"], []))
            r["origin"] = {"added": [tp_add, n_add], "kept": [tp_kept, n_kept], "deleted_fired": [fired, len(deleted)]}
        report[m] = r
        logger.info("%s: umbral %.2f, total P/R/F1 %s", NAMES.get(m, m), thr,
                    " / ".join(f"{100 * x:.1f}" for x in r["total"]))

    # ---- tabla en el formato de §7.12 (nº de eventos, precisión / recall)
    models = list(caches)
    head = "| Clase | Nº | " + " | ".join(NAMES.get(m, m) for m in models) + " |"
    lines = [f"**{args.title}**" if args.title else "", f"Tramo {span[0]:.0f}-{span[1]:.0f} s, ±{args.tol_s:g} s. "
             + "Umbrales " + " / ".join(f"{report[m]['threshold']:.2f}" for m in models) + ".", "", head,
             "|---|---|" + "---|" * len(models)]
    for k in sorted(bas_in_gt, key=lambda c: -n_gt[c]):
        cells = [fmt_pr(*report[m]["per_class"][k]["prf"][:2]) if k in caches[m].classes else "–" for m in models]
        lines.append(f"| {ES.get(k, k)} | {n_gt[k]} | " + " | ".join(cells) + " |")
    best = max(models, key=lambda m: report[m]["total"][2])
    tot = [fmt_pr(*report[m]["total"][:2]) for m in models]
    tot = [f"**{t}**" if m == best else t for m, t in zip(models, tot)]
    lines.append(f"| **Total (clases comunes)** | {sum(n_gt[c] for c in tuned)} | " + " | ".join(tot) + " |")
    lines.append(f"| Cualquier acción (sin etiqueta) | {sum(n_gt[c] for c in bas_in_gt)} | "
                 + " | ".join(fmt_pr(*report[m]["any"]["prf"][:2]) for m in models) + " |")
    for k in sn_in_gt:
        cells = [fmt_pr(*report[m]["sn"][k]["prf"][:2]) if "sn" in report[m] else "–" for m in models]
        lines.append(f"| {ES.get(k, k)} (cabeza SN-v2, umbral propio) | {n_gt[k]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Total a ±1 s: " + "; ".join(f"{NAMES.get(m, m)} {fmt_pr(*report[m]['total_tol1s'][:2])}"
                                              for m in models) + ".")
    for t in args.fixed:
        lines.append(f"Total a umbral fijo {t:g}: " + "; ".join(
            f"{NAMES.get(m, m)} {fmt_pr(*report[m]['fixed'][str(t)][:2])}" for m in models) + ".")
    if added is not None:
        lines.append("")
        lines.append("| Origen de la fila (clases comunes) | Nº | " + " | ".join(NAMES.get(m, m) for m in models) + " |")
        lines.append("|---|---|" + "---|" * len(models))
        for key, label in (("kept", "Ya la tenía el pipeline (sin tocar o corregida): recall"),
                           ("added", "Añadida a mano (el pipeline no la tenía): recall"),
                           ("deleted_fired", "Borrada por el revisor (falso positivo del pipeline): sigue disparando")):
            n = report[models[0]]["origin"][key][1]
            cells = [f"{100 * report[m]['origin'][key][0] / max(n, 1):.0f} %" for m in models]
            lines.append(f"| {label} | {n} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    print(md)
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(md + "\n")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"gt": str(args.gt), "span": span, "tol_s": args.tol_s, "n_gt": n_gt,
                                             "models": report}, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
