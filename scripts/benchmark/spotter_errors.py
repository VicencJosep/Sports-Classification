"""Anatomía de los errores de un spotter a su umbral de trabajo, a partir de las predicciones ya post-procesadas por
``eval_bas.py`` (``<out>/snms*/preds/<partido>/results_spotting.json``). Sin GPU.

Emparejamiento como en ``eval_prf.py`` (±0,5 s, 1 a 1, greedy por confianza, por clase). Cada evento no detectado (FN)
cae en la primera categoría que se cumple:

1. ``otra clase``: hay una predicción sobrante (FP) de otra clase a ±0,5 s -> el modelo vio el evento y erró la etiqueta.
2. ``fundido``: hay un acierto de su clase a ±0,5 s que se llevó un evento vecino -> un pico para dos eventos pegados.
3. ``a destiempo``: hay un FP de su clase entre ±0,5 y ±1,5 s -> lo vio, fuera de la tolerancia.
4. ``bajo el umbral``: hay una predicción de su clase a ±0,5 s con confianza entre 0,01 y el umbral.
5. ``sin respuesta``: nada de lo anterior.

Y cada predicción sobrante (FP): ``otra clase`` (un FN de otra clase a ±0,5 s), ``a destiempo`` (un FN de su clase a
±1,5 s), ``duplicado`` (un evento de su clase ya emparejado a ±0,5 s), ``sobre otro evento`` (un evento de otra clase
ya emparejado a ±0,5 s) o ``sin evento`` (ningún evento anotado a ±0,5 s).

Con ``--versus`` se añade la complementariedad con otro modelo: qué eventos detecta cada uno, los dos o ninguno.

Uso (contenedor):
    python -m scripts.benchmark.spotter_errors --model SoccerNetBall_test --versus AdaSpot_big
    python -m scripts.benchmark.spotter_errors --model AdaSpot_big --dataset pcbas --tag val_merged
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.benchmark.eval_bas import (LABELS_ROOT, OUT_ROOT, OUT_ROOT_PCBAS, PCBAS_CLASSES, TDEED_DIR,
                                        tdeed_eval_classes)

TOL_MS, FAR_MS = 500.0, 1500.0
EXCLUDED = {"FREE KICK", "GOAL"}  # AdaSpot no las predice
FN_KINDS = ("otra clase", "fundido", "a destiempo", "bajo el umbral", "sin respuesta")
FP_KINDS = ("otra clase", "a destiempo", "duplicado", "sobre otro evento", "sin evento")
ERR_ROOT = OUT_ROOT.parent / "errors"


def load(model: str, dataset: str, split: str, tag: str):
    """(clases, GT {clase: ms ordenados}, predicciones {clase: (ms, confianza)}, umbral óptimo de ``prf.json``)."""
    if dataset == "efl":
        games = [g["video"] for g in json.loads((TDEED_DIR / "data" / "soccernetball" / f"{split}.json").read_text())]
        label_root, out_root, classes = LABELS_ROOT, OUT_ROOT / model / split, tdeed_eval_classes()
    else:
        out_root, label_root, classes = OUT_ROOT_PCBAS / model / tag, OUT_ROOT_PCBAS / "labels", list(PCBAS_CLASSES.values())
        games = json.loads((out_root / "summary.json").read_text())["games"]
    preds_dir = next(out_root.glob("snms*")) / "preds"
    gt, pr = {c: [] for c in classes}, {c: [] for c in classes}
    for n, game in enumerate(games):  # cada partido se desplaza 10 h y cada parte 5 h para no emparejar entre ellos
        def ms(a: dict) -> float:
            half = int(a["half"]) if "half" in a else int(str(a["gameTime"])[0])
            return float(a["position"]) + n * 36_000_000 + (half - 1) * 18_000_000
        for a in json.loads((label_root / game / "Labels-ball.json").read_text())["annotations"]:
            if a["label"] in gt:
                gt[a["label"]].append(ms(a))
        for a in json.loads((preds_dir / game / "results_spotting.json").read_text())["predictions"]:
            if a["label"] in pr:
                pr[a["label"]].append((ms(a), float(a["confidence"])))
    prf = out_root / "prf.json"
    thr = json.loads(prf.read_text())["best"]["threshold"] if prf.exists() else None
    return classes, {c: np.array(sorted(v)) for c, v in gt.items()}, {c: np.array(sorted(v)).reshape(-1, 2) for c, v in pr.items()}, thr


def assign(gt_ms: np.ndarray, t: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras (GT detectado, predicción acertada): 1 a 1, greedy por confianza, GT libre más cercano a ±TOL_MS."""
    gt_hit, pr_hit = np.zeros(len(gt_ms), bool), np.zeros(len(t), bool)
    for j in np.argsort(-conf):
        if not len(gt_ms):
            break
        d = np.abs(gt_ms - t[j])
        d[gt_hit] = np.inf
        i = int(np.argmin(d))
        if d[i] <= TOL_MS:
            gt_hit[i], pr_hit[j] = True, True
    return gt_hit, pr_hit


def gap(times: np.ndarray, t: float) -> float:
    return float(np.min(np.abs(times - t))) if len(times) else np.inf


def anatomy(classes: list[str], gt: dict, pr: dict, thr: float) -> dict:
    m = {}
    for c in classes:
        t, conf = pr[c][:, 0], pr[c][:, 1]
        keep = conf >= thr
        g, p = assign(gt[c], t[keep], conf[keep])
        m[c] = {"hit": g, "tp": t[keep][p], "fp": t[keep][~p], "weak": t[~keep], "fn": gt[c][~g]}
    fn = {c: Counter() for c in classes}
    fp = {c: Counter() for c in classes}
    confusions = Counter()  # (clase anotada, clase predicha)
    for c in classes:
        others = [o for o in classes if o != c]
        for tg in m[c]["fn"]:
            near = min(others, key=lambda o: gap(m[o]["fp"], tg))
            if gap(m[near]["fp"], tg) <= TOL_MS:
                kind = "otra clase"
                confusions[(c, near)] += 1
            elif gap(m[c]["tp"], tg) <= TOL_MS:
                kind = "fundido"
            elif gap(m[c]["fp"], tg) <= FAR_MS:
                kind = "a destiempo"
            elif gap(m[c]["weak"], tg) <= TOL_MS:
                kind = "bajo el umbral"
            else:
                kind = "sin respuesta"
            fn[c][kind] += 1
        for tp in m[c]["fp"]:
            if min(gap(m[o]["fn"], tp) for o in others) <= TOL_MS:
                kind = "otra clase"
            elif gap(m[c]["fn"], tp) <= FAR_MS:
                kind = "a destiempo"
            elif gap(gt[c], tp) <= TOL_MS:
                kind = "duplicado"
            elif min(gap(gt[o], tp) for o in others) <= TOL_MS:
                kind = "sobre otro evento"
            else:
                kind = "sin evento"
            fp[c][kind] += 1
    return {"hit": {c: m[c]["hit"] for c in classes}, "fn": fn, "fp": fp, "confusions": confusions,
            "n_pred": {c: len(m[c]["tp"]) + len(m[c]["fp"]) for c in classes}}


def pct(a: float, b: float) -> str:
    return f"{100 * a / b:.0f}".replace(".", ",") if b else "—"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--versus", help="Segundo modelo: complementariedad de aciertos")
    ap.add_argument("--dataset", default="efl", choices=["efl", "pcbas"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="val_merged", help="pcbas: val | val_merged")
    ap.add_argument("--thr", type=float, help="Umbral de trabajo (por defecto, el óptimo de prf.json)")
    args = ap.parse_args()

    classes, gt, pr, thr = load(args.model, args.dataset, args.split, args.tag)
    thr = args.thr if args.thr is not None else thr
    if thr is None:
        raise SystemExit("Sin prf.json: ejecuta antes eval_prf.py o pasa --thr")
    classes = [c for c in classes if c not in EXCLUDED and len(gt[c])]
    a = anatomy(classes, gt, pr, thr)
    where = f"{args.dataset}/{args.split if args.dataset == 'efl' else args.tag}"
    n_gt = {c: len(gt[c]) for c in classes}
    tot = lambda d, k: sum(d[c][k] for c in classes)  # noqa: E731
    lines = [f"# Errores de {args.model} en {where} (umbral {thr:.2f}, ±0,5 s)", "",
             "## Eventos no detectados (% de los eventos anotados de la clase)", "",
             "| Clase | Eventos | Detectados | " + " | ".join(k.capitalize() for k in FN_KINDS) + " |",
             "|---|---:|---:|" + "---:|" * len(FN_KINDS)]
    for c in classes:
        lines.append(f"| {c.title()} | {n_gt[c]} | {pct(a['hit'][c].sum(), n_gt[c])} | "
                     + " | ".join(pct(a["fn"][c][k], n_gt[c]) for k in FN_KINDS) + " |")
    n = sum(n_gt.values())
    lines.append(f"| **Total** | {n} | {pct(sum(a['hit'][c].sum() for c in classes), n)} | "
                 + " | ".join(pct(tot(a["fn"], k), n) for k in FN_KINDS) + " |")
    lines += ["", "## Predicciones sobrantes (% de las predicciones de la clase)", "",
              "| Clase | Predicciones | Acertadas | " + " | ".join(k.capitalize() for k in FP_KINDS) + " |",
              "|---|---:|---:|" + "---:|" * len(FP_KINDS)]
    for c in classes:
        npred = a["n_pred"][c]
        lines.append(f"| {c.title()} | {npred} | {pct(a['hit'][c].sum(), npred)} | "
                     + " | ".join(pct(a["fp"][c][k], npred) for k in FP_KINDS) + " |")
    npred = sum(a["n_pred"].values())
    lines.append(f"| **Total** | {npred} | {pct(sum(a['hit'][c].sum() for c in classes), npred)} | "
                 + " | ".join(pct(tot(a["fp"], k), npred) for k in FP_KINDS) + " |")
    lines += ["", "## Confusiones más frecuentes (anotada -> predicha)", "",
              ", ".join(f"{g.title()} -> {p.title()} {k}" for (g, p), k in a["confusions"].most_common(12)) + "."]
    out = {"model": args.model, "dataset": where, "threshold": thr, "n_gt": n_gt, "n_pred": a["n_pred"],
           "detected": {c: int(a["hit"][c].sum()) for c in classes},
           "fn": {c: dict(a["fn"][c]) for c in classes}, "fp": {c: dict(a["fp"][c]) for c in classes},
           "confusions": [[g, p, k] for (g, p), k in a["confusions"].most_common()]}

    if args.versus:
        _, gt2, pr2, thr2 = load(args.versus, args.dataset, args.split, args.tag)
        b = anatomy(classes, gt2, pr2, thr2)
        lines += ["", f"## Complementariedad con {args.versus} (umbral {thr2:.2f}; % de los eventos de la clase)", "",
                  f"| Clase | Eventos | Los dos | Solo {args.model} | Solo {args.versus} | Ninguno |", "|---|---:|---:|---:|---:|---:|"]
        comp = {}
        for c in classes + ["Total"]:
            h1 = np.concatenate([a["hit"][k] for k in classes]) if c == "Total" else a["hit"][c]
            h2 = np.concatenate([b["hit"][k] for k in classes]) if c == "Total" else b["hit"][c]
            cells = [int((h1 & h2).sum()), int((h1 & ~h2).sum()), int((~h1 & h2).sum()), int((~h1 & ~h2).sum())]
            comp[c] = cells
            name = "**Total**" if c == "Total" else c.title()
            lines.append(f"| {name} | {len(h1)} | " + " | ".join(pct(v, len(h1)) for v in cells) + " |")
        out["versus"] = {"model": args.versus, "threshold": thr2, "both_only1_only2_none": comp}

    ERR_ROOT.mkdir(parents=True, exist_ok=True)
    stem = ERR_ROOT / f"{where.replace('/', '_')}_{args.model}"
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    stem.with_suffix(".json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
