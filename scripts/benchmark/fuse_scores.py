"""Fusión de spotters: media ponderada, frame a frame y clase a clase, de las cachés de probabilidad de varios modelos
(``<vídeo>.tdeed.<modelo>.npz``, mismas filas). Escribe una caché nueva ``<vídeo>.tdeed.<nombre>.npz`` que ``eval_bas.py``,
``eval_prf.py`` y ``spotter_errors.py`` leen como un modelo más. Sin GPU.

Los pesos se pasan con ``--weights`` o se ajustan con ``--tune``: barrido en el partido de validación de SN-BAS
(Middlesbrough) del mAP@1 oficial sobre las 10 clases comunes; nunca en Stoke/Reading ni en FOOTPASS. Con ``--per-class``
cada clase usa el peso que maximiza su propia AP en validación (un solo partido: con las clases raras es ruido). Una clase
que un modelo no tiene (FREE KICK y GOAL en AdaSpot) se queda con los modelos que sí la tienen.

Uso (contenedor):
    python -m scripts.benchmark.fuse_scores --models SoccerNetBall_test AdaSpot_big --name Fusion_T200_AdaB --tune
    python -m scripts.benchmark.eval_bas --model Fusion_T200_AdaB --split test --metrics at1 at2
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from core.ball_spotting import SpotScores, read_scores
from scripts.benchmark.eval_bas import (LABELS_ROOT, OUT_ROOT, TDEED_DIR, candidates, evaluate_candidates, pcbas_games,
                                        tdeed_eval_classes)

EXCLUDED = ("FREE KICK", "GOAL")


def efl_games(split: str) -> list[tuple[str, Path]]:
    names = [g["video"] for g in json.loads((TDEED_DIR / "data" / "soccernetball" / f"{split}.json").read_text())]
    return [(g, LABELS_ROOT / g / "720p") for g in names]


def fuse(scores: list[SpotScores], weights: np.ndarray, mode: str = "mean") -> tuple[np.ndarray, tuple[str, ...]]:
    """``weights``: [modelos, clases de ``tdeed_eval_classes()``]. Devuelve (probs con el fondo en la columna 0, clases)."""
    classes = tdeed_eval_classes()
    n = min(len(s.probs) for s in scores)
    out = np.zeros((n, 1 + len(classes)), np.float32)
    for j, c in enumerate(classes):
        have = [i for i, s in enumerate(scores) if c in s.classes]
        w = weights[have, j] / max(weights[have, j].sum(), 1e-9)
        cols = [scores[i].probs[:n, scores[i].classes.index(c)].astype(np.float32) for i in have]
        if mode == "geo":
            out[:, 1 + j] = np.exp(sum(wi * np.log(col + 1e-4) for wi, col in zip(w, cols)))
        else:
            out[:, 1 + j] = sum(wi * col for wi, col in zip(w, cols))
    out[:, 0] = np.clip(1 - out[:, 1:].sum(1), 0, 1)
    return out, ("background", *classes)


def write_cache(stem: Path, name: str, probs: np.ndarray, classes: tuple[str, ...], stride: int, fps: float) -> Path:
    """Escribe ``<stem>.tdeed.<name>.npz`` de forma atómica, con el formato de ``core.ball_spotting.read_scores``."""
    out = stem.with_name(f"{stem.name}.tdeed.{name}.npz")
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(tmp, probs=probs.astype(np.float16), classes=np.array(classes), stride=stride, fps=fps)
    tmp.rename(out)
    return out


def simplex(k: int, step: float) -> list[tuple[float, ...]]:
    m = round(1 / step)
    return [tuple(x / m for x in c) for c in itertools.product(range(m + 1), repeat=k) if sum(c) == m]


def tune(models: list[str], mode: str, step: float, per_class: bool, tmp: Path) -> np.ndarray:
    games = efl_games("val")
    loaded = [[read_scores(stem.with_name(f"{stem.name}.tdeed.{m}.npz")) for m in models] for _, stem in games]
    classes = tdeed_eval_classes()
    rows = []
    for w in simplex(len(models), step):
        W = np.repeat(np.array(w)[:, None], len(classes), 1)
        preds = []
        for (game, _), sc in zip(games, loaded):
            probs, names = fuse(sc, W, mode)
            preds.append({"video": game, "events": candidates(probs, names), "fps": sc[0].fps / sc[0].stride})
        _, res = evaluate_candidates(preds, sc[0].stride, [g for g, _ in games], LABELS_ROOT, tmp)
        r = next(iter(res.values()))
        rows.append((w, r["mAP_10cls"], r["per_class"]))
        print(f"  pesos {' / '.join(f'{x:.2f}' for x in w)}: mAP@1 (10 clases) {r['mAP_10cls']:.2f}", flush=True)
    best = max(rows, key=lambda r: r[1])
    print(f"Mejor en validación: {dict(zip(models, best[0]))} -> {best[1]:.2f}")
    W = np.repeat(np.array(best[0])[:, None], len(classes), 1)
    if per_class:
        for j, c in enumerate(classes):
            if c not in EXCLUDED:
                W[:, j] = max(rows, key=lambda r: r[2][c])[0]
        print("Pesos por clase: " + "; ".join(f"{c} {'/'.join(f'{x:.1f}' for x in W[:, j])}" for j, c in enumerate(classes)))
    return W


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--name", required=True, help="Nombre de la caché fusionada")
    ap.add_argument("--weights", type=float, nargs="+", help="Un peso por modelo (por defecto, iguales)")
    ap.add_argument("--tune", action="store_true", help="Ajusta los pesos en el partido de validación de SN-BAS")
    ap.add_argument("--per-class", action="store_true", help="Con --tune: un peso por clase")
    ap.add_argument("--step", type=float, default=0.1, help="Paso del barrido de pesos")
    ap.add_argument("--mode", default="mean", choices=["mean", "geo"], help="Media aritmética o geométrica")
    ap.add_argument("--datasets", nargs="+", default=["val", "test", "pcbas"], help="Dónde escribir la caché fusionada")
    args = ap.parse_args()

    out_dir = OUT_ROOT / args.name
    if args.tune:
        W = tune(args.models, args.mode, args.step, args.per_class, out_dir / "_tune")
    else:
        w = np.array(args.weights or [1.0] * len(args.models))
        W = np.repeat((w / w.sum())[:, None], len(tdeed_eval_classes()), 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fusion.json").write_text(json.dumps({
        "models": args.models, "mode": args.mode, "tuned_on": "efl/val" if args.tune else None,
        "weights": {c: [round(float(x), 3) for x in W[:, j]] for j, c in enumerate(tdeed_eval_classes())}}, indent=1))
    for ds in args.datasets:
        for game, stem in (pcbas_games("val") if ds == "pcbas" else efl_games(ds)):
            sc = [read_scores(stem.with_name(f"{stem.name}.tdeed.{m}.npz")) for m in args.models]
            probs, names = fuse(sc, W, args.mode)
            print(f"{ds}: {write_cache(stem, args.name, probs, names, sc[0].stride, sc[0].fps)}")


if __name__ == "__main__":
    main()
