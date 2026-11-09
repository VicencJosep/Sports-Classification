"""Evaluación oficial de SoccerNet-v2 (17 eventos, Average-mAP tight/loose) de la cabeza SoccerNet de T-DEED
sobre una lista de partidos (por defecto el piloto del split ``test`` oficial, limpio para todos los checkpoints).

Lee las cachés ``<partido>/{1,2}_720p.tdeed.<modelo>.npz`` (``probs_sn``, escritas por
``core.ball_spotting.TDEEDSpotter.ensure_scores``), aplica el postproceso de T-DEED para SoccerNet (NMS suave con
ventana ``WINDOWS_SN[1]`` = 6 filas a 12,5 fps ≈ 0,5 s, candidatos ≥ 0,01), escribe ``results_spotting.json`` por partido
(formato del reto) y calcula el Average-mAP con el código del paquete ``SoccerNet`` (``label2vector`` /
``predictions2vector`` / ``average_mAP``, framerate 2, ``EVENT_DICTIONARY_V2``), igual que ``evaluate`` del paquete
pero con nuestra lista de partidos. Da también las variantes *visible* y *unshown* del evaluador oficial.

Uso (contenedor):
    python -m scripts.benchmark.eval_snv2 --model SoccerNetBall_test --games output/benchmark/sn_v2_pilot20.txt \
        --tag pilot20 --metrics tight loose
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

from core.ball_spotting import read_scores
from scripts.benchmark.eval_bas import _tdeed_eval

ROOT = Path(__file__).resolve().parents[2]
LABELS_ROOT = ROOT / "datasets" / "SoccerNet"
OUT_ROOT = ROOT / "output" / "benchmark" / "snv2"
FRAMERATE_EVAL = 2  # el evaluador oficial trabaja a 2 fps (posiciones en ms -> cuantización de 0,5 s)
DELTAS = {"tight": np.arange(5) * 1 + 1, "loose": np.arange(12) * 5 + 5,
          **{f"at{k}": np.array([k]) for k in range(1, 6)}}


def candidates(probs: np.ndarray, classes: tuple[str, ...], thr: float = 0.01) -> list[dict]:
    """Filas (stride) x clases (col 0 = fondo) -> eventos candidatos {frame, label, score} por encima de ``thr``."""
    events = []
    rows, cols = np.where(probs[:, 1:] >= thr)
    for r, c in zip(rows.tolist(), cols.tolist()):
        events.append({"frame": r, "label": classes[c + 1], "score": float(probs[r, c + 1])})
    return events


def game_predictions(model: str, game: str, window: int) -> tuple[dict, int]:
    """Predicciones de las dos mitades de ``game`` tras el NMS suave, en el formato del reto; y el stride usado."""
    tdeed_eval = _tdeed_eval()
    preds, stride = [], None
    for half in (1, 2):
        sc = read_scores(LABELS_ROOT / game / f"{half}_720p.tdeed.{model}.npz")
        if sc.sn_probs is None:
            raise SystemExit(f"{game} mitad {half}: la caché no tiene probs_sn")
        stride = sc.stride
        preds.append({"video": f"{game}/{half}", "events": candidates(sc.sn_probs, sc.sn_classes), "fps": sc.fps / sc.stride})
    w = max(1, round(window * 2 / stride))
    preds = tdeed_eval.soft_non_maximum_supression(preds, window=w, threshold=0.01)
    out = {"UrlLocal": game, "predictions": []}
    for half, p in zip((1, 2), preds):
        for e in p["events"]:
            position = int(round(e["frame"] / p["fps"] * 1000))
            out["predictions"].append({"gameTime": f"{half} - {position // 60000}:{(position % 60000) // 1000:02d}",
                                       "label": e["label"], "position": position, "half": half,
                                       "confidence": round(float(e["score"]), 4)})
    return out, stride


def closest(label: np.ndarray) -> np.ndarray:
    """Copia de ``evaluate`` del paquete SoccerNet: para cada frame, la visibilidad de la acción más cercana."""
    closest_numpy = np.zeros(label.shape) - 1
    for c in np.arange(label.shape[-1]):
        indexes = np.where(label[:, c] != 0)[0].tolist()
        if len(indexes) == 0:
            continue
        indexes.insert(0, -indexes[0])
        indexes.append(2 * closest_numpy.shape[0])
        for i in np.arange(len(indexes) - 2) + 1:
            start = max(0, (indexes[i - 1] + indexes[i]) // 2)
            stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
            closest_numpy[start:stop, c] = label[indexes[i], c]
    return closest_numpy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Nombre de la caché: <mitad>.tdeed.<model>.npz")
    ap.add_argument("--games", type=Path, default=ROOT / "output" / "benchmark" / "sn_v2_pilot20.txt",
                    help="Lista de partidos (liga/temporada/partido), uno por línea")
    ap.add_argument("--tag", default="pilot20")
    ap.add_argument("--metrics", nargs="+", default=["tight", "loose"], choices=sorted(DELTAS))
    ap.add_argument("--window", type=int, default=6, help="Ventana del NMS suave en filas a stride 2 (WINDOWS_SN[1] de T-DEED)")
    args = ap.parse_args()

    from SoccerNet.Evaluation.ActionSpotting import average_mAP, label2vector, predictions2vector
    from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2

    games = [g.strip() for g in args.games.read_text().splitlines() if g.strip()]
    classes = list(EVENT_DICTIONARY_V2)
    out_root = OUT_ROOT / args.model / args.tag
    targets, detections, closests = [], [], []
    counts = {c: 0 for c in classes}
    counts_shown = {c: 0 for c in classes}
    n_pred = 0
    for game in games:
        pred, _ = game_predictions(args.model, game, args.window)
        n_pred += len(pred["predictions"])
        pdir = out_root / "preds" / game
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "results_spotting.json").write_text(json.dumps(pred, indent=1))
        labels = json.loads((LABELS_ROOT / game / "Labels-v2.json").read_text())
        for a in labels["annotations"]:
            if a["label"] in counts:
                counts[a["label"]] += 1
                counts_shown[a["label"]] += a.get("visibility") != "not shown"
        l1, l2 = label2vector(labels, num_classes=17, version=2, EVENT_DICTIONARY=EVENT_DICTIONARY_V2, framerate=FRAMERATE_EVAL)
        p1, p2 = predictions2vector(pred, num_classes=17, version=2, EVENT_DICTIONARY=EVENT_DICTIONARY_V2, framerate=FRAMERATE_EVAL)
        targets += [l1, l2]
        detections += [p1, p2]
        closests += [closest(l1), closest(l2)]

    summary = {"model": args.model, "games": games, "n_games": len(games), "n_predictions": n_pred,
               "gt_counts": counts, "gt_counts_shown": counts_shown, "results": {}}
    for metric in args.metrics:
        with contextlib.redirect_stdout(io.StringIO()):
            a_map, per_class, a_vis, per_vis, a_uns, per_uns = average_mAP(
                targets, detections, closests, FRAMERATE_EVAL, deltas=DELTAS[metric])
        res = {"a_mAP": round(100 * float(a_map), 2), "a_mAP_visible": round(100 * float(a_vis), 2),
               "a_mAP_unshown": round(100 * float(a_uns), 2),
               "per_class": {c: round(100 * float(v), 2) for c, v in zip(classes, per_class)},
               "per_class_visible": {c: round(100 * float(v), 2) for c, v in zip(classes, per_vis)},
               "per_class_unshown": {c: round(100 * float(v), 2) for c, v in zip(classes, per_uns)}}
        summary["results"][metric] = res
        print(f"{args.model} [{args.tag}, {len(games)} partidos, {metric}] Average-mAP {res['a_mAP']:.2f} "
              f"(visible {res['a_mAP_visible']:.2f}, unshown {res['a_mAP_unshown']:.2f})  "
              + "  ".join(f"{c} {v:.1f}" for c, v in res["per_class"].items()))
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
