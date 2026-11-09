"""Equipo de cada acción con el baseline de Team Ball Action Spotting
(caché ``720p.tdeed.SNTeam_baseline.npz`` de ``teamspot_scores.py``).

Nuestro GT (``Labels-ball.json`` de BAS-2024) da el equipo como home/away; el
modelo y la métrica de 2025 usan left/right en pantalla, que cambia en el
descanso. Sin acceso a las etiquetas de SN-BAS-2025 (HF gated con NDA), el
mapeo home/away -> left/right se elige a posteriori: una orientación y un
único punto de cambio por partido que maximizan el acierto. Es OPTIMISTA (dos
grados de libertad por partido) y así se marca en los resultados.

Dos medidas:
* ``team_acc``: para cada acción del GT, el frame de mayor P(clase) a ±0,5 s
  y el lado con más probabilidad. Comparable con ``team_accuracy`` de
  ``scripts/evaluate_match.py`` (mejor mapeo cluster -> lado).
* ``Team-mAP@1`` aproximado: evaluador del devkit (``util.score.compute_amAP``,
  event_team) con el GT remapeado y NMS suave de ventana 12, como
  ``mAPevaluateTest``.

Uso (contenedor):
    python -m scripts.benchmark.eval_team --split test
    python -m scripts.benchmark.eval_team --split val --minutes 20   # Middlesbrough, como evaluate_match
    python -m scripts.benchmark.eval_team --dataset st2              # SoccerTrack v2 eval, Team-mAP exacto
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "models" / "sn-teamspotting" / "repo"
TDEED_SPLITS = ROOT / "models" / "tdeed" / "T-DEED" / "data" / "soccernetball"
LABELS_ROOT = ROOT / "datasets" / "SoccerNet"
OUT_ROOT = ROOT / "output" / "benchmark" / "bas" / "SNTeam_baseline"
MODEL_NAME = "SNTeam_baseline"
ST2_VIEWS = ROOT / "datasets" / "soccertrack-v2" / "views"
ST2_LABELS = ROOT / "output" / "benchmark" / "bas_st2" / "labels"
ST2_OUT = ROOT / "output" / "benchmark" / "bas_st2" / MODEL_NAME
TOL_S = 0.5


def side_predictions(z, labels: list[dict], classes: list[str], fps_rows: float) -> list[tuple[float, str, int]]:
    """(t_s, equipo GT, lado predicho 0=left/1=right) por acción del GT."""
    team, probs = z["probs_team"].astype(np.float32), z["probs"].astype(np.float32)
    half = int(round(TOL_S * fps_rows))
    out = []
    for a in labels:
        c = classes.index(a["label"])  # columna en probs (0 = fondo)
        r = int(round(int(a["position"]) / 1000 * fps_rows))
        lo, hi = max(r - half, 0), min(r + half + 1, len(probs))
        if lo >= hi:
            continue
        best = lo + int(np.argmax(probs[lo:hi, c]))
        left, right = team[best, 2 * c - 1], team[best, 2 * c]
        out.append((int(a["position"]) / 1000, a["team"], int(right > left)))
    return out


def best_mapping(preds: list[tuple[float, str, int]], allow_switch: bool) -> tuple[float, float | None, bool]:
    """Acierto con la mejor orientación y (si se permite) un punto de cambio.

    Devuelve (acierto, t_cambio o None, home_es_left_al_inicio)."""
    gt = np.array([1 if team in ("away", "right") else 0 for _, team, _ in preds])
    pr = np.array([p for _, _, p in preds])
    ts = np.array([t for t, _, _ in preds])
    agree = (gt == pr).astype(int)  # orientación home=left
    best = (-1.0, None, True)
    cuts = range(len(preds) + 1) if allow_switch else [len(preds)]
    pre = np.concatenate([[0], np.cumsum(agree)])
    total = pre[-1]
    for k in cuts:
        a_before, a_after = pre[k], total - pre[k]
        n_before, n_after = k, len(preds) - k
        for home_left, hits in ((True, a_before + (n_after - a_after)), (False, (n_before - a_before) + a_after)):
            acc = hits / len(preds)
            if acc > best[0]:
                best = (acc, float(ts[k]) if k < len(preds) else None, home_left)
    return best


def remap_labels(labels: dict, t_switch: float | None, home_left: bool) -> dict:
    out = json.loads(json.dumps(labels))
    for a in out["annotations"]:
        hl = home_left if (t_switch is None or int(a["position"]) / 1000 < t_switch) else not home_left
        a["team"] = ("left" if a["team"] == "home" else "right") if hl else ("right" if a["team"] == "home" else "left")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--minutes", type=float, default=None, help="Solo las acciones de los primeros N minutos")
    ap.add_argument("--dataset", default="efl", choices=["efl", "st2"],
                    help="st2: SoccerTrack v2 por partes, GT ya en left/right (Team-mAP exacto, sin remapeo)")
    ap.add_argument("--matches", nargs="+", default=["128057", "132831"])
    ap.add_argument("--view", default="vcam796vl1")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    with contextlib.chdir(REPO):
        from util.eval import label2vector, predictions2vector, soft_non_maximum_supression
        from util.score import compute_amAP
        from util.utils import EVENT_DICTIONARY_BALL

    classes = ["background"] + [line.strip() for line in (REPO / "data" / "soccernetball" / "class.txt").read_text().splitlines() if line.strip()]
    team_dict = {}
    for k, v in EVENT_DICTIONARY_BALL.items():
        team_dict[f"{k}-left"], team_dict[f"{k}-right"] = 2 * v + 1, 2 * v + 2

    summary = {"model": MODEL_NAME, "split": args.split, "minutes": args.minutes, "games": {}}
    targets, detections, closests = [], [], []
    if args.dataset == "efl":
        games = [(g["video"], LABELS_ROOT / g["video"] / "Labels-ball.json", LABELS_ROOT / g["video"] / "720p")
                 for g in json.loads((TDEED_SPLITS / f"{args.split}.json").read_text())]
    else:  # etiquetas generadas por eval_bas.py --dataset st2
        games = [(f"{m}/{h}", ST2_LABELS / m / h / "Labels-ball.json", ST2_VIEWS / m / f"{h}_{args.view}")
                 for m in args.matches for h in ("1st", "2nd")]
    for game, label_path, stem in games:
        labels = json.loads(label_path.read_text())
        if args.minutes:
            labels["annotations"] = [a for a in labels["annotations"] if int(a["position"]) < args.minutes * 60_000]
        z = np.load(stem.with_name(f"{stem.name}.tdeed.{MODEL_NAME}.npz"))
        fps_rows = float(z["fps"]) / int(z["stride"])
        preds = side_predictions(z, labels["annotations"], classes, fps_rows)
        acc, t_switch, home_left = best_mapping(preds, allow_switch=args.minutes is None and args.dataset == "efl")
        summary["games"][game] = {"team_acc": round(100 * acc, 2), "n": len(preds),
                                  "t_switch_s": t_switch, "home_left_first": home_left}
        if args.dataset == "st2":  # GT ya en left/right: acierto directo, sin elegir mapeo
            direct = np.mean([(team == "right") == bool(p) for _, team, p in preds])
            summary["games"][game]["team_acc_direct"] = round(100 * float(direct), 2)
            print(f"{game}: acierto directo (left/right del GT) {100 * direct:.1f}%")
        print(f"{game}: acierto de equipo {100 * acc:.1f}% en {len(preds)} acciones "
              f"(cambio de lado en {t_switch and round(t_switch / 60, 1)} min)")

        # Team-mAP@1 aproximado con el GT remapeado
        gt_lr = labels if args.dataset == "st2" else remap_labels(labels, t_switch, home_left)
        team = z["probs_team"].astype(np.float32)
        n_rows = len(team) if not args.minutes else int(args.minutes * 60 * fps_rows)
        rows, cols = np.nonzero(team[:n_rows, 1:] >= 0.01)
        inv = {v: k for k, v in team_dict.items()}
        events = [{"label": inv[c + 1], "frame": int(r), "score": float(team[r, c + 1])} for r, c in zip(rows, cols)]
        events = soft_non_maximum_supression([{"video": game, "events": events}], window=12, threshold=0.01)[0]["events"]
        pred_json = {"predictions": [
            {"label": e["label"].rsplit("-", 1)[0], "team": e["label"].rsplit("-", 1)[1],
             "position": int(e["frame"] / fps_rows * 1000), "confidence": e["score"]} for e in events]}
        lab = label2vector(gt_lr, num_classes=len(team_dict), EVENT_DICTIONARY=team_dict, framerate=25, event_team=True)
        det = predictions2vector(pred_json, num_classes=len(team_dict), EVENT_DICTIONARY=team_dict, framerate=25, event_team=True)
        closest = np.zeros(lab.shape) - 1
        for c in range(lab.shape[-1]):
            idx = np.where(lab[:, c] != 0)[0].tolist()
            if not idx:
                continue
            idx = [-idx[0]] + idx + [2 * closest.shape[0]]
            for i in range(1, len(idx) - 1):
                start, stop = max(0, (idx[i - 1] + idx[i]) // 2), min(closest.shape[0], (idx[i] + idx[i + 1]) // 2)
                closest[start:stop, c] = lab[idx[i], c]
        targets.append(lab)
        detections.append(det)
        closests.append(closest)

    with contextlib.redirect_stderr(io.StringIO()):
        res = compute_amAP(targets, detections, closests, framerate=25, metric="at1", event_team=True)
    summary["team_mAP@1_approx" if args.dataset == "efl" else "team_mAP@1_exact"] = round(100 * float(res["mAP"]), 2)
    summary["team_acc_overall"] = round(
        sum(g["team_acc"] * g["n"] for g in summary["games"].values()) / sum(g["n"] for g in summary["games"].values()), 2)
    if args.dataset == "st2":
        summary["team_acc_direct_overall"] = round(
            sum(g["team_acc_direct"] * g["n"] for g in summary["games"].values()) / sum(g["n"] for g in summary["games"].values()), 2)
        print(f"Acierto de equipo: directo {summary['team_acc_direct_overall']:.1f}%, con mejor mapeo por parte "
              f"{summary['team_acc_overall']:.1f}%  ·  Team-mAP@1 exacto {summary['team_mAP@1_exact']:.2f}")
        out = ST2_OUT / args.view / "team.json"
    else:
        print(f"Acierto de equipo global {summary['team_acc_overall']:.1f}%  ·  Team-mAP@1 (aprox., mapeo optimista) "
              f"{summary['team_mAP@1_approx']:.2f}")
        out = OUT_ROOT / args.split / (f"team_{int(args.minutes)}min.json" if args.minutes else "team.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
