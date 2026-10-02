"""Evalúa la detección de eventos en PARTIDO COMPLETO contra el ground truth
de SoccerNet, a partir de la caché de ventanas de
``scripts/scan_event_windows.py``. Solo lee JSON: corre en segundos.

Compara dos post-procesos sobre la MISMA caché, así que la diferencia entre
ellos es solo el post-proceso, no el modelo:

* ``merge``: lo que hace hoy el pipeline (``analytics.engine.
  _merge_event_windows``): etiqueta ganadora por ventana + fusión de tramos
  contiguos. Un tramo acierta un GT si lo contiene (con ±tolerancia).
* ``spot``: ``core/spotting.py``: curva por clase + picos + NMS. Un pico
  acierta un GT si está a ±tolerancia.

Emparejado 1 a 1 (greedy por distancia, igual que ``evaluate_pipeline.
_match_greedy``): un segundo aviso sobre el mismo evento cuenta como falso
positivo, y un tramo ancho no puede "acertar" varios eventos a la vez.

Convención gol ⊂ tiro (ver ``core/spotting.py``): el GT de "shot" son todas
las anotaciones de tiro (las de los goles incluidas, ya vienen anotadas
aparte), y en ``merge`` un tramo "goal" cuenta también como predicción de
tiro.

GT soportado, buscado en la carpeta del vídeo:
* ``Labels-ball.json`` (Ball Action Spotting, 7 partidos EFL): SHOT, GOAL.
* ``Labels-v2.json`` (SoccerNet-v2, 500 partidos): la parte (1/2) sale del
  nombre del vídeo (``1_720p.mkv``). Si el vídeo está en
  ``/mnt/datasets/SoccerNet``, las etiquetas se buscan en el mismo árbol bajo
  ``datasets/SoccerNet`` del proyecto.

Uso (desde la raíz del proyecto, dentro del contenedor):
    python -m scripts.evaluate_spotting \\
        --cache datasets/SoccerNet/england_efl/2019-2020/*/720p.event_windows.vjepa2.w6s3.jsonl

    # Tiros no detectados, con lo que vio el modelo en su ventana
    python -m scripts.evaluate_spotting --cache ... --list-missed

    # Con los veredictos del VLM (scripts/verify_shot_candidates.py): añade el
    # método "spot+VLM", puntuado con el P(tiro) del verificador
    python -m scripts.evaluate_spotting --cache ... --verdicts
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.spotting import CURVE_COMPONENTS, class_curve, find_peaks

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MNT_SOCCERNET = Path("/mnt/datasets/SoccerNet")

BALL_LABEL_MAP: dict[str, str] = {"SHOT": "shot", "GOAL": "goal"}
V2_LABEL_MAP: dict[str, str] = {
    "Shots on target": "shot",
    "Shots off target": "shot",
    "Goal": "goal",
    "Corner": "corner",
    "Foul": "foul",
    "Yellow card": "card",
    "Red card": "card",
    "Yellow->red card": "card",
    "Substitution": "substitution",
    "Kick-off": "kick-off",
    "Throw-in": "throw-in",
    "Penalty": "penalty",
}

DEFAULT_TOLERANCES_S: tuple[float, ...] = (1.0, 3.0, 5.0)
MAIN_TOLERANCE_S: float = 3.0
SWEEP_THRESHOLDS: tuple[float, ...] = (
    0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.0,
)


# --------------------------------------------------------------------------- #
#  Ground truth
# --------------------------------------------------------------------------- #

def _find_labels(video_path: Path) -> Path:
    candidates = [video_path.parent]
    try:
        rel = video_path.parent.resolve().relative_to(MNT_SOCCERNET)
        candidates.append(PROJECT_ROOT / "datasets" / "SoccerNet" / rel)
    except ValueError:
        pass
    for d in candidates:
        for name in ("Labels-ball.json", "Labels-v2.json"):
            if (d / name).exists():
                return d / name
    raise FileNotFoundError(f"Sin Labels-ball.json ni Labels-v2.json para {video_path}")


def load_ground_truth(video_path: Path) -> tuple[dict[str, list[float]], Path]:
    """``{clase: [instantes en s]}`` para el vídeo, y la ruta del fichero leído."""
    labels_path = _find_labels(video_path)
    annotations = json.loads(labels_path.read_text())["annotations"]
    gt: dict[str, list[float]] = {}

    if labels_path.name == "Labels-ball.json":
        for a in annotations:
            label = BALL_LABEL_MAP.get(a["label"])
            if label:
                gt.setdefault(label, []).append(int(a["position"]) / 1000)
    else:
        half = video_path.name.split("_", 1)[0]
        for a in annotations:
            label = V2_LABEL_MAP.get(a["label"])
            if label and a["gameTime"].split(" - ", 1)[0] == half:
                gt.setdefault(label, []).append(int(a["position"]) / 1000)

    return {k: sorted(v) for k, v in gt.items()}, labels_path


# --------------------------------------------------------------------------- #
#  Predicciones (los dos post-procesos)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Prediction:
    t_s: float     # instante representativo (centro del tramo, o el pico)
    lo_s: float    # extremos del intervalo que "cubre" (igual a t_s en un pico)
    hi_s: float
    score: float


def _curve_parts(label: str) -> tuple[str, ...]:
    return CURVE_COMPONENTS.get(label, (label,))


def predictions_merge(meta: dict, rows: list[dict], label: str) -> list[Prediction]:
    """Tramos de ``_merge_event_windows`` (el post-proceso actual del pipeline)
    cuya etiqueta cuenta como ``label``."""
    from analytics.engine import _merge_event_windows
    from core.events import EventPrediction
    from core.windowing import Window

    fps = meta["fps"]
    pairs = [
        (
            Window(
                index=r["index"], start_frame=round(r["start_s"] * fps), end_frame=round(r["end_s"] * fps),
                start_s=r["start_s"], end_s=r["end_s"],
            ),
            EventPrediction(
                video_path=meta["source_video"], label=r["label"], confidence=r["confidence"],
                probabilities=r["probabilities"], is_background=r["is_background"],
            ),
        )
        for r in rows
    ]
    parts = _curve_parts(label)
    return [
        Prediction(t_s=(s.start_s + s.end_s) / 2, lo_s=s.start_s, hi_s=s.end_s, score=s.peak_confidence)
        for s in _merge_event_windows(pairs)
        if s.label in parts
    ]


def predictions_verified(verdicts: list[dict], label: str) -> list[Prediction]:
    """Candidatos verificados por el VLM. "shot": score = P(tiro) del
    verificador. "goal": solo los que respondió "goal" como desenlace, con
    score P(tiro) x P(goal)."""
    preds = []
    for v in verdicts:
        t = v["t_s"]
        if label == "shot":
            score = v["p_shot"]
        elif label == "goal" and v.get("outcome") == "goal":
            score = v["p_shot"] * v["outcome_probs"]["goal"]
        else:
            continue
        preds.append(Prediction(t_s=t, lo_s=t, hi_s=t, score=score))
    return preds


def predictions_spot(rows: list[dict], label: str, nms_s: float, offset_s: float) -> list[Prediction]:
    """Todos los picos de la curva de ``label`` (sin umbral: se barre después)."""
    times, scores = class_curve(rows, label)
    return [
        Prediction(t_s=t + offset_s, lo_s=t + offset_s, hi_s=t + offset_s, score=s)
        for t, s in find_peaks(times, scores, min_score=0.0, nms_s=nms_s)
    ]


# --------------------------------------------------------------------------- #
#  Emparejado y métricas
# --------------------------------------------------------------------------- #

def match_1to1(gt: list[float], preds: list[Prediction], tol: float) -> list[tuple[int, int]]:
    """Pares ``(i_gt, i_pred)`` 1 a 1, greedy por distancia creciente. Un GT
    es alcanzable por una predicción si cae en ``[lo - tol, hi + tol]``."""
    candidates = sorted(
        (abs(g - p.t_s), gi, pi)
        for gi, g in enumerate(gt)
        for pi, p in enumerate(preds)
        if p.lo_s - tol <= g <= p.hi_s + tol
    )
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    pairs = []
    for _, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        pairs.append((gi, pi))
    return pairs


@dataclass
class Counts:
    n_gt: int = 0
    n_pred: int = 0
    tp: int = 0

    def add(self, other: "Counts") -> None:
        self.n_gt += other.n_gt
        self.n_pred += other.n_pred
        self.tp += other.tp

    @property
    def precision(self) -> float:
        return self.tp / self.n_pred if self.n_pred else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_gt if self.n_gt else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def count(gt: list[float], preds: list[Prediction], tol: float, min_score: float = 0.0) -> Counts:
    kept = [p for p in preds if p.score >= min_score]
    return Counts(n_gt=len(gt), n_pred=len(kept), tp=len(match_1to1(gt, kept, tol)))


def average_precision(matches: list[tuple[list[float], list[Prediction]]], tol: float) -> float:
    """AP clásica de detección: predicciones de todos los partidos ordenadas
    por score; cada una se queda el GT libre más cercano de SU partido."""
    n_gt = sum(len(g) for g, _ in matches)
    if not n_gt:
        return 0.0
    ranked = sorted(
        ((p.score, mi, p) for mi, (_, preds) in enumerate(matches) for p in preds),
        key=lambda x: -x[0],
    )
    taken: list[set[int]] = [set() for _ in matches]
    tps: list[bool] = []
    for _, mi, p in ranked:
        gt = matches[mi][0]
        options = [
            (abs(g - p.t_s), gi) for gi, g in enumerate(gt)
            if gi not in taken[mi] and p.lo_s - tol <= g <= p.hi_s + tol
        ]
        if options:
            taken[mi].add(min(options)[1])
        tps.append(bool(options))

    precisions, recalls = [], []
    tp = 0
    for k, is_tp in enumerate(tps, start=1):
        tp += is_tp
        precisions.append(tp / k)
        recalls.append(tp / n_gt)
    # Precisión interpolada (máximo a la derecha), área por escalones de recall.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap, prev_r = 0.0, 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_r)
        prev_r = r
    return ap


# --------------------------------------------------------------------------- #
#  Informe
# --------------------------------------------------------------------------- #

@dataclass
class MatchData:
    name: str
    meta: dict
    rows: list[dict]
    gt: dict[str, list[float]]
    verdicts: Optional[list[dict]] = None


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _fmt_time(t: float) -> str:
    return f"{int(t // 60):3d}:{t % 60:04.1f}"


def _best_threshold(matches: list[tuple[list[float], list[Prediction]]], tol: float) -> float:
    """Umbral de mejor F1 en una rejilla de 0,005 (probar cada score único
    como umbral es O(scores × GT × pred) y se va a minutos con 7 partidos)."""
    best_f1, best_th = -1.0, 0.0
    for th in (i / 200 for i in range(201)):
        total = Counts()
        for gt, preds in matches:
            total.add(count(gt, preds, tol, th))
        if total.f1 > best_f1:
            best_f1, best_th = total.f1, th
    return best_th


def _window_summary(rows: list[dict], t: float, label: str, tol: float) -> str:
    """Qué vio el modelo alrededor de ``t``: máximo de la curva de ``label``
    en ±tol y las 3 clases más probables de la ventana centrada más cerca."""
    near = [r for r in rows if abs((r["start_s"] + r["end_s"]) / 2 - t) <= tol]
    if not near:
        return "(sin ventanas escaneadas aquí)"
    times, scores = class_curve(near, label)
    best = min(near, key=lambda r: abs((r["start_s"] + r["end_s"]) / 2 - t))
    top = sorted(best["probabilities"].items(), key=lambda kv: -kv[1])[:3]
    top_s = ", ".join(f"{k.replace('background_', 'bg_')}={v:.2f}" for k, v in top)
    return f"max curva ±{tol:g}s={max(scores):.2f} | ganó '{best['label']}' | {top_s}"


def report(
    data: list[MatchData], classes: list[str], nms_s: dict[str, float], offset_s: dict[str, float],
    tolerances: list[float], list_missed: bool,
) -> dict:
    out: dict = {"matches": [d.name for d in data], "classes": {}}
    print(f"Partidos: {len(data)} | horas de vídeo: "
          f"{sum(d.meta['total_frames'] / d.meta['fps'] for d in data) / 3600:.1f}")

    for label in classes:
        per_match_merge = [(d.gt.get(label, []), predictions_merge(d.meta, d.rows, label)) for d in data]
        per_match_spot = [
            (d.gt.get(label, []), predictions_spot(d.rows, label, nms_s[label], offset_s[label]))
            for d in data
        ]
        n_gt = sum(len(g) for g, _ in per_match_spot)
        th = _best_threshold(per_match_spot, MAIN_TOLERANCE_S)
        with_verdicts = all(d.verdicts is not None for d in data)
        per_match_ver = (
            [(d.gt.get(label, []), predictions_verified(d.verdicts, label)) for d in data]
            if with_verdicts else []
        )
        tau = _best_threshold(per_match_ver, MAIN_TOLERANCE_S) if with_verdicts else 0.0

        print(f"\n=== {label}  (GT: {n_gt} en {len(data)} partidos; spot: NMS ±{nms_s[label]:g}s, "
              f"offset {offset_s[label]:+g}s) ===")
        print(f"{'método':<26}{'tol':>5}{'pred':>7}{'TP':>6}{'P':>8}{'R':>8}{'F1':>8}")
        cls_out: dict = {"n_gt": n_gt, "spot_threshold_best_f1": th, "rows": []}
        methods = [
            ("merge (actual)", per_match_merge, 0.0),
            (f"spot θ={th:.3f} (mejor F1)", per_match_spot, th),
        ]
        if with_verdicts:
            methods += [
                ("candidatos al VLM", per_match_ver, 0.0),
                (f"spot+VLM τ={tau:.3f}", per_match_ver, tau),
            ]
            cls_out["vlm_threshold_best_f1"] = tau
        for method, per_match, min_score in methods:
            for tol in tolerances:
                total = Counts()
                for gt, preds in per_match:
                    total.add(count(gt, preds, tol, min_score))
                print(f"{method:<26}{'±' + format(tol, 'g'):>5}{total.n_pred:>7}{total.tp:>6}"
                      f"{_pct(total.precision):>8}{_pct(total.recall):>8}{_pct(total.f1):>8}")
                cls_out["rows"].append({"method": method, "tol": tol, "n_pred": total.n_pred,
                                        "tp": total.tp, "precision": total.precision,
                                        "recall": total.recall, "f1": total.f1})
        ap_merge = average_precision(per_match_merge, MAIN_TOLERANCE_S)
        ap_spot = average_precision(per_match_spot, MAIN_TOLERANCE_S)
        ap_line = f"AP ±{MAIN_TOLERANCE_S:g}s (sin umbral):  merge {_pct(ap_merge)}   spot {_pct(ap_spot)}"
        cls_out.update(ap_merge=ap_merge, ap_spot=ap_spot)
        if with_verdicts:
            ap_ver = average_precision(per_match_ver, MAIN_TOLERANCE_S)
            ap_line += f"   spot+VLM {_pct(ap_ver)}"
            cls_out["ap_verified"] = ap_ver
        print(ap_line)

        # Offset: instante real - instante predicho, sobre aciertos a ±5 s.
        diffs = []
        for gt, preds in per_match_spot:
            kept = [p for p in preds if p.score >= th]
            diffs += [gt[gi] - kept[pi].t_s for gi, pi in match_1to1(gt, kept, 5.0)]
        if len(diffs) >= 4:
            q = statistics.quantiles(diffs, n=4)
            print(f"Offset GT - pico (spot, aciertos ±5s, n={len(diffs)}): mediana {statistics.median(diffs):+.2f}s "
                  f"(p25 {q[0]:+.2f}, p75 {q[2]:+.2f}) -> pasar como --offset-{label}")
            cls_out["offset_median_s"] = statistics.median(diffs)

        # Barrido de umbral: coste (candidatos por partido) frente a recall.
        print(f"\nBarrido spot ±{MAIN_TOLERANCE_S:g}s  (para elegir el umbral de candidatos del verificador)")
        print(f"{'θ':>6}{'cand/partido':>14}{'recall':>9}{'precisión':>11}")
        cls_out["sweep"] = []
        for sweep_th in SWEEP_THRESHOLDS:
            total = Counts()
            for gt, preds in per_match_spot:
                total.add(count(gt, preds, MAIN_TOLERANCE_S, sweep_th))
            print(f"{sweep_th:>6.2f}{total.n_pred / len(data):>14.1f}{_pct(total.recall):>9}{_pct(total.precision):>11}")
            cls_out["sweep"].append({"threshold": sweep_th, "pred_per_match": total.n_pred / len(data),
                                     "recall": total.recall, "precision": total.precision})

        if with_verdicts:
            print(f"\nBarrido spot+VLM ±{MAIN_TOLERANCE_S:g}s  (τ sobre el score del verificador; el recall "
                  f"no puede superar el de los candidatos)")
            print(f"{'τ':>6}{'pred/partido':>14}{'recall':>9}{'precisión':>11}")
            cls_out["sweep_verified"] = []
            for sweep_th in SWEEP_THRESHOLDS:
                total = Counts()
                for gt, preds in per_match_ver:
                    total.add(count(gt, preds, MAIN_TOLERANCE_S, sweep_th))
                print(f"{sweep_th:>6.2f}{total.n_pred / len(data):>14.1f}{_pct(total.recall):>9}"
                      f"{_pct(total.precision):>11}")
                cls_out["sweep_verified"].append({"threshold": sweep_th, "pred_per_match": total.n_pred / len(data),
                                                  "recall": total.recall, "precision": total.precision})

        # Por partido, en el punto de mejor F1.
        print(f"\nPor partido (spot θ={th:.3f}, ±{MAIN_TOLERANCE_S:g}s)")
        for d, (gt, preds) in zip(data, per_match_spot):
            c = count(gt, preds, MAIN_TOLERANCE_S, th)
            print(f"  {d.name[:48]:<48} GT {c.n_gt:3d}  pred {c.n_pred:3d}  TP {c.tp:3d}  "
                  f"P {_pct(c.precision)}  R {_pct(c.recall)}")

        if list_missed:
            print(f"\nNo detectados (spot θ={th:.3f}, ±{MAIN_TOLERANCE_S:g}s)")
            for d, (gt, preds) in zip(data, per_match_spot):
                kept = [p for p in preds if p.score >= th]
                hit = {gi for gi, _ in match_1to1(gt, kept, MAIN_TOLERANCE_S)}
                for gi, g in enumerate(gt):
                    if gi not in hit:
                        print(f"  {d.name[:34]:<34} {_fmt_time(g)}  "
                              f"{_window_summary(d.rows, g, label, MAIN_TOLERANCE_S)}")

        out["classes"][label] = cls_out
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", required=True, nargs="+", type=Path,
                        help="Caché(s) *.event_windows.*.jsonl de scan_event_windows")
    parser.add_argument("--classes", nargs="+", default=["shot", "goal"])
    parser.add_argument("--tolerances", nargs="+", type=float, default=list(DEFAULT_TOLERANCES_S))
    # 4 s y no 8: con paso de 3 s dos máximos locales ya distan >= 6 s, así que
    # ±8 s solo suprimía rebotes reales (7 EFL, θ=0: recall ±3s 84% -> 87%).
    parser.add_argument("--nms-shot", type=float, default=4.0)
    parser.add_argument("--nms-goal", type=float, default=30.0)
    parser.add_argument("--nms-default", type=float, default=8.0, help="Resto de clases")
    parser.add_argument("--offset-shot", type=float, default=0.0)
    parser.add_argument("--offset-goal", type=float, default=0.0)
    parser.add_argument("--list-missed", action="store_true")
    parser.add_argument("--verdicts", action="store_true",
                        help="Añade el método spot+VLM con los veredictos de verify_shot_candidates")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from analytics.engine import read_event_windows

    data: list[MatchData] = []
    for cache in args.cache:
        meta, rows = read_event_windows(cache)
        video = Path(meta["source_video"])
        gt, labels_path = load_ground_truth(video)
        verdicts = None
        if args.verdicts:
            from core.events_vllm import SHOT_VERIFIER_PROMPT_VERSION
            from scripts.verify_shot_candidates import read_verdicts, verdicts_path_for

            verdicts_path = verdicts_path_for(cache, SHOT_VERIFIER_PROMPT_VERSION)
            if not verdicts_path.exists():
                parser.error(f"Sin veredictos para {cache} (esperaba {verdicts_path})")
            verdicts = read_verdicts(verdicts_path)[1]
        data.append(MatchData(name=video.parent.name, meta=meta, rows=rows, gt=gt, verdicts=verdicts))
        logger.info("%s: %d ventanas, GT %s (%s)", video.parent.name, len(rows),
                    {k: len(v) for k, v in gt.items()}, labels_path.name)

    nms = {c: getattr(args, f"nms_{c}", None) or args.nms_default for c in args.classes}
    offsets = {c: getattr(args, f"offset_{c}", 0.0) or 0.0 for c in args.classes}
    result = report(data, args.classes, nms, offsets, args.tolerances, args.list_missed)

    if args.json_out:
        result["params"] = {"nms_s": nms, "offset_s": offsets}
        args.json_out.write_text(json.dumps(result, indent=2))
        logger.info("Resultados en %s", args.json_out)


if __name__ == "__main__":
    main()
