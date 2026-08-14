"""Evalúa la telemetría del pipeline (``main.py --mode full-match``) contra un
ground truth de SoccerNet (``ground_truth/*.gt.json``).

Solo lee los dos JSON — ni vídeo, ni tracking, ni modelos —, así que corre en
segundos y sirve para A/B de cualquier cambio del pipeline sin re-inferir nada
que no haya cambiado.

Tres bloques de métricas, deliberadamente separados para no confundir "qué
puede hacer el disparador" con "qué hace el clasificador":

* **Acciones de balón**: emparejado greedy 1-a-1 (por frame más cercano dentro
  de ``--tolerance-frames``) entre las anotaciones GT y las predicciones NO
  background, con P/R/F1 por clase, macro-F1 y matriz de confusión.
* **Techos del disparador**: las mismas anotaciones GT emparejadas contra
  TODAS las predicciones (incluidas las ``background``). El recall que sale de
  aquí es el máximo alcanzable sin tocar el clasificador, y la precisión el
  máximo alcanzable sin tocar el disparador de cambio de posesión
  (``analytics/engine.py``).
* **Eventos**: recall/precisión por label de los tramos de ``core/events.py``,
  más el desglose de confianza (hoy no se filtra por confianza en ningún
  sitio, ver ``analytics/engine.py``).

Las anotaciones GT cuya clase NO está en el vocabulario del checkpoint (``out``,
``cross``, ``shot``, ``ball_player_block``, ``player_successful_tackle``) no
cuentan como ground truth — el modelo no puede predecirlas —, pero tampoco
penalizan: las predicciones que caen encima suyo se sacan del denominador de
precisión y se reportan aparte (ver ``IGNORED_GT_LABELS``).

Uso (desde la raíz del proyecto):
    python -m scripts.evaluate_pipeline \\
        --gt ground_truth/2019-10-01_Middlesbrough-Preston_North_End.gt.json \\
        --telemetry output/mi_run/telemetry.json

    # A/B contra un run anterior (columna extra con el delta):
    python -m scripts.evaluate_pipeline --gt ... --telemetry nuevo.json --compare viejo.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE_FRAMES = 25  # ±1s a 25fps

# Vocabulario del checkpoint de core/ball_actions.py (ver
# config.BallActionsConfig.classes / scripts/extract_ball_action_clips.py::
# BALL_LABEL_TO_CLASS). "background" no es una clase de ground truth: es la
# forma que tiene el modelo de decir "aquí no pasa nada".
ACTION_CLASSES: tuple[str, ...] = ("pass", "drive", "high_pass", "header", "throw_in")
BACKGROUND_LABEL = "background"
MISSED_LABEL = "(sin trigger)"

# Clases anotadas en el GT que el checkpoint no puede predecir (quedaron fuera
# a propósito de la primera fase, ver scripts/extract_ball_action_clips.py).
IGNORED_GT_LABELS: frozenset[str] = frozenset(
    {"out", "cross", "shot", "ball_player_block", "player_successful_tackle"}
)


# --------------------------------------------------------------------------- #
#  Emparejado GT <-> predicciones
# --------------------------------------------------------------------------- #

@dataclass
class ActionMetrics:
    """Resultado de evaluar el bloque de acciones de balón de un telemetry.json."""

    n_gt: int = 0
    n_predictions: int = 0          # predicciones no-background (las que se puntúan)
    n_triggers: int = 0             # TODAS las ventanas clasificadas (incl. background)
    n_matched: int = 0              # GT emparejadas con una predicción no-background
    n_correct: int = 0              # ...y además con el label correcto
    n_ignored_predictions: int = 0  # predicciones sobre GT fuera de vocabulario
    trigger_recall_n: int = 0       # GT emparejadas con CUALQUIER trigger
    tp: dict[str, int] = field(default_factory=dict)
    n_gt_by_class: dict[str, int] = field(default_factory=dict)
    n_pred_by_class: dict[str, int] = field(default_factory=dict)
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)

    @property
    def scored_predictions(self) -> int:
        return max(self.n_predictions - self.n_ignored_predictions, 0)

    @property
    def precision(self) -> float:
        return self.n_correct / self.scored_predictions if self.scored_predictions else 0.0

    @property
    def recall(self) -> float:
        return self.n_correct / self.n_gt if self.n_gt else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def matched_accuracy(self) -> float:
        return self.n_correct / self.n_matched if self.n_matched else 0.0

    @property
    def background_rate(self) -> float:
        """Fracción de acciones REALES que el modelo manda a ``background``."""
        n_bg = sum(self.confusion.get((c, BACKGROUND_LABEL), 0) for c in ACTION_CLASSES)
        return n_bg / self.trigger_recall_n if self.trigger_recall_n else 0.0

    @property
    def trigger_recall(self) -> float:
        """Techo de recall: GT que tiene ALGÚN trigger cerca."""
        return self.trigger_recall_n / self.n_gt if self.n_gt else 0.0

    @property
    def trigger_precision(self) -> float:
        """Techo de precisión: fracción de triggers que cae sobre una acción real."""
        return self.trigger_recall_n / self.n_triggers if self.n_triggers else 0.0

    def per_class(self, cls: str) -> tuple[float, float, float]:
        n_pred = self.n_pred_by_class.get(cls, 0)
        n_gt = self.n_gt_by_class.get(cls, 0)
        tp = self.tp.get(cls, 0)
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_gt if n_gt else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f1

    @property
    def macro_f1(self) -> float:
        scores = [self.per_class(c)[2] for c in ACTION_CLASSES if self.n_gt_by_class.get(c)]
        return sum(scores) / len(scores) if scores else 0.0


def _match_greedy(
    gt_frames: list[int], pred_frames: list[int], tolerance: int
) -> dict[int, int]:
    """Empareja 1-a-1 índices de GT con índices de predicción por cercanía.

    Recorre los pares candidatos ordenados por distancia creciente y va fijando
    los que aún tienen libres ambos extremos — así una predicción no se puede
    "gastar" en una anotación lejana habiendo otra pegada, que es lo que pasaría
    emparejando GT a GT en orden temporal. Devuelve ``{idx_gt: idx_pred}``.
    """
    candidates = [
        (abs(gf - pf), gi, pi)
        for gi, gf in enumerate(gt_frames)
        for pi, pf in enumerate(pred_frames)
        if abs(gf - pf) <= tolerance
    ]
    candidates.sort()

    matches: dict[int, int] = {}
    used_pred: set[int] = set()
    for _, gi, pi in candidates:
        if gi in matches or pi in used_pred:
            continue
        matches[gi] = pi
        used_pred.add(pi)
    return matches


def evaluate_ball_actions(
    annotations: list[dict], ball_actions: list[dict], tolerance: int
) -> ActionMetrics:
    gt = [a for a in annotations if a.get("ball_action_label") in ACTION_CLASSES]
    ignored_gt = [a for a in annotations if a.get("ball_action_label") in IGNORED_GT_LABELS]

    metrics = ActionMetrics(
        n_gt=len(gt),
        n_triggers=len(ball_actions),
        n_predictions=sum(1 for p in ball_actions if not p.get("is_background")),
    )
    for a in gt:
        cls = a["ball_action_label"]
        metrics.n_gt_by_class[cls] = metrics.n_gt_by_class.get(cls, 0) + 1
    for p in ball_actions:
        if not p.get("is_background"):
            metrics.n_pred_by_class[p["label"]] = metrics.n_pred_by_class.get(p["label"], 0) + 1

    gt_frames = [a["frame_idx"] for a in gt]

    # (a) Techo del disparador: contra TODAS las ventanas clasificadas.
    all_matches = _match_greedy(gt_frames, [p["frame_idx"] for p in ball_actions], tolerance)
    metrics.trigger_recall_n = len(all_matches)
    for gi, pi in all_matches.items():
        pred_label = ball_actions[pi]["label"]
        key = (gt[gi]["ball_action_label"], pred_label)
        metrics.confusion[key] = metrics.confusion.get(key, 0) + 1
    for gi, a in enumerate(gt):
        if gi not in all_matches:
            key = (a["ball_action_label"], MISSED_LABEL)
            metrics.confusion[key] = metrics.confusion.get(key, 0) + 1

    # (b) Métrica real: solo predicciones no-background.
    positives = [p for p in ball_actions if not p.get("is_background")]
    matches = _match_greedy(gt_frames, [p["frame_idx"] for p in positives], tolerance)
    metrics.n_matched = len(matches)
    for gi, pi in matches.items():
        if positives[pi]["label"] == gt[gi]["ball_action_label"]:
            cls = gt[gi]["ball_action_label"]
            metrics.tp[cls] = metrics.tp.get(cls, 0) + 1
            metrics.n_correct += 1

    # (c) Predicciones que caen sobre GT fuera de vocabulario: ni aciertan ni
    #     penalizan (el checkpoint no tiene esas clases). Solo se consideran las
    #     predicciones que no se emparejaron ya con una anotación puntuable.
    if ignored_gt:
        leftover = [pi for pi in range(len(positives)) if pi not in set(matches.values())]
        ignored_matches = _match_greedy(
            [a["frame_idx"] for a in ignored_gt], [positives[pi]["frame_idx"] for pi in leftover],
            tolerance,
        )
        metrics.n_ignored_predictions = len(ignored_matches)

    return metrics


# --------------------------------------------------------------------------- #
#  Eventos
# --------------------------------------------------------------------------- #

def evaluate_events(annotations: list[dict], events: list[dict]) -> dict:
    gt = [a for a in annotations if a.get("event_label")]
    labels = sorted({a["event_label"] for a in gt} | {e["label"] for e in events})

    per_label = {}
    for label in labels:
        g = [a for a in gt if a["event_label"] == label]
        p = [e for e in events if e["label"] == label]
        recall_n = sum(
            1 for a in g if any(e["start_s"] <= a["timestamp_s"] <= e["end_s"] for e in p)
        )
        precision_n = sum(
            1 for e in p if any(e["start_s"] <= a["timestamp_s"] <= e["end_s"] for a in g)
        )
        per_label[label] = {
            "n_gt": len(g), "n_pred": len(p),
            "recall_n": recall_n, "precision_n": precision_n,
        }

    confidences = [e.get("peak_confidence", 0.0) for e in events]
    return {
        "per_label": per_label,
        "n_spans": len(events),
        "n_low_confidence": sum(1 for c in confidences if c < 0.5),
        "n_single_window": sum(1 for e in events if e.get("n_windows", 1) == 1),
        "covered_s": sum(e["end_s"] - e["start_s"] for e in events),
    }


# --------------------------------------------------------------------------- #
#  Informe
# --------------------------------------------------------------------------- #

def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _delta(new: float, old: Optional[float]) -> str:
    if old is None:
        return ""
    d = 100 * (new - old)
    return f"  ({d:+5.1f})"


def print_report(
    metrics: ActionMetrics,
    events: dict,
    tolerance: int,
    baseline: Optional[ActionMetrics] = None,
    baseline_name: str = "",
) -> None:
    b = baseline
    print()
    print(f"══ ACCIONES DE BALÓN (tolerancia ±{tolerance} frames) ══")
    if b is not None:
        print(f"   (entre paréntesis: delta en puntos frente a {baseline_name})")
    print(f"   GT puntuable: {metrics.n_gt}   ventanas clasificadas: {metrics.n_triggers}   "
          f"predicciones no-background: {metrics.n_predictions}")
    if metrics.n_ignored_predictions:
        print(f"   predicciones sobre GT fuera de vocabulario (no penalizan): "
              f"{metrics.n_ignored_predictions}")
    print()
    print(f"   Precisión global          {_pct(metrics.precision)}"
          f"{_delta(metrics.precision, b.precision if b else None)}")
    print(f"   Recall global             {_pct(metrics.recall)}"
          f"{_delta(metrics.recall, b.recall if b else None)}")
    print(f"   F1 global                 {_pct(metrics.f1)}"
          f"{_delta(metrics.f1, b.f1 if b else None)}")
    print(f"   Macro-F1                  {_pct(metrics.macro_f1)}"
          f"{_delta(metrics.macro_f1, b.macro_f1 if b else None)}")
    print(f"   Accuracy s/ emparejadas   {_pct(metrics.matched_accuracy)}"
          f"{_delta(metrics.matched_accuracy, b.matched_accuracy if b else None)}"
          f"   ({metrics.n_correct}/{metrics.n_matched})")
    print(f"   Acciones reales -> background  {_pct(metrics.background_rate)}"
          f"{_delta(metrics.background_rate, b.background_rate if b else None)}")
    print()
    print("   Por clase:")
    print(f"     {'clase':<11}{'GT':>5}{'pred':>6}{'TP':>5}{'P':>9}{'R':>9}{'F1':>9}")
    for cls in ACTION_CLASSES:
        p, r, f1 = metrics.per_class(cls)
        print(f"     {cls:<11}{metrics.n_gt_by_class.get(cls, 0):>5}"
              f"{metrics.n_pred_by_class.get(cls, 0):>6}{metrics.tp.get(cls, 0):>5}"
              f"{_pct(p):>9}{_pct(r):>9}{_pct(f1):>9}")

    print()
    print("   Matriz de confusión (fila = GT, sobre TODAS las ventanas, incl. background):")
    cols = list(ACTION_CLASSES) + [BACKGROUND_LABEL, MISSED_LABEL]
    print("     " + " " * 11 + "".join(f"{c:>13}" for c in cols))
    for row in ACTION_CLASSES:
        cells = "".join(f"{metrics.confusion.get((row, c), 0):>13}" for c in cols)
        print(f"     {row:<11}{cells}")

    print()
    print("   Techos del disparador (independientes del clasificador):")
    print(f"     recall máximo alcanzable    {_pct(metrics.trigger_recall)}"
          f"   ({metrics.trigger_recall_n}/{metrics.n_gt} acciones con algún trigger cerca)")
    print(f"     precisión máxima alcanzable {_pct(metrics.trigger_precision)}"
          f"   ({metrics.trigger_recall_n}/{metrics.n_triggers} triggers sobre una acción real)")

    print()
    print("══ EVENTOS ══")
    print(f"   {events['n_spans']} tramos, cubren {events['covered_s']:.0f}s; "
          f"{events['n_low_confidence']} con peak_confidence<0.5, "
          f"{events['n_single_window']} de una sola ventana")
    print(f"     {'label':<14}{'GT':>5}{'pred':>6}{'recall':>12}{'precisión':>13}")
    for label, s in sorted(events["per_label"].items()):
        rec = f"{s['recall_n']}/{s['n_gt']}" if s["n_gt"] else "-"
        pre = f"{s['precision_n']}/{s['n_pred']}" if s["n_pred"] else "-"
        print(f"     {label:<14}{s['n_gt']:>5}{s['n_pred']:>6}{rec:>12}{pre:>13}")
    print()


def metrics_to_dict(metrics: ActionMetrics, events: dict) -> dict:
    return {
        "ball_actions": {
            "n_gt": metrics.n_gt,
            "n_triggers": metrics.n_triggers,
            "n_predictions": metrics.n_predictions,
            "n_matched": metrics.n_matched,
            "n_correct": metrics.n_correct,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "macro_f1": metrics.macro_f1,
            "matched_accuracy": metrics.matched_accuracy,
            "background_rate": metrics.background_rate,
            "trigger_recall": metrics.trigger_recall,
            "trigger_precision": metrics.trigger_precision,
            "per_class": {
                cls: dict(zip(("precision", "recall", "f1"), metrics.per_class(cls)))
                for cls in ACTION_CLASSES
            },
            "confusion": {f"{g}->{p}": n for (g, p), n in sorted(metrics.confusion.items())},
        },
        "events": events,
    }


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gt", required=True, type=Path, help="ground_truth/*.gt.json")
    parser.add_argument("--telemetry", required=True, type=Path,
                        help="telemetry.json del run a evaluar")
    parser.add_argument("--compare", default=None, type=Path,
                        help="telemetry.json de referencia para el A/B (columna de deltas)")
    parser.add_argument("--tolerance-frames", type=int, default=DEFAULT_TOLERANCE_FRAMES,
                        help=f"ventana de emparejado GT<->predicción (default {DEFAULT_TOLERANCE_FRAMES})")
    parser.add_argument("--json-out", default=None, type=Path,
                        help="Si se indica, vuelca las métricas a este JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for path in (args.gt, args.telemetry):
        if not path.exists():
            parser.error(f"No existe: {path}")

    gt_data = json.loads(args.gt.read_text())
    annotations = gt_data["annotations"]
    telemetry = json.loads(args.telemetry.read_text())

    metrics = evaluate_ball_actions(
        annotations, telemetry.get("ball_actions", []), args.tolerance_frames
    )
    events = evaluate_events(annotations, telemetry.get("events", []))

    baseline = None
    if args.compare:
        if not args.compare.exists():
            parser.error(f"No existe: {args.compare}")
        baseline = evaluate_ball_actions(
            annotations, json.loads(args.compare.read_text()).get("ball_actions", []),
            args.tolerance_frames,
        )

    print(f"\nGT:        {args.gt}  ({len(annotations)} anotaciones)")
    print(f"Telemetría: {args.telemetry}")
    if args.compare:
        print(f"Referencia: {args.compare}")
    print_report(metrics, events, args.tolerance_frames, baseline, str(args.compare))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics_to_dict(metrics, events), indent=2))
        logger.info("Métricas escritas en %s", args.json_out)


if __name__ == "__main__":
    main()
