"""Arnés único de evaluación de un partido: telemetría del pipeline contra un ground truth.

Sustituye a ``scripts/evaluate_pipeline.py`` (del que reutiliza el emparejado de acciones, así que reproduce
sus cifras) y añade lo que faltaba: eventos emparejados 1 a 1 con |Δt|, goles, equipo del ejecutor, jugador
(cuando el GT lo anotó a mano), fps real del vídeo y resolución de rutas del backend. Solo lee JSON: segundos.

Entradas:
* verdad: ``--gt ground_truth/<partido>.gt.json`` (``scripts/build_ground_truth.py``) o directamente la hoja de
  anotación manual ``--csv output/annotations/<partido>_Ground_Truth.csv`` (``scripts/annotation_csv.py``); con
  el CSV solo se puntúa el tramo anotado (de la primera a la última fila);
* predicción: ``--telemetry telemetry.json`` (+ ``--teams teams.json``, por defecto el de al lado) o
  ``--backend-match <id>`` (resuelve telemetry.json, teams.json y el fps del vídeo en ``backend/storage``);
* ``--compare otro_telemetry.json``: columna de deltas (A/B de dos versiones del pipeline).

Bloques:
* **Acciones de balón**: emparejado greedy 1 a 1 por frame (``±tolerance-s`` convertido con el fps real) entre
  las anotaciones del vocabulario del clasificador y las predicciones no-background; P/R/F1, macro-F1, matriz de
  confusión y techos del disparador (igual que evaluate_pipeline.py).
* **Equipo**: en las acciones emparejadas, el ``team_cluster`` del ejecutor (teams.json) contra ``home``/``away``
  del GT. El pipeline no sabe qué clúster es el local, así que se toma la asignación clúster->lado que más
  acierta (se informa cuál) y, aparte, la pureza de cada clúster.
* **Jugador**: si el GT lleva el ``tracker_id`` anotado a mano (CSV: columna ``dorsal``), fracción de acciones
  emparejadas con el mismo tracker (y con el mismo clúster de equipo).
* **Eventos**: emparejado 1 a 1 por |Δt| ≤ ``--event-tolerance-s`` por etiqueta; P/R/F1 y mediana de |Δt|.
  Convención gol ⊂ tiro: para "shot" cuentan también los GOAL del GT y los "goal" predichos; "goal" aparte.
* **Goles**: cada gol del GT, con el mejor "goal" y "shot" predicho cerca.

Uso (raíz del proyecto; sin numpy ni GPU: corre en el host o en el contenedor):
    python -m scripts.evaluate_match --gt ground_truth/2019-10-01_Middlesbrough-Preston_North_End.gt.json \\
        --backend-match 6565edb583a4a434
    python -m scripts.evaluate_match --csv output/annotations/espanyol_damm_Ground_Truth.csv \\
        --backend-match f2c9315d3394c6ee --list-errors
    python -m scripts.evaluate_match --gt ... --telemetry nuevo.json --compare viejo.json --json-out out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Optional

from scripts.build_ground_truth import build_annotations, from_csv
from scripts.evaluate_pipeline import (
    ACTION_CLASSES,
    BACKGROUND_LABEL,
    MISSED_LABEL,
    ActionMetrics,
    _match_greedy,
    evaluate_ball_actions,
)

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE_S = 1.0
DEFAULT_EVENT_TOLERANCE_S = 3.0
SPOT_EVENT_HALF_SPAN_S = 1.0          # analytics.engine: el pico está en start_s + 1 s
SHOT_LIKE = ("shot", "goal")          # gol ⊂ tiro, en el GT y en las predicciones


# --------------------------------------------------------------------------- #
#  Entradas
# --------------------------------------------------------------------------- #

def backend_paths(match_id: str, part_id: str, storage_dir: Path) -> dict:
    match = json.loads((storage_dir / "matches" / match_id / "match.json").read_text())
    part = next(p for p in match["parts"] if p["id"] == part_id)
    video_dir = storage_dir / "videos" / part["video_id"]
    video = json.loads((video_dir / "video.json").read_text())
    out_dir = storage_dir / "outputs" / match_id / part_id / "output"
    return {
        "name": match.get("name"), "home_team": match.get("home_team"), "away_team": match.get("away_team"),
        "video": video_dir / video.get("stored_filename", "source.mp4"), "fps": float(video["fps"]),
        "duration_s": float(video.get("duration_s") or 0.0),
        "telemetry": out_dir / "telemetry.json", "teams": out_dir / "teams.json",
    }


def load_team_by_tracker(path: Optional[Path]) -> dict[int, int]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {int(t["tracker_id"]): int(t["team_cluster"]) for t in raw.get("teams", [])}


def event_instants(telemetry: dict) -> list[dict]:
    """``EventSpan`` -> instante del pico (``start_s + 1`` salvo tramo recortado en 0)."""
    out = []
    for e in telemetry.get("events", []):
        start, end = float(e["start_s"]), float(e["end_s"])
        t = start + SPOT_EVENT_HALF_SPAN_S if end - start >= 2 * SPOT_EVENT_HALF_SPAN_S - 1e-6 else (start + end) / 2
        out.append({"t_s": t, "label": e["label"], "score": float(e.get("peak_confidence", 0.0)),
                    "team": e.get("team"), "tracker_id": e.get("tracker_id")})
    out.sort(key=lambda e: e["t_s"])
    return out


def restrict_span(telemetry: dict, fps: float, span: Optional[tuple[float, float]], tol_s: float, ev_tol_s: float) -> dict:
    """Deja solo las predicciones del tramo anotado (con margen de la tolerancia): fuera de él no hay verdad."""
    if span is None:
        return telemetry
    lo, hi = span
    out = dict(telemetry)
    out["ball_actions"] = [a for a in telemetry.get("ball_actions", [])
                           if lo - tol_s <= a["frame_idx"] / fps <= hi + tol_s]
    out["events"] = [e for e in telemetry.get("events", [])
                     if lo - ev_tol_s <= float(e["start_s"]) + SPOT_EVENT_HALF_SPAN_S <= hi + ev_tol_s]
    return out


# --------------------------------------------------------------------------- #
#  Equipo y jugador (sobre las acciones emparejadas)
# --------------------------------------------------------------------------- #

def matched_pairs(annotations: list[dict], ball_actions: list[dict], tolerance: int) -> list[tuple[dict, dict]]:
    """Las mismas parejas GT<->predicción no-background que puntúa ``evaluate_ball_actions``."""
    gt = [a for a in annotations if a.get("ball_action_label") in ACTION_CLASSES]
    positives = [p for p in ball_actions if not p.get("is_background")]
    matches = _match_greedy([a["frame_idx"] for a in gt], [p["frame_idx"] for p in positives], tolerance)
    return [(gt[gi], positives[pi]) for gi, pi in sorted(matches.items())]


def evaluate_team(pairs: list[tuple[dict, dict]], team_by_tracker: dict[int, int]) -> dict:
    counts: Counter = Counter()        # (lado GT, clúster) -> n
    n_side = 0
    for a, p in pairs:
        side = a.get("team")
        if side not in ("home", "away"):
            continue
        n_side += 1
        tid = int(p.get("tracker_id", -1))
        cluster = team_by_tracker.get(tid) if tid >= 0 else None
        counts[(side, cluster)] += 1
    with_cluster = sum(n for (_, c), n in counts.items() if c is not None)
    clusters = sorted({c for (_, c) in counts if c is not None})
    best = None
    for mapping in ({"home": 0, "away": 1}, {"home": 1, "away": 0}):
        ok = sum(n for (side, c), n in counts.items() if c is not None and mapping[side] == c)
        if best is None or ok > best[0]:
            best = (ok, mapping)
    purity = {}
    for c in clusters:
        tot = sum(n for (_, cc), n in counts.items() if cc == c)
        top = max(counts.get(("home", c), 0), counts.get(("away", c), 0))
        purity[str(c)] = {"n": tot, "purity": top / tot if tot else None,
                          "home": counts.get(("home", c), 0), "away": counts.get(("away", c), 0)}
    return {
        "n_matched_with_side": n_side, "n_with_cluster": with_cluster,
        "coverage": with_cluster / n_side if n_side else None,
        "accuracy": (best[0] / with_cluster) if with_cluster else None,
        "n_correct": best[0] if best else 0, "mapping": best[1] if best else None, "clusters": purity,
    }


def evaluate_possessor(pairs: list[tuple[dict, dict]], team_by_tracker: dict[int, int]) -> Optional[dict]:
    manual = [(a, p) for a, p in pairs
              if a.get("possession", {}).get("from_annotation") and a["possession"].get("tracker_id") is not None]
    if not manual:
        return None
    same = sum(1 for a, p in manual if int(p.get("tracker_id", -1)) == int(a["possession"]["tracker_id"]))
    same_cluster = n_cluster = 0
    for a, p in manual:
        cg, cp = team_by_tracker.get(int(a["possession"]["tracker_id"])), team_by_tracker.get(int(p.get("tracker_id", -1)))
        if cg is not None and cp is not None:
            n_cluster += 1
            same_cluster += int(cg == cp)
    return {"n": len(manual), "same_tracker": same, "accuracy": same / len(manual),
            "n_with_cluster": n_cluster, "same_cluster": same_cluster,
            "cluster_accuracy": same_cluster / n_cluster if n_cluster else None}


# --------------------------------------------------------------------------- #
#  Eventos y goles
# --------------------------------------------------------------------------- #

def _gt_events(annotations: list[dict], label: str) -> list[dict]:
    wanted = SHOT_LIKE if label == "shot" else (label,)
    return [a for a in annotations if a.get("event_label") in wanted]


def _pred_events(events: list[dict], label: str) -> list[dict]:
    wanted = SHOT_LIKE if label == "shot" else (label,)
    return [e for e in events if e["label"] in wanted]


def evaluate_events(annotations: list[dict], events: list[dict], tol_s: float,
                    team_by_tracker: dict[int, int], team_mapping: Optional[dict]) -> dict:
    labels = sorted({a["event_label"] for a in annotations if a.get("event_label")} | {e["label"] for e in events})
    per_label = {}
    for label in labels:
        g, p = _gt_events(annotations, label), _pred_events(events, label)
        matches = _match_greedy([a["timestamp_s"] for a in g], [e["t_s"] for e in p], tol_s)
        tp = len(matches)
        prec = tp / len(p) if p else None
        rec = tp / len(g) if g else None
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec else (0.0 if (g and p) else None)
        deltas = [abs(g[gi]["timestamp_s"] - p[pi]["t_s"]) for gi, pi in matches.items()]
        team_ok = team_n = 0
        for gi, pi in matches.items():
            side, pred_team = g[gi].get("team"), p[pi].get("team")
            if side in ("home", "away") and pred_team is not None and team_mapping:
                cluster = pred_team if isinstance(pred_team, int) else team_by_tracker.get(int(p[pi].get("tracker_id", -1)))
                if cluster is not None:
                    team_n += 1
                    team_ok += int(team_mapping[side] == cluster)
        per_label[label] = {
            "n_gt": len(g), "n_pred": len(p), "tp": tp, "precision": prec, "recall": rec, "f1": f1,
            "median_abs_dt_s": statistics.median(deltas) if deltas else None,
            "missed": [round(a["timestamp_s"], 1) for gi, a in enumerate(g) if gi not in matches],
            "false_alarms": [(round(e["t_s"], 1), round(e["score"], 2)) for pi, e in enumerate(p) if pi not in set(matches.values())],
            "team_n": team_n, "team_correct": team_ok,
        }
    return per_label


def evaluate_goals(annotations: list[dict], events: list[dict], tol_s: float) -> list[dict]:
    out = []
    for a in annotations:
        if a.get("event_label") != "goal":
            continue
        t = a["timestamp_s"]
        near = [e for e in events if abs(e["t_s"] - t) <= tol_s]
        best_goal = max((e["score"] for e in near if e["label"] == "goal"), default=None)
        best_shot = max((e["score"] for e in near if e["label"] in SHOT_LIKE), default=None)
        out.append({"t_s": t, "team": a.get("team"), "goal_pred": best_goal, "shot_pred": best_shot})
    return out


# --------------------------------------------------------------------------- #
#  Informe
# --------------------------------------------------------------------------- #

def _pct(x: Optional[float]) -> str:
    return "    - " if x is None else f"{100 * x:5.1f}%"


def _delta(new: Optional[float], old: Optional[float]) -> str:
    if old is None or new is None:
        return ""
    return f"  ({100 * (new - old):+5.1f})"


def _hms(t: float) -> str:
    t = float(t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:04.1f}"


def print_actions(m: ActionMetrics, tolerance: int, b: Optional[ActionMetrics]) -> None:
    print(f"══ ACCIONES DE BALÓN (tolerancia ±{tolerance} frames) ══")
    print(f"   GT puntuable: {m.n_gt}   ventanas clasificadas: {m.n_triggers}   predicciones no-background: {m.n_predictions}")
    if m.n_ignored_predictions:
        print(f"   predicciones sobre GT fuera de vocabulario (no penalizan): {m.n_ignored_predictions}")
    print()
    print(f"   Precisión global           {_pct(m.precision)}{_delta(m.precision, b.precision if b else None)}")
    print(f"   Recall global              {_pct(m.recall)}{_delta(m.recall, b.recall if b else None)}")
    print(f"   F1 global                  {_pct(m.f1)}{_delta(m.f1, b.f1 if b else None)}")
    print(f"   Macro-F1                   {_pct(m.macro_f1)}{_delta(m.macro_f1, b.macro_f1 if b else None)}")
    print(f"   Accuracy s/ emparejadas    {_pct(m.matched_accuracy)}"
          f"{_delta(m.matched_accuracy, b.matched_accuracy if b else None)}   ({m.n_correct}/{m.n_matched})")
    print(f"   Acciones reales -> background  {_pct(m.background_rate)}")
    print()
    print("   Por clase:")
    print(f"     {'clase':<11}{'GT':>5}{'pred':>6}{'TP':>5}{'P':>9}{'R':>9}{'F1':>9}")
    for cls in ACTION_CLASSES:
        p, r, f1 = m.per_class(cls)
        print(f"     {cls:<11}{m.n_gt_by_class.get(cls, 0):>5}{m.n_pred_by_class.get(cls, 0):>6}{m.tp.get(cls, 0):>5}"
              f"{_pct(p):>9}{_pct(r):>9}{_pct(f1):>9}")
    print()
    print("   Matriz de confusión (fila = GT, sobre TODAS las ventanas, incl. background):")
    cols = list(ACTION_CLASSES) + [BACKGROUND_LABEL, MISSED_LABEL]
    print("     " + " " * 11 + "".join(f"{c:>13}" for c in cols))
    for row in ACTION_CLASSES:
        print(f"     {row:<11}" + "".join(f"{m.confusion.get((row, c), 0):>13}" for c in cols))
    print()
    print("   Techos del disparador (independientes del clasificador):")
    print(f"     recall máximo alcanzable     {_pct(m.trigger_recall)}   ({m.trigger_recall_n}/{m.n_gt} acciones con algún trigger cerca)")
    print(f"     precisión máxima alcanzable  {_pct(m.trigger_precision)}   ({m.trigger_recall_n}/{m.n_triggers} triggers sobre una acción real)")
    print()


def print_team(t: dict, b: Optional[dict]) -> None:
    print("══ EQUIPO DEL EJECUTOR (acciones emparejadas con equipo en el GT) ══")
    if not t["n_matched_with_side"]:
        print("   sin anotaciones de equipo en el GT\n")
        return
    if not t["n_with_cluster"]:
        print(f"   {t['n_matched_with_side']} acciones emparejadas, ninguna con clúster de equipo (¿falta teams.json?)\n")
        return
    print(f"   exactitud                  {_pct(t['accuracy'])}{_delta(t['accuracy'], b['accuracy'] if b else None)}"
          f"   ({t['n_correct']}/{t['n_with_cluster']}; asignación {t['mapping']})")
    print(f"   cobertura (con clúster)    {_pct(t['coverage'])}   ({t['n_with_cluster']}/{t['n_matched_with_side']})")
    for c, s in t["clusters"].items():
        print(f"   clúster {c}: {s['n']} acciones, pureza {_pct(s['purity'])} (home {s['home']}, away {s['away']})")
    print()


def print_possessor(p: Optional[dict], b: Optional[dict]) -> None:
    if p is None:
        return
    print("══ JUGADOR (tracker anotado a mano frente al poseedor del pipeline) ══")
    print(f"   mismo tracker              {_pct(p['accuracy'])}{_delta(p['accuracy'], b['accuracy'] if b else None)}"
          f"   ({p['same_tracker']}/{p['n']})")
    if p["n_with_cluster"]:
        print(f"   mismo clúster de equipo    {_pct(p['cluster_accuracy'])}   ({p['same_cluster']}/{p['n_with_cluster']})")
    print()


def print_events(ev: dict, tol_s: float, b: Optional[dict], list_errors: bool) -> None:
    print(f"══ EVENTOS (emparejado 1 a 1, |Δt| ≤ {tol_s:g} s; gol ⊂ tiro) ══")
    print(f"     {'evento':<13}{'GT':>5}{'pred':>6}{'TP':>5}{'P':>9}{'R':>9}{'F1':>9}{'|Δt| med':>10}{'equipo':>10}")
    for label, s in ev.items():
        dt = "-" if s["median_abs_dt_s"] is None else f"{s['median_abs_dt_s']:.1f} s"
        team = f"{s['team_correct']}/{s['team_n']}" if s["team_n"] else "-"
        f1_delta = _delta(s["f1"], b[label]["f1"]) if b and label in b else ""
        print(f"     {label:<13}{s['n_gt']:>5}{s['n_pred']:>6}{s['tp']:>5}{_pct(s['precision']):>9}{_pct(s['recall']):>9}"
              f"{_pct(s['f1']):>9}{dt:>10}{team:>10}{f1_delta}")
        if list_errors:
            if s["missed"]:
                print(f"        no detectados: {', '.join(_hms(t) for t in s['missed'])}")
            if s["false_alarms"]:
                print(f"        falsas alarmas: {', '.join(f'{_hms(t)} ({sc})' for t, sc in s['false_alarms'])}")
    print()


def print_goals(goals: list[dict], tol_s: float) -> None:
    print(f"══ GOLES (±{tol_s:g} s) ══")
    if not goals:
        print("   el GT no tiene goles\n")
        return
    seen_goal = sum(1 for g in goals if g["goal_pred"] is not None)
    seen_shot = sum(1 for g in goals if g["shot_pred"] is not None)
    print(f"   {len(goals)} goles: como 'goal' {seen_goal}/{len(goals)}, como tiro {seen_shot}/{len(goals)}")
    for g in goals:
        gp = "-" if g["goal_pred"] is None else f"{g['goal_pred']:.2f}"
        sp = "-" if g["shot_pred"] is None else f"{g['shot_pred']:.2f}"
        print(f"     {_hms(g['t_s'])} {g['team'] or '?':<5} goal {gp:>5}  shot {sp:>5}")
    print()


def actions_to_dict(m: ActionMetrics) -> dict:
    return {
        "n_gt": m.n_gt, "n_triggers": m.n_triggers, "n_predictions": m.n_predictions, "n_matched": m.n_matched,
        "n_correct": m.n_correct, "precision": m.precision, "recall": m.recall, "f1": m.f1, "macro_f1": m.macro_f1,
        "matched_accuracy": m.matched_accuracy, "background_rate": m.background_rate,
        "trigger_recall": m.trigger_recall, "trigger_precision": m.trigger_precision,
        "per_class": {cls: dict(zip(("precision", "recall", "f1"), m.per_class(cls))) for cls in ACTION_CLASSES},
        "confusion": {f"{g}->{p}": n for (g, p), n in sorted(m.confusion.items())},
    }


# --------------------------------------------------------------------------- #
#  Todo junto
# --------------------------------------------------------------------------- #

def evaluate(annotations: list[dict], telemetry: dict, fps: float, team_by_tracker: dict[int, int],
             tol_s: float, ev_tol_s: float, span: Optional[tuple[float, float]]) -> dict:
    telemetry = restrict_span(telemetry, fps, span, tol_s, ev_tol_s)
    tolerance = int(round(tol_s * fps))
    ball_actions = telemetry.get("ball_actions", [])
    actions = evaluate_ball_actions(annotations, ball_actions, tolerance)
    pairs = matched_pairs(annotations, ball_actions, tolerance)
    team = evaluate_team(pairs, team_by_tracker)
    possessor = evaluate_possessor(pairs, team_by_tracker)
    events = event_instants(telemetry)
    per_label = evaluate_events(annotations, events, ev_tol_s, team_by_tracker, team["mapping"])
    goals = evaluate_goals(annotations, events, ev_tol_s)
    return {"tolerance_frames": tolerance, "_actions": actions, "ball_actions": actions_to_dict(actions),
            "team": team, "possessor": possessor, "events": per_label, "goals": goals,
            "n_pred_actions": len(ball_actions), "n_pred_events": len(events)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_argument_group("verdad")
    g.add_argument("--gt", type=Path, help="ground_truth/*.gt.json")
    g.add_argument("--csv", type=Path, nargs="+", help="hoja(s) de anotación manual (tiempo,etiqueta,dorsal,equipo)")
    p = parser.add_argument_group("predicción")
    p.add_argument("--telemetry", type=Path)
    p.add_argument("--teams", type=Path, default=None, help="teams.json (por defecto, el de al lado de la telemetría)")
    p.add_argument("--backend-match", help="id de partido del backend: resuelve telemetría, equipos y fps")
    p.add_argument("--part", default="full")
    p.add_argument("--storage-dir", type=Path, default=Path("backend/storage"))
    parser.add_argument("--compare", type=Path, default=None, help="telemetry.json de referencia (deltas)")
    parser.add_argument("--fps", type=float, default=None, help="fps real del vídeo (si no, backend/GT)")
    parser.add_argument("--tolerance-s", type=float, default=DEFAULT_TOLERANCE_S, help="acciones (±s)")
    parser.add_argument("--event-tolerance-s", type=float, default=DEFAULT_EVENT_TOLERANCE_S, help="eventos (±s)")
    parser.add_argument("--span", type=float, nargs=2, metavar=("INICIO_S", "FIN_S"), default=None,
                        help="solo puntuar este tramo (por defecto: el anotado)")
    parser.add_argument("--list-errors", action="store_true", help="listar eventos no detectados y falsas alarmas")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if bool(args.gt) == bool(args.csv):
        parser.error("indica --gt o --csv (uno de los dos)")
    if bool(args.telemetry) == bool(args.backend_match):
        parser.error("indica --telemetry o --backend-match (uno de los dos)")

    # Predicción y fps
    fps = args.fps
    teams_path = args.teams
    name = None
    if args.backend_match:
        bp = backend_paths(args.backend_match, args.part, args.storage_dir)
        telemetry_path, name = bp["telemetry"], bp["name"]
        fps = fps or bp["fps"]
        teams_path = teams_path or bp["teams"]
    else:
        telemetry_path = args.telemetry
        teams_path = teams_path or telemetry_path.with_name("teams.json")
    telemetry = json.loads(telemetry_path.read_text())
    team_by_tracker = load_team_by_tracker(teams_path)

    # Verdad
    span = tuple(args.span) if args.span else None
    if args.gt:
        gt = json.loads(args.gt.read_text())
        annotations = gt["annotations"]
        fps = fps or float((gt.get("video") or {}).get("fps") or 0) or None
        # Si la telemetría se pasa del tramo del GT en más de un minuto (partido entero contra un GT de
        # 20 min) se recorta al tramo; si no, se puntúa todo (el recorte histórico de Middlesbrough dura
        # 1207 s y su GT llega a 1200: así se reproducen sus cifras de siempre).
        if span is None and gt.get("slice") and gt["slice"].get("end_s") is not None:
            n_frames = int(telemetry.get("n_frames") or 0)
            if fps and n_frames and n_frames / fps > gt["slice"]["end_s"] + 60.0:
                span = (float(gt["slice"].get("start_s") or 0.0), float(gt["slice"]["end_s"]))
        gt_label = f"{args.gt}  ({len(annotations)} anotaciones)"
    else:
        if fps is None:
            parser.error("con --csv hace falta --fps o --backend-match (fps real del vídeo)")
        rows = from_csv(args.csv)
        if not rows:
            raise SystemExit("La hoja no tiene filas válidas")
        lo, hi = min(r["t_s"] for r in rows), max(r["t_s"] for r in rows)
        if span is None:
            span = (lo, hi)
        annotations = build_annotations(rows, fps, span[0], span[1])
        gt_label = f"{', '.join(str(c) for c in args.csv)}  ({len(annotations)} filas, {_hms(lo)}-{_hms(hi)})"
    if fps is None:
        fps = 25.0
        logger.warning("fps desconocido: supongo 25")

    result = evaluate(annotations, telemetry, fps, team_by_tracker, args.tolerance_s, args.event_tolerance_s, span)
    base = None
    if args.compare:
        base = evaluate(annotations, json.loads(args.compare.read_text()), fps,
                        load_team_by_tracker(args.compare.with_name("teams.json")) or team_by_tracker,
                        args.tolerance_s, args.event_tolerance_s, span)

    print(f"\nGT:         {gt_label}")
    print(f"Telemetría: {telemetry_path}" + (f"  [{name}]" if name else ""))
    if args.compare:
        print(f"Referencia: {args.compare}  (deltas entre paréntesis)")
    print(f"fps {fps:g}; equipos: {teams_path if team_by_tracker else 'sin teams.json'}; "
          f"tramo: {'todo' if span is None else f'{_hms(span[0])}-{_hms(span[1])}'}; "
          f"predicciones: {result['n_pred_actions']} acciones, {result['n_pred_events']} eventos\n")
    print_actions(result["_actions"], result["tolerance_frames"], base["_actions"] if base else None)
    print_team(result["team"], base["team"] if base else None)
    print_possessor(result["possessor"], base["possessor"] if base else None)
    print_events(result["events"], args.event_tolerance_s, base["events"] if base else None, args.list_errors)
    print_goals(result["goals"], args.event_tolerance_s)

    if args.json_out:
        out = {k: v for k, v in result.items() if not k.startswith("_")}
        out.update(gt=str(args.gt or args.csv), telemetry=str(telemetry_path), fps=fps, span=span,
                   tolerance_s=args.tolerance_s, event_tolerance_s=args.event_tolerance_s,
                   compare=str(args.compare) if args.compare else None,
                   compare_result={k: v for k, v in base.items() if not k.startswith("_")} if base else None)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        logger.info("Métricas escritas en %s", args.json_out)


if __name__ == "__main__":
    main()
