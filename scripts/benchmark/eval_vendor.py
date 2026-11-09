"""Evalúa los eventos que devuelve un servicio comercial (Scoutiqo u otro) sobre
un clip de prueba contra la verdad de terreno del clip, y nuestro T-DEED sobre
los mismos minutos con la misma regla, para que la comparación sea directa.

Clip y GT: ``output/benchmark/vendor/<clip>.mp4`` + ``<clip>.gt.json`` (de
SoccerTrack v2, CC BY 4.0; ver docs/comparativa_benchmark.md §7.9).

Salida del servicio: CSV transcrito a mano desde su línea de tiempo, formato
``tiempo,etiqueta[,equipo]`` con tiempo ``mm:ss`` o segundos relativos al clip
(mismo formato que ``scripts/annotation_csv.py``). Las etiquetas del servicio
se agrupan con las 12 clases de BAS en ``VENDOR_GROUPS``; cada grupo se evalúa
como una clase (un «shot» del servicio acierta si hay SHOT o GOAL del GT a
±tolerancia). Emparejamiento 1 a 1 greedy, P/R/F1 por grupo, a ±1 s y ±3 s.

``--ours <modelo>`` añade nuestro T-DEED (caché ``views/<m>/<parte>_<vista>
.tdeed.<modelo>.npz``, picos con ``core.spotting.find_peaks``, umbral 0,3 y
NMS 1 s) sobre la misma ventana y los mismos grupos. ``--ours-npz <ruta>`` usa
en su lugar la caché de T-DEED del propio clip (``<clip>.tdeed.<modelo>
.npz`` de ``TDEEDSpotter.ensure_scores``), con tiempos ya relativos al clip.

Clips propios (cámara que sigue el balón): el GT se construye desde la hoja de
anotación de ``scripts/annotation_csv.py`` con tiempos del CLIP
(``--gt-csv``): acciones de balón -> clase BAS; córner y penalti entran en
*set_piece*; falta, tarjeta, saque inicial... se guardan pero no puntúan.

Uso (contenedor):
    python -m scripts.benchmark.eval_vendor --clip output/benchmark/vendor/st2_128057_2nd_35m_10min \\
        --vendor-csv output/benchmark/vendor/scoutiqo_timeline.csv --ours SNB_pano1
    python -m scripts.benchmark.eval_vendor --clip output/benchmark/vendor/mallorca_barcelona_j28_72m_10min \\
        --gt-csv output/benchmark/vendor/mallorca_barcelona_j28_72m_10min.annotation.csv \\
        --vendor-csv output/benchmark/vendor/scoutiqo_mallorca_timeline.csv --vendor-name scoutiqo \\
        --ours-npz output/benchmark/vendor/mallorca_barcelona_j28_72m_10min.tdeed.SoccerNetBall_challenge2.npz
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Grupo del servicio -> clases de BAS que cuentan como acierto.
VENDOR_GROUPS: dict[str, tuple[str, ...]] = {
    "shot": ("SHOT", "GOAL"),
    "pass": ("PASS", "HIGH PASS", "CROSS"),
    "set_piece": ("FREE KICK", "THROW IN", "CORNER", "PENALTY"),  # CORNER/PENALTY solo en la hoja propia
    "turnover": ("PLAYER SUCCESSFUL TACKLE", "BALL PLAYER BLOCK", "OUT"),
}
# Sinónimos que puede usar la línea de tiempo del servicio.
ALIASES = {"shots": "shot", "tiro": "shot", "goal": "shot", "gol": "shot", "passes": "pass", "pase": "pass",
           "set pieces": "set_piece", "set piece": "set_piece", "setpiece": "set_piece", "free kick": "set_piece",
           "throw-in": "set_piece", "throw in": "set_piece", "corner": "set_piece", "turnovers": "turnover",
           "pérdida": "turnover", "perdida": "turnover", "recuperación": "turnover"}
# Hoja de anotación propia (scripts/annotation_csv.py) -> clase del GT.
ACTION_TO_GT = {"pass": "PASS", "drive": "DRIVE", "high_pass": "HIGH PASS", "header": "HEADER", "throw_in": "THROW IN",
                "cross": "CROSS", "shot": "SHOT", "goal": "GOAL", "out": "OUT", "free_kick": "FREE KICK",
                "player_successful_tackle": "PLAYER SUCCESSFUL TACKLE", "ball_player_block": "BALL PLAYER BLOCK"}
EVENT_TO_GT = {"corner": "CORNER", "penalty": "PENALTY", "foul": "FOUL", "kick-off": "KICK OFF", "card": "CARD",
               "offside": "OFFSIDE", "substitution": "SUBSTITUTION"}


def gt_from_st2(match: str, half: str, start_s: float, duration_s: float, view: str) -> dict:
    """GT del clip desde las 12 clases de SoccerTrack v2 (tiempos relativos al clip)."""
    src = ROOT / "datasets" / "soccertrack-v2" / "bas" / match / f"{match}_12_class_events.json"
    tag = {"1st": "1", "2nd": "2"}[half]
    events = [{"t_s": round(int(a["position"]) / 1000 - start_s, 3), "label": a["label"], "team": a.get("team"),
               "player_id": a.get("player_id")} for a in json.loads(src.read_text())["actions"]
              if a["gameTime"].startswith(tag + " -") and start_s <= int(a["position"]) / 1000 < start_s + duration_s]
    return {"match": match, "half": half, "start_s": start_s, "duration_s": duration_s, "view": view,
            "source": f"SoccerTrack v2 (CC BY 4.0), {match}_panorama_{half}_half.mp4 -> {view}; {src.name}", "events": events}


def gt_from_annotation_csv(path: Path, clip_name: str, duration_s: float) -> dict:
    """GT del clip desde la hoja de anotación propia; los tiempos de la hoja son los del CLIP."""
    from scripts.annotation_csv import read_annotation_csv

    rows, warnings = read_annotation_csv(path)
    for w in warnings:
        print(f"[aviso] {w}")
    events = []
    for r in rows:
        label = ACTION_TO_GT.get(r["ball_action_label"]) or EVENT_TO_GT.get(r["event_label"])
        if label is None:
            print(f"[aviso] línea {r['line']}: etiqueta {r['label_raw']!r} sin clase, se ignora")
            continue
        if r["t_s"] >= duration_s:
            print(f"[aviso] línea {r['line']}: {r['t_s']} s fuera del clip de {duration_s:g} s, se ignora")
            continue
        events.append({"t_s": r["t_s"], "label": label, "team": r["team"], "player_id": r["tracker_id"]})
    return {"match": clip_name, "half": None, "start_s": 0.0, "duration_s": duration_s,
            "source": f"hoja de anotación propia {path.name} (tiempos del clip)", "events": events}


def parse_time(s: str) -> float:
    s = s.strip()
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        return parts[-1] + 60 * parts[-2] + (3600 * parts[-3] if len(parts) == 3 else 0)
    return float(s)


def read_vendor_csv(path: Path) -> list[tuple[float, str]]:
    out = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or row[0].lower() in ("tiempo", "time"):
                continue
            label = row[1].strip().lower()
            label = ALIASES.get(label, label)
            if label not in VENDOR_GROUPS:
                print(f"[aviso] etiqueta sin grupo, se ignora: {row[1]!r}")
                continue
            out.append((parse_time(row[0]), label))
    return out


def ours_events(clip_meta: dict, model: str, view: str, min_score: float, nms_s: float,
                npz_path: Path | None = None) -> list[tuple[float, str]]:
    from core.spotting import find_peaks

    if npz_path:  # caché del propio clip: tiempos ya relativos al clip
        z, start = np.load(npz_path), 0.0
    else:
        stem = ROOT / "datasets" / "soccertrack-v2" / "views" / clip_meta["match"] / f"{clip_meta['half']}_{view}"
        z, start = np.load(stem.with_name(f"{stem.name}.tdeed.{model}.npz")), clip_meta["start_s"]
    probs, classes, stride = z["probs"].astype(np.float32), [str(c) for c in z["classes"]], int(z["stride"])
    fps_rows = (float(z["fps"]) if "fps" in z.files else 25.0) / stride
    t = np.arange(len(probs)) / fps_rows
    dur = clip_meta["duration_s"]
    out = []
    for group, labels in VENDOR_GROUPS.items():
        curve = np.sum([probs[:, classes.index(c)] for c in labels if c in classes], axis=0)
        for frame_t, score in find_peaks(t.tolist(), curve.tolist(), min_score, nms_s):
            if start <= frame_t < start + dur:
                out.append((frame_t - start, group))
    return out


def prf(gt_times: list[float], pred_times: list[float], tol: float) -> tuple[int, int, int]:
    gt = np.array(sorted(gt_times))
    used = np.zeros(len(gt), bool)
    tp = 0
    for p in sorted(pred_times):
        if not len(gt):
            break
        d = np.abs(gt - p)
        d[used] = np.inf
        i = int(np.argmin(d))
        if d[i] <= tol:
            used[i] = True
            tp += 1
    return tp, len(pred_times) - tp, len(gt) - tp


def report(name: str, events: list[tuple[float, str]], gt: list[dict], tols=(1.0, 3.0)) -> dict:
    res = {}
    print(f"\n== {name}: {len(events)} eventos")
    print(f"{'grupo':<10}{'n GT':>6}{'n pred':>8}" + "".join(f"{'P':>8}{'R':>7}{'F1':>7} (±{t:g}s)" for t in tols))
    for group, labels in VENDOR_GROUPS.items():
        g = [e["t_s"] for e in gt if e["label"] in labels]
        p = [t for t, l in events if l == group]
        cells, res[group] = "", {"n_gt": len(g), "n_pred": len(p)}
        if not g:
            res[group]["note"] = "sin GT"
            print(f"{group:<10}{0:>6}{len(p):>8}   (sin GT en la hoja: no puntúa)")
            continue
        for tol in tols:
            tp, fp, fn = prf(g, p, tol)
            P = tp / (tp + fp) if tp + fp else 0.0
            R = tp / (tp + fn) if tp + fn else 0.0
            F = 2 * P * R / (P + R) if P + R else 0.0
            res[group][f"tol{tol:g}"] = {"P": round(100 * P, 1), "R": round(100 * R, 1), "F1": round(100 * F, 1)}
            cells += f"{100 * P:8.1f}{100 * R:7.1f}{100 * F:7.1f}       "
        print(f"{group:<10}{len(g):>6}{len(p):>8}{cells}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, required=True, help="Ruta sin extensión del clip (lee <clip>.gt.json)")
    ap.add_argument("--vendor-csv", type=Path, default=None)
    ap.add_argument("--vendor-name", default="servicio")
    ap.add_argument("--ours", default=None, help="Modelo T-DEED de la caché de la cámara virtual")
    ap.add_argument("--ours-npz", type=Path, default=None, help="Caché T-DEED del propio clip (tiempos del clip)")
    ap.add_argument("--gt-csv", type=Path, default=None, help="Hoja de anotación propia del clip; escribe <clip>.gt.json")
    ap.add_argument("--gt-st2", nargs=3, metavar=("MATCH", "HALF", "START_S"), default=None,
                    help="GT desde SoccerTrack v2 para un clip cortado en START_S de esa parte; escribe <clip>.gt.json")
    ap.add_argument("--clip-duration", type=float, default=600.0, help="Duración del clip para --gt-csv")
    ap.add_argument("--view", default="vcam796vl1")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--nms-s", type=float, default=1.0)
    args = ap.parse_args()

    if args.gt_csv:
        args.clip.with_suffix(".gt.json").write_text(
            json.dumps(gt_from_annotation_csv(args.gt_csv, args.clip.name, args.clip_duration), indent=1))
    if args.gt_st2:
        m, h, t0 = args.gt_st2
        args.clip.with_suffix(".gt.json").write_text(
            json.dumps(gt_from_st2(m, h, float(t0), args.clip_duration, args.view), indent=1))
    meta = json.loads(args.clip.with_suffix(".gt.json").read_text())
    gt = meta["events"]
    counts = sorted(((sum(e["label"] == c for e in gt), c) for c in {e["label"] for e in gt}), reverse=True)
    print(f"Clip {args.clip.name}: {meta['match']} {meta['half']} desde {meta['start_s']} s, {len(gt)} eventos GT: "
          + ", ".join(f"{c} {n}" for n, c in counts))
    results = {}
    if args.vendor_csv:
        results[args.vendor_name] = report(args.vendor_name, read_vendor_csv(args.vendor_csv), gt)
    if args.ours or args.ours_npz:
        model = args.ours or args.ours_npz.name.split(".tdeed.")[-1].removesuffix(".npz")
        results[f"T-DEED {model}"] = report(f"T-DEED {model} (umbral {args.min_score}, NMS {args.nms_s} s)",
                                            ours_events(meta, model, args.view, args.min_score, args.nms_s, args.ours_npz), gt)
    args.clip.with_suffix(".eval.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
