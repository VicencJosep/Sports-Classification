"""Evalúa los highlights contra los goles de referencia (ground_truth/highlights_goals.json).

    # Lo ya generado (highlights.json de cada partido):
    python -m analytics.highlights.evaluate

    # Probar una variante sin cortar vídeo, reutilizando señales y caché del VLM
    # (solo se preguntan al VLM los candidatos nuevos):
    python -m analytics.highlights.evaluate --rerun --set vlm_max_candidates=30
    python -m analytics.highlights.evaluate --rerun --no-vlm            # solo reglas

Por gol: ¿está entre los candidatos?, ¿en qué puesto de reglas?, ¿lo vio el VLM?,
¿lo cubre un clip? Por partido: recall de goles, precisión de clips y cuántos
clips describen un gol que no existe.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import fields, replace
from pathlib import Path
from typing import Optional

from config import HighlightsConfig

from .pipeline import HIGHLIGHTS_JSON, claims_goal, generate_highlights
from .vlm import hms

GT_PATH = Path("ground_truth/highlights_goals.json")


def _covers(start_s: float, end_s: float, t_s: float, margin_s: float = 1.0) -> bool:
    return start_s - margin_s <= t_s <= end_s + margin_s


def evaluate_match(report: dict, goals: list[dict], tolerance_s: float, judged_ids: set[str]) -> dict:
    cands = report["candidates"]                       # ya ordenados por puntuación de reglas
    clips = report["highlights"]
    per_goal = []
    for g in goals:
        t = g["t_s"]
        hit = next(((rank, c) for rank, c in enumerate(cands, 1)
                    if abs(c["anchor_s"] - t) <= tolerance_s or _covers(c["start_s"], c["end_s"], t)), None)
        clip = next((c for c in clips if _covers(c["start_s"], c["end_s"], t)), None)
        per_goal.append({
            "note": g.get("note", ""), "t_s": t,
            "candidate": hit[1]["id"] if hit else None,
            "rule_rank": hit[0] if hit else None,
            "judged": bool(hit and hit[1]["id"] in judged_ids),
            "vlm_rank": hit[1].get("vlm_rank") if hit else None,
            "clip": clip["id"] if clip else None,
            "clip_label": clip["label"] if clip else None,
            "clip_goal_source": clip.get("goal_source") if clip else None,
        })
    clip_rows = []
    for c in clips:
        inside = [g for g in goals if _covers(c["start_s"], c["end_s"], g["t_s"])]
        says_goal = claims_goal(c.get("description") or "")
        clip_rows.append({
            "id": c["id"], "start_s": c["start_s"], "end_s": c["end_s"], "label": c["label"],
            "goal_source": c.get("goal_source"), "goal_fusion": c.get("goal_fusion"),
            "title": c["title"], "has_goal": bool(inside), "says_goal": says_goal,
            "hallucinated_goal": says_goal and not inside,
        })
    n = len(goals)
    return {
        "n_goals": n, "n_clips": len(clips), "n_candidates": len(cands),
        "cand_recall": sum(1 for p in per_goal if p["candidate"]) / n if n else None,
        "judged_recall": sum(1 for p in per_goal if p["judged"]) / n if n else None,
        "clip_recall": sum(1 for p in per_goal if p["clip"]) / n if n else None,
        "clip_precision": (sum(1 for r in clip_rows if r["has_goal"]) / len(clips)) if clips and n else None,
        "goal_labelled": sum(1 for p in per_goal if p["clip_label"] == "goal"),
        "false_goal_labels": sum(1 for r in clip_rows if r["label"] == "goal" and not r["has_goal"]),
        "hallucinated_goals": sum(1 for r in clip_rows if r["hallucinated_goal"]),
        "per_goal": per_goal, "clips": clip_rows,
    }


def _judged_ids(out_dir: Path) -> set[str]:
    path = out_dir / "vlm_cache.json"
    if not path.exists():
        return set()
    with open(path) as f:
        cache = json.load(f)
    return {k.split("@")[0] for k, v in cache.items() if not k.startswith("_") and "judge" in v}


def _typed_overrides(pairs: list[str]) -> dict:
    types = {f.name: f.type for f in fields(HighlightsConfig)}
    out = {}
    for p in pairs:
        k, v = p.split("=", 1)
        if k not in types:
            raise SystemExit(f"campo desconocido de HighlightsConfig: {k}")
        t = str(types[k])
        out[k] = (v.lower() in ("1", "true", "yes")) if "bool" in t else int(v) if "int" in t else float(v) if "float" in t else v
    return out


def _fmt(x: Optional[float]) -> str:
    return "  - " if x is None else f"{100 * x:3.0f}%"


def print_match(name: str, m: dict) -> None:
    print(f"\n=== {name}: {m['n_goals']} goles, {m['n_clips']} clips, {m['n_candidates']} candidatos")
    print(f"    recall candidatos {_fmt(m['cand_recall'])} | vistos por el VLM {_fmt(m['judged_recall'])} | "
          f"cubiertos por clip {_fmt(m['clip_recall'])} | precisión de clips {_fmt(m['clip_precision'])}")
    print(f"    etiquetados 'gol' bien {m['goal_labelled']}/{m['n_goals']} | etiqueta gol falsa {m['false_goal_labels']} | "
          f"descripción de gol inexistente {m['hallucinated_goals']}")
    for p in m["per_goal"]:
        cand = f"{p['candidate']} (puesto reglas {p['rule_rank']}, VLM {'sí' if p['judged'] else 'NO'})" if p["candidate"] else "SIN candidato"
        clip = f"{p['clip']} [{p['clip_label']}/{p['clip_goal_source']}]" if p["clip"] else "SIN clip"
        print(f"      {hms(p['t_s'])} {p['note']:<18} {cand:<44} {clip}")
    for r in m["clips"]:
        mark = "gol" if r["has_goal"] else ("FALSO" if r["says_goal"] else "-")
        fus = f" fusión={r['goal_fusion']:.2f}" if r.get("goal_fusion") is not None else ""
        print(f"      clip {r['id']} {hms(r['start_s'])}-{hms(r['end_s'])} [{r['label']}/{r.get('goal_source')}]{fus} {mark:<5} {r['title']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", type=Path, default=GT_PATH)
    parser.add_argument("--match", action="append", help="solo estos partidos (clave del GT)")
    parser.add_argument("--rerun", action="store_true", help="regenerar (sin cortar) con --set / --no-vlm y evaluar eso")
    parser.add_argument("--no-vlm", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="CAMPO=VALOR", help="override de HighlightsConfig")
    parser.add_argument("--storage-dir", type=Path, default=Path("backend/storage"))
    parser.add_argument("--json", type=Path, help="guardar las métricas")
    args = parser.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)
    tol = float(gt.get("tolerance_s", 15.0))
    cfg = replace(HighlightsConfig(), **_typed_overrides(args.set))
    results = {}
    for key, spec in gt["matches"].items():
        if args.match and key not in args.match:
            continue
        src = Path(spec["dir"])
        if args.rerun:
            from .__main__ import _backend_inputs

            work = Path(tempfile.mkdtemp(prefix=f"hl_eval_{key}_"))
            for name in ("signals.npz", "vlm_cache.json", "scoreboard.json", "vjepa_cache.json"):
                if (src / name).exists():
                    shutil.copy2(src / name, work / name)
            inputs = _backend_inputs(spec["backend_match"], spec.get("part", "full"), args.storage_dir)
            report = generate_highlights(inputs, work, cfg, use_vlm=not args.no_vlm, cut=False)
            judged = _judged_ids(work)
            for name in ("vlm_cache.json", "scoreboard.json", "vjepa_cache.json"):      # las cachés de los modelos se conservan
                if (work / name).exists():
                    shutil.copy2(work / name, src / name)
            shutil.rmtree(work, ignore_errors=True)
        else:
            with open(src / HIGHLIGHTS_JSON) as f:
                report = json.load(f)
            judged = _judged_ids(src)
        results[key] = evaluate_match(report, spec["goals"], tol, judged)
        print_match(f"{key} - {spec['name']}", results[key])

    tot = [m for m in results.values() if m["n_goals"]]
    if tot:
        n = sum(m["n_goals"] for m in tot)
        print("\n=== TOTAL (partidos con goles): "
              f"recall de goles en clips {sum(round(m['clip_recall'] * m['n_goals']) for m in tot)}/{n}, "
              f"candidatos {sum(round(m['cand_recall'] * m['n_goals']) for m in tot)}/{n}, "
              f"clips con gol {sum(round(m['clip_precision'] * m['n_clips']) for m in tot if m['clip_precision'] is not None)}"
              f"/{sum(m['n_clips'] for m in tot)}")
    if sum(m["hallucinated_goals"] for m in results.values()) or any(m["n_goals"] == 0 for m in results.values()):
        print("    descripciones de gol sin gol (todos los partidos): "
              f"{sum(m['hallucinated_goals'] for m in results.values())}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
