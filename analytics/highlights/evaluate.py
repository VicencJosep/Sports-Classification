"""Evalúa los highlights contra la referencia (ground_truth/highlights_goals.json): goles y duración.

    # Lo ya generado (highlights.json de cada partido, lo que ve el frontend):
    python -m analytics.highlights.evaluate

    # Probar una variante sin cortar vídeo, reutilizando señales y cachés del VLM, V-JEPA2 y marcador.
    # --vlm-cache-only no pregunta nada al modelo (sin Ollama ni GPU): sirve para medir cambios de la
    # política de corte o de selección con los veredictos ya guardados.
    python -m analytics.highlights.evaluate --rerun --vlm-cache-only
    python -m analytics.highlights.evaluate --rerun --set clip_max_goal_s=50 --vlm-cache-only
    python -m analytics.highlights.evaluate --rerun --no-vlm            # solo reglas

Por gol: ¿está entre los candidatos?, ¿en qué puesto de reglas?, ¿lo vio el VLM?, ¿lo cubre un clip?, cuánto
clip hay antes y después del gol. Por clip: duración y si está en el rango de la política (12-30 s; goles
hasta 45). Por partido: recall de goles, precisión de clips, clips que describen un gol que no existe, clips
en rango. ``ref_clips`` del GT (cortes de referencia dados a mano, p.ej. el gol de Messi 1:45:56-1:46:46):
diferencia de inicio y fin del clip que los cubre.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
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


def in_range(duration_s: float, is_goal: bool, cfg: HighlightsConfig) -> bool:
    hi = cfg.clip_max_goal_s if is_goal else cfg.clip_max_s
    return cfg.clip_min_s - 0.5 <= duration_s <= hi + 0.5


def evaluate_match(report: dict, goals: list[dict], tolerance_s: float, judged_ids: set[str],
                   cfg: HighlightsConfig = HighlightsConfig(), ref_clips: Optional[list[dict]] = None) -> dict:
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
            "anchor_delta_s": round(clip["anchor_s"] - t, 1) if clip else None,
            "start_before_goal_s": round(t - clip["start_s"], 1) if clip else None,
            "end_after_goal_s": round(clip["end_s"] - t, 1) if clip else None,
        })
    clip_rows = []
    for c in clips:
        inside = [g for g in goals if _covers(c["start_s"], c["end_s"], g["t_s"])]
        says_goal = claims_goal(c.get("description") or "")
        clip_rows.append({
            "id": c["id"], "start_s": c["start_s"], "end_s": c["end_s"], "label": c["label"],
            "duration_s": round(c["end_s"] - c["start_s"], 1), "in_range": in_range(c["end_s"] - c["start_s"], c["label"] == "goal", cfg),
            "goal_source": c.get("goal_source"), "goal_fusion": c.get("goal_fusion"),
            "title": c["title"], "has_goal": bool(inside), "says_goal": says_goal,
            "hallucinated_goal": says_goal and not inside,
        })
    refs = []
    for r in ref_clips or []:
        t = r.get("t_s", (r["start_s"] + r["end_s"]) / 2)
        clip = next((c for c in clips if _covers(c["start_s"], c["end_s"], t)), None)
        inter = max(0.0, min(clip["end_s"], r["end_s"]) - max(clip["start_s"], r["start_s"])) if clip else 0.0
        union = (max(clip["end_s"], r["end_s"]) - min(clip["start_s"], r["start_s"])) if clip else 0.0
        refs.append({
            "note": r.get("note", ""), "ref": [r["start_s"], r["end_s"]], "clip": clip["id"] if clip else None,
            "delta_start_s": round(clip["start_s"] - r["start_s"], 1) if clip else None,
            "delta_end_s": round(clip["end_s"] - r["end_s"], 1) if clip else None,
            "iou": round(inter / union, 2) if clip and union else None,
        })
    n = len(goals)
    durations = [r["duration_s"] for r in clip_rows]
    return {
        "n_goals": n, "n_clips": len(clips), "n_candidates": len(cands),
        "cand_recall": sum(1 for p in per_goal if p["candidate"]) / n if n else None,
        "judged_recall": sum(1 for p in per_goal if p["judged"]) / n if n else None,
        "clip_recall": sum(1 for p in per_goal if p["clip"]) / n if n else None,
        "clip_precision": (sum(1 for r in clip_rows if r["has_goal"]) / len(clips)) if clips and n else None,
        "goal_labelled": sum(1 for p in per_goal if p["clip_label"] == "goal"),
        "false_goal_labels": sum(1 for r in clip_rows if r["label"] == "goal" and not r["has_goal"]),
        "hallucinated_goals": sum(1 for r in clip_rows if r["hallucinated_goal"]),
        "n_in_range": sum(1 for r in clip_rows if r["in_range"]),
        "median_duration_s": statistics.median(durations) if durations else None,
        "min_end_after_goal_s": min((p["end_after_goal_s"] for p in per_goal if p["end_after_goal_s"] is not None), default=None),
        "per_goal": per_goal, "clips": clip_rows, "ref_clips": refs,
        "vlm": report.get("vlm"),
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
    vlm = m.get("vlm") or {}
    degraded = f"; VLM degradado ({vlm.get('n_failed_calls')} llamadas sin respuesta)" if vlm.get("degraded") else ""
    print(f"\n=== {name}: {m['n_goals']} goles, {m['n_clips']} clips, {m['n_candidates']} candidatos{degraded}")
    print(f"    recall candidatos {_fmt(m['cand_recall'])} | vistos por el VLM {_fmt(m['judged_recall'])} | "
          f"cubiertos por clip {_fmt(m['clip_recall'])} | precisión de clips {_fmt(m['clip_precision'])}")
    print(f"    etiquetados 'gol' bien {m['goal_labelled']}/{m['n_goals']} | etiqueta gol falsa {m['false_goal_labels']} | "
          f"descripción de gol inexistente {m['hallucinated_goals']} | en rango {m['n_in_range']}/{m['n_clips']} | "
          f"duración mediana {m['median_duration_s'] if m['median_duration_s'] is not None else '-'} s")
    for p in m["per_goal"]:
        cand = f"{p['candidate']} (puesto reglas {p['rule_rank']}, VLM {'sí' if p['judged'] else 'NO'})" if p["candidate"] else "SIN candidato"
        if p["clip"]:
            clip = (f"{p['clip']} [{p['clip_label']}/{p['clip_goal_source']}] ancla {p['anchor_delta_s']:+.0f} s, "
                    f"{p['start_before_goal_s']:.0f} s antes / {p['end_after_goal_s']:.0f} s después")
        else:
            clip = "SIN clip"
        print(f"      {hms(p['t_s'])} {p['note']:<18} {cand:<44} {clip}")
    for r in m["clips"]:
        mark = "gol" if r["has_goal"] else ("FALSO" if r["says_goal"] else "-")
        fus = f" fusión={r['goal_fusion']:.2f}" if r.get("goal_fusion") is not None else ""
        rng = "" if r["in_range"] else " FUERA DE RANGO"
        print(f"      clip {r['id']} {hms(r['start_s'])}-{hms(r['end_s'])} {r['duration_s']:5.1f} s [{r['label']}/{r.get('goal_source')}]{fus} {mark:<5} {r['title']}{rng}")
    for r in m["ref_clips"]:
        ref = f"{hms(r['ref'][0])}-{hms(r['ref'][1])} ({r['ref'][1] - r['ref'][0]:.0f} s)"
        if r["clip"]:
            print(f"      referencia {r['note']}: {ref} -> {r['clip']} inicio {r['delta_start_s']:+.1f} s, fin {r['delta_end_s']:+.1f} s, IoU {r['iou']}")
        else:
            print(f"      referencia {r['note']}: {ref} -> SIN clip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", type=Path, default=GT_PATH)
    parser.add_argument("--match", action="append", help="solo estos partidos (clave del GT)")
    parser.add_argument("--rerun", action="store_true", help="regenerar (sin cortar) con --set / --no-vlm y evaluar eso")
    parser.add_argument("--no-vlm", action="store_true")
    parser.add_argument("--no-vjepa", action="store_true", help="con --rerun: sin el voto de V-JEPA2")
    parser.add_argument("--vlm-cache-only", action="store_true",
                        help="con --rerun: solo veredictos ya cacheados (sin modelo ni GPU); lo no cacheado se omite")
    parser.add_argument("--set", action="append", default=[], metavar="CAMPO=VALOR", help="override de HighlightsConfig")
    parser.add_argument("--storage-dir", type=Path, default=Path("backend/storage"))
    parser.add_argument("--json", type=Path, help="guardar las métricas")
    args = parser.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)
    tol = float(gt.get("tolerance_s", 15.0))
    cfg = replace(HighlightsConfig(), **_typed_overrides(args.set))
    if args.vlm_cache_only:
        cfg = replace(cfg, vlm_api="cache")
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
            report = generate_highlights(inputs, work, cfg, use_vlm=not args.no_vlm, use_vjepa=not args.no_vjepa, cut=False)
            judged = _judged_ids(work)
            for name in ("vlm_cache.json", "scoreboard.json", "vjepa_cache.json"):      # las cachés de los modelos se conservan
                if (work / name).exists() and not args.vlm_cache_only:
                    shutil.copy2(work / name, src / name)
            shutil.rmtree(work, ignore_errors=True)
        else:
            with open(src / HIGHLIGHTS_JSON) as f:
                report = json.load(f)
            judged = _judged_ids(src)
        results[key] = evaluate_match(report, spec["goals"], tol, judged, cfg, spec.get("ref_clips"))
        print_match(f"{key} - {spec['name']}", results[key])

    tot = [m for m in results.values() if m["n_goals"]]
    if tot:
        n = sum(m["n_goals"] for m in tot)
        print("\n=== TOTAL (partidos con goles): "
              f"recall de goles en clips {sum(round(m['clip_recall'] * m['n_goals']) for m in tot)}/{n}, "
              f"candidatos {sum(round(m['cand_recall'] * m['n_goals']) for m in tot)}/{n}, "
              f"clips con gol {sum(round(m['clip_precision'] * m['n_clips']) for m in tot if m['clip_precision'] is not None)}"
              f"/{sum(m['n_clips'] for m in tot)}, etiquetados gol {sum(m['goal_labelled'] for m in tot)}/{n}, "
              f"etiquetas de gol falsas {sum(m['false_goal_labels'] for m in tot)}")
    if results:
        n_clips = sum(m["n_clips"] for m in results.values())
        durations = [r["duration_s"] for m in results.values() for r in m["clips"]]
        print(f"=== TODOS: clips en rango {sum(m['n_in_range'] for m in results.values())}/{n_clips}; "
              f"duración mediana {statistics.median(durations) if durations else '-'} s "
              f"(mín {min(durations) if durations else '-'}, máx {max(durations) if durations else '-'}); "
              f"descripciones de gol sin gol {sum(m['hallucinated_goals'] for m in results.values())}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
