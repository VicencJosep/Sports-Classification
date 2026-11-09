"""Convierte los clips de ``highlights.json`` (segundos del vídeo del backend) al formato de predicciones de
``scripts/evaluate_soccerhigh.py`` (segundos del vídeo HQ de SoccerNet, por mitad), para medir la selección actual
contra la verdad de SoccerHigh. También acepta ``soccerhigh.json`` (tramos del modelo sobre el mismo vídeo).

El mapeo por mitad se da en un JSON ``{"1": {"video_start_s": t0, "hq_start_s": h0}, "2": {...}}``: el instante
``t`` del vídeo cae en la mitad cuyo ``video_start_s`` es el mayor <= t, y ``t_HQ = t - video_start_s + hq_start_s``.
``scripts/soccerhigh_align.py --game`` deja esos valores en ``offsets.json`` para los 720p (``video_start_s`` = 0 y
``hq_start_s`` = offset) y ``--ours/--theirs`` los da para cualquier otro vídeo.

    python scripts/soccerhigh_predictions_from_highlights.py --highlights backend/storage/outputs/<m>/full/highlights/highlights.json \
        --mapping output/soccerhigh/clasico_backend/mapping.json --game "<liga/temporada/partido>" --out preds.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def to_hq(t: float, mapping: dict) -> tuple[str, float] | None:
    halves = sorted(((str(h), m) for h, m in mapping.items()), key=lambda kv: kv[1]["video_start_s"])
    chosen = None
    for h, m in halves:
        if t >= m["video_start_s"]:
            chosen = (h, m)
    if chosen is None:
        return None
    h, m = chosen
    return h, t - m["video_start_s"] + m["hq_start_s"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--highlights", type=Path, help="highlights.json del pipeline")
    p.add_argument("--soccerhigh", type=Path, help="soccerhigh.json (tramos del modelo)")
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--game", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--all-candidates", action="store_true", help="con --highlights: todos los candidatos, no solo los clips")
    args = p.parse_args()
    mapping = json.loads(args.mapping.read_text())
    if args.highlights:
        rep = json.loads(args.highlights.read_text())
        items = rep["candidates"] if args.all_candidates else rep["highlights"]
        rows = [(it["start_s"], it["end_s"], float(it.get("final_score", it.get("rule_score", 1.0)) or 0.0)) for it in items]
    else:
        rep = json.loads(args.soccerhigh.read_text())
        rows = [(sg["start_s"], sg["end_s"], sg["score"]) for sg in rep["segments"]]
    out = {"1": [], "2": []}
    for s, e, sc in rows:
        a, b = to_hq(s, mapping), to_hq(e, mapping)
        if a is None or b is None or a[0] != b[0]:
            continue
        out[a[0]].append([a[1], b[1], sc])
    args.out.write_text(json.dumps({args.game: out}, indent=1))
    print(f"{sum(len(v) for v in out.values())} tramos -> {args.out}")


if __name__ == "__main__":
    main()
