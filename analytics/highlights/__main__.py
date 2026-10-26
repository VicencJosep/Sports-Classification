"""CLI: mejores jugadas de un partido ya analizado.

    # Partido analizado por el backend (rutas resueltas desde backend/storage):
    python -m analytics.highlights --backend-match 433f62c92b727870 --out output/highlights/mallorca

    # Rutas explícitas (lo que deja `python -m main --mode full-match`):
    python -m analytics.highlights --video partido.mp4 --pipeline-output output/mipartido --out output/mipartido/highlights

    --no-vlm: solo reglas (sin modelo).  --no-vjepa: sin V-JEPA2.  --no-cut: no escribir clips.  --force: rehacer señales y cachés.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from config import HighlightsConfig

from .pipeline import HighlightInputs, generate_highlights
from .vlm import hms


def _backend_inputs(match_id: str, part_id: str, storage_dir: Path) -> HighlightInputs:
    with open(storage_dir / "matches" / match_id / "match.json") as f:
        match = json.load(f)
    part = next(p for p in match["parts"] if p["id"] == part_id)
    video_dir = storage_dir / "videos" / part["video_id"]
    with open(video_dir / "video.json") as f:
        video = json.load(f)
    return HighlightInputs.from_pipeline_layout(
        video_dir / video["stored_filename"], storage_dir / "outputs" / match_id / part_id / "output"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_argument_group("entrada")
    src.add_argument("--backend-match", help="id de partido del backend (backend/storage)")
    src.add_argument("--part", default="full", help="parte del partido en el backend (full | first-half | second-half)")
    src.add_argument("--storage-dir", type=Path, default=Path("backend/storage"))
    src.add_argument("--video", type=Path)
    src.add_argument("--pipeline-output", type=Path, help="carpeta --output-dir del pipeline (telemetry.json, teams.json, tracks_2d.jsonl)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-vlm", action="store_true")
    parser.add_argument("--no-vjepa", action="store_true", help="sin el voto de V-JEPA2 para gol")
    parser.add_argument("--no-cut", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--vlm-url", default=None)
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--vlm-api", choices=["ollama", "openai"], default=None)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--min-clips", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.backend_match:
        inputs = _backend_inputs(args.backend_match, args.part, args.storage_dir)
    elif args.video and args.pipeline_output:
        inputs = HighlightInputs.from_pipeline_layout(args.video, args.pipeline_output)
    else:
        parser.error("indica --backend-match o bien --video + --pipeline-output")

    cfg = HighlightsConfig()
    overrides = {k: v for k, v in {
        "vlm_base_url": args.vlm_url, "vlm_model": args.vlm_model, "vlm_api": args.vlm_api,
        "max_clips": args.max_clips, "min_clips": args.min_clips,
    }.items() if v is not None}
    cfg = replace(cfg, **overrides)

    report = generate_highlights(inputs, args.out, cfg, use_vlm=not args.no_vlm, use_vjepa=not args.no_vjepa,
                                 cut=not args.no_cut, force=args.force)
    print(f"\n{len(report['highlights'])} highlights (de {report['n_candidates']} candidatos; VLM: {report['vlm']['used']})")
    for h in report["highlights"]:
        print(f"  {h['rank']:2d}. {hms(h['start_s'])} - {hms(h['end_s'])} ({h['duration_s']:.0f} s) "
              f"[{h['label']}] {h['title']}  final={h['final_score']:.2f} reglas={h['rule_score']:.2f}")


if __name__ == "__main__":
    main()
