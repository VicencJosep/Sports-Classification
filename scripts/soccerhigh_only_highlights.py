"""Highlights de un partido del backend solo con SoccerHigh (sin reglas, sin VLM): selección "@T" del paper.

Tramos con confianza > --threshold, por confianza descendente mientras quepan en --total-s; los que quedan a
<= --merge-gap-s se funden en un clip. Copia lo actual a output/soccerhigh/before_soccerhigh_only/<id>/ y
sobrescribe highlights.json y los clip_*.mp4 de backend/storage/outputs/<id>/<parte>/highlights.

    python -m scripts.soccerhigh_only_highlights 433f62c92b727870 f2c9315d3394c6ee
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from analytics.highlights.cut import cut_clip
from analytics.highlights.soccerhigh import (
    BACKBONE_NAME, DEFAULT_CHECKPOINT, FPS, SoccerHighSummarizer, ensure_features, predict_frames,
    segments_from_frames, select_to_length,
)
from config import HighlightsConfig


def merge(segs, gap_s: float) -> list[dict]:
    out: list[dict] = []
    for s in segs:  # ordenados por inicio
        start, end = s.start / FPS, (s.end + 1) / FPS
        if out and start - out[-1]["end_s"] <= gap_s:
            out[-1]["end_s"] = max(out[-1]["end_s"], end)
            out[-1]["score"] = max(out[-1]["score"], s.score)
        else:
            out.append({"start_s": start, "end_s": end, "score": s.score})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("match_ids", nargs="+")
    p.add_argument("--part", default="full")
    p.add_argument("--storage-dir", type=Path, default=Path("backend/storage"))
    p.add_argument("--backup-dir", type=Path, default=Path("output/soccerhigh/before_soccerhigh_only"))
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--total-s", type=float, default=180.0)
    p.add_argument("--merge-gap-s", type=float, default=5.0)
    args = p.parse_args()

    cfg = HighlightsConfig()
    model = SoccerHighSummarizer.from_checkpoint(DEFAULT_CHECKPOINT)
    for match_id in args.match_ids:
        out_dir = args.storage_dir / "outputs" / match_id / args.part / "highlights"
        old = json.loads((out_dir / "highlights.json").read_text())
        video = Path(old["video"])

        backup = args.backup_dir / match_id
        if not backup.exists():
            backup.mkdir(parents=True)
            for f in [out_dir / "highlights.json", out_dir / "status.json", *out_dir.glob("clip_*.mp4")]:
                shutil.copy2(f, backup / f.name)

        prob, offsets = predict_frames(model, ensure_features(video))
        segs = segments_from_frames(prob, offsets, args.threshold)
        chosen = merge(select_to_length(segs, int(args.total_s * FPS)), args.merge_gap_s)

        for f in out_dir.glob("clip_*.mp4"):
            f.unlink()
        ranks = {id(c): r for r, c in enumerate(sorted(chosen, key=lambda c: -c["score"]), 1)}
        highlights = []
        for i, c in enumerate(chosen, 1):
            clip = f"clip_{i:02d}.mp4"
            cut_clip(video, c["start_s"], c["end_s"], out_dir / clip, cfg.ffmpeg_crf, cfg.ffmpeg_preset)
            highlights.append({
                "id": f"hl-{i:02d}", "rank": ranks[id(c)], "clip": clip,
                "start_s": c["start_s"], "end_s": c["end_s"], "anchor_s": (c["start_s"] + c["end_s"]) / 2,
                "duration_s": round(c["end_s"] - c["start_s"], 2), "label": "play", "team": -1,
                "title": f"Jugada SoccerHigh ({c['score']:.2f})", "description": "",
                "final_score": round(c["score"], 3), "soccerhigh_score": round(c["score"], 3),
                "sources": ["soccerhigh"],
            })

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report = {
            "video": str(video), "generated_at": now, "mode": "soccerhigh_only",
            "vlm": {"used": False}, "vjepa": {"used": False},
            "soccerhigh": {"used": True, "backbone": BACKBONE_NAME, "checkpoint": str(DEFAULT_CHECKPOINT),
                           "threshold": args.threshold, "total_s_budget": args.total_s, "merge_gap_s": args.merge_gap_s,
                           "n_segments": len(segs), "total_s": round(sum(h["duration_s"] for h in highlights), 1)},
            "n_candidates": len(segs), "highlights": highlights,
        }
        (out_dir / "highlights.json").write_text(json.dumps(report, indent=2))
        (out_dir / "status.json").write_text(json.dumps({"stage": "completed", "updated_at": now,
                                                         "n_highlights": len(highlights)}, indent=2))
        print(f"{match_id}: {len(segs)} tramos > {args.threshold}, {len(highlights)} clips, "
              f"{report['soccerhigh']['total_s']} s")
        for h in highlights:
            print(f"  {h['id']} #{h['rank']:2d} {h['start_s']:7.1f}-{h['end_s']:7.1f} ({h['duration_s']:4.1f} s) "
                  f"{h['soccerhigh_score']:.2f}")


if __name__ == "__main__":
    main()
