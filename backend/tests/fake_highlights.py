"""Doble de ``python -m analytics.highlights`` para los tests: acepta los mismos
argumentos y deja ``status.json``, ``highlights.json`` y un ``clip_01.mp4`` de
mentira en ``--out``, sin modelos ni ffmpeg.

Con ``FAKE_HIGHLIGHTS_FAIL=1`` termina con código 2 tras escribir un error.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend-match", required=True)
    p.add_argument("--part", required=True)
    p.add_argument("--storage-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    print(f"fake highlights: match={args.backend_match} part={args.part} force={args.force}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "status.json").write_text(json.dumps({"stage": "vlm", "done": 2, "total": 5}))
    if os.environ.get("FAKE_HIGHLIGHTS_FAIL") == "1":
        print("RuntimeError: VLM sin respuesta (simulated)", flush=True)
        return 2
    time.sleep(0.2)
    (args.out / "clip_01.mp4").write_bytes(b"\x00" * 4096)
    report = {
        "video": str(args.storage_dir / "videos" / "x" / "source.mp4"),
        "generated_at": "2026-10-16T12:00:00Z",
        "vlm": {"used": True, "model": "fake-vlm", "api": "ollama", "degraded": True, "n_failed_calls": 1},
        "has_audio": True,
        "scoreboard": {"available": True, "n_samples": 10, "n_valid": 9, "changes": [{"t_lo": 0.5, "t_hi": 1.5, "before": [0, 0], "after": [1, 0]}]},
        "vjepa": {"used": False},
        "n_candidates": 7,
        "highlights": [{
            "id": "hl-01", "rank": 1, "clip": "clip_01.mp4", "start_s": 0.0, "end_s": 1.5, "anchor_s": 1.0,
            "duration_s": 1.5, "label": "goal", "team": 1, "title": "Gol (0:00:01)", "description": "Gol de prueba.",
            "final_score": 0.9, "rule_score": 3.2, "goal_source": "restart", "vlm_rank": 1, "vlm_reason": "goal",
            "vlm": {"is_shot": True, "is_goal": True, "goal_confidence": 0.9, "skill": 6, "excitement": 8, "worth_highlight": True},
            "candidate_id": "cand-000", "merged_from": None, "sources": ["shot", "audio"],
        }],
        "candidates": [],
    }
    (args.out / "highlights.json").write_text(json.dumps(report))
    (args.out / "status.json").write_text(json.dumps({"stage": "completed", "n_highlights": 1}))
    print("fake highlights: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
