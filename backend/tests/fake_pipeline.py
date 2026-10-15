"""Doble de ``python -m main`` para los tests: acepta los mismos argumentos y
deja los mismos ficheros (copiados de recortes de salidas reales), sin GPU.

Con la variable de entorno ``FAKE_PIPELINE_FAIL=1`` termina con código 3 tras escribir
un error en el log, para probar el camino de fallo.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--skip-homography", action="store_true")
    p.add_argument("--skip-classification", action="store_true")
    args = p.parse_args()

    print(f"fake pipeline: mode={args.mode} source={args.source}", flush=True)
    if os.environ.get("FAKE_PIPELINE_FAIL") == "1":
        print("RuntimeError: CUDA out of memory (simulated)", flush=True)
        return 3

    time.sleep(0.2)
    stem = args.source.stem
    shutil.copy(FIXTURES / "tracks_2d.sample.jsonl", args.source.with_name(f"{stem}.tracks.jsonl"))
    shutil.copy(FIXTURES / "possession.sample.jsonl", args.source.with_name(f"{stem}.possession.jsonl"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    telemetry = "telemetry.clip.json" if args.mode == "clip" else "telemetry.full_match.json"
    shutil.copy(FIXTURES / telemetry, args.output_dir / "telemetry.json")
    if not args.skip_classification:
        shutil.copy(FIXTURES / "teams.json", args.output_dir / "teams.json")
    if not args.skip_homography:
        shutil.copy(FIXTURES / "tracks_2d.sample.jsonl", args.output_dir / "tracks_2d.jsonl")
    print("fake pipeline: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
