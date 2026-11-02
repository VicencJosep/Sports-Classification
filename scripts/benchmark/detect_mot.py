"""Paso 1 del benchmark de tracking: corre un detector del pipeline sobre
secuencias MOT y guarda las detecciones (única parte con GPU). Los trackers
(``track_mot.py``) leen después estas mismas detecciones, así que la
comparación de asociación es con detecciones idénticas.

Uso (raíz del proyecto, dentro del contenedor):
    python -m scripts.benchmark.detect_mot \\
        --seqs datasets/SoccerNet/tracking/extracted/test \\
        --detector hybrid --out output/benchmark/snmot_test/dets/hybrid

Reanudable: salta las secuencias cuyo .txt ya existe.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import torch

from core.tracking import build_detector
from scripts.benchmark.mot import list_sequences, seq_info, write_mot

logger = logging.getLogger(__name__)


def detect_sequence(detector, seq_dir: Path) -> tuple[list[tuple], float]:
    info = seq_info(seq_dir)
    rows: list[tuple] = []
    t0 = time.perf_counter()
    for start in range(1, info["length"] + 1, detector.chunk_size):
        frames_idx = list(range(start, min(start + detector.chunk_size, info["length"] + 1)))
        frames = [cv2.imread(str(info["img_dir"] / f"{i:06d}{info['img_ext']}")) for i in frames_idx]
        for frame_no, dets in zip(frames_idx, detector.detect(frames)):
            jerseys = dets.data.get("jersey", [None] * len(dets))
            for (x1, y1, x2, y2), conf, cls, jersey in zip(dets.xyxy, dets.confidence, dets.class_id, jerseys):
                rows.append((frame_no, -1, x1, y1, x2 - x1, y2 - y1, conf, cls,
                             jersey if jersey is not None else -1))
    return rows, time.perf_counter() - t0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", type=Path, required=True, help="Carpeta de secuencias MOT (o una sola)")
    ap.add_argument("--detector", default="hybrid", help="hybrid | soccermaster | yolo (core.tracking.build_detector)")
    ap.add_argument("--yolo-model", default="yolo11x.pt", help="Pesos YOLO (balón en hybrid, todo en yolo)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="Solo las N primeras secuencias (prueba)")
    args = ap.parse_args()

    detector = build_detector(args.detector, yolo_model_path=args.yolo_model)
    args.out.mkdir(parents=True, exist_ok=True)
    timing_path = args.out / "timing.jsonl"
    for seq_dir in list_sequences(args.seqs)[: args.limit]:
        out = args.out / f"{seq_dir.name}.txt"
        if out.exists():
            continue
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        rows, secs = detect_sequence(detector, seq_dir)
        write_mot(out, rows)
        n = seq_info(seq_dir)["length"]
        vram = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        with open(timing_path, "a") as f:
            f.write(json.dumps({"seq": seq_dir.name, "frames": n, "seconds": round(secs, 2),
                                "fps": round(n / secs, 2), "vram_peak_gb": round(vram, 2)}) + "\n")
        logger.info("%s: %d detecciones, %.1f fps, VRAM pico %.1f GB", seq_dir.name, len(rows), n / secs, vram)


if __name__ == "__main__":
    main()
