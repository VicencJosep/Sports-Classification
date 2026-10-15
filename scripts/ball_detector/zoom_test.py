"""¿Ayudaría una cámara virtual (zoom) a detectar el balón? Techo rápido: inferir el frame entero
reescalado a 1280/1920/2560 px equivale a un zoom x1 / x1.5 / x2 sobre toda la imagen. Si a 2560 no
mejora respecto a 1280 (resolución nativa del 720p), recortar y ampliar tampoco ayudará.

Uso: python scripts/ball_detector/zoom_test.py --video v.mp4 --start-frame 15000 --n 1000 --out r.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--n", type=int, default=1000, help="frames a evaluar")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1280, 1920, 2560])
    ap.add_argument("--weights", default="/app/yolo11x.pt")
    ap.add_argument("--ball-class", type=int, default=32, help="32 = 'sports ball' de COCO; 0 en un detector de balón propio")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)
    frames = []
    for i in range(a.n * a.step):
        ok, f = cap.read()
        if not ok:
            break
        if i % a.step == 0:
            frames.append(f)
    cap.release()
    print(f"{len(frames)} frames desde {a.start_frame} de {a.video}", flush=True)

    out = {"video": a.video, "start_frame": a.start_frame, "n_frames": len(frames), "weights": a.weights, "sizes": {}}
    for imgsz in a.sizes:
        confs = []
        t0 = time.time()
        for f in frames:
            r = model(f, conf=0.05, classes=[a.ball_class], imgsz=imgsz, verbose=False)[0]
            c = r.boxes.conf.cpu().numpy()
            confs.append(float(c.max()) if len(c) else 0.0)
        confs_arr = np.array(confs)
        res = {
            "frames_with_ball_ge_0.15": round(float((confs_arr >= 0.15).mean()), 4),
            "frames_with_ball_ge_0.30": round(float((confs_arr >= 0.30).mean()), 4),
            "median_conf_when_found": round(float(np.median(confs_arr[confs_arr >= 0.15])), 3) if (confs_arr >= 0.15).any() else None,
            "ms_per_frame": round((time.time() - t0) / max(1, len(frames)) * 1000, 1),
        }
        out["sizes"][str(imgsz)] = res
        print(imgsz, json.dumps(res), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
