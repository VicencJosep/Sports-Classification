"""Imágenes para revisar a ojo el detector de balón ajustado (y, de paso, las etiquetas de entrenamiento).

Para cada tramo pedido corre el detector ajustado (clase 0) y el COCO (clase 32) a 1280 px sobre frames
muestreados, y guarda por detección una imagen compuesta: frame entero a 960 px con la caja, y a la derecha
un recorte 3x alrededor de la detección. El nombre lleva el tramo, el frame, la confianza y si COCO también
vio el balón ahí. Además escribe un resumen numérico por tramo (frames con detección, acuerdo con COCO, etc.).

Uso (contenedor, cwd /app):
    python scripts/ball_detector/visual_check.py --out output/ball_experiments/visual_check \\
        --slice mallorca_play=backend/storage/videos/80be98a792008a28/source.mp4:15000:3000 \\
        --slice espanyol_play=backend/storage/videos/ced1633da35cf0a6/source.mp4:60000:3000 \\
        --slice espanyol_prematch=backend/storage/videos/ced1633da35cf0a6/source.mp4:1800:3000 \\
        --slice espanyol_halftime=backend/storage/videos/ced1633da35cf0a6/source.mp4:133000:3000 \\
        --per-slice 16 --train-labels 12
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def compose(frame, cx, cy, w, h, text, zoom=3, crop=160):
    H, W = frame.shape[:2]
    full = frame.copy()
    x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
    cv2.rectangle(full, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (0, 255, 255), 2)
    cv2.circle(full, (int(cx), int(cy)), 28, (0, 255, 255), 2)
    scale = 960 / W
    full = cv2.resize(full, (960, int(H * scale)))
    cx0, cy0 = int(np.clip(cx - crop / 2, 0, W - crop)), int(np.clip(cy - crop / 2, 0, H - crop))
    patch = frame[cy0:cy0 + crop, cx0:cx0 + crop]
    patch = cv2.resize(patch, (crop * zoom, crop * zoom), interpolation=cv2.INTER_CUBIC)
    px1, py1 = int((x1 - cx0) * zoom), int((y1 - cy0) * zoom)
    px2, py2 = int((x2 - cx0) * zoom), int((y2 - cy0) * zoom)
    cv2.rectangle(patch, (px1 - 4, py1 - 4), (px2 + 4, py2 + 4), (0, 255, 255), 2)
    canvas_h = max(full.shape[0], patch.shape[0]) + 40
    canvas = np.full((canvas_h, full.shape[1] + patch.shape[1] + 20, 3), 30, dtype=np.uint8)
    canvas[40:40 + full.shape[0], :full.shape[1]] = full
    canvas[40:40 + patch.shape[0], full.shape[1] + 20:] = patch
    cv2.putText(canvas, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


def dets_of(model, frame, cls, imgsz):
    r = model(frame, conf=0.05, classes=[cls], imgsz=imgsz, verbose=False)[0]
    out = []
    if r.boxes is not None and len(r.boxes):
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(c)))
    return sorted(out, key=lambda d: -d[4])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slice", action="append", required=True, help="nombre=video:frame_inicial:n_frames")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--per-slice", type=int, default=16)
    ap.add_argument("--train-labels", type=int, default=12, help="imágenes de entrenamiento con su etiqueta GT dibujada")
    ap.add_argument("--weights", default="models/ball/yolo11s_snt_ball.pt")
    ap.add_argument("--coco", default="yolo11x.pt")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    from ultralytics import YOLO
    ft, coco = YOLO(a.weights), YOLO(a.coco)
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    summary = {}
    for spec in a.slice:
        name, rest = spec.split("=", 1)
        video, start, n = rest.rsplit(":", 2)
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        rows = []
        for i in range(int(n)):
            ok, f = cap.read()
            if not ok:
                break
            if i % a.step:
                continue
            d_ft, d_coco = dets_of(ft, f, 0, 1280), dets_of(coco, f, 32, 1280)
            rows.append((int(start) + i, f, d_ft, d_coco))
        cap.release()
        s = {"frames": len(rows)}
        for thr in (0.15, 0.30):
            s[f"ft_frames_with_det_ge_{thr}"] = int(sum(bool([d for d in r[2] if d[4] >= thr]) for r in rows))
            s[f"coco_frames_with_det_ge_{thr}"] = int(sum(bool([d for d in r[3] if d[4] >= thr]) for r in rows))
            s[f"ft_mean_dets_per_frame_ge_{thr}"] = round(sum(len([d for d in r[2] if d[4] >= thr]) for r in rows) / max(1, len(rows)), 2)
            agree = both = 0
            for _, _, d_ft, d_coco in rows:
                dft = [d for d in d_ft if d[4] >= thr]
                dco = [d for d in d_coco if d[4] >= thr]
                if dft and dco:
                    both += 1
                    agree += ((dft[0][0] - dco[0][0]) ** 2 + (dft[0][1] - dco[0][1]) ** 2) ** 0.5 < 30
            s[f"both_detect_ge_{thr}"] = int(both)
            s[f"agree_within_30px_ge_{thr}"] = int(agree)
        summary[name] = s
        # imágenes: detecciones del ajustado >= 0.15, prioridad a las que COCO no ve (las sospechosas)
        with_det = [r for r in rows if r[2] and r[2][0][4] >= 0.15]
        rng.shuffle(with_det)
        with_det.sort(key=lambda r: 0 if not [d for d in r[3] if d[4] >= 0.15] else 1)
        for idx, frame, d_ft, d_coco in with_det[: a.per_slice]:
            cx, cy, w, h, c = d_ft[0]
            coco_ok = any(((d[0] - cx) ** 2 + (d[1] - cy) ** 2) ** 0.5 < 30 for d in d_coco if d[4] >= 0.15)
            tag = "coco_agrees" if coco_ok else "coco_missed"
            text = f"{name} frame {idx} conf {c:.2f} ({tag}) - caja amarilla = deteccion del detector ajustado; derecha: zoom x3"
            cv2.imwrite(str(a.out / f"{name}_f{idx}_c{int(c * 100):03d}_{tag}.jpg"), compose(frame, cx, cy, w, h, text), [cv2.IMWRITE_JPEG_QUALITY, 85])
    # etiquetas de entrenamiento
    img_dir, lbl_dir = Path("datasets/soccernet_tracking_ball/images/train"), Path("datasets/soccernet_tracking_ball/labels/train")
    if a.train_labels and img_dir.exists():
        imgs = sorted(img_dir.glob("*.jpg"))
        rng.shuffle(imgs)
        done = 0
        for p in imgs:
            rows_l = [l.split() for l in (lbl_dir / (p.stem + ".txt")).read_text().splitlines() if l.strip()]
            if not rows_l:
                continue
            frame = cv2.imread(str(p))
            H, W = frame.shape[:2]
            _, cxn, cyn, wn, hn = (float(v) for v in rows_l[0])
            text = f"TRAIN LABEL {p.stem} - caja amarilla = etiqueta GT de SoccerNet Tracking (debe ser el balon)"
            cv2.imwrite(str(a.out / f"trainlabel_{p.stem}.jpg"), compose(frame, cxn * W, cyn * H, wn * W, hn * H, text), [cv2.IMWRITE_JPEG_QUALITY, 85])
            done += 1
            if done >= a.train_labels:
                break
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
