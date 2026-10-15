"""Compara detectores de balón sin tocar el pipeline.

1. Sobre el val de datasets/soccernet_tracking_ball (TV 1080p, balón anotado, partidos no vistos): recall y
   precisión por frame a confianza 0.15 y 0.30. Un acierto = alguna detección cuyo centro cae a menos de
   max(24 px, 2 x ancho del balón anotado) del balón real.
2. Sobre tramos de 2 min de nuestros vídeos de YouTube (sin anotar): % de frames con alguna detección, igual
   que en zoom_test.py, para ver si el detector ajustado "ve" más balón en el dominio real.

Cada modelo se da como nombre=pesos:clase:imgsz (clase 32 = 'sports ball' de COCO, 0 = detector propio).

Uso (contenedor, cwd /app):
    python scripts/ball_detector/eval_ball_detectors.py --out output/ball_experiments/eval \\
        --model coco_yolo11x_640=yolo11x.pt:32:640 --model coco_yolo11x_1280=yolo11x.pt:32:1280 \\
        --model snt_yolo11s_1280=models/ball/yolo11s_snt_ball.pt:0:1280 \\
        --slice backend/storage/videos/80be98a792008a28/source.mp4:15000 \\
        --slice backend/storage/videos/ced1633da35cf0a6/source.mp4:60000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

VAL_IMAGES = Path("datasets/soccernet_tracking_ball/images/val")
VAL_LABELS = Path("datasets/soccernet_tracking_ball/labels/val")


def parse_model(spec: str) -> tuple[str, str, int, int]:
    name, rest = spec.split("=", 1)
    weights, cls, imgsz = rest.rsplit(":", 2)
    return name, weights, int(cls), int(imgsz)


def best_boxes(result, cls: int) -> list[tuple[float, float, float, float, float]]:
    """[(cx, cy, w, conf)] de las detecciones de la clase pedida."""
    out = []
    b = result.boxes
    if b is None or len(b) == 0:
        return out
    xyxy = b.xyxy.cpu().numpy()
    conf = b.conf.cpu().numpy()
    kls = b.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, kls):
        if k == cls:
            out.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(c)))
    return out


def _near_any(cx: float, cy: float, gts, tol_of) -> bool:
    return any(((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5 <= tol_of(gw) for gx, gy, gw in gts)


def eval_val(model, cls: int, imgsz: int, images: list[Path]) -> dict:
    """Por umbral de confianza, dos tolerancias (laxa: max(24 px, 2 x ancho GT); estricta: max(16 px, 1 x ancho))
    y tres vistas: 'any' (alguna detección sobre un balón: recall / precisión por detección), 'top1' (la mejor
    detección del frame, que es la que usa PlayerTracker._resolve_ball: acierta / señala otra cosa / no hay) y
    frames sin balón anotado con alguna detección (falsa alarma por frame)."""
    TOLS = {"laxa": lambda gw: max(24.0, 2 * gw), "estricta": lambda gw: max(16.0, gw)}
    stats = {(thr, tol): {"gt": 0, "tp": 0, "fp_with_gt": 0, "fp_no_gt": 0, "frames_no_gt": 0, "frames_no_gt_with_det": 0,
                          "top1_hit": 0, "top1_wrong": 0, "top1_none": 0}
             for thr in (0.15, 0.30) for tol in TOLS}
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        rows = [l.split() for l in (VAL_LABELS / (img_path.stem + ".txt")).read_text().splitlines() if l.strip()]
        gts = [(float(r[1]) * W, float(r[2]) * H, float(r[3]) * W) for r in rows]
        dets = best_boxes(model(img, conf=0.05, classes=[cls], imgsz=imgsz, verbose=False)[0], cls)
        for (thr, tol), s in stats.items():
            tol_of = TOLS[tol]
            d = [x for x in dets if x[4] >= thr]
            if gts:
                s["gt"] += 1
                hits = [_near_any(cx, cy, gts, tol_of) for cx, cy, _, _, _ in d]
                s["tp"] += int(any(hits))
                s["fp_with_gt"] += int(sum(not h for h in hits))
                if not d:
                    s["top1_none"] += 1
                elif hits[0]:
                    s["top1_hit"] += 1
                else:
                    s["top1_wrong"] += 1
            else:
                s["frames_no_gt"] += 1
                s["fp_no_gt"] += len(d)
                s["frames_no_gt_with_det"] += int(bool(d))
    out = {}
    for (thr, tol), s in stats.items():
        fp = s["fp_with_gt"] + s["fp_no_gt"]
        out[f"{thr}|{tol}"] = {
            "recall": round(s["tp"] / s["gt"], 4) if s["gt"] else None,
            "precision": round(s["tp"] / (s["tp"] + fp), 4) if (s["tp"] + fp) else None,
            "top1_acierto": round(s["top1_hit"] / s["gt"], 4) if s["gt"] else None,
            "top1_otra_cosa": round(s["top1_wrong"] / s["gt"], 4) if s["gt"] else None,
            "top1_sin_deteccion": round(s["top1_none"] / s["gt"], 4) if s["gt"] else None,
            "falsa_alarma_sin_balon": round(s["frames_no_gt_with_det"] / s["frames_no_gt"], 4) if s["frames_no_gt"] else None,
            "frames_with_gt": s["gt"], "frames_without_gt": s["frames_no_gt"], "false_positives": int(fp),
        }
    return out


def eval_slice(model, cls: int, imgsz: int, video: str, start: int, n: int = 1000, step: int = 2) -> dict:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    confs = []
    for i in range(n * step):
        ok, f = cap.read()
        if not ok:
            break
        if i % step:
            continue
        dets = best_boxes(model(f, conf=0.05, classes=[cls], imgsz=imgsz, verbose=False)[0], cls)
        confs.append(max((d[4] for d in dets), default=0.0))
    cap.release()
    c = np.array(confs)
    return {"frames": len(confs), "ge_0.15": round(float((c >= 0.15).mean()), 4) if len(c) else None,
            "ge_0.30": round(float((c >= 0.30).mean()), 4) if len(c) else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="nombre=pesos:clase:imgsz")
    ap.add_argument("--slice", action="append", default=[], help="video.mp4:frame_inicial")
    ap.add_argument("--val-images", type=int, default=2450, help="todas las del val por defecto (cuesta ~1 min/modelo)")
    ap.add_argument("--out", required=True, type=Path, help="prefijo de salida (.json y .txt)")
    a = ap.parse_args()

    from ultralytics import YOLO

    images = sorted(VAL_IMAGES.glob("*.jpg")) if VAL_IMAGES.exists() else []
    random.Random(0).shuffle(images)
    images = images[: a.val_images]
    report: dict = {"val_images": len(images), "models": {}}
    lines = [f"Val SoccerNet Tracking: {len(images)} imágenes; tolerancia laxa = max(24 px, 2 x ancho balón), "
             "estricta = max(16 px, 1 x ancho). 'top-1' = la detección de mayor confianza del frame, que es la que usa el pipeline.", ""]
    for spec in a.model:
        name, weights, cls, imgsz = parse_model(spec)
        if not Path(weights).exists():
            lines.append(f"{name}: pesos no encontrados ({weights}), saltado")
            continue
        model = YOLO(weights)
        entry: dict = {"weights": weights, "class": cls, "imgsz": imgsz}
        if images:
            entry["val"] = eval_val(model, cls, imgsz, images)
            for key, v in entry["val"].items():
                thr, tol = key.split("|")
                lines.append(f"{name:22s} conf>={thr} tol {tol:8s}: recall {v['recall']}, precisión {v['precision']}, "
                             f"top-1 acierta {v['top1_acierto']} / otra cosa {v['top1_otra_cosa']} / nada {v['top1_sin_deteccion']}, "
                             f"falsa alarma en frames sin balón {v['falsa_alarma_sin_balon']}, FP {v['false_positives']}")
        entry["slices"] = {}
        for sl in a.slice:
            video, start = sl.rsplit(":", 1)
            r = eval_slice(model, cls, imgsz, video, int(start))
            entry["slices"][sl] = r
            lines.append(f"{name:22s} {Path(video).parent.name} desde frame {start}: balón en {r['ge_0.15']} de los frames (>=0.15), {r['ge_0.30']} (>=0.30)")
        report["models"][name] = entry
        lines.append("")
        print("\n".join(lines[-6:]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
