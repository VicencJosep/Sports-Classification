"""Ajuste fino de YOLO11s (COCO) como detector de balón de una clase sobre datasets/soccernet_tracking_ball
(ver prepare_soccernet_tracking_ball.py). Hiperparámetros heredados del detector panorámico que ya funcionó
(output/ball_panorama/runs/yolo11s_pano/args.yaml), con imgsz 1280 porque el pipeline infiere a esa
resolución. Reanudable: si hay weights/last.pt sin terminar, continúa; si ya terminó, no hace nada.

Uso (contenedor, cwd /app): python scripts/ball_detector/train_ball_yolo.py [--epochs 25 --batch 8]
Salida: output/ball_experiments/runs/yolo11s_snt_ball/ y copia de best.pt en models/ball/yolo11s_snt_ball.pt
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets/soccernet_tracking_ball/data.yaml")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="/app/output/ball_experiments/runs",
                    help="ruta ABSOLUTA: con una relativa Ultralytics la cuelga de runs/detect/")
    ap.add_argument("--name", default="yolo11s_snt_ball")
    a = ap.parse_args()

    from ultralytics import YOLO

    run_dir = Path(a.project) / a.name
    done = run_dir / ".train_done"
    last = run_dir / "weights" / "last.pt"
    best = run_dir / "weights" / "best.pt"

    def _finished(ckpt: Path) -> bool:
        """Ultralytics deja last.pt con epoch=-1 y sin optimizador al acabar
        (strip_optimizer); train(resume=True) sobre eso NO reanuda: arranca un
        entrenamiento nuevo con los args por defecto."""
        import torch
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        return int(ck.get("epoch", -1)) < 0 or ck.get("optimizer") is None

    if (done.exists() and best.exists()) or (last.exists() and _finished(last)):
        done.touch()
        print("entrenamiento ya terminado:", best)
    elif last.exists():
        print("reanudando desde", last)
        results = YOLO(str(last)).train(resume=True)
        (Path(results.save_dir) / ".train_done").touch()
    else:
        results = YOLO(a.model).train(
            data=a.data, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, workers=a.workers,
            project=a.project, name=a.name, exist_ok=True, patience=8, cache=False, device=0,
            single_cls=True, close_mosaic=5, mosaic=1.0, scale=0.3, translate=0.1, fliplr=0.5,
            degrees=0.0, mixup=0.0, copy_paste=0.0, erasing=0.4, plots=True, verbose=True,
        )
        best = Path(results.save_dir) / "weights" / "best.pt"
        (Path(results.save_dir) / ".train_done").touch()
    if best.exists():
        dst = Path("models/ball") / f"{a.name}.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst)
        print("pesos copiados a", dst)


if __name__ == "__main__":
    main()
