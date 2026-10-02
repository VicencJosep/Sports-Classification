"""Escanea partidos completos con T-DEED (ganador SoccerNet Ball Action Spotting 2024) y guarda la
probabilidad por frame de las 12 clases (PASS, DRIVE, ..., CROSS, SHOT, ..., GOAL) en un .npz por partido.

Checkpoints (models/tdeed/<modelo>/checkpoint_best.pt):
  SoccerNetBall_test        rny002, entrenado con el split train (Leeds, Hull, Brentford, Blackburn);
                            val = Middlesbrough, test = Stoke y Reading  -> estos 3 son evaluación limpia
  SoccerNetBall_challenge2  rny008, entrenado con 6 partidos (todos menos Middlesbrough)

Uso (dentro del contenedor, cwd = models/tdeed/T-DEED):
  python /app/output/shot_benchmark/experiments/tdeed_scan.py MODELO VIDEO [VIDEO...] [--max-clips N]
Salida: /app/output/shot_benchmark/tdeed/MODELO/<partido>.npz  (probs [n, 13] float16, col 0 = fondo;
        stride 2 -> la fila i es el frame 2*i del vídeo). Reanudable por partido."""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
from util.io import load_json
from util.dataset import load_classes
from model.model import TDEEDModel
from dataset.frame import ActionSpotInferenceDataset

CONFIG_OF = {"SoccerNetBall_test": "SoccerNetBall_challenge1", "SoccerNetBall_challenge1": "SoccerNetBall_challenge1",
             "SoccerNetBall_challenge2": "SoccerNetBall_challenge2"}
STRIDE = 2

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("videos", nargs="+"); ap.add_argument("--max-clips", type=int, default=None); ap.add_argument("--out-tag", default="")
a = ap.parse_args()

cfg = load_json(f"config/SoccerNetBall/{CONFIG_OF[a.model]}.json")
args = argparse.Namespace(**{k: v for k, v in cfg.items() if k != "pretrain"}, pretrain=cfg.get("pretrain"))
args.crop_dim = None if args.crop_dim <= 0 else args.crop_dim
classes = load_classes("data/soccernetball/class.txt")
pre_classes = load_classes("data/soccernet/class.txt")
model = TDEEDModel(args=args)
n_classes = [len(classes) + 1, len(pre_classes) + 1]
model._model.update_pred_head(n_classes)
model._num_classes = int(np.sum(n_classes))
model.load(torch.load(f"/app/models/tdeed/{a.model}/checkpoint_best.pt"))
names = ["background"] + [c for c, _ in sorted(classes.items(), key=lambda kv: kv[1])]
print("clases:", names, flush=True)

out_dir = Path(f"/app/output/shot_benchmark/tdeed/{a.model}{a.out_tag}")
out_dir.mkdir(parents=True, exist_ok=True)
for video in a.videos:
    stem = Path(video).stem
    out = out_dir / (f"{Path(video).parent.name}.npz" if stem == "720p" else f"{Path(video).parent.name} [{stem}].npz")
    if out.exists() and a.max_clips is None:
        print("ya hecho:", out.name, flush=True); continue
    ds = ActionSpotInferenceDataset(video, clip_len=args.clip_len, overlap_len=args.clip_len // 4 * 3,
                                    stride=STRIDE, dataset="soccernetball", size=(796, 448))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    n = ds._video_len // STRIDE
    pred = np.zeros((n, len(names)), np.float32); support = np.zeros(n, np.int32)
    t0 = time.perf_counter()
    for b, (frames, starts) in enumerate(loader):
        _, scores = model.predict(frames)
        if b == 0:
            print("frames", tuple(frames.shape), "scores", np.asarray(scores).shape, flush=True)
        for i in range(frames.shape[0]):
            s, start = scores[i][:, :len(names)], starts[i].item()
            if start < 0:
                s, start = s[-start:], 0
            end = min(start + s.shape[0], n)
            pred[start:end] += s[:end - start]; support[start:end] += 1
        if b % 200 == 0:
            done = (b + 1) * args.batch_size * (args.clip_len // 4) * STRIDE
            print(f"  {Path(video).parent.name[13:35]} {100 * done / ds._video_len:5.1f}%  {time.perf_counter() - t0:.0f}s", flush=True)
        if a.max_clips and b + 1 >= a.max_clips:
            break
    probs = pred / np.maximum(support, 1)[:, None]
    if a.max_clips:
        top = probs[: (b + 1) * args.batch_size * 25]
        print("prueba: máx por clase", {k: round(float(top[:, j].max()), 2) for j, k in enumerate(names)}); continue
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, probs=probs.astype(np.float16), classes=np.array(names), stride=STRIDE)
    tmp.rename(out)
    print(f"FIN {out.name} en {time.perf_counter() - t0:.0f}s", flush=True)
