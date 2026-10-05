"""Como tdeed_scan.py, pero guarda las DOS cabezas de T-DEED: acciones de balón (13 columnas) y
SoccerNet-v2 (18: fondo + Penalty, Kick-off, Goal, Substitution, Offside, Shots on/off target, Clearance,
Ball out of play, Throw-in, Foul, Indirect/Direct free-kick, Corner, Yellow card, Red card, Yellow->red).
model.predict() del repo descarta la segunda; aquí se replica su post-proceso para ambas.

Uso (cwd = models/tdeed/T-DEED):  python tdeed_scan_heads.py MODELO VIDEO [VIDEO...] [--max-clips N]
Salida: /app/output/shot_benchmark/tdeed/MODELO_heads/<partido> [<mitad>].npz  (probs, probs_sn, classes, classes_sn)"""
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

CONFIG_OF = {"SoccerNetBall_test": "SoccerNetBall_challenge1", "SoccerNetBall_challenge2": "SoccerNetBall_challenge2"}
STRIDE = 2
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("videos", nargs="+"); ap.add_argument("--max-clips", type=int, default=None)
a = ap.parse_args()

cfg = load_json(f"config/SoccerNetBall/{CONFIG_OF[a.model]}.json")
args = argparse.Namespace(**cfg); args.crop_dim = None
names = lambda f: ["background"] + [c for c, _ in sorted(load_classes(f).items(), key=lambda kv: kv[1])]
cls1, cls2 = names("data/soccernetball/class.txt"), names("data/soccernet/class.txt")
model = TDEEDModel(args=args)
model._model.update_pred_head([len(cls1), len(cls2)])
model._num_classes = len(cls1) + len(cls2)
model.load(torch.load(f"/app/models/tdeed/{a.model}/checkpoint_best.pt"))
net = model._model


def predict_heads(frames):
    """Softmax por cabeza + recolocación temporal con el desplazamiento predicho (máximo), como
    model.modules.process_double_head pero para las dos cabezas y vectorizado."""
    seq = frames.to(model.device).float()
    net.eval()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        pred, _ = net(seq, inference=True)
    logits, displ = pred["im_feat"], pred["displ_feat"]
    if isinstance(logits, list): logits = logits[0]
    if isinstance(displ, list): displ = displ[0]
    logits, displ = logits.float(), displ.float()
    B, T = logits.shape[:2]
    d = displ.reshape(B, T, -1)[:, :, 0].round().long()
    target = (torch.arange(T, device=d.device)[None] - d).clamp(0, T - 1)
    out = []
    for sl in (slice(0, len(cls1)), slice(len(cls1), len(cls1) + len(cls2))):
        p = torch.softmax(logits[:, :, sl], dim=2)
        agg = torch.zeros_like(p).scatter_reduce(1, target[:, :, None].expand_as(p), p, reduce="amax", include_self=True)
        out.append(agg.cpu().numpy())
    return out


out_dir = Path(f"/app/output/shot_benchmark/tdeed/{a.model}_heads"); out_dir.mkdir(parents=True, exist_ok=True)
for video in a.videos:
    stem = Path(video).stem
    out = out_dir / (f"{Path(video).parent.name}.npz" if stem == "720p" else f"{Path(video).parent.name} [{stem}].npz")
    if out.exists() and a.max_clips is None:
        print("ya hecho:", out.name, flush=True); continue
    ds = ActionSpotInferenceDataset(video, clip_len=args.clip_len, overlap_len=args.clip_len // 4 * 3,
                                    stride=STRIDE, dataset="soccernetball", size=(796, 448))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    n = ds._video_len // STRIDE
    acc = [np.zeros((n, len(cls1)), np.float32), np.zeros((n, len(cls2)), np.float32)]; support = np.zeros(n, np.int32)
    t0 = time.perf_counter()
    for b, (frames, starts) in enumerate(loader):
        heads = predict_heads(frames)
        if b == 0 and a.max_clips:
            _, ref = model.predict(frames)
            print("check cabeza 1 vs model.predict: max |Δ| =", float(np.abs(ref - heads[0]).max()), "| shapes", heads[0].shape, heads[1].shape, flush=True)
        for i, start in enumerate(starts.tolist()):
            lo = -start if start < 0 else 0
            s0 = max(start, 0); end = min(s0 + heads[0].shape[1] - lo, n)
            for h in (0, 1):
                acc[h][s0:end] += heads[h][i][lo: lo + end - s0]
            support[s0:end] += 1
        if a.max_clips and b + 1 >= a.max_clips:
            break
    if a.max_clips:
        k = (b + 1) * args.batch_size * 25
        print("máx por clase SN:", {c: round(float((acc[1][:k] / np.maximum(support[:k], 1)[:, None])[:, j].max()), 2) for j, c in enumerate(cls2)}); continue
    div = np.maximum(support, 1)[:, None]
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, probs=(acc[0] / div).astype(np.float16), probs_sn=(acc[1] / div).astype(np.float16),
             classes=np.array(cls1), classes_sn=np.array(cls2), stride=STRIDE)
    tmp.rename(out)
    print(f"FIN {out.name} en {time.perf_counter() - t0:.0f}s", flush=True)
