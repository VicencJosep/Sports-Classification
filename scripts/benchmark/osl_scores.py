"""Caché de puntuaciones del E2E-Spot de OpenSportsLab entrenado en SoccerNet
Ball Action Spotting (HF ``OpenSportsLab/OSL-loc-snbas-2025-e2e``, AGPL-3.0;
pesos en ``models/osl_e2e/weights``) en el formato de ``core.ball_spotting``
para ``eval_bas.py``.

Se construye ``E2EModel.Impl`` (RegNetY-008 + GSM + GRU) con el código del
wheel de ``opensportslib`` 0.3.3 copiado en ``models/osl_e2e/src`` (sin
instalarlo: el paquete arrastra DALI). Entrada como su pipeline DALI de test:
RGB 398x224, normalización ImageNet, 1 de cada ``get_stride(25, 2)`` = 12
frames (≈2 fps, ``extract_fps: 2`` del config), clips de 100 con solape 50.

Uso (contenedor):
    python -m scripts.benchmark.osl_scores --split test
    python -m scripts.benchmark.osl_scores --videos datasets/soccertrack-v2/views/128057/1st_vcam796.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "models" / "osl_e2e" / "src"
WEIGHTS = ROOT / "models" / "osl_e2e" / "weights" / "model.pt"
TDEED_SPLITS = ROOT / "models" / "tdeed" / "T-DEED" / "data" / "soccernetball"
LABELS_ROOT = ROOT / "datasets" / "SoccerNet"
MODEL_NAME = "OSL_E2E_snbas"
CLASSES = ["PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", "THROW IN", "SHOT",
           "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"]
SIZE, STRIDE, CLIP_LEN, OVERLAP = (398, 224), 12, 100, 50
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255

logger = logging.getLogger(__name__)


def load_model():
    sys.path.insert(0, str(SRC))
    from opensportslib.models.base.e2e import E2EModel

    net = E2EModel.Impl(len(CLASSES) + 1, SimpleNamespace(type="rny008_gsm"), SimpleNamespace(type="gru"),
                        CLIP_LEN, "rgb")
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=False)["model_state_dict"]
    net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
    return net.cuda().eval()


def predict(net, clips: torch.Tensor) -> torch.Tensor:
    """Lote uint8 ``[B, L, 3, H, W]`` -> softmax ``[B, L, 13]``. Si los frames no
    son 398x224 se redimensionan (bilineal) y se normalizan como su pipeline."""
    b, l = clips.shape[:2]
    x = clips.reshape(b * l, *clips.shape[2:]).float()
    if tuple(x.shape[-2:]) != (SIZE[1], SIZE[0]):
        x = torch.nn.functional.interpolate(x, size=(SIZE[1], SIZE[0]), mode="bilinear", align_corners=False)
    x = ((x - MEAN) / STD).reshape(b, l, 3, SIZE[1], SIZE[0]).cuda()
    with torch.no_grad(), torch.autocast("cuda"):
        pred = net(x)
    if isinstance(pred, tuple):
        pred = pred[0]
    if pred.dim() > 3:
        pred = pred[-1]
    return torch.softmax(pred.float(), dim=2)


def score_video(net, video: Path) -> np.ndarray:
    """Probabilidades por frame muestreado (1 de cada 12) de un vídeo entero, media sobre clips solapados."""
    frames = torch.stack(list(sampled_frames(video)))  # [n, 3, 224, 398] uint8 (~ 470 filas/partido-min)
    n = len(frames)
    acc = np.zeros((n, len(CLASSES) + 1), np.float32)
    support = np.zeros(n, np.int32)
    starts = list(range(0, max(n - CLIP_LEN, 0) + 1, CLIP_LEN - OVERLAP))
    if starts[-1] + CLIP_LEN < n:
        starts.append(n - CLIP_LEN)
    for b in range(0, len(starts), 4):
        batch_starts = starts[b: b + 4]
        pred = predict(net, torch.stack([frames[s: s + CLIP_LEN] for s in batch_starts])).cpu().numpy()
        for p, s in zip(pred, batch_starts):
            acc[s: s + len(p)] += p
            support[s: s + len(p)] += 1
    return acc / np.maximum(support, 1)[:, None]


def sampled_frames(video: Path):
    cap = cv2.VideoCapture(str(video))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % STRIDE == 0:
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), SIZE)
            yield torch.from_numpy(frame).permute(2, 0, 1)
        i += 1
    cap.release()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--videos", nargs="*", type=Path, help="Vídeos sueltos (p. ej. cámara virtual) en vez del split EFL")
    args = ap.parse_args()

    net = load_model()
    videos = args.videos or [LABELS_ROOT / g["video"] / "720p.mp4"
                             for g in json.loads((TDEED_SPLITS / f"{args.split}.json").read_text())]
    for video in videos:
        game = video
        out = video.with_name(f"{video.stem}.tdeed.{MODEL_NAME}.npz")
        if out.exists():
            continue
        t0 = time.perf_counter()
        probs = score_video(net, video)
        n = len(probs)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, probs=probs.astype(np.float16), classes=np.array(["background"] + CLASSES),
                 stride=STRIDE, fps=25.0)
        tmp.rename(out)
        logger.info("%s: %d filas en %.0fs", game, n, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
