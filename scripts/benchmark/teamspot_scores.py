"""Caché de puntuaciones del baseline de SoccerNet Team Ball Action Spotting
2025 (T-DEED RegNetY-200MF + cabeza de equipo, repo ``models/sn-teamspotting/
repo``, pesos de su Drive en ``models/sn-teamspotting/weights``), en el mismo
formato que ``core.ball_spotting`` para evaluarla con ``eval_bas.py``:

    720p.tdeed.SNTeam_baseline.npz
        probs       [n, 13]  fondo + 12 acciones (izquierda + derecha sumadas)
        probs_team  [n, 25]  fondo + 12 acciones x {left, right} intercalados
        classes, stride, fps

Mismo muestreo que nuestro T-DEED (796x448, 1 de cada 2 frames, clips de 100
con solape del 75 %) y ``model.predict`` del repo (desplazamiento + equipo).
Se ejecuta aparte de ``core.ball_spotting`` porque los dos repos tienen
paquetes ``model``/``util``/``dataset`` con el mismo nombre.

Uso (contenedor):
    python -m scripts.benchmark.teamspot_scores --split test
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "models" / "sn-teamspotting" / "repo"
CHECKPOINT = ROOT / "models" / "sn-teamspotting" / "weights" / "SoccerNetBall_baseline" / "checkpoint_best.pt"
TDEED_SPLITS = ROOT / "models" / "tdeed" / "T-DEED" / "data" / "soccernetball"
LABELS_ROOT = ROOT / "datasets" / "SoccerNet"
MODEL_NAME = "SNTeam_baseline"
FRAME_SIZE, STRIDE, CLIP_LEN, PAD_LEN = (796, 448), 2, 100, 5

logger = logging.getLogger(__name__)


def load_model():
    sys.path.insert(0, str(REPO))
    with contextlib.chdir(REPO):
        from model.model import TDEEDModel
        from util.io import load_json

        cfg = load_json(os.path.join("config", "SoccerNetBall", "SoccerNetBall_baseline.json"))
    args = argparse.Namespace(**cfg)
    args.crop_dim = None
    model = TDEEDModel(args=args)
    # Dos cabezas (BAS 12+1 y SoccerNet-v2 17+1) como en train_tdeed_bas.py con event_team.
    model._model.update_pred_head([13, 18])
    model._num_classes = 31
    model.load(torch.load(CHECKPOINT, map_location="cuda", weights_only=False))
    classes = [line.strip() for line in (REPO / "data" / "soccernetball" / "class.txt").read_text().splitlines() if line.strip()]
    return model, classes


def clips(video: Path):
    """Igual que ``ActionSpotInferenceDataset`` del T-DEED vendorizado (relleno
    inicial de ``PAD_LEN`` frames negros), devolviendo (clip uint8 [L,3,H,W], inicio)."""
    stream = cv2.VideoCapture(str(video))
    buf: deque = deque()
    i = -PAD_LEN * STRIDE
    black = torch.zeros((3, FRAME_SIZE[1], FRAME_SIZE[0]), dtype=torch.uint8)
    while True:
        if i < 0:
            if i % STRIDE == 0:
                buf.append(black)
            i += 1
            continue
        ok, frame = stream.read()
        if not ok:
            break
        if i % STRIDE == 0:
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), FRAME_SIZE)
            buf.append(torch.from_numpy(frame).permute(2, 0, 1))
            if len(buf) == CLIP_LEN:
                yield torch.stack(list(buf)), (i + 1 + STRIDE - 1) // STRIDE - CLIP_LEN
                for _ in range(CLIP_LEN // 4):
                    buf.popleft()
        i += 1
    stream.release()


def scan(model, video: Path, n_cls_team: int) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // STRIDE
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    acc = np.zeros((n, n_cls_team), np.float32)
    support = np.zeros(n, np.int32)
    batch, starts = [], []

    def flush():
        _, pred = model.predict(torch.stack(batch))
        for p, start in zip(pred, starts):
            lo = -start if start < 0 else 0
            s0 = max(start, 0)
            end = min(s0 + p.shape[0] - lo, n)
            acc[s0:end] += p[lo: lo + end - s0]
            support[s0:end] += 1
        batch.clear()
        starts.clear()

    for clip, start in clips(video):
        batch.append(clip)
        starts.append(start)
        if len(batch) == 4:
            flush()
    if batch:
        flush()
    return acc / np.maximum(support, 1)[:, None], fps


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--videos", nargs="*", type=Path, help="Vídeos 796x448 sueltos (p. ej. cámara virtual) en vez del split EFL")
    args = ap.parse_args()

    model, classes = load_model()
    videos = args.videos or [LABELS_ROOT / g["video"] / "720p.mp4"
                             for g in json.loads((TDEED_SPLITS / f"{args.split}.json").read_text())]
    for video in videos:
        game = video
        out = video.with_name(f"{video.stem}.tdeed.{MODEL_NAME}.npz")
        if out.exists():
            continue
        t0 = time.perf_counter()
        team, fps = scan(model, video, 1 + 2 * len(classes))
        probs = np.concatenate([team[:, :1], team[:, 1::2] + team[:, 2::2]], axis=1)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, probs=probs.astype(np.float16), probs_team=team.astype(np.float16),
                 classes=np.array(["background"] + classes), stride=STRIDE, fps=fps)
        tmp.rename(out)
        logger.info("%s: %d filas en %.0fs", game, len(probs), time.perf_counter() - t0)


if __name__ == "__main__":
    main()
