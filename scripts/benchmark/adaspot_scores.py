"""Caché de puntuaciones de AdaSpot (Xarles et al., arXiv 2602.22073; repo
``models/adaspot/repo``, pesos de HF ``arturxe/AdaSpot`` en
``models/adaspot/weights``) en el formato de ``core.ball_spotting``
(``720p.tdeed.<modelo>.npz``, ``probs`` [n, 1 + 10]) para ``eval_bas.py``.

Inferencia como su ``inference.py`` (clips de 100 frames muestreados, solape
del 50 %, 1 de cada 2 frames, softmax por frame y media por soporte), salvo el
tamaño: su ``inference.py`` pasa ``(448, 796)`` a ``cv2.resize``, que espera
(ancho, alto); aquí se usa 796x448, que es con lo que se entrenó.

Solo 10 clases (sin FREE KICK ni GOAL): compárese con ``--classes10``.

Uso (contenedor):
    python -m scripts.benchmark.adaspot_scores --model SoccerNetBall_big --split test
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "models" / "adaspot" / "repo"
WEIGHTS = ROOT / "models" / "adaspot" / "weights"
TDEED_SPLITS = ROOT / "models" / "tdeed" / "T-DEED" / "data" / "soccernetball"
LABELS_ROOT = ROOT / "datasets" / "SoccerNet"

logger = logging.getLogger(__name__)


def load(model_name: str):
    sys.path.insert(0, str(REPO))
    with contextlib.chdir(REPO):
        import inference as ada_inf  # update_args / dict_to_namespace del repo
        from dataset.frame import ActionSpotInferenceDataset
        from model.model import AdaSpot
        from util.dataset import load_classes
        from util.io import load_json

        config = load_json(f"config/SoccerNetBall/{model_name}.json")
        args = ada_inf.update_args(argparse.Namespace(model_name=model_name, seed=1), config)
        classes = load_classes("data/soccernetball/class.txt")
    model = AdaSpot(args_model=args.model, args_training=args.training, classes=classes, elements=None)
    model.load(torch.load(WEIGHTS / f"{model_name}-1" / "checkpoint_best.pt", map_location="cuda", weights_only=False))
    model.clean_modules()
    names = ["background"] + [k for k, _ in sorted(classes.items(), key=lambda kv: kv[1])]
    return model, args, names, ActionSpotInferenceDataset


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="SoccerNetBall_big", choices=["SoccerNetBall_big", "SoccerNetBall_small"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--videos", nargs="*", type=Path, help="Vídeos 796x448 sueltos (p. ej. cámara virtual) en vez del split EFL")
    args = ap.parse_args()

    model, cfg, names, Dataset = load(args.model)
    tag = "AdaSpot_" + args.model.split("_")[1]
    videos = args.videos or [LABELS_ROOT / g["video"] / "720p.mp4"
                             for g in json.loads((TDEED_SPLITS / f"{args.split}.json").read_text())]
    for video in videos:
        game = video
        out = video.with_name(f"{video.stem}.tdeed.{tag}.npz")
        if out.exists():
            continue
        t0 = time.perf_counter()
        ds = Dataset(str(video), clip_len=cfg.data.clip_len, overlap_len=cfg.data.clip_len // 2, stride=2,
                     dataset="soccernetball", size=(cfg.model.hr_dim[1], cfg.model.hr_dim[0]))
        n = ds._video_len // 2
        acc = np.zeros((n, len(names)), np.float32)
        support = np.zeros(n, np.int32)
        for frames, starts in DataLoader(ds, batch_size=cfg.training.batch_size, num_workers=2, pin_memory=True):
            _, pred = model.predict(frames)
            for p, start in zip(pred, starts.tolist()):
                if start < 0:
                    p, start = p[-start:], 0
                end = min(start + len(p), n)
                acc[start:end] += p[: end - start]
                support[start:end] += 1
        probs = acc / np.maximum(support, 1)[:, None]
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, probs=probs.astype(np.float16), classes=np.array(names), stride=2, fps=25.0)
        tmp.rename(out)
        logger.info("%s %s: %d filas en %.0fs", tag, game, n, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
