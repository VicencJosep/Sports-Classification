"""Caché de puntuaciones de AdaSpot (Xarles et al., arXiv 2602.22073; repo
``models/adaspot/repo``, pesos de HF ``arturxe/AdaSpot`` en
``models/adaspot/weights``) en el formato de ``core.ball_spotting``
(``720p.tdeed.<modelo>.npz``, ``probs`` [n, 1 + 10]) para ``eval_bas.py``.

Inferencia como su ``inference.py`` (clips de 100 frames muestreados, solape
del 50 %, 1 de cada 2 frames, softmax por frame y media por soporte), salvo el
tamaño: su ``inference.py`` pasa ``(448, 796)`` a ``cv2.resize``, que espera
(ancho, alto); aquí se usa 796x448, que es con lo que se entrenó.

Solo 10 clases (sin FREE KICK ni GOAL): compárese con ``--classes10``.

``--use_gt_roi`` (o ``--roi gt center det``) cambia de dónde sale la región de la rama de alta resolución: en vez del
selector de saliencia, una caja de 112x112 centrada en coordenadas anotadas (``scripts/benchmark/adaspot_gt_roi.py``).
Cada variante se guarda con su sufijo (``AdaSpot_big_gtroi``, ``_ctrroi``, ``_detroi``) y todas las pedidas se puntúan
en una sola lectura del vídeo, que es lo que más tarda. La caché guarda además los centros de RoI por fila
(``roi_used``, ``roi_gt``, ``roi_saliency``: (y, x) normalizado) para ``adaspot_roi_report.py``.

Uso (contenedor):
    python -m scripts.benchmark.adaspot_scores --model SoccerNetBall_big --split test
    python -m scripts.benchmark.adaspot_scores --model SoccerNetBall_big --use_gt_roi --roi gt center \
        --videos datasets/SoccerNet/pcbas-2026/videos/game_18.mp4
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

from scripts.benchmark.adaspot_gt_roi import TAGS, RoIOverride, with_rois

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "models" / "adaspot" / "repo"
WEIGHTS = ROOT / "models" / "adaspot" / "weights"
TDEED_SPLITS = ROOT / "models" / "tdeed" / "T-DEED" / "data" / "soccernetball"
LABELS_ROOT = ROOT / "datasets" / "SoccerNet"

logger = logging.getLogger(__name__)


def load(model_name: str, device: str = "cuda"):
    """Carga el modelo en ``device`` (``cuda`` o ``cpu``; en CPU se sustituye el ``torch.cuda.FloatTensor`` que GSM
    usa como ayudante para que no cree un contexto CUDA)."""
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
    if device == "cpu":
        from model.impl import gsm  # noqa: E402  (repo de AdaSpot)
        gsm.ftens = torch.FloatTensor
    model = AdaSpot(device=torch.device(device), args_model=args.model, args_training=args.training, classes=classes,
                    elements=None)
    model.load(torch.load(WEIGHTS / f"{model_name}-1" / "checkpoint_best.pt", map_location=device, weights_only=False))
    model.clean_modules()
    names = ["background"] + [k for k, _ in sorted(classes.items(), key=lambda kv: kv[1])]
    return model, args, names, ActionSpotInferenceDataset


def _window(start: int, length: int, n: int) -> tuple[slice, slice]:
    """Filas de la caché y del clip que se solapan (el primer clip empieza en el relleno: start < 0)."""
    a, b = max(start, 0), min(start + length, n)
    return slice(a, b), slice(a - start, b - start)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="SoccerNetBall_big", choices=["SoccerNetBall_big", "SoccerNetBall_small"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--videos", nargs="*", type=Path, help="Vídeos 796x448 sueltos (p. ej. cámara virtual) en vez del split EFL")
    ap.add_argument("--use_gt_roi", action="store_true",
                    help="RoI centrada en las coordenadas anotadas del dataset en vez de la saliencia (= --roi gt)")
    ap.add_argument("--roi", nargs="+", choices=list(TAGS), help="Variantes de RoI a puntuar en la misma pasada: saliency "
                    "(original), gt (anotación), center (control), det (pseudo-GT de nuestro detector de balón), gtfeet "
                    "(anotación a la altura de los pies) y fix (saliencia corregida para contener al ejecutor)")
    ap.add_argument("--batch-size", type=int, help="Clips por lote (por defecto el del config, 4: ~7 GB de VRAM; con 2, ~4 GB)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="cpu: sin GPU (lento; ver --threads)")
    ap.add_argument("--threads", type=int, help="Hilos de torch en CPU (por defecto todos los núcleos)")
    ap.add_argument("--no-log-saliency", action="store_true",
                    help="No calcula la saliencia en los frames que ya tienen RoI dada (por defecto se guarda para comparar)")
    args = ap.parse_args()
    modes = list(args.roi or ["saliency"])
    if args.use_gt_roi and "gt" not in modes:
        modes = ["gt"] + [m for m in modes if m != "saliency" or args.roi]

    if args.threads:
        torch.set_num_threads(args.threads)
    model, cfg, names, Dataset = load(args.model, args.device)
    selector = model._model.roi_selector = RoIOverride(model._model.roi_selector, log_saliency=not args.no_log_saliency)
    Dataset = with_rois(Dataset)
    tag = "AdaSpot_" + args.model.split("_")[1]
    videos = args.videos or [LABELS_ROOT / g["video"] / "720p.mp4"
                             for g in json.loads((TDEED_SPLITS / f"{args.split}.json").read_text())]
    for video in videos:
        outs = {m: video.with_name(f"{video.stem}.tdeed.{tag}{TAGS[m]}.npz") for m in modes}
        todo = [m for m in modes if not outs[m].exists()]
        if not todo:
            continue
        t0 = time.perf_counter()
        ds = Dataset(str(video), clip_len=cfg.data.clip_len, overlap_len=cfg.data.clip_len // 2, stride=2,
                     dataset="soccernetball", size=(cfg.model.hr_dim[1], cfg.model.hr_dim[0]), modes=todo)
        n = ds._video_len // 2
        acc = {m: np.zeros((n, len(names)), np.float32) for m in todo}
        used = {m: np.zeros((n, 2), np.float32) for m in todo}
        saliency, n_saliency, support = np.zeros((n, 2), np.float32), np.zeros(n, np.int32), np.zeros(n, np.int32)
        # Un solo worker: el dataset es iterable y no se reparte, así que cada worker de más repetiría todos los clips.
        for frames, starts, rois in DataLoader(ds, batch_size=args.batch_size or cfg.training.batch_size, num_workers=1,
                                          pin_memory=args.device == "cuda"):
            frames = frames.to(args.device, non_blocking=True)
            selector.new_batch()
            pred, centers = {}, {}
            for m in todo:
                selector.mode, selector.given = m, rois.get(m)
                pred[m] = model.predict(frames, use_amp=args.device == "cuda")[1]
                centers[m] = selector.used.float().cpu().numpy()
            sal = None if selector.saliency is None else selector.saliency[0].float().cpu().numpy()
            for i, start in enumerate(starts.tolist()):
                rows, clip = _window(start, cfg.data.clip_len, n)
                support[rows] += 1
                for m in todo:
                    acc[m][rows] += pred[m][i][clip]
                    used[m][rows] += centers[m][i][clip]
                if sal is not None:
                    saliency[rows] += sal[i][clip]
                    n_saliency[rows] += 1
        div = np.maximum(support, 1)[:, None]
        for m in todo:
            extra = {"roi_used": used[m] / div}
            if m in ds.tracks:
                extra["roi_gt"] = ds.tracks[m][:n]
            if n_saliency.any():
                extra["roi_saliency"] = saliency / np.maximum(n_saliency, 1)[:, None]
            tmp = outs[m].with_name(outs[m].stem + ".tmp.npz")
            np.savez(tmp, probs=(acc[m] / div).astype(np.float16), classes=np.array(names), stride=2, fps=25.0, **extra)
            tmp.rename(outs[m])
        cover = {m: f"{100 * np.isfinite(t[:n, 0]).mean():.0f} %" for m, t in ds.tracks.items()}
        logger.info("%s %s %s: %d filas en %.0fs; frames con RoI dada %s", tag, todo, video, n, time.perf_counter() - t0, cover)


if __name__ == "__main__":
    main()
