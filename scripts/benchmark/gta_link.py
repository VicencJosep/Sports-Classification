"""GTA-Link (Sun et al., ACCV-W 2024, arXiv 2411.08216; MIT) como
post-proceso offline de unos tracks MOT (pensado para los de Deep-EIoU).

Repo original clonado en ``models/gta_link/repo`` (el mismo código que usa
GTATrack, ganador de SoccerTrack 2025). Se importan ``Tracklet.py`` y
``refine_tracklets.py`` TAL CUAL; solo se calzan ``seaborn`` (únicamente lo usa
un heatmap de depuración) y ``loguru`` (logging estándar), que no están en el
contenedor.

Pasos, como sus dos scripts:
1. ``generate_tracklets.py``: por cada fila del track, recorte PIL RGB
   ``img.crop((l, t, l + w, t + h))`` -> Resize(256, 128) -> Normalize ImageNet
   -> OSNet x1_0 "sports" (el de Deep-EIoU) -> embedding L2-normalizado.
2. ``refine_tracklets.py``: split por DBSCAN + connect por distancia coseno
   media con restricción espacial. Parámetros por defecto = los del README de
   gta-link (min_len 100, eps 0,6, min_samples 10, max_k 3, spatial 1,0,
   merge_dist 0,4).

Los embeddings por fila se cachean en ``--emb-dir`` (float16).

Uso:
    python -m scripts.benchmark.gta_link --seqs datasets/SoccerNet/tracking/extracted/test \\
        --tracks output/benchmark/snmot_test/tracks/oracle__deep_eiou \\
        --emb-dir output/benchmark/snmot_test/emb/gta_oracle__deep_eiou \\
        --out output/benchmark/snmot_test/tracks/oracle__deep_eiou_gta
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import logging
import sys
import time
import types
from pathlib import Path

import numpy as np

from scripts.benchmark.mot import list_sequences, read_mot, seq_info
from scripts.benchmark.track_deep_eiou import build_osnet

logger = logging.getLogger(__name__)

GTA_REPO = Path(__file__).resolve().parents[2] / "models" / "gta_link" / "repo"


def _import_gta():
    sys.modules.setdefault("seaborn", types.ModuleType("seaborn"))
    if "loguru" not in sys.modules:
        sys.modules["loguru"] = types.SimpleNamespace(logger=logging.getLogger("gta_link"))
    if str(GTA_REPO) not in sys.path:
        sys.path.insert(0, str(GTA_REPO))
    spec = importlib.util.spec_from_file_location("gta_refine", GTA_REPO / "refine_tracklets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from Tracklet import Tracklet  # noqa: E402

    return mod, Tracklet


class _TrackCrops:
    """Recortes por frame como ``generate_tracklets.py`` (PIL, RGB)."""

    def __init__(self, info: dict, rows_by_frame: dict[int, np.ndarray]) -> None:
        import torchvision.transforms as T

        self.info, self.rows = info, rows_by_frame
        self.frames = sorted(rows_by_frame)
        self.prep = T.Compose([T.Resize([256, 128]), T.ToTensor(),
                               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int):
        import torch
        from PIL import Image

        f = self.frames[i]
        img = Image.open(self.info["img_dir"] / f"{f:06d}{self.info['img_ext']}")
        return torch.stack([self.prep(img.crop((l, t, l + w, t + h)).convert("RGB"))
                            for l, t, w, h in self.rows[f]])


def embed_rows(model, info: dict, tr: np.ndarray, device: str, batch: int, workers: int) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    frames = tr[:, 0].astype(int)
    ds = _TrackCrops(info, {int(f): tr[frames == f, 2:6] for f in np.unique(frames)})
    dl = DataLoader(ds, batch_size=None, num_workers=workers, prefetch_factor=4 if workers else None)
    feats = []
    with torch.no_grad():
        for crops in dl:
            x = crops.to(device)
            feats.append(torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)]).float().cpu().numpy())
    emb = np.concatenate(feats) if feats else np.zeros((0, 512), np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-len", type=int, default=100)
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--max-k", type=int, default=3)
    ap.add_argument("--spatial-factor", type=float, default=1.0)
    ap.add_argument("--merge-dist-thres", type=float, default=0.4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    gta, Tracklet = _import_gta()
    model = None
    args.out.mkdir(parents=True, exist_ok=True)
    args.emb_dir.mkdir(parents=True, exist_ok=True)
    timing = []
    for seq_dir in list_sequences(args.seqs):
        tr_path = args.tracks / f"{seq_dir.name}.txt"
        if not tr_path.exists():
            continue
        info = seq_info(seq_dir)
        tr = read_mot(tr_path)
        tr = tr[np.lexsort((tr[:, 1], tr[:, 0]))]  # frame, id (generate_tracklets recorre por frame)
        cache = args.emb_dir / f"{seq_dir.name}.npz"
        t0 = time.perf_counter()
        if cache.exists() and np.array_equal((z := np.load(cache))["rows"], tr[:, :6].astype(np.float32)):
            emb = z["emb"].astype(np.float32)
        else:
            if model is None:
                model = build_osnet(args.device)
            emb = embed_rows(model, info, tr, args.device, args.batch, args.workers)
            np.savez(cache, rows=tr[:, :6].astype(np.float32), emb=emb.astype(np.float16))
            emb = emb.astype(np.float16).astype(np.float32)
        t_emb = time.perf_counter() - t0

        # Tracklets como generate_tracklets.py (clave = id float, como np.genfromtxt)
        tracks: dict = {}
        for (frame, tid, l, t, w, h, score), feat in zip(tr[:, :7], emb):
            if tid not in tracks:
                tracks[tid] = Tracklet(tid, frame, score, [l, t, w, h])
            else:
                tracks[tid].append_det(frame, score, [l, t, w, h])
            tracks[tid].append_feat(feat)

        t0 = time.perf_counter()
        n_in = len(tracks)  # split/merge modifican los dicts en sitio: contar antes
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            max_x, max_y = gta.get_spatial_constraints(tracks, args.spatial_factor)
            split = gta.split_tracklets(tracks, eps=args.eps, max_k=args.max_k,
                                        min_samples=args.min_samples, len_thres=args.min_len)
            n_split = len(split)
            dist = gta.get_distance_matrix(split)
            merged = gta.merge_tracklets(split, {}, dist, seq_name=seq_dir.name, max_x_range=max_x,
                                         max_y_range=max_y, merge_dist_thres=args.merge_dist_thres)
            gta.save_results(str(args.out / f"{seq_dir.name}.txt"), merged)
        t_ref = time.perf_counter() - t0
        logger.info("%s: %d -> %d -> %d tracklets, emb %.1fs, refine %.1fs",
                    seq_dir.name, n_in, n_split, len(merged), t_emb, t_ref)
        timing.append({"seq": seq_dir.name, "tracklets_in": n_in, "after_split": n_split,
                       "after_merge": len(merged), "emb_seconds": round(t_emb, 2), "refine_seconds": round(t_ref, 2)})
    (args.out / "timing.json").write_text(json.dumps({"params": {k: v for k, v in vars(args).items()
                                                                  if k not in ("seqs", "tracks", "emb_dir", "out")},
                                                      "seqs": timing}, indent=1, default=str))


if __name__ == "__main__":
    main()
