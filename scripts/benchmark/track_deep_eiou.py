"""Deep-EIoU (Huang et al., WACVW 2024, arXiv 2306.13074) en el benchmark de
tracking, sobre las MISMAS detecciones que ``track_mot.py``.

Repo original (sin licencia: solo benchmark interno) clonado en
``models/deep_eiou/repo`` y ReID OSNet x1_0 "sports" en
``models/deep_eiou/weights/sports_model.pth.tar-60``. Se importa su tracker
(``tracker/Deep_EIoU.py``) y su ``osnet.py`` TAL CUAL, con dos calzos para no
instalar nada: ``cython_bbox.bbox_overlaps`` reimplementado en numpy (con el +1
de píxel del original) y ``np.float`` (quitado en numpy >= 1.24) = ``float``.
``lap`` ya está en el contenedor.

Réplica de ``tools/sport_track.py`` (SportsMOT) + ``tools/demo.py`` (ReID):
* parámetros por defecto: track_high 0,6, track_low 0,1, new_track 0,7,
  track_buffer 60 con ``frame_rate=30`` fijo (60 frames), match 0,8,
  proximity 0,5, appearance 0,25, EIoU 0,7 -> 0,8 en dos iteraciones, ReID on;
* salida: ``last_tlwh`` de cada track vivo con área > 10 px²;
* ReID: recorte ``frame[int(y1):int(y2), int(x1):int(x2)]`` del frame cv2,
  ``ToPILImage`` -> ``Resize((256, 128))`` -> ``ToTensor`` -> ``Normalize``
  ImageNet, como ``torchreid.utils.FeatureExtractor``. El demo le pasa recortes
  BGR (cv2) a un modelo entrenado en RGB; se respeta por defecto y
  ``--rgb`` da la variante corregida;
* opcional (``--dti-out``): su interpolación ``tools/sport_interpolation.py``
  (n_min 5, n_dti 20, distancia 200 px).

Entrada: el mismo filtro que los trackers de ultralytics en ``track_mot.py``
(sin balón, conf >= ``UL_MIN_CONF``, área >= ``UL_MIN_AREA``). No se aplica el
"remove edge detection" del demo (``sport_track.py`` tampoco lo hace).

Los embeddings se cachean por secuencia en ``--emb-dir`` (float16, con las
cajas para comprobar que casan), así que repetir el tracking es barato.

Uso:
    python -m scripts.benchmark.track_deep_eiou --seqs datasets/SoccerNet/tracking/extracted/test \\
        --dets gt --emb-dir output/benchmark/snmot_test/emb/oracle \\
        --out output/benchmark/snmot_test/tracks/oracle__deep_eiou \\
        --dti-out output/benchmark/snmot_test/tracks/oracle__deep_eiou_dti
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

import cv2
import numpy as np

from scripts.benchmark.mot import BALL_CLASS_ID, gt_as_detections, list_sequences, read_mot, seq_info, write_mot
from scripts.benchmark.track_mot import UL_MIN_AREA, UL_MIN_CONF

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "models" / "deep_eiou" / "repo" / "Deep-EIoU"
WEIGHTS = ROOT / "models" / "deep_eiou" / "weights" / "sports_model.pth.tar-60"


# --------------------------------------------------------------------------- calzos
def _bbox_overlaps(boxes: np.ndarray, query: np.ndarray) -> np.ndarray:
    """``cython_bbox.bbox_overlaps`` en numpy (áreas e intersección con +1)."""
    boxes, query = np.asarray(boxes, dtype=float), np.asarray(query, dtype=float)
    if len(boxes) == 0 or len(query) == 0:
        return np.zeros((len(boxes), len(query)))
    area_b = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    area_q = (query[:, 2] - query[:, 0] + 1) * (query[:, 3] - query[:, 1] + 1)
    iw = np.minimum(boxes[:, None, 2], query[None, :, 2]) - np.maximum(boxes[:, None, 0], query[None, :, 0]) + 1
    ih = np.minimum(boxes[:, None, 3], query[None, :, 3]) - np.maximum(boxes[:, None, 1], query[None, :, 1]) + 1
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    ua = area_b[:, None] + area_q[None, :] - inter
    return np.where(inter > 0, inter / ua, 0.0)


def _import_repo():
    """Módulos ``tracker.Deep_EIoU`` y ``sport_interpolation`` del repo original."""
    if "float" not in np.__dict__:
        np.float = float  # type: ignore[attr-defined]  # el repo usa np.float (numpy < 1.24)
    sys.modules.setdefault("cython_bbox", types.SimpleNamespace(bbox_overlaps=_bbox_overlaps))
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tracker.Deep_EIoU import Deep_EIoU  # noqa: E402

    spec = importlib.util.spec_from_file_location("deep_eiou_interp", REPO / "tools" / "sport_interpolation.py")
    interp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interp)
    interp.args = types.SimpleNamespace(distance_thres=200)  # dti() lee el global ``args``
    return Deep_EIoU, interp


def default_args() -> types.SimpleNamespace:
    """Defaults de ``tools/sport_track.py`` que usa ``Deep_EIoU``."""
    return types.SimpleNamespace(
        track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=0.7, track_buffer=60,
        match_thresh=0.8, aspect_ratio_thresh=1.6, min_box_area=10, with_reid=True,
        proximity_thresh=0.5, appearance_thresh=0.25,
    )


# --------------------------------------------------------------------------- ReID
def build_osnet(device: str):
    import torch

    spec = importlib.util.spec_from_file_location("deep_eiou_osnet", REPO / "reid" / "torchreid" / "models" / "osnet.py")
    osnet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(osnet)
    model = osnet.osnet_x1_0(num_classes=1, pretrained=False)  # como FeatureExtractor
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    own = model.state_dict()
    # torchreid.utils.load_pretrained_weights: descarta 'module.' y capas que no casan (el clasificador)
    matched = {k.removeprefix("module."): v for k, v in state.items()
               if k.removeprefix("module.") in own and own[k.removeprefix("module.")].shape == v.shape}
    own.update(matched)
    model.load_state_dict(own)
    logger.info("OSNet: %d/%d capas cargadas", len(matched), len(state))
    return model.eval().to(device)


class _FrameCrops:
    """Dataset de frames: recortes ya preprocesados como FeatureExtractor."""

    def __init__(self, info: dict, by_frame: dict[int, np.ndarray], rgb: bool) -> None:
        import torchvision.transforms as T

        self.info, self.by_frame, self.rgb = info, by_frame, rgb
        self.frames = sorted(by_frame)
        self.to_pil = T.ToPILImage()
        self.prep = T.Compose([T.Resize((256, 128)), T.ToTensor(),
                               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int):
        import torch

        frame_no = self.frames[i]
        img = cv2.imread(str(self.info["img_dir"] / f"{frame_no:06d}{self.info['img_ext']}"))
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        crops = []
        for x1, y1, x2, y2 in self.by_frame[frame_no]:
            ya, yb, xa, xb = max(0, int(y1)), min(h, int(y2)), max(0, int(x1)), min(w, int(x2))
            if yb <= ya or xb <= xa:  # caja degenerada: 1 px (el original fallaría)
                yb, xb = min(h, ya + 1), min(w, xa + 1)
                ya, xa = yb - 1, xb - 1
            crops.append(self.prep(self.to_pil(np.ascontiguousarray(img[ya:yb, xa:xb]))))
        return torch.stack(crops)


def compute_embeddings(model, info: dict, xyxy_by_frame: dict[int, np.ndarray], rgb: bool,
                       device: str, batch: int, workers: int) -> dict[int, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    ds = _FrameCrops(info, xyxy_by_frame, rgb)
    dl = DataLoader(ds, batch_size=None, num_workers=workers, prefetch_factor=4 if workers else None)
    out: dict[int, np.ndarray] = {}
    pend_frames, pend = [], []

    def flush():
        x = torch.cat(pend).to(device)
        with torch.no_grad():
            f = torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)]).float().cpu().numpy()
        k = 0
        for fr, n in pend_frames:
            out[fr] = f[k:k + n]
            k += n
        pend_frames.clear()
        pend.clear()

    for frame_no, crops in zip(ds.frames, dl):
        pend_frames.append((frame_no, len(crops)))
        pend.append(crops)
        if sum(n for _, n in pend_frames) >= batch:
            flush()
    if pend:
        flush()
    return out


# --------------------------------------------------------------------------- tracking
def load_dets(dets_src: str, seq_dir: Path) -> np.ndarray | None:
    if dets_src == "gt":
        dets = gt_as_detections(seq_dir)
    else:
        path = Path(dets_src) / f"{seq_dir.name}.txt"
        if not path.exists():
            return None
        dets = read_mot(path)
    keep = (dets[:, 7] != BALL_CLASS_ID) & (dets[:, 6] >= UL_MIN_CONF) & (dets[:, 4] * dets[:, 5] >= UL_MIN_AREA)
    dets = dets[keep]
    return dets[np.argsort(dets[:, 0], kind="stable")]


def run_deep_eiou(Deep_EIoU, dets: np.ndarray, embs: np.ndarray, length: int, params) -> list[tuple]:
    tracker = Deep_EIoU(params, frame_rate=30)  # sport_track.py fija 30
    rows: list[tuple] = []
    for frame_no in range(1, length + 1):
        m = dets[:, 0] == frame_no
        r = dets[m]
        det = np.column_stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5], r[:, 6]])
        for t in tracker.update(det, embs[m].astype(np.float32)):
            x, y, w, h = t.last_tlwh
            if w * h > params.min_box_area:
                rows.append((frame_no, t.track_id, x, y, w, h, float(t.score), -1, -1))
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", type=Path, required=True)
    ap.add_argument("--dets", required=True, help="Carpeta de detecciones o 'gt' (oráculo)")
    ap.add_argument("--emb-dir", type=Path, required=True, help="Caché de embeddings ReID por secuencia")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dti-out", type=Path, default=None, help="Además, tracks con su interpolación")
    ap.add_argument("--rgb", action="store_true", help="Recortes en RGB (el demo original pasa BGR)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    Deep_EIoU, interp = _import_repo()
    params = default_args()
    model = None
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dti_out:
        args.dti_out.mkdir(parents=True, exist_ok=True)
    emb_dir = args.emb_dir / ("rgb" if args.rgb else "bgr")
    emb_dir.mkdir(parents=True, exist_ok=True)
    timing = []
    for seq_dir in list_sequences(args.seqs):
        info = seq_info(seq_dir)
        dets = load_dets(args.dets, seq_dir)
        if dets is None:
            logger.warning("Sin detecciones para %s, se salta", seq_dir.name)
            continue
        xyxy = np.column_stack([dets[:, 2], dets[:, 3], dets[:, 2] + dets[:, 4], dets[:, 3] + dets[:, 5]])
        cache = emb_dir / f"{seq_dir.name}.npz"
        t_emb = 0.0
        if cache.exists() and np.array_equal((z := np.load(cache))["xyxy"], xyxy.astype(np.float32)):
            embs = z["emb"]
        else:
            if model is None:
                model = build_osnet(args.device)
            t0 = time.perf_counter()
            frames = dets[:, 0].astype(int)
            by_frame = {int(f): xyxy[frames == f] for f in np.unique(frames)}
            per_frame = compute_embeddings(model, info, by_frame, args.rgb, args.device, args.batch, args.workers)
            embs = np.concatenate([per_frame[int(f)] for f in np.unique(frames)]) if len(dets) else np.zeros((0, 512))
            t_emb = time.perf_counter() - t0
            np.savez(cache, xyxy=xyxy.astype(np.float32), emb=embs.astype(np.float16))
            embs = embs.astype(np.float16)  # misma precisión que desde la caché
        t0 = time.perf_counter()
        rows = run_deep_eiou(Deep_EIoU, dets, embs, info["length"], params)
        t_trk = time.perf_counter() - t0
        out_path = args.out / f"{seq_dir.name}.txt"
        write_mot(out_path, rows)
        if args.dti_out:
            with contextlib.redirect_stdout(io.StringIO()):
                interp.dti(str(out_path), str(args.dti_out / f"{seq_dir.name}.txt"), n_min=5, n_dti=20)
        timing.append({"seq": seq_dir.name, "n_dets": int(len(dets)), "emb_seconds": round(t_emb, 2),
                       "track_seconds": round(t_trk, 2), "track_fps": round(info["length"] / t_trk, 1)})
        logger.info("%s: %d dets, emb %.1fs, tracking %.1fs", seq_dir.name, len(dets), t_emb, t_trk)
    (args.out / "timing.json").write_text(json.dumps(timing, indent=1))
    logger.info("%d secuencias -> %s", len(timing), args.out)


if __name__ == "__main__":
    main()
