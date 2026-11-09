"""Modelo de SoccerHigh (Díaz-Juan, Ballester y Haro, MMSports 2025) para localizar los tramos de una retransmisión
que entrarían en el resumen oficial. Reimplementación en torch puro del baseline del repo IPCV/SoccerHigh:

1. **Rasgos**: VideoMAEv2-Giant (K710) sobre un clip de 16 fotogramas a 2 fps (224x224, recorte central) centrado en
   cada fotograma -> un vector de 1408 por cada 0,5 s (``extract_features``; caché ``<vídeo>.soccerhigh.<backbone>.npy``).
2. **Resumidor** (``SoccerHighSummarizer``): proyección 1408->256, codificación posicional sinusoidal, 2 capas de
   transformer (8 cabezas) y cabeza anchor-free (DSNet-AF): confianza por fotograma y desplazamientos (inicio, fin).
   Pesos publicados: ``Baseline_VideoMAEv2-Giant.ckpt`` (F1 0,3956 en su test), aquí ``soccerhigh_baseline_giant.pt``.
3. **Inferencia** (``predict``): ventanas de 60 s sin solape, fotogramas con confianza > 0,5 proponen el tramo
   [i - d_ini, i + d_fin] y una NMS sin solape se queda con los de más confianza. ``select_to_length`` es la selección
   "@T" del paper: los mejores tramos hasta la duración del resumen.

Licencias: modelo y datos de SoccerHigh CC BY-NC-SA 4.0 (I+D y evaluación); VideoMAEv2 CC BY-NC 4.0.

    python -m analytics.highlights.soccerhigh --video partido.mp4 --out output/soccerhigh/partido
    python -m analytics.highlights.soccerhigh --features 1_HQ_224p_VideoMAEv2_Giant_K710_1408.npy --out dir
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FPS = 2                                      # fotogramas por segundo de la rejilla de SoccerHigh
CLIP_LEN = 16                                # fotogramas por clip de VideoMAEv2
IMG = 224
FEATURE_DIM = 1408
BACKBONE_NAME = "VideoMAEv2_Giant_K710"
MODELS_DIR = Path("models/soccerhigh")
DEFAULT_BACKBONE = MODELS_DIR / "vit_g_hybrid_pt_1200e_k710_ft.pth"
DEFAULT_CHECKPOINT = MODELS_DIR / "soccerhigh_baseline_giant.pt"


# --------------------------------------------------------------------------- #
#  Rasgos
# --------------------------------------------------------------------------- #

def features_cache_path(video: Path, backbone: str = BACKBONE_NAME) -> Path:
    """``partido.mp4`` -> ``partido.soccerhigh.VideoMAEv2_Giant_K710.npy`` (junto al vídeo, como la caché de T-DEED)."""
    video = Path(video)
    return video.with_name(f"{video.stem}.soccerhigh.{backbone}.npy")


def iter_frames(video: Path, fps: int = FPS, size: int = IMG, start_s: float = 0.0,
                duration_s: Optional[float] = None) -> Iterator[np.ndarray]:
    """Fotogramas RGB ``size x size`` a ``fps`` (lado corto a ``size`` y recorte central), vía ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(video)]
    if duration_s is not None:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += ["-vf", f"fps={fps},scale=-2:{size}:flags=bicubic,crop={size}:{size}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 7)
    n = size * size * 3
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(size, size, 3)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace").strip()
        proc.wait()
        if proc.returncode not in (0, None) and err:
            logger.warning("ffmpeg (%s): %s", video, err[-300:])


def _frame_reader(video: Path, q: "queue.Queue", **kw) -> None:
    try:
        for f in iter_frames(video, **kw):
            q.put(f)
    finally:
        q.put(None)


@torch.no_grad()
def extract_features(
    video: Path, backbone_path: Path | str = DEFAULT_BACKBONE, batch_size: int = 8, device: str = "cuda",
    dtype: torch.dtype = torch.float16, progress=None, centered: bool = True, **reader_kw,
) -> np.ndarray:
    """``[n_frames, 1408]`` float16: un rasgo por fotograma a 2 fps. El clip del fotograma ``i`` son los 16
    fotogramas ``[i-8, i+7]`` (``centered``) o ``[i-15, i]``, replicando el primero/último en los bordes."""
    from .videomaev2 import load_giant_k710, normalize_clip

    model = load_giant_k710(backbone_path, device=device, dtype=dtype)
    before = CLIP_LEN // 2 if centered else CLIP_LEN - 1
    after = CLIP_LEN - 1 - before
    q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=256)
    threading.Thread(target=_frame_reader, args=(video, q), kwargs=reader_kw, daemon=True).start()

    frames: list[np.ndarray] = []          # ventana deslizante de fotogramas pendientes
    feats: list[np.ndarray] = []
    done = False
    t0 = time.time()
    n_out = 0

    def flush(batch: list[np.ndarray]) -> None:
        nonlocal n_out
        x = normalize_clip(np.stack(batch), device, dtype)
        feats.append(model(x).float().cpu().numpy().astype(np.float16))
        n_out += len(batch)
        if progress and n_out % (batch_size * 50) < batch_size:
            progress(n_out, time.time() - t0)

    batch: list[np.ndarray] = []
    first_pad = True
    while not done:
        f = q.get()
        if f is None:
            done = True
            if frames:
                frames.extend([frames[-1]] * after)      # borde derecho: se replica el último
        else:
            if first_pad:
                frames.extend([f] * before)             # borde izquierdo: se replica el primero
                first_pad = False
            frames.append(f)
        # Cada vez que hay CLIP_LEN fotogramas en la ventana sale el rasgo del fotograma `before`-ésimo
        while len(frames) >= CLIP_LEN:
            batch.append(np.stack(frames[:CLIP_LEN]))
            frames.pop(0)
            if len(batch) == batch_size:
                flush(batch)
                batch = []
    if batch:
        flush(batch)
    del model
    torch.cuda.empty_cache()
    out = np.concatenate(feats) if feats else np.zeros((0, FEATURE_DIM), np.float16)
    logger.info("SoccerHigh: %d rasgos de %s en %.0f s", len(out), video, time.time() - t0)
    return out


def ensure_features(video: Path, backbone_path: Path | str = DEFAULT_BACKBONE, batch_size: int = 8,
                    progress=None, force: bool = False) -> np.ndarray:
    path = features_cache_path(video)
    if path.exists() and not force:
        return np.load(path, mmap_mode="r")
    feats = extract_features(video, backbone_path, batch_size=batch_size, progress=progress)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, feats)
    tmp.replace(path)
    return feats


# --------------------------------------------------------------------------- #
#  Resumidor
# --------------------------------------------------------------------------- #

class PositionalEncoding1D(nn.Module):
    """Codificación sinusoidal (seno y coseno intercalados) del repo de SoccerHigh; ``inv_freq`` va en el checkpoint."""

    def __init__(self, channels: int):
        super().__init__()
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type())
        sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1).flatten(-2, -1)
        return emb[None, :, : x.shape[2]].type(x.type()).expand(x.shape[0], -1, -1)


class TransformerEncoder(nn.Module):
    def __init__(self, ndim: int, nlayers: int, nheads: int, dropout: float):
        super().__init__()
        self.pos_encoder = PositionalEncoding1D(ndim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=ndim, nhead=nheads, dropout=dropout, batch_first=True),
            num_layers=nlayers, norm=nn.LayerNorm(ndim), enable_nested_tensor=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x + self.pos_encoder(x))


class DSNetAFHead(nn.Module):
    def __init__(self, ndim: int, num_classes: int = 1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(ndim)
        self.fc = nn.Sequential(nn.Linear(ndim, ndim), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.LayerNorm(ndim))
        self.cls = nn.Linear(ndim, num_classes)
        self.loc = nn.Linear(ndim, 2 * num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.fc(self.layer_norm(x))
        return self.cls(x), self.loc(x)


class SoccerHighSummarizer(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, ndim: int = 256, nlayers: int = 2, nheads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim_reduction = nn.Linear(input_dim, ndim)
        self.encoder = TransformerEncoder(ndim, nlayers, nheads, dropout)
        self.head = DSNetAFHead(ndim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` [B, N, input_dim] -> (logits [B, N, 1], desplazamientos [B, N, 2] en fotogramas)."""
        return self.head(self.encoder(self.dim_reduction(x)))

    @classmethod
    def from_checkpoint(cls, path: Path | str = DEFAULT_CHECKPOINT, device: str = "cpu") -> "SoccerHighSummarizer":
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        model = cls()
        model.load_state_dict(sd, strict=True)
        return model.to(device).eval()


# --------------------------------------------------------------------------- #
#  Inferencia
# --------------------------------------------------------------------------- #

@dataclass
class Segment:
    start: int          # fotograma (2 fps), inclusive
    end: int            # fotograma, inclusive
    score: float        # confianza del fotograma que lo propuso

    @property
    def n_frames(self) -> int:
        return self.end - self.start + 1

    def to_dict(self, fps: int = FPS) -> dict:
        return {"start_s": self.start / fps, "end_s": (self.end + 1) / fps, "score": round(self.score, 4),
                "start_frame": self.start, "end_frame": self.end}


@torch.no_grad()
def predict_frames(model: SoccerHighSummarizer, features: np.ndarray, window: int = 60 * FPS,
                   batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Ventanas de ``window`` fotogramas sin solape (la última se rellena con el último fotograma):
    ``(prob [n], offsets [n, 2])``."""
    n = len(features)
    if n == 0:
        return np.zeros(0, np.float32), np.zeros((0, 2), np.float32)
    device = next(model.parameters()).device
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    pad = (-n) % window
    if pad:
        x = torch.cat([x, x[-1:].expand(pad, -1)])
    x = x.view(-1, window, x.shape[-1])
    probs, locs = [], []
    for i in range(0, len(x), batch_size):
        cls, loc = model(x[i:i + batch_size].to(device))
        probs.append(torch.sigmoid(cls).squeeze(-1).cpu())
        locs.append(loc.cpu())
    prob = torch.cat(probs).reshape(-1).numpy()[:n]
    offsets = torch.cat(locs).reshape(-1, 2).numpy()[:n]
    return prob, offsets


def nms(segments: list[Segment], iou_threshold: float = 0.0) -> list[Segment]:
    """Como en SoccerHigh: por confianza descendente, fuera todo tramo que solape (IoU > umbral) con uno elegido."""
    kept: list[Segment] = []
    for s in sorted(segments, key=lambda s: -s.score):
        ok = True
        for k in kept:
            inter = min(s.end, k.end) - max(s.start, k.start)
            if inter > 0:
                union = max(s.end, k.end) - min(s.start, k.start)
                if inter / max(union, 1e-8) > iou_threshold:
                    ok = False
                    break
        if ok:
            kept.append(s)
    return kept


def segments_from_frames(prob: np.ndarray, offsets: np.ndarray, threshold: float = 0.5, n_frames: Optional[int] = None) -> list[Segment]:
    n_frames = len(prob) if n_frames is None else n_frames
    props = []
    for i in np.where(prob > threshold)[0]:
        start = int(round(i - offsets[i, 0]))
        end = int(round(i + offsets[i, 1]))
        if end > start:
            props.append(Segment(max(start, 0), min(end, n_frames - 1), float(prob[i])))
    return sorted(nms(props), key=lambda s: s.start)


def predict(model: SoccerHighSummarizer, features: np.ndarray, threshold: float = 0.5, window_s: float = 60.0) -> tuple[list[Segment], np.ndarray]:
    prob, offsets = predict_frames(model, features, window=int(round(window_s * FPS)))
    return segments_from_frames(prob, offsets, threshold), prob


def select_to_length(segments: list[Segment], total_frames: int) -> list[Segment]:
    """Selección "@T" del paper: tramos por confianza descendente mientras quepan en ``total_frames`` (sin recortar)."""
    chosen, used = [], 0
    for s in sorted(segments, key=lambda s: -s.score):
        if used + s.n_frames <= total_frames:
            chosen.append(s)
            used += s.n_frames
    return sorted(chosen, key=lambda s: s.start)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def run_on_video(video: Path, out_dir: Path, checkpoint: Path | str = DEFAULT_CHECKPOINT,
                 backbone: Path | str = DEFAULT_BACKBONE, threshold: float = 0.5, batch_size: int = 8,
                 progress=None, force: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = ensure_features(Path(video), backbone, batch_size=batch_size, progress=progress, force=force)
    model = SoccerHighSummarizer.from_checkpoint(checkpoint)
    segs, prob = predict(model, feats, threshold)
    np.save(out_dir / "soccerhigh_scores.npy", prob.astype(np.float32))
    report = {
        "video": str(video), "backbone": BACKBONE_NAME, "checkpoint": str(checkpoint), "fps": FPS,
        "threshold": threshold, "n_frames": int(len(prob)),
        "segments": [s.to_dict() for s in segs],
        "total_s": round(sum(s.n_frames for s in segs) / FPS, 1),
    }
    with open(out_dir / "soccerhigh.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path)
    p.add_argument("--features", type=Path, help=".npy ya extraído (p.ej. los de SoccerHigh en Hugging Face)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--backbone", default=str(DEFAULT_BACKBONE))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--force", action="store_true", help="volver a extraer los rasgos aunque haya caché")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.features:
        feats = np.load(args.features, mmap_mode="r")
        model = SoccerHighSummarizer.from_checkpoint(args.checkpoint)
        segs, prob = predict(model, feats, args.threshold)
        args.out.mkdir(parents=True, exist_ok=True)
        report = {"features": str(args.features), "n_frames": int(len(prob)), "segments": [s.to_dict() for s in segs],
                  "total_s": round(sum(s.n_frames for s in segs) / FPS, 1)}
        with open(args.out / "soccerhigh.json", "w") as f:
            json.dump(report, f, indent=2)
    elif args.video:
        def progress(n, elapsed):
            print(f"  {n} fotogramas ({n / FPS / 60:.1f} min de vídeo) en {elapsed:.0f} s", flush=True)
        report = run_on_video(args.video, args.out, args.checkpoint, args.backbone, args.threshold, args.batch_size,
                              progress, args.force)
    else:
        p.error("indica --video o --features")
    print(f"{len(report['segments'])} tramos, {report['total_s']} s en total")
    for s in report["segments"]:
        print(f"  {s['start_s']:8.1f} - {s['end_s']:8.1f}  ({s['end_s'] - s['start_s']:4.1f} s)  {s['score']:.2f}")


if __name__ == "__main__":
    main()
