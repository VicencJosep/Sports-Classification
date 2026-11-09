"""Desfase entre los vídeos HQ de SoccerHigh y nuestros 720p (``video.ini`` no es público): se alinean los rasgos
VideoMAEv2 extraídos de nuestro vídeo con los de SoccerHigh (Hugging Face) buscando el desplazamiento que maximiza la
similitud coseno media. Escribe ``offsets.json`` en la carpeta del partido: ``t_HQ = t_720p + offset`` por mitad.

    docker exec -w /app Sports-Classification python scripts/soccerhigh_align.py --game "spain_laliga/2016-2017/2017-04-23 - 21-45 Real Madrid 2 - 3 Barcelona"
    python scripts/soccerhigh_align.py --ours a.npy --theirs b.npy          # un par de ficheros sueltos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.highlights.soccerhigh import FPS, ensure_features, features_cache_path  # noqa: E402

ROOT = Path("datasets/soccerhigh")
HF_FEATURES = "HQ_224p_VideoMAEv2_Giant_K710_1408.npy"


def best_lag(ours: np.ndarray, theirs: np.ndarray, max_lag_s: float = 900.0, ours_start_s: float = 0.0,
             device: str = "cuda") -> dict:
    """Desplazamiento ``lag`` (fotogramas) tal que ours[i] ~ theirs[i + lag]; devuelve lag, similitud media en el
    pico, similitud mediana fuera del pico y el segundo mejor (para ver si el pico es nítido)."""
    a = torch.nn.functional.normalize(torch.from_numpy(np.asarray(ours, np.float32)).to(device), dim=1)
    b = torch.nn.functional.normalize(torch.from_numpy(np.asarray(theirs, np.float32)).to(device), dim=1)
    a = a - a.mean(0, keepdim=True)          # sin la componente común (todos los frames de fútbol se parecen)
    b = b - b.mean(0, keepdim=True)
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    base = int(round(ours_start_s * FPS))
    max_lag = int(max_lag_s * FPS)
    lags = list(range(max(-len(a) + 50, base - max_lag // 4), base + max_lag))
    sims = []
    for lag in lags:
        lo_a, hi_a = max(0, -lag), min(len(a), len(b) - lag)
        if hi_a - lo_a < 50:
            sims.append(-1.0)
            continue
        sims.append(float((a[lo_a:hi_a] * b[lo_a + lag: hi_a + lag]).sum(1).mean()))
    sims = np.array(sims)
    k = int(sims.argmax())
    order = np.argsort(-sims)
    second = next((sims[j] for j in order if abs(lags[j] - lags[k]) > 4), float("nan"))
    return {"lag_frames": lags[k], "offset_s": lags[k] / FPS, "peak_sim": float(sims[k]),
            "second_sim": float(second), "median_sim": float(np.median(sims))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game")
    p.add_argument("--ours", type=Path)
    p.add_argument("--theirs", type=Path)
    p.add_argument("--ours-start-s", type=float, default=0.0, help="segundo del vídeo en que empieza --ours")
    p.add_argument("--max-lag-s", type=float, default=900.0)
    args = p.parse_args()
    if args.ours and args.theirs:
        print(best_lag(np.load(args.ours), np.load(args.theirs), args.max_lag_s, args.ours_start_s))
        return
    game_dir = ROOT / "videos" / args.game
    out = {}
    for h in (1, 2):
        video = game_dir / f"{h}_720p.mkv"
        ours = ensure_features(video, progress=lambda n, e: print(f"  mitad {h}: {n} fotogramas en {e:.0f} s", flush=True))
        theirs = np.load(ROOT / args.game / f"{h}_{HF_FEATURES}", mmap_mode="r")
        r = best_lag(ours, theirs, args.max_lag_s)
        r["n_ours"], r["n_theirs"] = int(len(ours)), int(len(theirs))
        out[str(h)] = r["offset_s"]
        out[f"{h}_detail"] = r
        print(f"mitad {h}: offset {r['offset_s']:.1f} s (pico {r['peak_sim']:.3f}, segundo {r['second_sim']:.3f}, "
              f"mediana {r['median_sim']:.3f}); {len(ours)} vs {len(theirs)} fotogramas")
    (game_dir / "offsets.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
