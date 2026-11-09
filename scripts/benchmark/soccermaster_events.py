"""Sonda del clasificador de eventos de SoccerMaster (arXiv 2512.11016; repo vendorizado en ``Soccer_Master/SoccerMaster``,
cabeza ``CaptionClassification.pt``: 23 clases, clip de 30 frames a 1 fps y 512x512, una etiqueta por clip, sin instante)
sobre los partidos del piloto de SoccerNet-v2 (``output/benchmark/sn_v2_pilot20.txt``, test oficial).

Dos modos:
  oracle  ventana de 30 s centrada en cada evento anotado (hasta ``--per-class`` por clase) + ventanas de fondo: exactitud
          de clasificación con el instante dado (el protocolo del paper).
  slide   ventana deslizante (paso ``--stride`` s) sobre partidos enteros: probabilidades por ventana, para medir
          falsas alarmas y compararlo como detector a tolerancia holgada.

Hay que lanzarlo con el repo como directorio de trabajo (sus YAML usan rutas relativas):
    docker exec -w /app/Soccer_Master/SoccerMaster Sports-Classification \
        python /app/scripts/benchmark/soccermaster_events.py oracle
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SN_ROOT = ROOT / "datasets" / "SoccerNet"
OUT = ROOT / "output" / "benchmark" / "soccermaster"
SIZE, T = 512, 30

# Clase de SoccerNet-v2 -> clases de SoccerMaster que cuentan como acierto.
SN_TO_SM = {
    "Penalty": {"penalty", "foul lead to penalty"}, "Kick-off": {"start of half game"}, "Goal": {"goal"},
    "Substitution": {"substitution"}, "Offside": {"off side"}, "Shots on target": {"saved by goal-keeper"},
    "Shots off target": {"shot off target"}, "Clearance": {"clearance"}, "Ball out of play": {"ball out of play"},
    "Throw-in": {"throw in"}, "Foul": {"foul with no card"}, "Indirect free-kick": {"free kick"},
    "Direct free-kick": {"free kick"}, "Corner": {"corner"}, "Yellow card": {"yellow card"}, "Red card": {"red card"},
    "Yellow->red card": {"second yellow card"},
}


def load_model():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, os.getcwd())
    from configs.util import load_super_config, yaml_to_dict
    from data.video_caption import keywords_list
    from models.multi_task import MultiTaskingModel

    cfg = yaml_to_dict("./configs/pretrain.yaml")
    cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))
    model = MultiTaskingModel(config=cfg)
    model.load_checkpoint("./pretrained_models/SoccerMaster", logger=None, load_heads=True)
    return model.eval().to("cuda"), list(keywords_list)


def ffmpeg_frames(video: Path, start: float | None = None, duration: float | None = None) -> np.ndarray:
    """Frames RGB a 1 fps y 512x512 (uint8, [n, 512, 512, 3])."""
    cmd = ["ffmpeg", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{max(start, 0):.3f}"]
    cmd += ["-i", str(video)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vf", f"fps=1,scale={SIZE}:{SIZE}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, SIZE, SIZE, 3)


def to_clip(frames: np.ndarray) -> torch.Tensor:
    if len(frames) < T:
        frames = np.concatenate([frames, np.repeat(frames[-1:], T - len(frames), 0)])
    x = torch.from_numpy(frames[:T].copy()).float().div(255).sub(0.5).div(0.5)
    return x.permute(0, 3, 1, 2)


@torch.no_grad()
def classify(model, clips: list[torch.Tensor]) -> np.ndarray:
    x = torch.stack(clips).to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, dataset_name="VideoCaption", text=["dummy"] * len(clips))
    return torch.softmax(out["CaptionClassification"]["logits"].float(), dim=-1).cpu().numpy()


def read_labels(game: str) -> list[dict]:
    ann = json.loads((SN_ROOT / game / "Labels-v2.json").read_text())["annotations"]
    return [{"label": a["label"], "half": int(a["gameTime"][0]), "t": int(a["position"]) / 1000,
             "visible": a.get("visibility") != "not shown"} for a in ann if a["label"] in SN_TO_SM]


def run_oracle(model, names, games, per_class: int, n_background: int, batch: int) -> None:
    rng = random.Random(0)
    items = []
    by_class: dict[str, list] = {}
    for g in games:
        ev = read_labels(g)
        for e in ev:
            by_class.setdefault(e["label"], []).append({**e, "game": g})
        for _ in range(max(1, n_background // len(games))):  # fondo: sin ningún evento a menos de 20 s
            for _try in range(50):
                half, t = rng.choice((1, 2)), rng.uniform(60, 2600)
                if all(e["half"] != half or abs(e["t"] - t) > 20 for e in ev):
                    items.append({"label": "background", "half": half, "t": t, "visible": True, "game": g})
                    break
    for label, lst in by_class.items():
        rng.shuffle(lst)
        items += lst[:per_class]
    print(f"{len(items)} ventanas ({ {k: min(len(v), per_class) for k, v in by_class.items()} })", flush=True)
    res, buf, meta, t_dec, t_inf = [], [], [], 0.0, 0.0

    def flush():
        nonlocal t_inf
        t0 = time.time()
        probs = classify(model, buf)
        t_inf += time.time() - t0
        for m, p in zip(meta, probs):
            res.append({**m, "probs": [round(float(v), 4) for v in p]})
        buf.clear()
        meta.clear()

    for i, it in enumerate(items):
        t0 = time.time()
        fr = ffmpeg_frames(SN_ROOT / it["game"] / f"{it['half']}_720p.mkv", it["t"] - T / 2, T)
        t_dec += time.time() - t0
        if not len(fr):
            continue
        buf.append(to_clip(fr))
        meta.append(it)
        if len(buf) == batch:
            flush()
        if i % 100 == 0:
            print(f"  {i}/{len(items)}  decodificar {t_dec:.0f}s  inferir {t_inf:.0f}s", flush=True)
    if buf:
        flush()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "oracle.json").write_text(json.dumps({"classes": names, "seconds_decode": t_dec, "seconds_infer": t_inf,
                                                 "results": res}))
    print(f"hecho: {len(res)} ventanas, decodificar {t_dec:.0f}s, inferir {t_inf:.0f}s", flush=True)


def run_slide(model, names, games, stride: float, batch: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for g in games:
        for half in (1, 2):
            out = OUT / f"slide_{g.replace('/', '__')}_{half}.npz"
            if out.exists():
                continue
            t0 = time.time()
            frames = ffmpeg_frames(SN_ROOT / g / f"{half}_720p.mkv")
            t_dec = time.time() - t0
            centers = np.arange(T / 2, len(frames) - T / 2 + 1e-6, stride)
            probs, t0 = [], time.time()
            for i in range(0, len(centers), batch):
                clips = [to_clip(frames[int(c - T / 2): int(c - T / 2) + T]) for c in centers[i:i + batch]]
                probs.append(classify(model, clips))
            np.savez(out, centers=centers, probs=np.concatenate(probs).astype(np.float32), classes=np.array(names))
            print(f"{g} parte {half}: {len(frames)} s, {len(centers)} ventanas, decodificar {t_dec:.0f}s, "
                  f"inferir {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["oracle", "slide"])
    ap.add_argument("--games", type=Path, default=ROOT / "output" / "benchmark" / "sn_v2_pilot20.txt")
    ap.add_argument("--only", nargs="*", help="slide: subcadenas de los partidos a procesar")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--background", type=int, default=120)
    ap.add_argument("--stride", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    games = [g.strip() for g in args.games.read_text().splitlines() if g.strip()]
    if args.only:
        games = [g for g in games if any(s in g for s in args.only)]
    model, names = load_model()
    if args.mode == "oracle":
        run_oracle(model, names, games, args.per_class, args.background, args.batch)
    else:
        run_slide(model, names, games, args.stride, args.batch)


if __name__ == "__main__":
    main()
