"""Segunda etapa ligera sobre spotters congelados: una red convolucional temporal pequeña que lee las curvas de probabilidad
de uno o varios modelos (cachés ``<vídeo>.tdeed.<modelo>.npz``) en una ventana de ±5 s y corrige, clase a clase, su
fusión por media. Arranca siendo la fusión (capa de salida a cero) y aprende un residuo sobre sus logits. CPU, minutos.

Solo se entrena con partidos que los spotters no usaron para entrenar (sus probabilidades ahí son las de uso real):

  --exp pcbas-logo  FOOTPASS val dejando un partido fuera: entrena con 2 y predice el tercero (8 clases).
  --exp efl         Entrena con Middlesbrough (validación de SN-BAS) y predice Stoke y Reading (10 clases). Con
                    ``--extra-pcbas`` añade los 3 partidos de FOOTPASS solo para las clases que se anotan igual en los dos
                    datasets (centro, saque de banda, tiro, cabezazo, entrada, bloqueo).

El último 15 % de cada partido de entrenamiento decide cuándo parar. Escribe ``<vídeo>.tdeed.<nombre>.npz``, que
``eval_bas.py`` y ``eval_prf.py`` leen como un modelo más.

Uso (contenedor):
    python -m scripts.benchmark.stack_scores --exp pcbas-logo --name Stack_logo
    python -m scripts.benchmark.eval_bas --model Stack_logo --dataset pcbas --merge-high-pass
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from core.ball_spotting import read_scores
from scripts.benchmark.eval_bas import LABELS_ROOT, OUT_ROOT_PCBAS, PCBAS_CLASSES, merge_high_pass, pcbas_games, tdeed_eval_classes
from scripts.benchmark.fuse_scores import efl_games, fuse, write_cache

CLASSES = tdeed_eval_classes()
BAS10 = tuple(c for c in CLASSES if c not in ("FREE KICK", "GOAL"))
SHARED = ("CROSS", "THROW IN", "SHOT", "HEADER", "PLAYER SUCCESSFUL TACKLE", "BALL PLAYER BLOCK")
SIGMA, RADIUS = 1.5, 4  # objetivo gaussiano en filas (12,5 fps)


def load_game(game: str, stem: Path, label_root: Path, models: list[str], train_classes, merged: bool, sn_head: bool) -> dict:
    sc = [read_scores(stem.with_name(f"{stem.name}.tdeed.{m}.npz")) for m in models]
    base, names = fuse(sc, np.full((len(sc), len(CLASSES)), 1 / len(sc)))
    if merged:
        base = merge_high_pass(base, names)
    n = len(base)
    feats = [s.probs[:n, 1:].astype(np.float32) for s in sc]
    if sn_head:
        feats.append(sc[0].sn_probs[:n, 1:].astype(np.float32))
    y = np.zeros((n, len(CLASSES)), np.float32)
    kernel = np.exp(-0.5 * (np.arange(-RADIUS, RADIUS + 1) / SIGMA) ** 2).astype(np.float32)
    rows_per_s = sc[0].fps / sc[0].stride
    for a in json.loads((label_root / game / "Labels-ball.json").read_text())["annotations"]:
        if a["label"] in CLASSES:
            r, j = int(round(float(a["position"]) / 1000 * rows_per_s)), CLASSES.index(a["label"])
            lo, hi = max(0, r - RADIUS), min(n, r + RADIUS + 1)
            if lo < hi:
                y[lo:hi, j] = np.maximum(y[lo:hi, j], kernel[lo - (r - RADIUS): hi - (r - RADIUS)])
    return {"game": game, "stem": stem, "x": np.sqrt(np.concatenate(feats, 1)), "y": y, "base": base[:, 1:],
            "mask": np.array([c in train_classes for c in CLASSES], np.float32), "stride": sc[0].stride, "fps": sc[0].fps}


class TCN(nn.Module):
    def __init__(self, f_in: int, hidden: int = 64, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.inp = nn.Conv1d(f_in, hidden, 1)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2 * d, dilation=d), nn.GELU(),
                                                  nn.Dropout(0.1)) for d in dilations)
        self.out = nn.Conv1d(hidden, len(CLASSES), 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for block in self.blocks:
            h = h + block(h)
        return torch.logit(base.clamp(1e-4, 1 - 1e-4)) + self.out(h)


def tensors(g: dict, lo: int = 0, hi: int | None = None):
    sl = slice(lo, hi)
    return tuple(torch.from_numpy(np.ascontiguousarray(g[k][sl].T)) for k in ("x", "base", "y"))


def loss_of(out: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per = F.binary_cross_entropy_with_logits(out, y, pos_weight=torch.tensor(5.0), reduction="none")
    return (per * mask).sum() / (mask.sum() * out.shape[-1] / mask.shape[-1]).clamp(min=1)


def train(games: list[dict], seed: int, steps: int, window: int = 1024, batch: int = 16) -> TCN:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = TCN(games[0]["x"].shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    cut = [int(0.85 * len(g["x"])) for g in games]
    weights = np.array(cut, float) / sum(cut)

    def val_loss() -> float:
        net.eval()
        tot = 0.0
        with torch.no_grad():
            for g, c in zip(games, cut):
                x, b, y = tensors(g)
                out = net(x[None], b[None])[..., c:]
                tot += float(loss_of(out, y[None, :, c:], torch.from_numpy(g["mask"])[None, :, None].expand_as(out)))
        net.train()
        return tot / len(games)

    best, best_state, bad = val_loss(), {k: v.clone() for k, v in net.state_dict().items()}, 0
    start = best
    for step in range(1, steps + 1):
        xs, bs, ys, ms = [], [], [], []
        for _ in range(batch):
            i = int(rng.choice(len(games), p=weights))
            lo = int(rng.integers(0, cut[i] - window))
            x, b, y = tensors(games[i], lo, lo + window)
            xs.append(x), bs.append(b), ys.append(y), ms.append(torch.from_numpy(games[i]["mask"])[:, None].expand_as(y))
        loss = loss_of(net(torch.stack(xs), torch.stack(bs)), torch.stack(ys), torch.stack(ms))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            v = val_loss()
            if v < best - 1e-5:
                best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 8:
                    break
    net.load_state_dict(best_state)
    print(f"    semilla {seed}: pérdida de validación {start:.4f} -> {best:.4f} ({step} pasos)", flush=True)
    return net.eval()


def predict(nets: list[TCN], g: dict) -> np.ndarray:
    x, b, _ = tensors(g)
    with torch.no_grad():
        p = torch.stack([torch.sigmoid(net(x[None], b[None]))[0] for net in nets]).mean(0).numpy().T
    probs = np.where(g["mask"][None] > 0, p, g["base"])  # las clases no entrenadas se quedan con la fusión
    return np.concatenate([np.clip(1 - probs.sum(1, keepdims=True), 0, 1), probs], 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, choices=["pcbas-logo", "efl"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", nargs="+", default=["SoccerNetBall_test", "AdaSpot_big"])
    ap.add_argument("--sn-head", action="store_true", help="Añade la cabeza de 17 eventos de SoccerNet-v2 del primer modelo")
    ap.add_argument("--extra-pcbas", action="store_true", help="efl: añade FOOTPASS val para las clases de definición común")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    torch.set_num_threads(8)

    pcbas = lambda classes, merged: [load_game(g, s, OUT_ROOT_PCBAS / "labels", args.models, classes, merged, args.sn_head)  # noqa: E731
                                     for g, s in pcbas_games("val")]
    names = ("background", *CLASSES)
    if args.exp == "pcbas-logo":
        games = pcbas(tuple(PCBAS_CLASSES.values()), True)
        for held in games:
            print(f"  fuera: {held['game']}", flush=True)
            nets = [train([g for g in games if g is not held], seed, args.steps) for seed in range(args.seeds)]
            print(f"  {write_cache(held['stem'], args.name, predict(nets, held), names, held['stride'], held['fps'])}")
    else:
        load = lambda split: [load_game(g, s, LABELS_ROOT, args.models, BAS10, False, args.sn_head) for g, s in efl_games(split)]  # noqa: E731
        train_games = load("val") + (pcbas(SHARED, False) if args.extra_pcbas else [])
        nets = [train(train_games, seed, args.steps) for seed in range(args.seeds)]
        for g in load("test"):
            print(f"  {write_cache(g['stem'], args.name, predict(nets, g), names, g['stride'], g['fps'])}")


if __name__ == "__main__":
    main()
