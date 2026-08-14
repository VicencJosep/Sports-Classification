"""Fine-tuning de R3D-18 (Kinetics-400 preentrenado) sobre los clips extraídos
por ``scripts/extract_ball_action_clips.py`` (``dataset_ball_actions/<clase>/*.mp4``),
para el clasificador de ``core/ball_actions.py`` (pass/drive/high_pass/header/throw_in/background).

Split train/val por PARTIDO (no por clip): el nombre de fichero que genera el
script de extracción es ``<match_dir_name>_h{half}_f{frame}.mp4``, así que el
prefijo antes de ``_h`` identifica el partido. Split por partido evita fuga de
información entre clips del mismo partido/misma jugada (dos clips separados
por 1s de la misma jugada no deberían acabar uno en train y otro en val).

Desbalanceo de clases (Pass/Drive dominan frente a Throw In/Header, ver README): por defecto
usa ``WeightedRandomSampler`` en el DataLoader de train; con ``--balance loss``
usa en su lugar ``CrossEntropyLoss(weight=...)`` sin resampling.

Uso:
    python -m training.train_ball_actions --data-dir dataset_ball_actions/ \\
        --epochs 15 --batch-size 8 --checkpoint models/r3d18_ball_actions.pt

    # Smoke test rápido (no espera buenas métricas, solo que el loop corra):
    python -m training.train_ball_actions --data-dir dataset_ball_actions/ \\
        --epochs 1 --limit-per-class 5 --checkpoint /tmp/smoke.pt
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.video import R3D_18_Weights, r3d_18

from core.ball_actions import DEFAULT_NUM_FRAMES, ensure_num_frames, read_clip_frames

logger = logging.getLogger(__name__)

DEFAULT_CLASSES = ("pass", "drive", "high_pass", "header", "throw_in", "background")
# "tackle"/"cross"/"ball_player_block" quedan fuera en esta primera fase
# (clases minoritarias o con riesgo de confusión visual, ver
# scripts/extract_ball_action_clips.py::BALL_LABEL_TO_CLASS); añadir de
# vuelta aquí + reactivar su extracción cuando haya suficientes ejemplos.


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #

@dataclass
class ClipSample:
    path: Path
    label_idx: int
    match_id: str  # prefijo antes de "_h{half}_f{frame}" -> partido de origen


def _match_id_from_filename(path: Path) -> str:
    stem = path.stem  # "<match>_h1_f12345"
    if "_h" in stem:
        return stem.rsplit("_h", 1)[0]
    return stem


def discover_samples(data_dir: Path, classes: tuple[str, ...]) -> list[ClipSample]:
    samples = []
    for label_idx, cls in enumerate(classes):
        class_dir = data_dir / cls
        if not class_dir.exists():
            logger.warning("Carpeta de clase ausente (0 clips): %s", class_dir)
            continue
        for clip_path in sorted(class_dir.glob("*.mp4")):
            samples.append(ClipSample(clip_path, label_idx, _match_id_from_filename(clip_path)))
    return samples


def split_by_match(
    samples: list[ClipSample], val_fraction: float, seed: int
) -> tuple[list[ClipSample], list[ClipSample]]:
    match_ids = sorted({s.match_id for s in samples})
    rng = random.Random(seed)
    rng.shuffle(match_ids)
    n_val_matches = max(1, round(len(match_ids) * val_fraction)) if len(match_ids) > 1 else 0
    val_matches = set(match_ids[:n_val_matches])
    logger.info("Partidos en validación: %s", val_matches)

    train = [s for s in samples if s.match_id not in val_matches]
    val = [s for s in samples if s.match_id in val_matches]
    if not val:
        logger.warning(
            "Split por partido no produjo val set (¿solo hay 1 partido en el dataset?); "
            "las métricas de val no serán fiables hasta tener más partidos."
        )
    return train, val


class BallActionsDataset(Dataset):
    """Lee un clip .mp4 ya recortado (ver extract_ball_action_clips.py) y aplica
    el preprocesado estándar de R3D-18/Kinetics-400 (``weights.transforms()``)."""

    def __init__(self, samples: list[ClipSample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        frames = ensure_num_frames(read_clip_frames(sample.path), DEFAULT_NUM_FRAMES)
        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)  # (T, C, H, W)
        clip = self.transform(video)  # -> (C, T, H, W) normalizado, listo para r3d_18
        return clip, sample.label_idx


# --------------------------------------------------------------------------- #
#  Entrenamiento
# --------------------------------------------------------------------------- #

def build_model(num_classes: int) -> nn.Module:
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def class_weights(samples: list[ClipSample], num_classes: int) -> torch.Tensor:
    counts = Counter(s.label_idx for s in samples)
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def make_train_loader(
    train_samples: list[ClipSample],
    transform,
    batch_size: int,
    num_classes: int,
    balance: str,
) -> DataLoader:
    dataset = BallActionsDataset(train_samples, transform)
    if balance == "sampler":
        weights = class_weights(train_samples, num_classes)
        sample_weights = [weights[s.label_idx].item() for s in train_samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def evaluate(model: nn.Module, loader: DataLoader, device: str, num_classes: int) -> dict:
    model.eval()
    correct = 0
    total = 0
    per_class_correct = Counter()
    per_class_total = Counter()
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            preds = model(clips).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for label, pred in zip(labels.tolist(), preds.tolist()):
                per_class_total[label] += 1
                if label == pred:
                    per_class_correct[label] += 1
    per_class_acc = {
        i: (per_class_correct[i] / per_class_total[i] if per_class_total[i] else None)
        for i in range(num_classes)
    }
    return {"accuracy": correct / total if total else 0.0, "per_class_accuracy": per_class_acc}


def train(
    data_dir: Path,
    checkpoint_path: Path,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-4,
    val_fraction: float = 0.2,
    balance: str = "sampler",
    limit_per_class: Optional[int] = None,
    device: Optional[str] = None,
    seed: int = 0,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = R3D_18_Weights.KINETICS400_V1
    transform = weights.transforms()

    samples = discover_samples(data_dir, classes)
    if not samples:
        raise SystemExit(f"No se encontraron clips .mp4 en {data_dir} (¿corriste extract_ball_action_clips.py?)")

    if limit_per_class is not None:
        by_class: dict[int, list[ClipSample]] = {}
        for s in samples:
            by_class.setdefault(s.label_idx, []).append(s)
        rng = random.Random(seed)
        samples = [s for group in by_class.values() for s in rng.sample(group, min(limit_per_class, len(group)))]

    train_samples, val_samples = split_by_match(samples, val_fraction, seed)
    logger.info(
        "Dataset: %d clips train, %d clips val (%d clases). Distribución train: %s",
        len(train_samples), len(val_samples), len(classes),
        Counter(classes[s.label_idx] for s in train_samples),
    )

    train_loader = make_train_loader(train_samples, transform, batch_size, len(classes), balance)
    val_loader = (
        DataLoader(BallActionsDataset(val_samples, transform), batch_size=batch_size)
        if val_samples else None
    )

    model = build_model(len(classes)).to(device)
    loss_weight = class_weights(train_samples, len(classes)).to(device) if balance == "loss" else None
    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(clips), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        avg_loss = running_loss / len(train_samples)
        msg = f"[epoch {epoch}/{epochs}] train_loss={avg_loss:.4f}"
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, len(classes))
            msg += f" val_acc={metrics['accuracy']:.3f}"
        logger.info(msg)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "classes": list(classes)}, checkpoint_path)
    logger.info("Checkpoint guardado en %s", checkpoint_path)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, type=Path, help="dataset_ball_actions/ (salida de extract_ball_action_clips.py)")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Ruta de salida del .pt entrenado")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--balance", choices=["sampler", "loss", "none"], default="sampler")
    parser.add_argument("--limit-per-class", type=int, default=None, help="Smoke test: limita clips por clase")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    train(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        balance=args.balance,
        limit_per_class=args.limit_per_class,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
