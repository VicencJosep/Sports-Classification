"""SoccerNet Tracking (train.zip / test.zip, clips de TV a 1080p con el balón anotado) -> dataset YOLO de una clase.

Las cajas del balón salen de ``gt/gt.txt`` (formato MOT: frame, id, x, y, w, h, ...) tomando los tracklets que
``gameinfo.ini`` etiqueta como ``ball``. Las imágenes no se copian: ``images/{train,val}`` son enlaces simbólicos a
los jpg extraídos (Ultralytics localiza la etiqueta por la ruta del enlace, así que funciona).

Partición: train.zip (57 clips) -> train, 1 de cada ``--train-stride`` frames; test.zip (49 clips) -> val, 1 de cada
``--val-stride``. Son partidos distintos de los nuestros, así que el val es limpio. Idempotente: no re-extrae ni
re-escribe lo que ya existe.

Uso (contenedor, cwd /app): python scripts/ball_detector/prepare_soccernet_tracking_ball.py
Salida: datasets/soccernet_tracking_ball/{images,labels}/{train,val}, data.yaml, stats.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import zipfile
from pathlib import Path

ZIP_DIR = Path("datasets/SoccerNet/tracking")
OUT = Path("datasets/soccernet_tracking_ball")


def extract(split: str) -> Path:
    dest = ZIP_DIR / "extracted"
    marker = dest / f".done_{split}"
    if marker.exists():
        return dest / split
    dest.mkdir(parents=True, exist_ok=True)
    print(f"extrayendo {split}.zip ...", flush=True)
    with zipfile.ZipFile(ZIP_DIR / f"{split}.zip") as z:
        z.extractall(dest)
    marker.touch()
    return dest / split


def ball_ids(gameinfo: Path) -> set[int]:
    ids: set[int] = set()
    for line in gameinfo.read_text(errors="replace").splitlines():
        if line.lower().startswith("trackletid_") and "=" in line:
            key, val = line.split("=", 1)
            if "ball" in val.lower():
                ids.add(int(key.split("_", 1)[1].strip()))
    return ids


def seq_size(seqinfo: Path) -> tuple[int, int]:
    w = h = None
    for line in seqinfo.read_text(errors="replace").splitlines():
        if line.startswith("imWidth"):
            w = int(line.split("=")[1])
        elif line.startswith("imHeight"):
            h = int(line.split("=")[1])
    return w or 1920, h or 1080


def convert(split_dir: Path, out_split: str, stride: int, stats: dict) -> None:
    img_out = OUT / "images" / out_split
    lbl_out = OUT / "labels" / out_split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    n_img = n_with_ball = n_seq_ball = 0
    sizes_w: list[float] = []
    sizes_h: list[float] = []
    for seq in seqs:
        ids = ball_ids(seq / "gameinfo.ini")
        W, H = seq_size(seq / "seqinfo.ini")
        boxes: dict[int, list[tuple[float, float, float, float]]] = {}
        for line in (seq / "gt" / "gt.txt").read_text().splitlines():
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame, tid = int(parts[0]), int(parts[1])
            if tid not in ids:
                continue
            x, y, w, h = (float(v) for v in parts[2:6])
            boxes.setdefault(frame, []).append((x, y, w, h))
        if ids and boxes:
            n_seq_ball += 1
        imgs = sorted((seq / "img1").glob("*.jpg"))
        for img in imgs:
            frame = int(img.stem)
            if frame % stride:
                continue
            name = f"{seq.name}_{img.stem}"
            link = img_out / f"{name}.jpg"
            if not link.exists():
                link.symlink_to(img.resolve())
            rows = []
            for x, y, w, h in boxes.get(frame, []):
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                if 0 <= cx <= 1 and 0 <= cy <= 1 and w > 0 and h > 0:
                    rows.append(f"0 {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
                    sizes_w.append(w)
                    sizes_h.append(h)
            (lbl_out / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            n_img += 1
            n_with_ball += bool(rows)
    stats[out_split] = {
        "sequences": len(seqs), "sequences_with_ball": n_seq_ball, "images": n_img, "images_with_ball": n_with_ball,
        "ball_w_px_median": statistics.median(sizes_w) if sizes_w else None,
        "ball_h_px_median": statistics.median(sizes_h) if sizes_h else None,
    }
    print(out_split, json.dumps(stats[out_split]), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-stride", type=int, default=3)
    ap.add_argument("--val-stride", type=int, default=15)
    a = ap.parse_args()
    stats: dict = {}
    convert(extract("train"), "train", a.train_stride, stats)
    convert(extract("test"), "val", a.val_stride, stats)
    if stats["train"]["sequences_with_ball"] < 20:
        raise SystemExit(f"Solo {stats['train']['sequences_with_ball']} clips de train con balón: revisa el parseo de gameinfo.ini")
    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n"
    )
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print("data.yaml escrito en", OUT / "data.yaml")


if __name__ == "__main__":
    main()
