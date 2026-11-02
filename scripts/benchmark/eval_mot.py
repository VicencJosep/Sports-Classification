"""Paso 3 del benchmark de tracking: HOTA / CLEAR (MOTA) / Identity (IDF1) de
unos tracks MOT contra el GT, con las métricas de TrackEval vendorizadas en
``Soccer_Master/tracklab/trackeval`` (sin instalar nada).

Se importan solo los módulos de métricas: el ``__init__`` de esa copia
importa un ``datasets`` que no existe. La carga de datos es la de
MOTChallenge sin preprocesado de distractores (SoccerNet-Tracking y
SoccerTrack v2 tienen una sola clase y ninguna fila ignorada).

Modos (``--mode``):
* ``all``: todo el GT, balón incluido (como el GT oficial de SoccerNet-Tracking).
* ``persons``: quita del GT los tracklets de balón (``gameinfo.ini``) y de los
  tracks las filas con ``class_id`` de balón. Es el HOTA de jugadores del plan.

Comprobación del arnés: ``--tracks gt`` evalúa el GT contra sí mismo (HOTA = 1).

Uso:
    python -m scripts.benchmark.eval_mot --seqs datasets/SoccerNet/tracking/extracted/test \\
        --tracks output/benchmark/snmot_test/tracks/hybrid__ours --mode persons \\
        --out output/benchmark/snmot_test/results/hybrid__ours.persons.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np

from scripts.benchmark.mot import BALL_CLASS_ID, gt_roles, list_sequences, read_mot, seq_info

_TRACKEVAL_DIR = Path(__file__).resolve().parents[2] / "Soccer_Master" / "tracklab" / "trackeval"


def _load_metrics():
    """HOTA, CLEAR e Identity de la copia vendorizada, sin ejecutar sus __init__."""
    for name, path in (("trackeval", _TRACKEVAL_DIR), ("trackeval.metrics", _TRACKEVAL_DIR / "metrics")):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(path)]
            sys.modules[name] = pkg
    cfg = {"PRINT_CONFIG": False}
    return [
        importlib.import_module("trackeval.metrics.hota").HOTA(cfg),
        importlib.import_module("trackeval.metrics.clear").CLEAR(cfg),
        importlib.import_module("trackeval.metrics.identity").Identity(cfg),
    ]


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre cajas x,y,w,h (como trackeval._BaseDataset._calculate_box_ious)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.column_stack([a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]])
    b = np.column_stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def build_data(gt: np.ndarray, tr: np.ndarray, num_timesteps: int) -> dict:
    """Diccionario de entrada de las métricas de TrackEval (ids contiguos desde 0)."""
    gt_map = {v: i for i, v in enumerate(np.unique(gt[:, 1]).astype(int))}
    tr_map = {v: i for i, v in enumerate(np.unique(tr[:, 1]).astype(int))}
    data = {"num_timesteps": num_timesteps, "gt_ids": [], "tracker_ids": [], "similarity_scores": [],
            "num_gt_ids": len(gt_map), "num_tracker_ids": len(tr_map),
            "num_gt_dets": len(gt), "num_tracker_dets": len(tr)}
    for t in range(1, num_timesteps + 1):
        g, r = gt[gt[:, 0] == t], tr[tr[:, 0] == t]
        data["gt_ids"].append(np.array([gt_map[int(i)] for i in g[:, 1]], dtype=int))
        data["tracker_ids"].append(np.array([tr_map[int(i)] for i in r[:, 1]], dtype=int))
        data["similarity_scores"].append(_iou(g[:, 2:6], r[:, 2:6]))
    return data


def summarize(hota: dict, clear: dict, ident: dict) -> dict:
    return {
        "HOTA": float(np.mean(hota["HOTA"])), "DetA": float(np.mean(hota["DetA"])),
        "AssA": float(np.mean(hota["AssA"])), "LocA": float(np.mean(hota["LocA"])),
        "MOTA": float(clear["MOTA"]), "IDF1": float(ident["IDF1"]),
        "IDSW": int(clear["IDSW"]), "FP": int(clear["CLR_FP"]), "FN": int(clear["CLR_FN"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", type=Path, required=True)
    ap.add_argument("--tracks", required=True, help="Carpeta con <seq>.txt, o 'gt' (comprobación)")
    ap.add_argument("--mode", choices=["all", "persons"], default="persons")
    ap.add_argument("--gt-stride", type=int, default=1,
                    help="Evalúa 1 de cada N frames (FutTraffic usa 2 en SoccerTrack v2)")
    ap.add_argument("--out", type=Path, default=None, help="JSON con el resumen global y por secuencia")
    args = ap.parse_args()

    metrics = _load_metrics()
    per_seq_raw: dict[str, list[dict]] = {}
    for seq_dir in list_sequences(args.seqs):
        tr_path = seq_dir / "gt" / "gt.txt" if args.tracks == "gt" else Path(args.tracks) / f"{seq_dir.name}.txt"
        if not tr_path.exists():
            print(f"[aviso] sin tracks para {seq_dir.name}, se salta")
            continue
        gt, tr = read_mot(seq_dir / "gt" / "gt.txt"), read_mot(tr_path)
        if args.mode == "persons":
            ball_ids = [i for i, role in gt_roles(seq_dir).items() if role == "ball"]
            gt = gt[~np.isin(gt[:, 1], ball_ids)]
            tr = tr[~np.isin(tr[:, 1], ball_ids)] if args.tracks == "gt" else tr[tr[:, 7] != BALL_CLASS_ID]
        length = seq_info(seq_dir)["length"]
        if args.gt_stride > 1:
            keep = lambda a: a[(a[:, 0] - 1) % args.gt_stride == 0]  # noqa: E731
            gt, tr = keep(gt), keep(tr)
            gt[:, 0] = (gt[:, 0] - 1) // args.gt_stride + 1
            tr[:, 0] = (tr[:, 0] - 1) // args.gt_stride + 1
            length = (length - 1) // args.gt_stride + 1
        data = build_data(gt, tr, length)
        per_seq_raw[seq_dir.name] = [m.eval_sequence(data) for m in metrics]

    if not per_seq_raw:
        raise SystemExit("Ninguna secuencia evaluada")
    combined = [m.combine_sequences({s: r[k] for s, r in per_seq_raw.items()}) for k, m in enumerate(metrics)]
    result = {
        "tracks": str(args.tracks), "mode": args.mode, "gt_stride": args.gt_stride,
        "num_sequences": len(per_seq_raw), "combined": summarize(*combined),
        "per_sequence": {s: summarize(*r) for s, r in per_seq_raw.items()},
    }
    c = result["combined"]
    print(f"{args.tracks} [{args.mode}] {len(per_seq_raw)} secuencias: HOTA {100 * c['HOTA']:.2f}  "
          f"DetA {100 * c['DetA']:.2f}  AssA {100 * c['AssA']:.2f}  MOTA {100 * c['MOTA']:.2f}  "
          f"IDF1 {100 * c['IDF1']:.2f}  IDSW {c['IDSW']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
