"""Análisis de identidades del benchmark de tracking: ¿de dónde vienen los cambios de id?

Para cada tracker (carpeta de tracks MOT) y cada secuencia:
* **paneo de cámara**: mediana por frame del desplazamiento de las cajas del GT que
  siguen presentes de un frame al siguiente (un paneo mueve a todos a la vez; los
  jugadores se anulan en la mediana), en px por frame; la secuencia se resume por
  su media. Las secuencias se reparten en terciles (quieta / media / mucho paneo)
  y se suman los IDSW de cada tracker por tercil.
* **reentradas**: una identidad del GT que desaparece > ``--gap`` frames y vuelve
  (sale del plano o queda tapada). Se empareja GT <-> tracker por frame (IoU >= 0,5,
  asignación húngara); se toma el último id del tracker emparejado en los
  ``--window`` frames anteriores al hueco y el primero en los ``--window``
  posteriores. La reentrada cuenta como *recuperada* si es el mismo id,
  *cambiada* si es otro y *sin emparejar* si en una de las dos ventanas el
  tracker no tiene caja para ese GT (en el borde del plano el detector falla).
* **fragmentos**: nº de ids distintos del tracker que pasan por cada identidad
  del GT (1 = perfecta), y nº de identidades del GT que comparten un id del tracker.

Uso (contenedor):
    python -m scripts.benchmark.id_analysis --seqs datasets/SoccerNet/tracking/extracted/test \\
        --tracks output/benchmark/snmot_test/tracks/hybrid__ours_bytetrack \\
                 output/benchmark/snmot_test/tracks/hybrid__ours_botsort_osnet
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.benchmark.mot import BALL_CLASS_ID, gt_roles, list_sequences, read_mot, seq_info


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter + 1e-9)


def load_gt_persons(seq_dir: Path) -> np.ndarray:
    gt = read_mot(seq_dir / "gt" / "gt.txt")
    roles = gt_roles(seq_dir)
    keep = np.array([roles.get(int(i), "player") != "ball" for i in gt[:, 1]])
    return gt[keep]


def camera_pan(gt: np.ndarray, length: int) -> np.ndarray:
    """|mediana dx| + |mediana dy| por frame (px) de las cajas presentes en t-1 y t."""
    by_frame = {f: gt[gt[:, 0] == f] for f in range(1, length + 1)}
    pan = np.zeros(length)
    for f in range(2, length + 1):
        a, b = by_frame[f - 1], by_frame[f]
        ids = np.intersect1d(a[:, 1], b[:, 1])
        if len(ids) < 3:
            continue
        ca = {int(r[1]): (r[2] + r[4] / 2, r[3] + r[5] / 2) for r in a}
        cb = {int(r[1]): (r[2] + r[4] / 2, r[3] + r[5] / 2) for r in b}
        d = np.array([(cb[i][0] - ca[i][0], cb[i][1] - ca[i][1]) for i in ids.astype(int)])
        pan[f - 1] = abs(np.median(d[:, 0])) + abs(np.median(d[:, 1]))
    return pan


def match_per_frame(gt: np.ndarray, tr: np.ndarray, length: int, thr: float = 0.5) -> dict[int, dict[int, int]]:
    """frame -> {gt_id: tracker_id} con asignación húngara por IoU."""
    out: dict[int, dict[int, int]] = {}
    gt_by = {f: gt[gt[:, 0] == f] for f in range(1, length + 1)}
    tr_by = {f: tr[tr[:, 0] == f] for f in range(1, length + 1)}
    for f in range(1, length + 1):
        g, t = gt_by[f], tr_by[f]
        m: dict[int, int] = {}
        if len(g) and len(t):
            iou = _iou(g[:, 2:6], t[:, 2:6])
            r, c = linear_sum_assignment(-iou)
            for i, j in zip(r, c):
                if iou[i, j] >= thr:
                    m[int(g[i, 1])] = int(t[j, 1])
        out[f] = m
    return out


def reentries(gt: np.ndarray, matches: dict[int, dict[int, int]], min_gap: int, window: int) -> dict[str, int]:
    res = {"total": 0, "recuperada": 0, "cambiada": 0, "sin_emparejar": 0}
    for gid in np.unique(gt[:, 1]).astype(int):
        frames = np.sort(gt[gt[:, 1] == gid, 0]).astype(int)
        gaps = np.flatnonzero(np.diff(frames) > min_gap)
        for k in gaps:
            before, after = frames[k], frames[k + 1]
            res["total"] += 1
            a = next((matches[f].get(gid) for f in range(before, max(before - window, 0), -1)
                      if matches.get(f, {}).get(gid) is not None), None)
            b = next((matches[f].get(gid) for f in range(after, after + window)
                      if matches.get(f, {}).get(gid) is not None), None)
            if a is None or b is None:
                res["sin_emparejar"] += 1
            elif a == b:
                res["recuperada"] += 1
            else:
                res["cambiada"] += 1
    return res


def fragments(matches: dict[int, dict[int, int]]) -> tuple[float, float]:
    """(ids del tracker por identidad del GT, identidades del GT por id del tracker), medias."""
    per_gt: dict[int, set] = defaultdict(set)
    per_tr: dict[int, set] = defaultdict(set)
    for m in matches.values():
        for g, t in m.items():
            per_gt[g].add(t)
            per_tr[t].add(g)
    return (float(np.mean([len(v) for v in per_gt.values()])) if per_gt else 0.0,
            float(np.mean([len(v) for v in per_tr.values()])) if per_tr else 0.0)


def idsw_per_seq(results_json: Path) -> dict[str, int]:
    if not results_json.exists():
        return {}
    d = json.load(open(results_json))
    return {s: int(v["IDSW"]) for s, v in d["per_sequence"].items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, nargs="+", required=True)
    ap.add_argument("--gap", type=int, default=5, help="frames mínimos de ausencia para contar reentrada")
    ap.add_argument("--window", type=int, default=10, help="frames alrededor del hueco donde buscar el id emparejado")
    ap.add_argument("--out", type=Path, default=None, help="JSON con el detalle por secuencia")
    args = ap.parse_args()

    seqs = list_sequences(args.seqs)
    pans: dict[str, float] = {}
    gts: dict[str, np.ndarray] = {}
    infos: dict[str, dict] = {}
    for s in seqs:
        info = seq_info(s)
        gt = load_gt_persons(s)
        gts[s.name], infos[s.name] = gt, info
        pans[s.name] = float(camera_pan(gt, info["length"]).mean())
    order = sorted(pans, key=pans.get)
    terciles = [order[: len(order) // 3], order[len(order) // 3: 2 * len(order) // 3], order[2 * len(order) // 3:]]
    labels = ["quieta", "media", "mucho paneo"]
    print("Paneo medio (px/frame) por tercil:",
          ", ".join(f"{l} {np.mean([pans[s] for s in t]):.2f} (n={len(t)})" for l, t in zip(labels, terciles)))

    detail: dict[str, dict] = {"pan": pans, "trackers": {}}
    for tdir in args.tracks:
        name = tdir.name
        res_json = tdir.parent.parent / "results" / f"{name}.persons.json"
        idsw = idsw_per_seq(res_json)
        re_tot = {"total": 0, "recuperada": 0, "cambiada": 0, "sin_emparejar": 0}
        frag_gt, frag_tr = [], []
        per_seq = {}
        for s in seqs:
            tr = read_mot(tdir / f"{s.name}.txt")
            tr = tr[tr[:, 7] != BALL_CLASS_ID] if len(tr) else tr
            m = match_per_frame(gts[s.name], tr, infos[s.name]["length"])
            r = reentries(gts[s.name], m, args.gap, args.window)
            fg, ft = fragments(m)
            frag_gt.append(fg); frag_tr.append(ft)
            for k in re_tot:
                re_tot[k] += r[k]
            per_seq[s.name] = {"reentradas": r, "frag_gt": fg, "frag_tr": ft, "idsw": idsw.get(s.name)}
        by_terc = [sum(idsw.get(s, 0) for s in t) for t in terciles] if idsw else None
        detail["trackers"][name] = {"reentradas": re_tot, "idsw_por_tercil": by_terc, "per_seq": per_seq}
        print(f"\n== {name}")
        if by_terc:
            print("  IDSW por tercil de paneo:", dict(zip(labels, by_terc)), "total", sum(by_terc))
        t = re_tot["total"] or 1
        print(f"  reentradas (hueco > {args.gap} frames, ventana {args.window}): {re_tot['total']}  recuperada {re_tot['recuperada']} "
              f"({100 * re_tot['recuperada'] / t:.1f} %)  cambiada {re_tot['cambiada']} "
              f"({100 * re_tot['cambiada'] / t:.1f} %)  sin emparejar {re_tot['sin_emparejar']}")
        print(f"  ids del tracker por identidad del GT: {np.mean(frag_gt):.2f}   identidades del GT por id del tracker: {np.mean(frag_tr):.2f}")
    if args.out:
        args.out.write_text(json.dumps(detail, indent=1))


if __name__ == "__main__":
    main()
