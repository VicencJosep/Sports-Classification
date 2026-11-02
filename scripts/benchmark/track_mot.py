"""Paso 2 del benchmark de tracking: corre un tracker sobre detecciones ya
guardadas (``detect_mot.py`` o el oráculo del GT) y escribe tracks MOT.

Trackers:
* ``ours``: el ``core.tracking.PlayerTracker`` de producción, tal cual (mismos
  filtros, ByteTrack de supervision y balón por frame), alimentado por un
  detector que lee de la caché. Solo cambia de dónde salen las cajas.
* ``ul_bytetrack`` / ``ul_botsort``: los trackers de ultralytics con sus YAML
  por defecto (BoT-SORT con GMC sparseOptFlow, sin ReID) sobre las mismas
  detecciones; filtro de entrada en ``UL_MIN_CONF`` / ``UL_MIN_AREA``.

Fuente de detecciones (``--dets``):
* una carpeta con ``<seq>.txt`` de ``detect_mot.py``;
* ``gt``: oráculo, el GT de la secuencia sin identidades (como la columna
  "oracle" de Deep-EIoU, arXiv 2306.13074).

Uso:
    python -m scripts.benchmark.track_mot --seqs datasets/SoccerNet/tracking/extracted/test \\
        --dets output/benchmark/snmot_test/dets/hybrid --tracker ours \\
        --out output/benchmark/snmot_test/tracks/hybrid__ours
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import supervision as sv

from core.tracking import CLASS_ID_TO_ROLE, PlayerTracker, _with_role_data
from scripts.benchmark.mot import (
    BALL_CLASS_ID, BALL_TRACK_ID, gt_as_detections, list_sequences, read_mot, seq_info, write_mot,
)

logger = logging.getLogger(__name__)


class CachedDetector:
    """``core.tracking.Detector`` que devuelve detecciones MOT ya calculadas.

    ``detect`` recibe como "frames" los números de frame (1-based): el tracker no
    mira los píxeles, solo las cajas.
    """

    chunk_size = 1

    def __init__(self, rows: np.ndarray) -> None:
        self._by_frame = {int(f): rows[rows[:, 0] == f] for f in np.unique(rows[:, 0])} if len(rows) else {}

    def detect(self, frames: Sequence[int]) -> list[sv.Detections]:
        out = []
        for frame_no in frames:
            r = self._by_frame.get(int(frame_no))
            if r is None or len(r) == 0:
                out.append(_with_role_data(sv.Detections.empty(), [], []))
                continue
            xyxy = np.column_stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]])
            dets = sv.Detections(xyxy=xyxy, confidence=r[:, 6].astype(float), class_id=r[:, 7].astype(int))
            roles = [CLASS_ID_TO_ROLE.get(int(c), "other") for c in r[:, 7]]
            jerseys = [int(j) if j >= 0 else None for j in r[:, 8]]
            out.append(_with_role_data(dets, roles, jerseys))
        return out


def track_ours(dets: np.ndarray, info: dict, seq_dir: Path) -> list[tuple]:
    tracker = PlayerTracker(CachedDetector(dets), frame_rate=info["frame_rate"])
    rows: list[tuple] = []
    for frame_no in range(1, info["length"] + 1):
        # process_batch con el nº de frame como "frame": CachedDetector lo usa de clave.
        (res,) = tracker.process_batch([frame_no], [frame_no], [frame_no / info["frame_rate"]])
        for t in res.tracks:
            x1, y1, x2, y2 = t.xyxy
            rows.append((frame_no, t.tracker_id, x1, y1, x2 - x1, y2 - y1, t.confidence, t.class_id,
                         t.jersey if t.jersey is not None else -1))
        if res.ball is not None:
            x1, y1, x2, y2 = res.ball.xyxy
            rows.append((frame_no, BALL_TRACK_ID, x1, y1, x2 - x1, y2 - y1, res.ball.confidence, 0, -1))
    return rows


class _UlDets:
    """Lo mínimo de ``ultralytics.engine.results.Boxes`` que piden sus trackers."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy, self.conf, self.cls = xyxy, conf, cls
        self.xywh = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
                                     xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]]) if len(xyxy) else np.zeros((0, 4))

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "_UlDets":
        return _UlDets(self.xyxy[mask], self.conf[mask], self.cls[mask])


# Mismo filtro de personas que PlayerTracker (sin balón, área >= 300 px²) pero
# con conf >= 0,1 (track_low_thresh): la segunda pasada de ByteTrack/BoT-SORT
# vive de las detecciones flojas, y PlayerTracker las corta en 0,3.
UL_MIN_CONF = 0.1
UL_MIN_AREA = 300.0


def _track_ultralytics(kind: str, dets: np.ndarray, info: dict, seq_dir: Path) -> list[tuple]:
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml

    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(f"{kind}.yaml")))
    tracker = (BOTSORT if kind == "botsort" else BYTETracker)(args=cfg)  # track_buffer en frames, sin escalar por fps
    keep = (dets[:, 7] != BALL_CLASS_ID) & (dets[:, 6] >= UL_MIN_CONF) & (dets[:, 4] * dets[:, 5] >= UL_MIN_AREA)
    dets = dets[keep]
    rows: list[tuple] = []
    for frame_no in range(1, info["length"] + 1):
        r = dets[dets[:, 0] == frame_no]
        xyxy = np.column_stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]])
        img = None
        if kind == "botsort":  # GMC (sparseOptFlow) necesita la imagen
            img = cv2.imread(str(info["img_dir"] / f"{frame_no:06d}{info['img_ext']}"))
        out = tracker.update(_UlDets(xyxy, r[:, 6], r[:, 7]), img)
        for x1, y1, x2, y2, tid, score, cls, idx in out:
            jersey = r[int(idx), 8] if len(r) else -1
            rows.append((frame_no, tid, x1, y1, x2 - x1, y2 - y1, score, cls, jersey))
    return rows


TRACKERS = {
    "ours": track_ours,
    "ul_bytetrack": lambda d, i, s: _track_ultralytics("bytetrack", d, i, s),
    "ul_botsort": lambda d, i, s: _track_ultralytics("botsort", d, i, s),
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", type=Path, required=True)
    ap.add_argument("--dets", required=True, help="Carpeta de detecciones o 'gt' (oráculo)")
    ap.add_argument("--tracker", choices=sorted(TRACKERS), default="ours")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    timing = []
    for seq_dir in list_sequences(args.seqs):
        info = seq_info(seq_dir)
        if args.dets == "gt":
            dets = gt_as_detections(seq_dir)
        else:
            det_path = Path(args.dets) / f"{seq_dir.name}.txt"
            if not det_path.exists():
                logger.warning("Sin detecciones para %s, se salta", seq_dir.name)
                continue
            dets = read_mot(det_path)
        t0 = time.perf_counter()
        rows = TRACKERS[args.tracker](dets, info, seq_dir)
        secs = time.perf_counter() - t0
        write_mot(args.out / f"{seq_dir.name}.txt", rows)
        timing.append({"seq": seq_dir.name, "seconds": round(secs, 3), "fps": round(info["length"] / secs, 1)})
    (args.out / "timing.json").write_text(json.dumps(timing, indent=1))
    logger.info("%d secuencias -> %s", len(timing), args.out)


if __name__ == "__main__":
    main()
