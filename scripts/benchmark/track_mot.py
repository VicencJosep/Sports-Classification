"""Paso 2 del benchmark de tracking: corre un tracker sobre detecciones ya
guardadas (``detect_mot.py`` o el oráculo del GT) y escribe tracks MOT.

Trackers:
* ``ours``: el ``core.tracking.PlayerTracker`` de producción con sus valores por
  defecto (mismos filtros y balón por frame), alimentado por un detector que
  lee de la caché. Solo cambia de dónde salen las cajas. Variantes explícitas:
  ``ours_bytetrack`` (supervision, el tracker anterior), ``ours_botsort`` (GMC,
  sin ReID), ``ours_botsort_osnet`` y ``ours_botsort_prtreid`` (GMC + ReID).
  Las variantes BoT-SORT leen los JPG de la secuencia (GMC y ReID miran píxeles).
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

    ``detect`` ignora el contenido de ``frames`` (imagen o número) y sirve las
    cajas del frame ``self.frame_no`` (1-based), que ``track_ours`` fija antes
    de cada llamada; con lotes de más de un frame, las de los consecutivos.
    """

    chunk_size = 1

    def __init__(self, rows: np.ndarray) -> None:
        self._by_frame = {int(f): rows[rows[:, 0] == f] for f in np.unique(rows[:, 0])} if len(rows) else {}
        self.frame_no = 1

    def detect(self, frames: Sequence) -> list[sv.Detections]:
        out = []
        for k in range(len(frames)):
            r = self._by_frame.get(self.frame_no + k)
            if r is None or len(r) == 0:
                out.append(_with_role_data(sv.Detections.empty(), [], []))
                continue
            xyxy = np.column_stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]])
            dets = sv.Detections(xyxy=xyxy, confidence=r[:, 6].astype(float), class_id=r[:, 7].astype(int))
            roles = [CLASS_ID_TO_ROLE.get(int(c), "other") for c in r[:, 7]]
            jerseys = [int(j) if j >= 0 else None for j in r[:, 8]]
            out.append(_with_role_data(dets, roles, jerseys))
        return out


def track_ours(dets: np.ndarray, info: dict, seq_dir: Path, **tracker_kwargs) -> list[tuple]:
    detector = CachedDetector(dets)
    tracker = PlayerTracker(detector, frame_rate=info["frame_rate"], **tracker_kwargs)
    rows: list[tuple] = []
    for frame_no in range(1, info["length"] + 1):
        detector.frame_no = frame_no
        # BoT-SORT (GMC/ReID) necesita la imagen; ByteTrack solo mira cajas y recibe el nº de frame.
        frame = (cv2.imread(str(info["img_dir"] / f"{frame_no:06d}{info['img_ext']}"))
                 if tracker.needs_frames else frame_no)
        (res,) = tracker.process_batch([frame], [frame_no], [frame_no / info["frame_rate"]])
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


def _botsort(d, i, s, **kw):
    """BoT-SORT con los parámetros base de las filas de §7.10 (buffer 30, NMS 0,7, puerta de IoU, apariencia 0,8)."""
    base = dict(tracker="botsort", lost_track_buffer=30, nms_iou=0.7, reid_lost_without_gate=False, appearance_thresh=0.8)
    base.update(kw)
    return track_ours(d, i, s, **base)


TRACKERS = {
    "ours": track_ours,  # valores por defecto de PlayerTracker (= producción; hoy = ours_botsort_osnet_nms_buf60_lost95)
    # Cada variante fija TODOS los parámetros que cambian entre filas de docs/comparativa_benchmark.md §7.10, para que
    # sigan siendo reproducibles aunque cambien los valores por defecto de PlayerTracker.
    "ours_bytetrack": lambda d, i, s: track_ours(d, i, s, tracker="bytetrack", lost_track_buffer=30),
    "ours_botsort": lambda d, i, s: _botsort(d, i, s, reid="none", nms_iou=None),
    "ours_botsort_osnet": lambda d, i, s: _botsort(d, i, s, reid="osnet", nms_iou=None),
    "ours_botsort_prtreid": lambda d, i, s: _botsort(d, i, s, reid="prtreid", nms_iou=None),
    # variantes del ajuste en el split de train
    "ours_botsort_nms": lambda d, i, s: _botsort(d, i, s, reid="none"),
    "ours_botsort_osnet_nms": lambda d, i, s: _botsort(d, i, s, reid="osnet"),
    "ours_botsort_osnet_nms_buf60": lambda d, i, s: _botsort(d, i, s, reid="osnet", lost_track_buffer=72),
    "ours_botsort_osnet_nms_lost": lambda d, i, s: _botsort(d, i, s, reid="osnet", reid_lost_without_gate=True),
    "ours_botsort_osnet_nms_buf60_lost": lambda d, i, s: _botsort(d, i, s, reid="osnet", lost_track_buffer=72,
                                                                  reid_lost_without_gate=True),
    "ours_botsort_osnet_nms_buf125_lost": lambda d, i, s: _botsort(d, i, s, reid="osnet", lost_track_buffer=150,
                                                                   reid_lost_without_gate=True),
    "ours_botsort_osnet_nms_buf60_lost90": lambda d, i, s: _botsort(d, i, s, reid="osnet", lost_track_buffer=72,
                                                                    reid_lost_without_gate=True, appearance_thresh=0.9),
    "ours_botsort_osnet_nms_buf60_lost95": lambda d, i, s: _botsort(d, i, s, reid="osnet", lost_track_buffer=72,
                                                                    reid_lost_without_gate=True, appearance_thresh=0.95),
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
