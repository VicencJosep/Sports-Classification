"""Registros de ``tracks.jsonl`` / ``possession.jsonl`` / ``tracks_2d.jsonl``
(core/tracking.py, core/possession.py, core/homography.py) -> TrackingFrame.

Los tres ficheros comparten la forma base por frame; ``possession.jsonl``
añade ``possessor_tracker_id`` y ``tracks_2d.jsonl`` añade ``homography_ok``
y ``field_xy`` por punto. El adaptador recibe el registro base (el más rico
disponible) y, aparte, el poseedor leído de ``possession.jsonl``.
"""

from __future__ import annotations

from app.core.errors import MalformedOutputError
from app.schemas import BallDetection, PersonDetection, TrackingFrame

_ROLES = frozenset({"player", "goalkeeper", "referee", "other"})


def _xy(value) -> tuple[float, float] | None:
    if value is None:
        return None
    return (float(value[0]), float(value[1]))


def to_tracking_frame(record: dict, possessor_tracker_id: int | None = None) -> TrackingFrame:
    try:
        persons = [
            PersonDetection(
                tracker_id=int(t["tracker_id"]),
                box=tuple(float(v) for v in t["xyxy"]),
                anchor=_xy(t["anchor_xy"]),
                confidence=float(t["confidence"]),
                role=t["role"] if t.get("role") in _ROLES else "other",
                jersey=int(t["jersey"]) if isinstance(t.get("jersey"), int) else None,
                field_xy=_xy(t.get("field_xy")),
            )
            for t in record.get("tracks", [])
        ]
        ball_raw = record.get("ball")
        ball = (
            BallDetection(
                box=tuple(float(v) for v in ball_raw["xyxy"]),
                center=_xy(ball_raw["center_xy"]),
                confidence=float(ball_raw["confidence"]),
                field_xy=_xy(ball_raw.get("field_xy")),
            )
            if ball_raw
            else None
        )
        if possessor_tracker_id is None and isinstance(record.get("possessor_tracker_id"), int):
            possessor_tracker_id = record["possessor_tracker_id"]
        homography_ok = record.get("homography_ok")
        return TrackingFrame(
            frame_idx=int(record["frame_idx"]),
            timestamp_s=float(record["timestamp_s"]),
            persons=persons,
            ball=ball,
            possessor_tracker_id=possessor_tracker_id,
            homography_ok=homography_ok if isinstance(homography_ok, bool) else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MalformedOutputError(f"tracking record: {exc!r}") from exc
