from __future__ import annotations

from pydantic import BaseModel

Point = tuple[float, float]
Box = tuple[float, float, float, float]


class PersonDetection(BaseModel):
    tracker_id: int
    box: Box                        # píxeles del frame original, [x1, y1, x2, y2]
    anchor: Point                   # centro del borde inferior (punto de apoyo)
    confidence: float
    role: str                       # player | goalkeeper | referee | other
    jersey: int | None
    field_xy: Point | None          # metros, origen en la esquina, campo 105 x 68


class BallDetection(BaseModel):
    box: Box
    center: Point
    confidence: float
    field_xy: Point | None          # proyectado suponiendo balón en el suelo


class TrackingFrame(BaseModel):
    frame_idx: int
    timestamp_s: float
    persons: list[PersonDetection]
    ball: BallDetection | None
    possessor_tracker_id: int | None
    homography_ok: bool | None


class TrackingWindow(BaseModel):
    match_id: str
    part_id: str
    from_s: float
    to_s: float
    fps: float
    frames: list[TrackingFrame]
