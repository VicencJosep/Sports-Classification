"""``telemetry.json`` (analytics/engine.py) -> eventos y acciones de balón del dominio.

Dos formas reales:
* partido completo (``FullMatchTelemetry``): ``events[]`` como tramos;
* clip (``MatchTelemetry``): un único ``event`` para todo el clip, que puede
  ser una clase ``background_*`` (sin evento).

Decisiones que absorbe este adaptador (ver análisis del repo):
* ``tracker_id == -1`` significa "sin poseedor" -> ``None``;
* ``n_possession_changes`` no se expone (en partido completo es el nº de acciones);
* etiquetas al vocabulario canónico (``throw-in`` -> ``throw_in``).
"""

from __future__ import annotations

from app.core.errors import MalformedOutputError
from app.schemas import AnalysisMode, BallAction, MatchEvent, PartAnalysis, TeamAssignments

from .labels import canonical_label


def telemetry_mode(raw: dict) -> AnalysisMode:
    if isinstance(raw.get("events"), list):
        return "full-match"
    if "event" in raw:
        return "clip"
    raise MalformedOutputError("telemetry.json: neither 'events' nor 'event' present")


def to_events(raw: dict, duration_s: float) -> list[MatchEvent]:
    try:
        if telemetry_mode(raw) == "full-match":
            spans = sorted(raw["events"], key=lambda e: float(e["start_s"]))
            return [
                MatchEvent(
                    id=f"evt-{i:04d}",
                    label=canonical_label(str(e["label"])),
                    start_s=float(e["start_s"]),
                    end_s=float(e["end_s"]),
                    peak_confidence=float(e["peak_confidence"]),
                    mean_confidence=float(e["mean_confidence"]),
                    n_windows=int(e["n_windows"]),
                    estimated=bool(e.get("estimated", False)),
                )
                for i, e in enumerate(spans)
            ]
        event = raw["event"]
        if event is None or event.get("is_background"):
            return []
        confidence = float(event["confidence"])
        return [
            MatchEvent(
                id="evt-0000",
                label=canonical_label(str(event["label"])),
                start_s=0.0,
                end_s=float(duration_s),
                peak_confidence=confidence,
                mean_confidence=confidence,
                n_windows=1,
            )
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise MalformedOutputError(f"telemetry.json events: {exc!r}") from exc


def to_ball_actions(raw: dict, fps: float, teams: TeamAssignments) -> list[BallAction]:
    try:
        actions = sorted(raw["ball_actions"], key=lambda a: int(a["frame_idx"]))
        result: list[BallAction] = []
        for a in actions:
            frame_idx = int(a["frame_idx"])
            tracker_id: int | None = int(a["tracker_id"])
            if tracker_id is not None and tracker_id < 0:
                tracker_id = None
            spot = a.get("spot_score")
            result.append(
                BallAction(
                    id=f"act-{frame_idx}",
                    frame_idx=frame_idx,
                    timestamp_s=round(frame_idx / fps, 3),
                    tracker_id=tracker_id,
                    team_cluster=teams.assignments.get(tracker_id) if tracker_id is not None else None,
                    label=canonical_label(str(a["label"])),
                    confidence=float(a["confidence"]),
                    probabilities={canonical_label(str(k)): float(v) for k, v in a["probabilities"].items()},
                    is_background=bool(a["is_background"]),
                    spot_score=float(spot) if isinstance(spot, (int, float)) else None,
                )
            )
        return result
    except (KeyError, TypeError, ValueError) as exc:
        raise MalformedOutputError(f"telemetry.json ball_actions: {exc!r}") from exc


def build_part_analysis(
    telemetry: dict, teams_raw: dict | None, *, match_id: str, part_id: str, fps: float, duration_s: float
) -> PartAnalysis:
    from .teams import to_team_assignments

    teams = to_team_assignments(teams_raw)
    try:
        frame_count = int(telemetry["n_frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MalformedOutputError(f"telemetry.json n_frames: {exc!r}") from exc
    source = telemetry.get("source_video")
    return PartAnalysis(
        match_id=match_id,
        part_id=part_id,
        mode=telemetry_mode(telemetry),
        frame_count=frame_count,
        source_video_path=str(source) if source else None,
        events=to_events(telemetry, duration_s),
        ball_actions=to_ball_actions(telemetry, fps, teams),
        teams=teams,
    )
