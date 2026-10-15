"""``teams.json`` (core/classification.py::TeamClusteringResult) -> TeamAssignments."""

from __future__ import annotations

from app.core.errors import MalformedOutputError
from app.schemas import TeamAssignments


def to_team_assignments(raw: dict | None) -> TeamAssignments:
    if raw is None:
        return TeamAssignments(clusters=[], assignments={}, excluded={})
    try:
        assignments = {int(t["tracker_id"]): int(t["team_cluster"]) for t in raw["teams"]}
        excluded = {int(e["tracker_id"]): str(e["reason"]) for e in raw.get("excluded", [])}
        n_clusters = int(raw.get("n_clusters", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise MalformedOutputError(f"teams.json: {exc!r}") from exc
    clusters = sorted(set(assignments.values()) | set(range(n_clusters)))
    return TeamAssignments(clusters=clusters, assignments=assignments, excluded=excluded)
