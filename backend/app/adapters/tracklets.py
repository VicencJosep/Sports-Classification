"""Resumen por tracklet a partir de un JSONL de tracking (una pasada):
rol y dorsal mayoritarios, primer/último frame. Es lo que permite decir
"quién" hizo una acción de balón (dorsal cuando el detector lo leyó)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.schemas import TeamAssignments, Tracklet


def summarize_tracklets(jsonl_path: Path, teams: TeamAssignments) -> list[Tracklet]:
    roles: dict[int, Counter] = {}
    jerseys: dict[int, Counter] = {}
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    count: dict[int, int] = {}
    with open(jsonl_path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            frame_idx = int(rec["frame_idx"])
            for t in rec.get("tracks", []):
                tid = int(t["tracker_id"])
                if tid not in first:
                    first[tid] = frame_idx
                    roles[tid] = Counter()
                    jerseys[tid] = Counter()
                    count[tid] = 0
                last[tid] = frame_idx
                count[tid] += 1
                roles[tid][str(t.get("role", "other"))] += 1
                jersey = t.get("jersey")
                if isinstance(jersey, int):
                    jerseys[tid][jersey] += 1
    return [
        Tracklet(
            tracker_id=tid,
            team_cluster=teams.assignments.get(tid),
            role=roles[tid].most_common(1)[0][0],
            jersey=jerseys[tid].most_common(1)[0][0] if jerseys[tid] else None,
            first_frame=first[tid],
            last_frame=last[tid],
            frame_count=count[tid],
        )
        for tid in sorted(first)
    ]
