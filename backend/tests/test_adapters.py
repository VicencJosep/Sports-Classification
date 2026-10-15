import json
from pathlib import Path

import pytest

from app.adapters.telemetry import build_part_analysis, to_events
from app.adapters.tracks import to_tracking_frame
from app.core.errors import MalformedOutputError
from app.repositories.tracking_index import TrackingIndex

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_clip_background_event_is_dropped():
    raw = {"source_video": "x", "event": {"label": "background_public", "confidence": 0.3, "is_background": True,
                                          "probabilities": {}, "video_path": "x"}, "ball_actions": [], "n_frames": 50}
    assert to_events(raw, 2.0) == []


def test_malformed_telemetry():
    with pytest.raises(MalformedOutputError):
        build_part_analysis({"source_video": "x"}, None, match_id="m", part_id="full", fps=25, duration_s=1)


def test_tracking_index_roundtrip(tmp_path):
    src = FIXTURES / "tracks_2d.sample.jsonl"
    idx = TrackingIndex.build(src)
    assert len(idx) == 50 and idx.frames[0] == 0 and idx.frames[-1] == 49
    idx.save(tmp_path / "x.idx")
    loaded = TrackingIndex.load(tmp_path / "x.idx")
    records = loaded.read_window(src, 10, 12)
    assert [r["frame_idx"] for r in records] == [10, 11, 12]
    assert records == [json.loads(l) for l in src.read_text().splitlines()[10:13]]
    assert loaded.read_window(src, 500, 600) == []


def test_tracking_frame_from_real_record():
    record = json.loads((FIXTURES / "possession.sample.jsonl").read_text().splitlines()[0])
    frame = to_tracking_frame(record)
    assert frame.frame_idx == 0 and frame.ball is None
    assert frame.homography_ok is None and frame.possessor_tracker_id is None
    assert frame.persons[0].field_xy is None
