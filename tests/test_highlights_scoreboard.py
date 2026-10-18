"""Lógica del marcador en analytics/highlights (sin vídeo ni VLM)."""

from config import HighlightsConfig

from analytics.highlights.candidates import Candidate
from analytics.highlights.pipeline import claims_goal, goal_decision, select_highlights
from analytics.highlights.scoreboard import ScoreTimeline, _confirmed_changes


def _valid(scores, step=30.0):
    return [(i * step, s) for i, s in enumerate(scores)]


def test_confirmed_change_is_located_between_reads():
    valid = _valid([(0, 0), (0, 0), (1, 0), (1, 0), (1, 1), (1, 1)])
    assert _confirmed_changes(valid) == [(1, 2, (0, 0), (1, 0)), (3, 4, (1, 0), (1, 1))]


def test_isolated_misread_is_not_a_goal():
    valid = _valid([(0, 0), (0, 0), (3, 0), (0, 0), (0, 0)])
    assert _confirmed_changes(valid) == []


def test_decreasing_score_is_discarded():
    valid = _valid([(1, 1), (1, 1), (0, 1), (0, 1), (1, 1)])
    assert _confirmed_changes(valid) == []


def test_lower_bound_skips_discarded_misread():
    # La lectura (5, 0) se descarta; el límite inferior sigue siendo la última lectura con (0, 0).
    valid = _valid([(0, 0), (0, 0), (5, 0), (0, 0), (1, 0), (1, 0)])
    assert _confirmed_changes(valid) == [(3, 4, (0, 0), (1, 0))]


def test_unconfirmed_last_read_is_ignored():
    assert _confirmed_changes(_valid([(0, 0), (0, 0), (1, 0)])) == []


def test_brackets_needs_reads_on_both_sides():
    tl = ScoreTimeline(True, [], [0.0, 30.0, 60.0, 90.0, 120.0], 5)
    assert tl.brackets(60.0)
    assert not tl.brackets(0.0)        # nada antes
    assert not tl.brackets(115.0)      # nada después


def test_scoreboard_decides_goal_over_vlm_and_restart():
    vlm_goal = {"is_goal": True, "goal_confidence": 1.0}
    assert goal_decision(vlm_goal, {"score_goal": 0.0}, False) == (False, "scoreboard_unchanged")
    assert goal_decision({"is_goal": False}, {"score_goal": 1.0}, False) == (True, "scoreboard")
    assert goal_decision({"is_goal": False}, {"score_goal": 1.0, "center_restart": 0.0}, True)[0] is True
    # Sin marcador legible la cascada anterior sigue igual: VLM solo, sin audio, no basta.
    assert goal_decision(vlm_goal, {}, False) == (False, "contradicted")


def test_claims_goal_ignores_saves():
    assert claims_goal("Un jugador marca un gol espectacular.")
    assert claims_goal("Golazo en el área")
    assert not claims_goal("El portero realiza una gran parada para evitar el gol.")
    assert not claims_goal("Un jugador rojo remata, pero el portero salva el gol.")


def _cand(i, anchor, score):
    return Candidate(id=f"cand-{i:03d}", anchor_s=anchor, start_s=anchor - 10, end_s=anchor + 10, team=0,
                     sources=["shot"], features={"shot": 0.5}, rule_score=score)


def test_every_goal_is_selected_beyond_max_clips():
    cfg = HighlightsConfig(max_clips=2, min_clips=1)
    cands = [_cand(i, 100.0 * i, 3.0 - 0.1 * i) for i in range(6)]
    for c in cands[2:5]:
        c.features["score_goal"] = 1.0
    chosen = select_highlights(cands, {}, None, cfg, has_audio=False)
    ids = {r["candidate"].id for r in chosen}
    assert {"cand-002", "cand-003", "cand-004"} <= ids
    assert sum(1 for r in chosen if not r["is_goal"]) <= cfg.max_clips


def test_find_region_picks_first_region_with_readable_scoreboard():
    from analytics.highlights import scoreboard

    original = scoreboard.read_score
    try:
        scoreboard.read_score = lambda client, video, t, region="top-left": (
            {"home": 0, "away": 0, "clock": None} if region == "top-right" else None)
        assert scoreboard.find_region(None, "v.mp4", 5400.0) == "top-right"
        scoreboard.read_score = lambda client, video, t, region="top-left": None
        assert scoreboard.find_region(None, "v.mp4", 5400.0) is None
    finally:
        scoreboard.read_score = original


def test_goal_in_last_seconds_is_confirmed_with_neighbouring_reads():
    from analytics.highlights import scoreboard

    original = scoreboard.read_score
    try:
        scoreboard.read_score = lambda client, video, t, region="top-left": (
            {"home": 1 if t >= 95.0 else 0, "away": 0, "clock": None})
        tl = scoreboard.build_timeline(None, "v.mp4", duration_s=100.0, step_s=10.0, min_valid=0.25)
        assert [c.after for c in tl.changes] == [(1, 0)]
        assert tl.changes[0].t_lo < 95.0 <= tl.changes[0].t_hi
    finally:
        scoreboard.read_score = original
