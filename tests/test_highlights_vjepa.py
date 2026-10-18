"""V-JEPA2 como juez de gol en analytics/highlights (sin modelo ni vídeo)."""

from config import HighlightsConfig

from analytics.highlights.candidates import Candidate
from analytics.highlights.pipeline import _description, _title, goal_decision
from analytics.highlights.vjepa import VJEPAWindows, apply_vjepa, goal_fusion


class _Pred:
    def __init__(self, probs):
        self.probabilities = probs


def _fake_windows(duration_s, probs_by_k):
    vw = VJEPAWindows("v.mp4", duration_s, None, 6.0)

    class Clf:
        def predict_window(self, video, t0, t1):
            base = {c: 0.0 for c in ("goal", "shot", "kick-off", "corner", "throw-in")}
            base.update(probs_by_k.get(int(t0 // 6), {}))
            return _Pred(base)

    vw._clf = Clf()
    return vw


def test_keys_overlapping_follow_the_grid_and_the_video_end():
    vw = _fake_windows(60.0, {})
    assert vw.keys_overlapping(10.0, 20.0) == [1, 2, 3]
    assert vw.keys_overlapping(55.0, 80.0) == [9]          # la ventana 10 no cabe en el vídeo
    assert vw.keys_overlapping(5.0, 5.0) == []


def test_goal_fusion_needs_vjepa_and_sums_votes():
    assert goal_fusion({"is_goal": True}, {"shot": 0.9}) is None
    f = {"vj_goal": 0.8, "vj_kickoff_after": 0.5, "vj_deadball_after": 0.4, "shot": 0.6, "vj_shot": 0.2}
    assert abs(goal_fusion({"is_goal": True, "goal_confidence": 0.9}, f) - (0.8 + 0.9 + 0.5 - 0.2 + 0.3)) < 1e-9
    assert abs(goal_fusion({"is_goal": False}, f) - (0.8 + 0.5 - 0.2 + 0.3)) < 1e-9


def test_goal_decision_fusion_needs_celebration_and_scoreboard_wins():
    f = {"vj_goal": 0.9, "vj_kickoff_after": 0.6, "shot": 0.5}
    seen = {"_celebration_seen": True}
    assert goal_decision(seen, f, False, 1.5) == (True, "fusion")
    assert goal_decision({}, f, False, 1.5) == (False, "none")                # sin celebración vista, no
    assert goal_decision(seen, f, False, None)[0] is False                    # sin umbral no hay fusión
    assert goal_decision(seen, {"vj_goal": 0.2, "shot": 0.5}, False, 1.5) == (False, "none")
    assert goal_decision(seen, {**f, "next_deadball_s": 20.0}, False, 1.5) == (False, "none")   # córner justo después
    assert goal_decision(seen, {**f, "score_goal": 0.0}, False, 1.5) == (False, "scoreboard_unchanged")


def test_goal_decision_celebration_plus_crowd_without_vjepa():
    f = {"shot": 0.4, "audio_sustained": 6.8}
    assert goal_decision({"_celebration_seen": True}, f, True, 1.5, 6.0) == (True, "celebration+audio")
    assert goal_decision({"_celebration_seen": True}, f, True, 1.5, 7.0) == (False, "none")
    assert goal_decision({"_celebration_seen": False}, {**f, "audio_sustained": 12.0}, True, 1.5, 6.0) == (False, "none")


def test_apply_vjepa_scores_all_anchors_and_after_windows_for_the_best():
    cfg = HighlightsConfig(vjepa_max_candidates=1)
    c1 = Candidate(id="cand-000", anchor_s=100.0, start_s=90.0, end_s=112.0, team=0, sources=["shot"],
                   features={"shot": 0.5}, rule_score=3.0)
    c2 = Candidate(id="cand-001", anchor_s=400.0, start_s=390.0, end_s=412.0, team=1, sources=["shot"],
                   features={"shot": 0.3}, rule_score=1.0)
    vw = _fake_windows(600.0, {17: {"goal": 0.85}, 20: {"kick-off": 0.7}, 66: {"goal": 0.05}})
    cands = [c2, c1]
    n = apply_vjepa(vw, cands, cfg)
    assert n > 0 and [c.id for c in cands] == ["cand-000", "cand-001"]
    assert c1.features["vj_goal"] == 0.85 and c1.features["vj_kickoff_after"] == 0.7
    assert abs(c1.rule_score - (3.0 + 2.0 * 0.85 + 0.5 * 0.7)) < 1e-9 and "vjepa" in c1.sources
    assert c2.features["vj_goal"] == 0.05 and "vj_kickoff_after" not in c2.features and c2.rule_score == 1.0


def test_probable_goal_wording():
    assert _title("goal", {"title_es": "Parada"}, 100.0, None, "fusion") == "Gol probable (0:01:40)"
    assert _title("goal", {"title_es": "Golazo"}, 100.0, None, "celebration+audio") == "Gol probable (0:01:40)"
    assert _title("goal", {"title_es": "Golazo"}, 100.0, None, "restart") == "Golazo"
    assert _description(True, {"description_es": "El portero para."}, None, "fusion") == "Probable gol. El portero para."
    assert _description(True, {"description_es": "Marca un gol."}, None, "fusion") == "Marca un gol."


def test_probable_goals_do_not_bypass_the_clip_cap():
    from analytics.highlights.pipeline import select_highlights

    cfg = HighlightsConfig(max_clips=2, min_clips=1)
    cands = []
    for i in range(5):           # cinco "goles probables" por fusión, de puntuación decreciente
        cands.append(Candidate(id=f"cand-{i:03d}", anchor_s=100.0 * i, start_s=100.0 * i - 10, end_s=100.0 * i + 10,
                               team=0, sources=["shot"], features={"shot": 0.5, "vj_goal": 0.9, "vj_kickoff_after": 0.8},
                               rule_score=3.0 - 0.1 * i))
    judges = {c.id: {"_celebration_seen": True} for c in cands}
    cands.append(Candidate(id="cand-sb", anchor_s=900.0, start_s=890.0, end_s=910.0, team=0, sources=["shot"],
                           features={"shot": 0.5, "score_goal": 1.0}, rule_score=0.5))
    chosen = select_highlights(cands, judges, None, cfg, has_audio=False)
    ids = [r["candidate"].id for r in chosen]
    assert "cand-sb" in ids                                   # el confirmado entra aunque haya tope
    assert sum(1 for r in chosen if r["goal_source"] == "fusion") <= cfg.max_clips


def test_probable_goal_survives_the_vlm_ranking_rejection():
    from analytics.highlights.pipeline import select_highlights

    cfg = HighlightsConfig(max_clips=3, min_clips=1)
    cands = [Candidate(id=f"cand-{i:03d}", anchor_s=100.0 * i, start_s=100.0 * i - 10, end_s=100.0 * i + 10, team=0,
                       sources=["shot"], features={"shot": 0.5, "audio_sustained": 9.0}, rule_score=3.0 - 0.5 * i) for i in range(4)]
    judges = {"cand-003": {"_celebration_seen": True, "excitement": 5, "skill": 5}}      # gol probable, el peor por reglas
    ranking = {"rank": {"cand-000": 1, "cand-001": 2}, "rejected": ["cand-003"], "reasons": {}}
    chosen = select_highlights(cands, judges, ranking, cfg, has_audio=True)
    assert [r["candidate"].id for r in chosen][0] == "cand-003"                 # los goles van primero
    assert chosen[0]["goal_source"] == "celebration+audio"


def test_vlm_candidates_adds_goal_hints_beyond_the_top():
    from analytics.highlights.pipeline import vlm_candidates

    cfg = HighlightsConfig(vlm_max_candidates=1)
    def cand(i, **f):
        return Candidate(id=f"cand-{i:03d}", anchor_s=100.0 * i, start_s=0, end_s=0, team=0, sources=[], features=f, rule_score=5.0 - i)
    cands = [cand(0, shot=0.9), cand(1, shot=0.2), cand(2, audio_sustained=6.8, vj_goal=0.22), cand(3, audio_sustained=9.0, vj_goal=0.05),
             cand(4, vj_goal=0.35), cand(5, score_goal=1.0)]
    ids = [c.id for c in vlm_candidates(cands, cfg, audio_ok=True)]
    assert ids == ["cand-000", "cand-002", "cand-004", "cand-005"]
    ids = [c.id for c in vlm_candidates(cands, cfg, audio_ok=False)]      # sin audio útil, el griterío no cuenta
    assert ids == ["cand-000", "cand-004", "cand-005"]
