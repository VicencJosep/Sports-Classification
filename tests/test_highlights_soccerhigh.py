"""SoccerHigh dentro de analytics/highlights, sin vídeo ni GPU: curva y tramos en las señales, anclas y puntuación
de candidatos, inferencia del resumidor sobre rasgos sintéticos (NMS y selección @T).

    docker exec -w /app Sports-Classification python -c "import tests.test_highlights_soccerhigh as t; t.run()"
"""

import numpy as np

from analytics.highlights.candidates import build_candidates
from analytics.highlights.signals import GRID_S, MatchSignals, attach_soccerhigh
from analytics.highlights.soccerhigh import Segment, nms, segments_from_frames, select_to_length
from config import HighlightsConfig

CFG = HighlightsConfig()


def _signals(duration_s: float = 600.0) -> MatchSignals:
    n = int(duration_s / GRID_S)
    z = np.zeros(n, np.float32)
    return MatchSignals(
        duration_s=duration_s, fps=25.0, t=((np.arange(n) + 0.5) * GRID_S).astype(np.float32), has_audio=False,
        audio_rel=z.copy(), audio_sustained=z.copy(), curves={}, possessor=np.full(n, -1), team=np.full(n, -1),
        ball_xy=np.full((n, 2), np.nan, np.float32), possessor_xy=np.full((n, 2), np.nan, np.float32),
        players_left_frac=np.full(n, np.nan, np.float32), n_players_xy=z.copy(),
    )


def test_attach_soccerhigh_builds_curve_and_segments():
    sig = _signals(100.0)
    scores = np.zeros(200, np.float32)
    scores[100:120] = 0.9                       # 50-60 s
    attach_soccerhigh(sig, scores, [{"start_s": 50.0, "end_s": 60.0, "score": 0.9}])
    assert sig.curve("soccerhigh")[sig.bin_of(55.0)] == np.float32(0.9)
    assert sig.curve("soccerhigh")[sig.bin_of(20.0)] == 0.0
    assert sig.soccerhigh_segments == [{"start_s": 50.0, "end_s": 60.0, "score": 0.9}]


def test_segment_without_tdeed_becomes_a_candidate_and_scores():
    sig = _signals()
    scores = np.zeros(1200, np.float32)
    scores[600:640] = 0.8                       # 300-320 s, sin tiro de T-DEED
    attach_soccerhigh(sig, scores, [{"start_s": 300.0, "end_s": 320.0, "score": 0.8}])
    cands = build_candidates(sig, CFG)
    assert len(cands) == 1
    c = cands[0]
    assert c.sources == ["soccerhigh"]
    assert 300.0 <= c.anchor_s <= 320.0
    assert c.features["soccerhigh"] == np.float32(0.8)
    assert abs(c.rule_score - CFG.w_soccerhigh * 0.8) < 1e-5
    assert c.end_s >= 320.0                     # el tramo alarga el fin provisional


def test_weak_segment_is_not_an_anchor():
    sig = _signals()
    scores = np.zeros(1200, np.float32)
    scores[600:640] = 0.3
    attach_soccerhigh(sig, scores, [{"start_s": 300.0, "end_s": 320.0, "score": 0.3}])
    assert build_candidates(sig, CFG) == []


def test_shot_peak_inside_segment_is_the_anchor():
    sig = _signals()
    shot = np.zeros(sig.n_bins, np.float32)
    shot[sig.bin_of(312.0)] = 0.4
    sig.curves["shot"] = shot
    scores = np.zeros(1200, np.float32)
    scores[600:640] = 0.9
    attach_soccerhigh(sig, scores, [{"start_s": 300.0, "end_s": 320.0, "score": 0.9}])
    cands = build_candidates(sig, CFG)
    assert len(cands) == 1 and abs(cands[0].anchor_s - 312.25) < 0.3
    assert set(cands[0].sources) == {"shot", "soccerhigh"}


def test_segments_from_frames_and_nms():
    prob = np.zeros(100, np.float32)
    prob[10:14] = 0.9                           # propone [i-2, i+3] para cada i
    offsets = np.tile(np.array([2.0, 3.0], np.float32), (100, 1))
    segs = segments_from_frames(prob, offsets, 0.5)
    assert [(sg.start, sg.end) for sg in segs] == [(8, 13)] and abs(segs[0].score - 0.9) < 1e-6   # los demás solapan
    kept = nms([Segment(0, 5, 0.9), Segment(5, 9, 0.8), Segment(6, 9, 0.95)])
    assert [(s.start, s.end) for s in kept] == [(6, 9), (0, 5)]      # (5, 9) solapa con (6, 9)


def test_select_to_length_keeps_best_that_fit():
    segs = [Segment(0, 9, 0.9), Segment(20, 39, 0.8), Segment(50, 54, 0.7)]
    chosen = select_to_length(segs, total_frames=16)
    assert [(s.start, s.end) for s in chosen] == [(0, 9), (50, 54)]


def run() -> None:
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)


if __name__ == "__main__":
    run()
