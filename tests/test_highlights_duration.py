"""Política de duración de los clips de highlights (12-30 s; goles hasta 45 s), sin vídeo ni VLM.

Sin pytest en el contenedor:
    docker exec -w /app Sports-Classification python -c "import tests.test_highlights_duration as t; t.run()"
"""

from config import HighlightsConfig

from analytics.highlights.pipeline import _merge_close, fit_duration
from analytics.highlights.vlm import clip_end_from_kinds, first_celebration

CFG = HighlightsConfig()


def test_play_is_capped_to_30s_trimming_the_build_up_first():
    # 35 s de jugada + 40 s de cola: primero se recorta lo anterior al ancla a 12 s, luego la cola a 30 s.
    start, end = fit_duration(65.0, 140.0, 100.0, False, CFG, 6000.0)
    assert start == 88.0 and end == 118.0


def test_short_play_is_extended_to_12s():
    start, end = fit_duration(96.0, 104.0, 100.0, False, CFG, 6000.0)
    assert end - start == 12.0 and start == 96.0


def test_goal_keeps_celebration_counted_from_the_real_goal():
    # Ancla 26 s antes del gol (Mallorca 1-0): el tope de 45 s se cuenta desde el gol, no desde el ancla.
    anchor, goal = 4418.75, 4449.0
    start, end = fit_duration(4399.0, 4463.75, anchor, True, CFG, 6000.0, goal_s=goal)
    assert start == goal - CFG.clip_back_max_goal_s
    assert end == 4463.75                       # la celebración entera cabe (29.75 s)
    assert end - start <= CFG.clip_max_goal_s


def test_goal_longer_than_45s_is_cut_at_the_tail():
    start, end = fit_duration(6347.0, 6410.0, 6362.75, True, CFG, 6000.0 + 1000)
    assert start == 6347.75 and end == 6347.75 + 45.0


def test_replays_only_count_after_a_goal():
    kinds = [(103.0, "action"), (107.0, "replay"), (111.0, "replay"), (115.0, "replay"), (119.0, "walking_back"),
             (123.0, "play")]
    # Parada seguida de repeticiones de TV: acaba con la jugada (1 frame de acción -> 107 s).
    assert clip_end_from_kinds(kinds, 100.0, 4.0, 45.0, 12.0, replay_as_reaction=False) == 107.0
    # Gol: las repeticiones cuentan, con tope de 8 s desde la "última reacción real" (ninguna -> ancla + 12 - 4 + 8 = 116 -> último replay 115 + 4).
    assert clip_end_from_kinds(kinds, 100.0, 4.0, 45.0, 12.0, replay_as_reaction=True, replay_tail_max_s=8.0) == 119.0


def test_replay_tail_after_celebration_is_bounded():
    kinds = [(103.0, "action"), (107.0, "celebration"), (111.0, "celebration"), (115.0, "replay"), (119.0, "replay"),
             (123.0, "replay"), (127.0, "replay"), (131.0, "walking_back")]
    # Última celebración 111 + 8 s de repeticiones = 119 -> fin 123 (sin tope sería 127 + 4 = 131).
    assert clip_end_from_kinds(kinds, 100.0, 4.0, 45.0, 12.0, replay_as_reaction=True, replay_tail_max_s=8.0) == 123.0
    assert clip_end_from_kinds(kinds, 100.0, 4.0, 45.0, 12.0, replay_as_reaction=True) == 131.0
    assert first_celebration(kinds) == 107.0


def test_replays_between_two_celebration_runs_do_not_cut_a_goal():
    # Messi 2-3: celebración, 3 repeticiones, vuelta, y otra vez celebración hasta +43 s.
    a = 6362.75
    kinds = [(a + 3, "action"), (a + 7, "replay"), (a + 11, "celebration"), (a + 15, "celebration"), (a + 19, "replay"),
             (a + 23, "replay"), (a + 27, "replay"), (a + 31, "walking_back"), (a + 35, "celebration"),
             (a + 39, "celebration"), (a + 43, "celebration")]
    assert clip_end_from_kinds(kinds, a, 4.0, 45.0, 12.0, replay_as_reaction=True, replay_tail_max_s=8.0) == a + 45.0


def test_goal_time_prefers_scoreboard_and_ignores_late_celebration_on_tv():
    from analytics.highlights.pipeline import goal_time

    # Rakitic 1-1: 7 repeticiones y la celebración a +30 s; el rótulo cambia a +6 s -> el ancla ya es el gol.
    kinds = [(2289.25 + 2 + 4 * i, "replay") for i in range(7)] + [(2319.25, "celebration"), (2323.25, "play")]
    assert goal_time(2289.25, {"t_lo": 2291.25, "t_hi": 2295.0}, kinds, CFG) is None
    # Sin rótulo, el mismo caso: solo cuenta la celebración si el desplazamiento es claro (aquí 26 s: se usa).
    assert goal_time(2289.25, None, kinds, CFG) == 2315.25
    # Ancla 27 s antes del cambio de rótulo (Mallorca 1-0): el gol es t_hi - 4.
    assert goal_time(4418.75, {"t_lo": 4447.5, "t_hi": 4451.25}, [], CFG) == 4447.25
    # Celebración desde el primer frame: el ancla ya es el gol.
    assert goal_time(4445.25, None, [(4447.25, "celebration"), (4451.25, "celebration")], CFG) is None


def test_scoreboard_goal_goes_to_the_candidate_near_the_change():
    from analytics.highlights.candidates import Candidate
    from analytics.highlights.pipeline import apply_scoreboard
    from analytics.highlights.scoreboard import ScoreChange, ScoreTimeline

    def cand(cid, anchor, shot):
        return Candidate(cid, anchor, anchor - 10.0, anchor + 12.0, 0, ["shot"], {"shot": shot, "goal": 0.0}, 2.0 * shot)

    early, late = cand("cand-039", 4418.75, 0.35), cand("cand-040", 4445.25, 0.14)
    tl = ScoreTimeline(True, [ScoreChange(4447.5, 4451.25, (0, 0), (1, 0))], [4400.0, 4447.5, 4451.25, 4500.0], 4)
    info = apply_scoreboard(None, [early, late], tl, CFG)
    assert list(info) == ["cand-040"] and info["cand-040"]["t_hi"] == 4451.25
    assert late.features["score_goal"] == 1.0 and early.features.get("score_goal") != 1.0


def test_merge_is_bounded_by_max_merged_duration():
    a = {"start_s": 100.0, "end_s": 125.0, "is_goal": False, "final_score": 0.5, "candidate": type("C", (), {"id": "a"})()}
    b = {"start_s": 135.0, "end_s": 160.0, "is_goal": False, "final_score": 0.6, "candidate": type("C", (), {"id": "b"})()}
    assert len(_merge_close([dict(a), dict(b)], 20.0, 40.0)) == 2          # 60 s fundidos: no
    merged = _merge_close([dict(a), dict(b)], 20.0, 80.0)
    assert len(merged) == 1 and merged[0]["start_s"] == 100.0 and merged[0]["end_s"] == 160.0


def run() -> None:
    import inspect
    import sys

    mod = sys.modules[__name__]
    tests = [f for name, f in inspect.getmembers(mod, inspect.isfunction) if name.startswith("test_")]
    failed = 0
    for f in tests:
        try:
            f()
            print(f"ok    {f.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FALLA {f.__name__}: {exc!r}")
    print(f"{len(tests) - failed}/{len(tests)} tests OK")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
