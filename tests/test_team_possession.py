"""Pruebas de core/team_possession con datos sintéticos (sin ficheros externos)."""
import numpy as np
import pytest

from core.team_possession import control, decode, io_metrica, labels as L, spell_metrics as pm
from core.team_possession.features import FeatureConfig, compute_features, feature_matrix, feature_names
from core.team_possession.frames import AWAY, HOME, PositionalData, resample, expand_labels
from core.team_possession.simulate import anonymize_slots, drop_ball


def _synthetic(T=600, fps=25.0, seed=0):
    """Dos jugadores (uno por equipo) y un balón que va pegado al local la primera
    mitad del tramo y al visitante la segunda; un tramo de balón muerto en medio."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fps
    home = np.stack([20 + 2 * t, 30 + np.sin(t)], axis=1)
    away = np.stack([60 - 2 * t, 40 + np.cos(t)], axis=1)
    players = np.stack([home, away], axis=1).astype(np.float32)
    ball = np.where((np.arange(T) < T // 2)[:, None], home + 0.3, away + 0.3).astype(np.float32)
    ball += rng.normal(0, 0.05, ball.shape).astype(np.float32)
    labels = np.where(np.arange(T) < T // 2, L.H, L.A).astype(np.int8)
    # parada: balón quieto y lejos de todos durante 3 s
    dead = slice(T // 2 - 40, T // 2 + 40)
    ball[dead] = (5.0, 5.0)
    labels[dead] = L.D
    return PositionalData(fps=fps, frame_index=np.arange(T), period=np.ones(T, np.int16),
                          players_xy=players, team=np.array([HOME, AWAY], np.int8), ball_xy=ball,
                          player_ids=["h", "a"], labels=labels, name="synth")


def test_labels_segments_and_merge():
    codes = np.array([0, 0, 0, 1, 0, 0, 2, 2, 1, 1, 1], np.int8)
    assert L.segments(codes) == [(0, 3, 0), (3, 4, 1), (4, 6, 0), (6, 8, 2), (8, 11, 1)]
    merged = L.merge_short_spells(codes, 2)
    assert L.segments(merged) == [(0, 6, 0), (6, 8, 2), (8, 11, 1)]
    assert L.codes_from_strings(["H", "A", "D", "X", "?"]).tolist() == [0, 1, 2, 3, 3]


def test_viterbi_recovers_clean_sequence_from_noise():
    rng = np.random.default_rng(1)
    true = np.repeat([0, 1, 2, 0], [120, 80, 50, 100])
    emit = np.full((len(true), 3), -1.0)
    emit[np.arange(len(true)), true] = 1.0
    emit += rng.normal(0, 1.5, emit.shape)
    hard = emit.argmax(1)
    vit = decode.viterbi_decode(emit, switch_cost=6.0)
    assert (vit == true).mean() > 0.95 > (hard == true).mean()
    vit_d = decode.viterbi_decode(emit, switch_cost=3.0, min_dur=10)
    assert all(b - a >= 10 for a, b, _ in L.segments(vit_d)[:-1])
    assert (vit_d == true).mean() > 0.95


def test_majority_smooth_matches_challenge_baseline_semantics():
    codes = np.array([2, 2, 0, 2, 2, 1, 1, 1, 2], np.int8)
    out = decode.majority_smooth(codes, 1)
    # ventana ±1: solo votan H y A; D si nadie vota; empate -> H
    assert out.tolist() == [2, 0, 0, 0, 1, 1, 1, 1, 1]


def test_min_segment_filter_removes_blips():
    codes = np.array([0] * 30 + [1] * 2 + [0] * 30 + [1] * 40, np.int8)
    out = decode.min_segment_filter(codes, 5)
    assert L.segments(out) == [(0, 62, 0), (62, 102, 1)]


def test_features_and_rules_on_synthetic():
    d = _synthetic()
    f = compute_features(d, FeatureConfig())
    assert f.ball_obs.all()
    assert (f.nearest_team[:250] == HOME).all() and (f.nearest_team[-250:] == AWAY).all()
    cfg = control.ControlConfig()
    pred = control.rule_labels(f, cfg)
    res = pm.evaluate(d.labels, pred)
    assert res["macro_f1"] > 0.8, res["macro_f1"]  # las reglas detectan la parada con 1,5 s de retraso
    emit = control.log_softmax(control.emission_scores(f, cfg))
    vit = decode.viterbi_decode(emit, switch_cost=4.0, runs=f.period_runs)
    assert pm.evaluate(d.labels, vit)["spell_f1_mean"] >= 0.9
    X = feature_matrix(f, d)
    assert X.shape == (d.n_frames, len(feature_names())) and np.isfinite(X).all()


def test_drop_ball_and_bridge_keeps_possession():
    d = _synthetic()
    dd = drop_ball(d, 0.4, seed=3)
    assert 0.3 < (~dd.ball_observed()).mean() < 0.55
    f = compute_features(dd)
    emit = control.log_softmax(control.emission_scores(f, control.ControlConfig()))
    vit = decode.viterbi_decode(emit, switch_cost=4.0, runs=f.period_runs)
    assert pm.evaluate(d.labels, vit)["spell_f1_mean"] >= 0.8
    an = anonymize_slots(d, cuts_per_minute=30, seed=0)
    assert an.n_slots == 4 and set(an.team.tolist()) <= {HOME, AWAY, -1}
    # las posiciones se conservan (solo cambian de columna)
    assert np.allclose(np.nansum(an.players_xy), np.nansum(d.players_xy))


def test_resample_and_expand_labels_roundtrip():
    d = _synthetic(T=100)
    s = resample(d, 5)
    assert s.n_frames == 20 and s.fps == 5.0
    full = expand_labels(s.frame_index, s.labels, d.frame_index)
    assert (full == d.labels).mean() > 0.9


def test_labels_from_events_follow_challenge_semantics():
    fi = np.arange(1, 101)
    per = np.ones(100, np.int16)
    ev = [
        {"team": HOME, "type": "SET PIECE", "subtype": "KICK OFF", "period": 1, "start": 10, "end": 10},
        {"team": HOME, "type": "PASS", "subtype": "", "period": 1, "start": 12, "end": 20},
        {"team": HOME, "type": "BALL LOST", "subtype": "INTERCEPTION", "period": 1, "start": 25, "end": 30},
        {"team": AWAY, "type": "RECOVERY", "subtype": "INTERCEPTION", "period": 1, "start": 30, "end": 30},
        {"team": AWAY, "type": "BALL OUT", "subtype": "", "period": 1, "start": 50, "end": 55},
        {"team": HOME, "type": "SET PIECE", "subtype": "THROW IN", "period": 1, "start": 70, "end": 70},
        {"team": HOME, "type": "PASS", "subtype": "", "period": 1, "start": 71, "end": 80},
    ]
    lab = io_metrica.labels_from_events(ev, fi, per)
    segs = L.segments(lab)
    assert segs == [(0, 9, L.D), (9, 29, L.H), (29, 54, L.A), (54, 69, L.D), (69, 80, L.H), (80, 100, L.D)]


def test_spell_metrics_vendored_api():
    gt = np.array([0] * 50 + [2] * 10 + [1] * 40, np.int8)
    assert pm.evaluate(gt, gt)["spell_f1_mean"] == 1.0
    assert pm.evaluate(gt, np.full(100, 0, np.int8))["spell_f1_mean"] < 0.5
