"""BoT-SORT (GMC + ReID) en ``core.tracking.PlayerTracker`` con un detector stub: sin GPU ni pesos.

    docker exec -w /app Sports-Classification python -m pytest -q tests/test_tracking_botsort.py
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from core.tracking import BALL_CLASS_ID, PlayerTracker, _with_role_data

W, H, N = 640, 360, 12
_RNG = np.random.default_rng(0)
_BACKDROP = _RNG.integers(0, 255, (H, W + 2 * N, 3), dtype=np.uint8)


def _frame(i: int) -> np.ndarray:
    """Fondo con textura (esquinas para el flujo óptico de la GMC) que panea 2 px por frame."""
    return np.ascontiguousarray(_BACKDROP[:, 2 * i: 2 * i + W])


class StubDetector:
    """Un jugador que cruza, un árbitro que sale del plano en el frame 6 y un balón."""

    chunk_size = 1

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frames):
        out = []
        for _ in frames:
            i = self.calls
            self.calls += 1
            boxes = [[100 + 3 * i, 100, 140 + 3 * i, 200]]
            conf, cls, roles, jerseys = [0.9], [3], ["player"], [10]
            if i != 6:
                boxes.append([400, 150 + 2 * i, 440, 250 + 2 * i])
                conf.append(0.8); cls.append(4); roles.append("referee"); jerseys.append(None)
            boxes.append([300, 50, 310, 60])
            conf.append(0.5); cls.append(BALL_CLASS_ID); roles.append("ball"); jerseys.append(None)
            d = sv.Detections(xyxy=np.array(boxes, dtype=float), confidence=np.array(conf), class_id=np.array(cls))
            out.append(_with_role_data(d, roles, jerseys))
        return out


class MeanColorEmbedder:
    """Embedder de juguete (protocolo core.classification.Embedder)."""

    embedding_dim = 3

    def __init__(self) -> None:
        self.calls = 0
        self.crops = 0

    def embed_batch(self, crops):
        self.calls += 1
        self.crops += len(crops)
        return np.stack([c.reshape(-1, 3).mean(axis=0) for c in crops]).astype(np.float32)


def _run(**kwargs):
    tracker = PlayerTracker(StubDetector(), frame_rate=25, **kwargs)
    results = []
    for i in range(N):
        (res,) = tracker.process_batch([_frame(i)], [i], [i / 25])
        results.append(res)
    return results


def _ids(results, role):
    return {t.tracker_id for r in results for t in r.tracks if t.role == role}


@pytest.mark.parametrize("kind", ["botsort", "bytetrack"])
def test_ids_stable_and_detection_data_preserved(kind):
    results = _run(tracker=kind, reid="none")
    # Un frame de activación como mucho; después, una persona por detección (dos salvo en el frame 6).
    for i, r in enumerate(results[2:], start=2):
        assert len(r.tracks) == (1 if i == 6 else 2), (kind, i, r.tracks)
    assert len(_ids(results[2:], "player")) == 1
    # El árbitro sale un frame y vuelve con el mismo id (buffer de tracks perdidos).
    assert len(_ids(results[2:], "referee")) == 1
    for r in results:
        assert all(t.class_id != BALL_CLASS_ID for t in r.tracks)
        assert r.ball is not None and r.ball.confidence == pytest.approx(0.5)
        for t in r.tracks:
            assert t.jersey == (10 if t.role == "player" else None)
            assert t.anchor_xy == ((t.xyxy[0] + t.xyxy[2]) / 2, t.xyxy[3])


def test_botsort_emits_the_detected_box_not_the_kalman_prediction():
    results = _run(tracker="botsort", reid="none")
    player = [t for t in results[5].tracks if t.role == "player"][0]
    assert player.xyxy == (115.0, 100.0, 155.0, 200.0)
    assert player.confidence == pytest.approx(0.9)
    assert results[5].detections.tracker_id is not None and len(results[5].detections) == 2


def test_botsort_with_injected_embedder_calls_reid_per_frame():
    emb = MeanColorEmbedder()
    tracker = PlayerTracker(StubDetector(), frame_rate=25, tracker="botsort", embedder=emb)
    assert tracker.tracker.with_reid and tracker.needs_frames
    results = []
    for i in range(N):
        (res,) = tracker.process_batch([_frame(i)], [i], [i / 25])
        results.append(res)
    assert emb.calls >= N                   # al menos una llamada por frame (personas de confianza alta)
    assert emb.crops == 2 * N - 1           # un crop por persona detectada; el balón nunca entra
    assert len(_ids(results[2:], "player")) == 1 and len(_ids(results[2:], "referee")) == 1


def test_botsort_reset_restarts_ids_and_survives_empty_frames():
    tracker = PlayerTracker(StubDetector(), frame_rate=25, tracker="botsort", reid="none", gmc_method="none")
    (first,) = tracker.process_batch([_frame(0)], [0], [0.0])
    tracker.tracker.reset()
    empty = _with_role_data(sv.Detections.empty(), [], [])
    res = tracker._track_frame(empty, _frame(1), 1, 0.04)
    assert res.tracks == [] and res.ball is None
    # Tras el reset los ids vuelven a empezar; un track nuevo tarda dos frames en confirmarse.
    tracker.process_batch([_frame(2)], [2], [0.08])
    (again,) = tracker.process_batch([_frame(3)], [3], [0.12])
    assert {t.tracker_id for t in again.tracks} == {t.tracker_id for t in first.tracks} == {1, 2}


def test_missing_osnet_weights_fall_back_to_no_reid(tmp_path):
    tracker = PlayerTracker(StubDetector(), frame_rate=25, tracker="botsort", reid="osnet",
                            reid_weights_path=tmp_path / "no_existe.pth")
    assert tracker.tracker.with_reid is False


def test_unknown_tracker_is_rejected():
    with pytest.raises(ValueError):
        PlayerTracker(StubDetector(), tracker="sort")
