"""Copia revisada: corregir, borrar, añadir y exportar la hoja de verdad."""

import importlib.util
import json
from pathlib import Path

import pytest

from app.core.config import PROJECT_ROOT

from .conftest import create_match, upload, wait_for


@pytest.fixture
def analysed(client, sample_video):
    video = upload(client, sample_video)
    match = create_match(client, video["id"])
    client.post(f"/api/matches/{match['id']}/analysis")
    assert wait_for(client, match["id"])["status"] == "completed"
    return match["id"]


def _review_dir(settings, match_id: str) -> Path:
    return settings.storage_dir / "outputs" / match_id / "full" / "review"


def test_review_starts_as_the_pipeline_output(client, analysed, settings):
    base = f"/api/matches/{analysed}/parts/full"
    analysis = client.get(f"{base}/analysis").json()
    review = client.get(f"{base}/review").json()
    assert review["review"]["exists"] is False and review["review"]["n_edited"] == 0
    assert [a["id"] for a in review["ball_actions"]] == [a["id"] for a in analysis["ball_actions"]]
    assert [e["id"] for e in review["events"]] == [e["id"] for e in analysis["events"]]
    assert all(a["review"] is None and a["side"] is None for a in review["ball_actions"])
    assert not _review_dir(settings, analysed).exists()            # leer no crea la copia


def test_correct_a_ball_action(client, analysed, settings):
    base = f"/api/matches/{analysed}/parts/full"
    original = client.get(f"{base}/analysis").json()

    # etiqueta + ejecutor + lado en una sola corrección
    r = client.patch(f"{base}/review/items/act-22", json={"label": "High pass", "tracker_id": 3, "side": "away"})
    assert r.status_code == 200, r.text
    body = r.json()
    action = body["action"]
    assert body["kind"] == "action" and body["event"] is None and body["deleted"] is False
    assert action["label"] == "high_pass" and action["review"] == "edited"
    assert action["confidence"] == 1.0 and action["probabilities"] == {"high_pass": 1.0}
    assert action["tracker_id"] == 3 and action["team_cluster"] == 1 and action["side"] == "away"
    assert body["review"]["exists"] is True and body["review"]["n_edited"] == 1
    assert body["review"]["tracker_sides"] == {"3": "away"}

    # el análisis del pipeline no cambia; la copia sí
    assert client.get(f"{base}/analysis").json() == original
    stored = json.loads((_review_dir(settings, analysed) / "analysis.json").read_text())
    assert next(a for a in stored["ball_actions"] if a["id"] == "act-22")["label"] == "high_pass"
    review = client.get(f"{base}/review").json()
    assert next(a for a in review["ball_actions"] if a["id"] == "act-22")["label"] == "high_pass"
    assert review["review"]["exists"] is True and review["review"]["stale"] is False

    # mover el instante: el frame sale del fps real y la lista sigue ordenada
    r = client.patch(f"{base}/review/items/act-22", json={"time_s": 2.52})
    action = r.json()["action"]
    assert action["frame_idx"] == 63 and action["timestamp_s"] == 2.52 and action["label"] == "high_pass"
    frames = [a["frame_idx"] for a in client.get(f"{base}/review").json()["ball_actions"]]
    assert frames == sorted(frames) and 63 in frames and 22 not in frames

    # ejecutor desconocido (null explícito) frente a campo ausente (no se toca)
    r = client.patch(f"{base}/review/items/act-22", json={"tracker_id": None})
    assert r.json()["action"]["tracker_id"] is None and r.json()["action"]["team_cluster"] is None
    assert r.json()["action"]["side"] == "away"

    log = (_review_dir(settings, analysed) / "edits.jsonl").read_text().splitlines()
    assert [json.loads(line)["op"] for line in log] == ["update", "update", "update"]
    assert json.loads(log[0])["before"]["label"] == "pass" and json.loads(log[0])["after"]["label"] == "high_pass"


def test_a_correction_without_changes_does_not_create_the_copy(client, analysed, settings):
    base = f"/api/matches/{analysed}/parts/full"
    r = client.patch(f"{base}/review/items/act-22", json={"label": "pass"})
    assert r.status_code == 200 and r.json()["action"]["review"] is None
    assert r.json()["review"]["exists"] is False
    assert not _review_dir(settings, analysed).exists()


def test_correct_an_event(client, analysed):
    base = f"/api/matches/{analysed}/parts/full"
    events = client.get(f"{base}/review").json()["events"]
    foul = next(e for e in events if e["label"] == "foul")           # tramo 24-30 s -> instante 25 s
    r = client.patch(f"{base}/review/items/{foul['id']}", json={"label": "corner", "time_s": 27.0, "tracker_id": 2})
    assert r.status_code == 200, r.text
    event = r.json()["event"]
    assert event["label"] == "corner" and event["review"] == "edited" and event["peak_confidence"] == 1.0
    assert event["start_s"] == 26.0 and event["end_s"] == 32.0       # el tramo se desplaza entero
    assert event["tracker_id"] == 2 and event["team_cluster"] == 0
    starts = [e["start_s"] for e in client.get(f"{base}/review").json()["events"]]
    assert starts == sorted(starts)


def test_add_delete_and_restore(client, analysed):
    base = f"/api/matches/{analysed}/parts/full"

    r = client.post(f"{base}/review/items", json={"kind": "action", "label": "drive", "time_s": 6.0, "tracker_id": 1, "side": "home"})
    assert r.status_code == 201, r.text
    added = r.json()["action"]
    assert added["id"].startswith("act-new-") and added["review"] == "added" and added["frame_idx"] == 150
    assert added["team_cluster"] == 1 and added["spot_score"] is None and r.json()["review"]["n_added"] == 1

    r = client.post(f"{base}/review/items", json={"kind": "event", "label": "shot", "time_s": 50.0})
    event = r.json()["event"]
    assert event["id"].startswith("evt-new-") and (event["start_s"], event["end_s"]) == (49.0, 51.0)
    assert r.json()["review"]["n_added"] == 2

    # borrar algo del pipeline y deshacerlo
    r = client.delete(f"{base}/review/items/act-50")
    assert r.status_code == 200 and r.json()["deleted"] is True and r.json()["review"]["n_deleted"] == 1
    ids = [a["id"] for a in client.get(f"{base}/review").json()["ball_actions"]]
    assert "act-50" not in ids and added["id"] in ids
    assert client.patch(f"{base}/review/items/act-50", json={"label": "pass"}).status_code == 404

    r = client.post(f"{base}/review/items/act-50/restore")
    assert r.status_code == 200 and r.json()["action"]["id"] == "act-50" and r.json()["review"]["n_deleted"] == 0
    frames = [a["frame_idx"] for a in client.get(f"{base}/review").json()["ball_actions"]]
    assert 50 in frames and frames == sorted(frames)
    assert client.post(f"{base}/review/items/act-50/restore").status_code == 404

    # borrar algo añadido a mano no cuenta como falso positivo del pipeline
    r = client.delete(f"{base}/review/items/{added['id']}")
    assert r.json()["review"]["n_deleted"] == 0 and r.json()["review"]["n_added"] == 1


def test_invalid_requests(client, analysed, sample_video):
    base = f"/api/matches/{analysed}/parts/full"
    assert client.patch(f"{base}/review/items/act-22", json={"label": "dribble"}).status_code == 422
    assert client.patch(f"{base}/review/items/act-22", json={"side": "left"}).status_code == 422
    assert client.patch(f"{base}/review/items/act-999999", json={"label": "pass"}).status_code == 404
    assert client.post(f"{base}/review/items", json={"kind": "action", "label": "pass", "time_s": -1}).status_code == 422
    assert client.post(f"{base}/review/export").status_code == 409             # aún no hay copia

    video = upload(client, sample_video)
    pending = create_match(client, video["id"])
    assert client.get(f"/api/matches/{pending['id']}/parts/full/review").status_code == 409


def test_export_ground_truth_sheet(client, analysed, settings):
    base = f"/api/matches/{analysed}/parts/full"
    # tracklet 6 = local: vale para sus dos acciones (frames 50 y 72), aunque solo se toque una
    client.patch(f"{base}/review/items/act-50", json={"side": "home"})
    client.patch(f"{base}/review/items/act-104", json={"label": "player_successful_tackle"})
    client.delete(f"{base}/review/items/act-190")
    # un saque de banda como acción junto al evento de banda del pipeline (tramo 33-42 s, instante 34 s): una fila
    client.post(f"{base}/review/items", json={"kind": "action", "label": "throw_in", "time_s": 34.4, "tracker_id": 9, "side": "away"})
    client.post(f"{base}/review/items", json={"kind": "event", "label": "goal", "time_s": 50.0, "tracker_id": 5})

    r = client.post(f"{base}/review/export", json={"from_s": 0, "to_s": 52})
    assert r.status_code == 200, r.text
    export = r.json()
    lines = export["csv"].splitlines()
    assert lines[0] == "tiempo,etiqueta,dorsal,equipo"
    assert "00:02.00,drive,6,home" in lines and "00:02.88,pass,6,home" in lines     # lado heredado del tracklet
    assert "00:04.16,tackle,20," in lines
    assert not any(",high pass," in line for line in lines)                          # borrada
    assert "00:34.40,throw in,9,away" in lines
    assert sum(1 for line in lines if ",throw in," in line) == 1                     # evento + acción = una fila
    assert "00:50.00,goal,5," in lines
    assert "00:00.88,pass,14," in lines and "00:01.00,kick off,," in lines
    assert not any(line.startswith("00:55") or line.startswith("01:") for line in lines)   # fuera del tramo
    assert export["n_rows"] == len(lines) - 1
    times = [line.split(",")[0] for line in lines[1:]]
    assert times == sorted(times)

    saved = _review_dir(settings, analysed) / export["filename"]
    assert export["filename"] == "ground_truth_00m00-00m52.csv" and saved.read_text() == export["csv"]

    # la hoja la entiende el lector de anotaciones del proyecto, sin avisos
    reader_path = PROJECT_ROOT / "scripts" / "annotation_csv.py"
    if not reader_path.exists():
        pytest.skip("scripts/annotation_csv.py not available")
    spec = importlib.util.spec_from_file_location("annotation_csv", reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    rows, warnings = reader.read_annotation_csv(saved)
    assert warnings == [] and len(rows) == export["n_rows"]
    tackle = next(row for row in rows if row["label_raw"] == "tackle")
    assert tackle["ball_action_label"] == "player_successful_tackle" and tackle["tracker_id"] == 20
    goal = next(row for row in rows if row["label_raw"] == "goal")
    assert goal["event_label"] == "goal" and goal["t_s"] == 50.0

    # sin to_s: hasta el final del vídeo (2 s en la muestra)
    r = client.post(f"{base}/review/export", json={"from_s": 1})
    assert r.json()["to_s"] == 2.0 and r.json()["filename"] == "ground_truth_00m01-00m02.csv"
    assert client.post(f"{base}/review/export", json={"from_s": 10, "to_s": 5}).status_code == 422
