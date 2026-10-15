from .conftest import create_match, upload, wait_for


def test_analysis_lifecycle(client, sample_video):
    video = upload(client, sample_video)
    match = create_match(client, video["id"])
    mid = match["id"]

    assert client.get(f"/api/matches/{mid}/parts/full/analysis").status_code == 409  # aún nada

    r = client.post(f"/api/matches/{mid}/analysis")
    assert r.status_code == 202, r.text
    assert r.json()["status"] in ("pending", "processing")

    status = wait_for(client, mid)
    assert status["status"] == "completed", status
    part = status["parts"][0]
    assert part["started_at"] and part["finished_at"] and part["error"] is None

    match = client.get(f"/api/matches/{mid}").json()
    assert match["status"] == "completed" and match["parts"][0]["status"] == "completed"

    analysis = client.get(f"/api/matches/{mid}/parts/full/analysis").json()
    assert analysis["mode"] == "full-match" and analysis["frame_count"] == 30180
    labels = {e["label"] for e in analysis["events"]}
    assert "kick_off" in labels and "throw_in" in labels      # etiquetas canónicas
    assert analysis["events"] == sorted(analysis["events"], key=lambda e: e["start_s"])
    actions = analysis["ball_actions"]
    assert actions[0]["timestamp_s"] == round(actions[0]["frame_idx"] / 25.0, 3)
    assert any(a["tracker_id"] is None for a in actions)          # -1 -> null
    assert all(a["spot_score"] is not None for a in actions)
    assert analysis["teams"]["clusters"] == [0, 1]
    assert analysis["teams"]["assignments"]["1"] == 1
    assert all("jersey" in a for a in actions)

    tracklets = client.get(f"/api/matches/{mid}/parts/full/tracklets").json()
    assert tracklets and tracklets[0]["tracker_id"] == 1 and tracklets[0]["role"] == "player"
    assert tracklets[0]["first_frame"] == 0 and tracklets[0]["frame_count"] > 0
    assert any(t["jersey"] is not None for t in tracklets)            # el detector leyó algún dorsal en la muestra
    assert tracklets[0]["team_cluster"] == 1

    window = client.get(f"/api/matches/{mid}/parts/full/tracking", params={"from_s": 0, "to_s": 1}).json()
    assert window["fps"] == 25.0
    frames = window["frames"]
    assert [f["frame_idx"] for f in frames] == list(range(0, 26))
    first = frames[0]
    assert first["homography_ok"] is True
    assert first["persons"][0]["tracker_id"] == 1 and first["persons"][0]["field_xy"] is not None
    assert first["persons"][0]["role"] == "player"
    assert "possessor_tracker_id" in first

    # ventana demasiado grande / inválida
    assert client.get(f"/api/matches/{mid}/parts/full/tracking", params={"from_s": 0, "to_s": 60}).status_code == 422
    assert client.get(f"/api/matches/{mid}/parts/full/tracking", params={"from_s": 5, "to_s": 1}).status_code == 422

    # re-ingesta sin pipeline: mismo resultado
    r = client.post(f"/api/matches/{mid}/parts/full/reingest")
    assert r.status_code == 200 and r.json() == analysis

    # relanzar un análisis completado con el mismo modo no hace nada
    r = client.post(f"/api/matches/{mid}/analysis")
    assert r.status_code == 202 and r.json()["status"] == "completed"


def test_analysis_failure_is_reported(client, sample_video, monkeypatch):
    monkeypatch.setenv("FAKE_PIPELINE_FAIL", "1")
    video = upload(client, sample_video)
    match = create_match(client, video["id"])
    client.post(f"/api/matches/{match['id']}/analysis")
    status = wait_for(client, match["id"])
    assert status["status"] == "error"
    assert "exited with code 3" in status["parts"][0]["error"]
    assert "CUDA out of memory" in status["parts"][0]["error"]
    assert client.get(f"/api/matches/{match['id']}").json()["error_message"]


def test_clip_mode(client, sample_video):
    video = upload(client, sample_video)
    match = create_match(client, video["id"])
    client.post(f"/api/matches/{match['id']}/analysis", json={"mode": "clip"})
    assert wait_for(client, match["id"])["status"] == "completed"
    analysis = client.get(f"/api/matches/{match['id']}/parts/full/analysis").json()
    assert analysis["mode"] == "clip"
    assert [e["label"] for e in analysis["events"]] == ["shot"]
    assert analysis["events"][0]["end_s"] == 2.0


def test_start_twice_conflicts(client, sample_video):
    video = upload(client, sample_video)
    match = create_match(client, video["id"])
    assert client.post(f"/api/matches/{match['id']}/analysis").status_code == 202
    assert client.post(f"/api/matches/{match['id']}/analysis").status_code == 409
    wait_for(client, match["id"])


def test_skip_classification(tmp_path, sample_video, monkeypatch):
    import sys

    from fastapi.testclient import TestClient

    from app.core.config import Settings
    from app.main import create_app
    from .conftest import FAKE_PIPELINE

    settings = Settings(
        storage_dir=tmp_path / "storage", pipeline_command=f"{sys.executable} {FAKE_PIPELINE}", project_root=tmp_path,
        skip_classification=True, _env_file=None,
    )
    with TestClient(create_app(settings)) as c:
        video = upload(c, sample_video)
        match = create_match(c, video["id"])
        c.post(f"/api/matches/{match['id']}/analysis")
        assert wait_for(c, match["id"])["status"] == "completed"
        analysis = c.get(f"/api/matches/{match['id']}/parts/full/analysis").json()
        assert analysis["teams"] == {"clusters": [], "assignments": {}, "excluded": {}}
        assert all(a["team_cluster"] is None for a in analysis["ball_actions"])
