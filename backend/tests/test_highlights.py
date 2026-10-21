import time

from .conftest import create_match, upload, wait_for


def wait_for_highlights(client, mid: str, part: str = "full", timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = client.get(f"/api/matches/{mid}/parts/{part}/highlights")
        if r.status_code == 200 and r.json()["status"] in ("completed", "error"):
            return r.json()
        time.sleep(0.1)
    raise AssertionError("highlights did not finish in time")


def test_highlights_chained_to_analysis(client, sample_video):
    video = upload(client, sample_video)
    mid = create_match(client, video["id"])["id"]

    # Sin análisis completado no se pueden generar, y sin generar no hay nada que leer.
    assert client.post(f"/api/matches/{mid}/parts/full/highlights").status_code == 409
    assert client.get(f"/api/matches/{mid}/parts/full/highlights").status_code == 404
    assert client.get(f"/api/matches/{mid}").json()["parts"][0]["highlights"] is None

    r = client.post(f"/api/matches/{mid}/analysis", json={"mode": "full-match", "highlights": True})
    assert r.status_code == 202, r.text
    assert wait_for(client, mid)["status"] == "completed"

    result = wait_for_highlights(client, mid)
    assert result["status"] == "completed", result
    assert result["vlm_model"] == "fake-vlm" and result["n_candidates"] == 7
    # Origen y estado degradado, tal cual los deja highlights.json (la UI los enseña).
    assert result["vlm_used"] is True and result["vlm_degraded"] is True and result["vjepa_used"] is False
    assert result["scoreboard_available"] is True and result["n_scoreboard_changes"] == 1
    clip = result["clips"][0]
    assert clip["id"] == "hl-01" and clip["label"] == "goal" and clip["team_cluster"] == 1
    assert clip["goal_source"] == "restart"
    assert clip["content_url"] == f"/api/matches/{mid}/parts/full/highlights/hl-01/content"

    summary = client.get(f"/api/matches/{mid}").json()["parts"][0]["highlights"]
    assert summary == {"status": "completed", "stage": None, "n_clips": 1, "error": None}

    content = client.get(clip["content_url"])
    assert content.status_code == 200 and content.headers["content-type"] == "video/mp4"
    partial = client.get(clip["content_url"], headers={"Range": "bytes=0-99"})
    assert partial.status_code == 206 and len(partial.content) == 100
    assert client.get(f"/api/matches/{mid}/parts/full/highlights/hl-02/content").status_code == 404

    # Regenerar sobre una parte ya analizada: vuelve a pasar por pending/processing -> completed.
    r = client.post(f"/api/matches/{mid}/parts/full/highlights", json={"force": True})
    assert r.status_code == 202 and r.json()["status"] in ("pending", "processing")
    assert client.post(f"/api/matches/{mid}/parts/full/highlights").status_code == 409   # ya en marcha
    assert wait_for_highlights(client, mid)["status"] == "completed"


def test_highlights_failure_is_reported(client, sample_video, monkeypatch):
    monkeypatch.setenv("FAKE_HIGHLIGHTS_FAIL", "1")
    video = upload(client, sample_video)
    mid = create_match(client, video["id"])["id"]
    assert client.post(f"/api/matches/{mid}/analysis").status_code == 202
    assert wait_for(client, mid)["status"] == "completed"

    assert client.post(f"/api/matches/{mid}/parts/full/highlights").status_code == 202
    result = wait_for_highlights(client, mid)
    assert result["status"] == "error" and "code 2" in result["error"] and "VLM" in result["error"]
    assert result["clips"] == []
    assert client.get(f"/api/matches/{mid}/parts/full/highlights/hl-01/content").status_code == 409


def test_analysis_with_highlights_on_completed_part(client, sample_video):
    """Pedir análisis + highlights sobre una parte ya completada no relanza el pipeline pero sí encola los highlights."""
    video = upload(client, sample_video)
    mid = create_match(client, video["id"])["id"]
    assert client.post(f"/api/matches/{mid}/analysis").status_code == 202
    assert wait_for(client, mid)["status"] == "completed"
    assert client.get(f"/api/matches/{mid}/parts/full/highlights").status_code == 404

    r = client.post(f"/api/matches/{mid}/analysis", json={"highlights": True})
    assert r.status_code == 202 and r.json()["status"] == "completed"
    assert wait_for_highlights(client, mid)["status"] == "completed"
