from .conftest import create_match, upload


def test_create_and_get_match(client, sample_video):
    video = upload(client, sample_video)
    match = create_match(client, video["id"])
    assert match["name"] == "Test match" and match["status"] == "pending"
    assert match["duration_s"] == 2.0
    part = match["parts"][0]
    assert part["id"] == "full" and part["kind"] == "full" and part["status"] == "pending"
    assert part["video"]["id"] == video["id"]

    assert client.get(f"/api/matches/{match['id']}").json() == match
    listed = client.get("/api/matches").json()
    assert [m["id"] for m in listed] == [match["id"]]


def test_create_match_with_halves(client, sample_video):
    v1, v2 = upload(client, sample_video), upload(client, sample_video)
    r = client.post("/api/matches", json={"name": "Halves", "parts": [
        {"kind": "first-half", "video_id": v1["id"]}, {"kind": "second-half", "video_id": v2["id"]}]})
    assert r.status_code == 201
    assert [p["id"] for p in r.json()["parts"]] == ["first-half", "second-half"]
    assert r.json()["duration_s"] == 4.0


def test_create_match_validation(client, sample_video):
    video = upload(client, sample_video)
    r = client.post("/api/matches", json={"name": "x", "parts": [{"kind": "full", "video_id": "0123456789abcdef"}]})
    assert r.status_code == 422
    r = client.post("/api/matches", json={"name": "x", "parts": [
        {"kind": "full", "video_id": video["id"]}, {"kind": "first-half", "video_id": video["id"]}]})
    assert r.status_code == 422
    assert client.get("/api/matches/0123456789abcdef").status_code == 404
