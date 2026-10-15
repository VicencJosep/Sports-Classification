from .conftest import upload


def test_upload_video(client, sample_video):
    video = upload(client, sample_video)
    assert video["original_filename"] == "sample.mp4"
    assert video["video_codec"] == "h264" and video["container"] == "mp4"
    assert video["width"] == 320 and video["height"] == 240
    assert video["fps"] == 25.0 and video["frame_count"] == 50 and video["duration_s"] == 2.0
    assert video["browser_playable"] is True
    assert video["content_url"] == f"/api/videos/{video['id']}/content"

    r = client.get(f"/api/videos/{video['id']}")
    assert r.status_code == 200 and r.json() == video


def test_upload_rejects_unknown_extension(client, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with open(bad, "rb") as f:
        r = client.post("/api/videos/upload", files={"file": ("notes.txt", f, "text/plain")})
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_media"


def test_upload_rejects_non_video_bytes(client, tmp_path):
    bad = tmp_path / "fake.mp4"
    bad.write_bytes(b"this is not a video" * 100)
    with open(bad, "rb") as f:
        r = client.post("/api/videos/upload", files={"file": ("fake.mp4", f, "video/mp4")})
    assert r.status_code == 415
    settings = client.app.state.services.settings
    assert list(settings.videos_dir.glob("*/")) == []          # nada queda almacenado
    assert list(settings.tmp_dir.iterdir()) == []


def test_video_content_supports_range(client, sample_video):
    video = upload(client, sample_video)
    full = client.get(video["content_url"])
    assert full.status_code == 200
    assert full.headers["content-type"] == "video/mp4"
    assert full.headers["accept-ranges"] == "bytes"
    assert len(full.content) == video["size_bytes"]

    partial = client.get(video["content_url"], headers={"Range": "bytes=0-99"})
    assert partial.status_code == 206
    assert partial.headers["content-range"] == f"bytes 0-99/{video['size_bytes']}"
    assert partial.content == full.content[:100]


def test_unknown_video(client):
    assert client.get("/api/videos/0123456789abcdef").status_code == 404
    assert client.get("/api/videos/not-an-id").status_code == 422
