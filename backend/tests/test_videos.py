from .conftest import upload


def test_upload_video(client, sample_video):
    video = upload(client, sample_video)
    assert video["original_filename"] == "sample.mp4"
    assert video["video_codec"] == "h264" and video["container"] == "mp4"
    assert video["width"] == 320 and video["height"] == 240
    assert video["fps"] == 25.0 and video["frame_count"] == 50 and video["duration_s"] == 2.0
    assert video["browser_playable"] is True
    assert video["content_url"] == f"/api/videos/{video['id']}/content"
    assert video["thumbnail_url"] == f"/api/videos/{video['id']}/thumbnail"

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


def test_video_range_responses_are_capped(client, sample_video):
    """El navegador pide "desde aquí hasta el final": se le da un trozo y vuelve a pedir, en vez de dejar la
    conexión abierta con el resto del fichero."""
    video = upload(client, sample_video)
    size = video["size_bytes"]
    full = client.get(video["content_url"]).content
    client.app.state.services.settings.video_range_chunk_bytes = 1000

    r = client.get(video["content_url"], headers={"Range": "bytes=100-"})
    assert r.status_code == 206 and r.headers["content-range"] == f"bytes 100-1099/{size}"
    assert r.headers["content-length"] == "1000" and r.content == full[100:1100]
    assert r.headers["content-type"] == "video/mp4" and r.headers["accept-ranges"] == "bytes"

    # un rango cerrado más largo que el tope también se acorta; uno corto se sirve exacto
    r = client.get(video["content_url"], headers={"Range": f"bytes=0-{size - 1}"})
    assert r.headers["content-range"] == f"bytes 0-999/{size}" and r.content == full[:1000]
    r = client.get(video["content_url"], headers={"Range": "bytes=5-14"})
    assert r.headers["content-range"] == f"bytes 5-14/{size}" and r.content == full[5:15]

    # final del fichero: lo que quede; sufijo; y recorriendo trozo a trozo sale el fichero entero
    r = client.get(video["content_url"], headers={"Range": f"bytes={size - 10}-"})
    assert r.headers["content-range"] == f"bytes {size - 10}-{size - 1}/{size}" and r.content == full[-10:]
    r = client.get(video["content_url"], headers={"Range": "bytes=-20"})
    assert r.status_code == 206 and r.content == full[-20:]
    pieces, position = [], 0
    while position < size:
        r = client.get(video["content_url"], headers={"Range": f"bytes={position}-"})
        assert r.status_code == 206
        pieces.append(r.content)
        position += len(r.content)
    assert b"".join(pieces) == full

    # fuera del fichero: 416, como antes; sin Range: el fichero entero; tope 0: sin acortar
    assert client.get(video["content_url"], headers={"Range": f"bytes={size + 5}-"}).status_code == 416
    assert len(client.get(video["content_url"]).content) == size
    client.app.state.services.settings.video_range_chunk_bytes = 0
    r = client.get(video["content_url"], headers={"Range": "bytes=100-"})
    assert r.status_code == 206 and len(r.content) == size - 100


def test_unknown_video(client):
    assert client.get("/api/videos/0123456789abcdef").status_code == 404
    assert client.get("/api/videos/not-an-id").status_code == 422


def test_video_thumbnail(client, sample_video):
    video = upload(client, sample_video)
    settings = client.app.state.services.settings
    r = client.get(video["thumbnail_url"], params={"t": 1.04, "h": 72})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"
    assert "immutable" in r.headers["cache-control"]
    assert len(r.content) < 20_000                               # unos KB, no un trozo de vídeo

    # los instantes se redondean a décimas: 1,04 y 0,96 son la misma miniatura, ya guardada en disco
    cached = settings.videos_dir / video["id"] / "thumbs" / "h72" / "0000010.jpg"
    assert cached.read_bytes() == r.content
    cached.write_bytes(b"\xff\xd8cached")
    assert client.get(video["thumbnail_url"], params={"t": 0.96, "h": 72}).content == b"\xff\xd8cached"

    # más allá del final: el último fotograma, no un error
    r = client.get(video["thumbnail_url"], params={"t": 999, "h": 72})
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"
    assert client.get(video["thumbnail_url"]).status_code == 422               # falta t
    assert client.get(video["thumbnail_url"], params={"t": -1}).status_code == 422
    assert client.get(video["thumbnail_url"], params={"t": 1, "h": 4000}).status_code == 422
    assert client.get("/api/videos/0123456789abcdef/thumbnail", params={"t": 1}).status_code == 404
