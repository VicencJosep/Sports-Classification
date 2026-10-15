import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

TESTS_DIR = Path(__file__).resolve().parent
FAKE_PIPELINE = TESTS_DIR / "fake_pipeline.py"


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """2 s de testsrc en H.264 a 25 fps (requiere ffmpeg)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    path = tmp_path_factory.mktemp("video") / "sample.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=25",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        storage_dir=tmp_path / "storage",
        pipeline_command=f"{sys.executable} {FAKE_PIPELINE}",
        project_root=tmp_path,
        max_upload_bytes=50 * 1024 * 1024,
        tracking_max_window_s=10.0,
        _env_file=None,
    )


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as c:
        yield c


def upload(client: TestClient, path: Path, filename: str | None = None) -> dict:
    with open(path, "rb") as f:
        r = client.post("/api/videos/upload", files={"file": (filename or path.name, f, "video/mp4")})
    assert r.status_code == 201, r.text
    return r.json()


def create_match(client: TestClient, video_id: str, kind: str = "full", **extra) -> dict:
    body = {"name": "Test match", "home_team": "Home", "away_team": "Away", "date": "2026-10-09",
            "parts": [{"kind": kind, "video_id": video_id}], **extra}
    r = client.post("/api/matches", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def wait_for(client: TestClient, match_id: str, statuses=("completed", "error"), timeout_s: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout_s
    status = None
    while time.monotonic() < deadline:
        status = client.get(f"/api/matches/{match_id}/analysis/status").json()
        if status["status"] in statuses:
            return status
        time.sleep(0.1)
    raise AssertionError(f"analysis did not reach {statuses} in time: {status}")
