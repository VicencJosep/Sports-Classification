"""Configuración del backend (variables de entorno con prefijo ``BACKEND_`` y
``backend/.env``). Separada a propósito del ``config.py`` del pipeline: aquí
solo vive lo que necesita la API, nunca hiperparámetros de modelos."""

from __future__ import annotations

import shlex
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR: Path = Path(__file__).resolve().parents[2]
PROJECT_ROOT: Path = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BACKEND_", env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    host: str = "127.0.0.1"          # el contenedor usa la red del host: no exponer por defecto
    port: int = 8100                  # 8000 está ocupado en esta máquina
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    public_base_url: str = ""         # prefijo de las URLs de contenido; vacío = relativas a la API

    storage_dir: Path = BACKEND_DIR / "storage"
    max_upload_bytes: int = 8 * 1024 ** 3
    allowed_extensions: str = ".mp4,.mkv,.mov,.webm,.avi"
    ffprobe_bin: str = "ffprobe"

    # Pipeline: se ejecuta como subproceso con el intérprete de ML, desde la raíz del repo.
    project_root: Path = PROJECT_ROOT
    pipeline_python: Path = Path("/opt/venv/bin/python")
    pipeline_command: str | None = None   # sustituye a "<pipeline_python> -m main" (tests)
    highlights_command: str | None = None  # sustituye a "<pipeline_python> -m analytics.highlights" (tests)
    render_video: bool = False            # final.mp4 no se reproduce en navegadores y pesa ~5 GB
    skip_homography: bool = False
    # Equipos (PRTReID + K-Means) exige el paquete `prtreid`, que no siempre está instalado en el
    # contenedor; con True el pipeline corre con --skip-classification (sin teams.json).
    skip_classification: bool = False

    tracking_max_window_s: float = 120.0

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_extension_set(self) -> frozenset[str]:
        return frozenset(e.strip().lower() for e in self.allowed_extensions.split(",") if e.strip())

    def pipeline_base_command(self) -> list[str]:
        if self.pipeline_command:
            return shlex.split(self.pipeline_command)
        return [str(self.pipeline_python), "-m", "main"]

    def highlights_base_command(self) -> list[str]:
        if self.highlights_command:
            return shlex.split(self.highlights_command)
        return [str(self.pipeline_python), "-m", "analytics.highlights"]

    @property
    def videos_dir(self) -> Path:
        return self.storage_dir / "videos"

    @property
    def matches_dir(self) -> Path:
        return self.storage_dir / "matches"

    @property
    def outputs_dir(self) -> Path:
        return self.storage_dir / "outputs"

    @property
    def tmp_dir(self) -> Path:
        return self.storage_dir / "tmp"


@lru_cache
def get_settings() -> Settings:
    return Settings()
