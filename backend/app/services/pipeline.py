"""Lanzamiento del pipeline existente como subproceso (``python -m main``).

Todo lo que el backend sabe de la IA está aquí: los argumentos del CLI y qué
ficheros va dejando cada etapa (para deducir una etapa aproximada sin tocar
el código del pipeline, que los escribe de forma atómica).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

from app.core.config import Settings
from app.schemas import AnalysisMode

# Orden real en main.py --mode full-match: T-DEED -> tracking -> posesión +
# acciones + eventos -> telemetry.json -> teams.json -> homografía -> render.
STAGES_FULL_MATCH = ("spotting", "tracking", "analysis", "teams", "homography", "finishing")


class PipelineLauncher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_command(self, source: Path, output_dir: Path, mode: AnalysisMode) -> list[str]:
        cmd = [*self.settings.pipeline_base_command(), "--mode", mode, "--source", str(source), "--output-dir", str(output_dir)]
        if not self.settings.render_video:
            cmd.append("--skip-render")
        if self.settings.skip_homography:
            cmd.append("--skip-homography")
        if self.settings.skip_classification:
            cmd.append("--skip-classification")
        return cmd

    def run(self, command: list[str], log_path: Path, on_start: Callable[[subprocess.Popen], None]) -> int:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        # La GPU es compartida: evitar fragmentación del allocator es gratis y a veces decide un OOM.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as log:
            proc = subprocess.Popen(
                command, cwd=str(self.settings.project_root), stdout=log, stderr=subprocess.STDOUT,
                env=env, start_new_session=True,
            )
            on_start(proc)
            return proc.wait()

    def pipeline_commit(self) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(self.settings.project_root), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() or None if out.returncode == 0 else None

    @staticmethod
    def detect_stage(
        source: Path, output_dir: Path, mode: AnalysisMode, skip_homography: bool, skip_classification: bool = False
    ) -> str:
        """Etapa en curso deducida de los ficheros que ya existen."""
        stem = source.stem
        video_dir = source.parent
        has = lambda p: p.exists()  # noqa: E731
        if mode == "full-match" and not any(video_dir.glob(f"{stem}.tdeed.*.npz")):
            return "spotting"
        if not has(video_dir / f"{stem}.tracks.jsonl"):
            return "tracking"
        if not has(output_dir / "telemetry.json"):
            return "analysis"
        if not skip_classification and not has(output_dir / "teams.json"):
            return "teams"
        if not skip_homography and not has(output_dir / "tracks_2d.jsonl"):
            return "homography"
        return "finishing"
