"""Recorte de clips con ffmpeg: recodificando (corte exacto al frame, no al
keyframe), audio original, resolución original."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def cut_clip(
    video: Path, start_s: float, end_s: float, out_path: Path,
    crf: int = 20, preset: str = "medium", fade_s: float = 0.4,
) -> Path:
    duration = max(0.5, end_s - start_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.mp4")
    vf = f"fade=t=in:st=0:d={fade_s},fade=t=out:st={max(0.0, duration - fade_s):.3f}:d={fade_s}"
    af = f"afade=t=in:st=0:d={fade_s},afade=t=out:st={max(0.0, duration - fade_s):.3f}:d={fade_s}"
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-ss", f"{start_s:.3f}", "-i", str(video), "-t", f"{duration:.3f}",
        "-vf", vf, "-af", af,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace")
        # Sin pista de audio el filtro afade falla: reintento sin audio.
        if "afade" in err or "Stream specifier" in err or "does not contain any stream" in err:
            cmd_noaudio = [c for c in cmd if c not in ("-af", af)]
            cmd_noaudio = [c for i, c in enumerate(cmd_noaudio) if not (c in ("-c:a", "aac", "-b:a", "160k"))]
            cmd_noaudio.insert(cmd_noaudio.index("-c:v"), "-an")
            proc = subprocess.run(cmd_noaudio, capture_output=True)
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg falló recortando {out_path.name}: {proc.stderr.decode(errors='replace')[-400:]}")
    tmp.replace(out_path)
    logger.info("Clip %s: %.1f s", out_path.name, duration)
    return out_path
