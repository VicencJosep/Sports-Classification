"""Un fotograma del vídeo como JPEG pequeño, con ``ffmpeg``.

El editor de clips enseña una tira de miniaturas. Sacarlas en el navegador (un ``<video>`` aparte que va saltando
por el fichero) obliga a bajar el vídeo a trozos: decenas de MB cada vez que se abre el editor, que por un túnel
SSH atascan todo lo demás. Aquí cada miniatura son unos pocos KB y se guarda en disco.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from app.core.errors import MalformedOutputError

_TIMEOUT_S = 20


def grab_frame(path: Path, time_s: float, height: int, ffmpeg_bin: str = "ffmpeg") -> bytes:
    """JPEG del fotograma en ``time_s`` (segundos de vídeo), escalado a ``height`` px de alto."""
    command = [
        ffmpeg_bin, "-v", "error", "-nostdin", "-threads", "2",
        "-ss", f"{max(0.0, time_s):.3f}", "-i", str(path),      # -ss antes de -i: salta al keyframe y decodifica hasta el instante
        "-an", "-sn", "-frames:v", "1", "-vf", f"scale=-2:{height}", "-q:v", "6", "-f", "mjpeg", "pipe:1",
    ]
    try:
        proc = subprocess.run(command, capture_output=True, timeout=_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MalformedOutputError(f"could not run ffmpeg for a thumbnail: {exc}") from exc
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()[-1:] or ["no frame"]
        raise MalformedOutputError(f"ffmpeg produced no thumbnail at {time_s:.2f} s: {detail[0]}")
    return proc.stdout
