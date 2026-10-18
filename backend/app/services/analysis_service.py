"""Orquesta el análisis: cola en memoria, un pipeline a la vez (la GPU es una
y compartida), estado persistido en ``run.json`` y, al terminar, ingesta en
Football Memory.

El trabajador es un hilo: el pipeline tarda horas y ``subprocess.wait`` es
bloqueante; así las rutas síncronas de FastAPI no se ven afectadas.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

from app.core.config import Settings
from app.core.errors import ConflictError, NotFoundError
from app.repositories.records import MatchRepository, RunStore, VideoRepository
from app.schemas import (
    AnalysisMode, AnalysisRun, AnalysisStatusResponse, MatchRecord, PartAnalysisStatus,
)
from app.schemas.common import utcnow_iso

from .football_memory import FootballMemory
from .pipeline import PipelineLauncher

logger = logging.getLogger(__name__)

_LOG_TAIL_BYTES = 4000


def derive_status(statuses: list[str]) -> str:
    if not statuses:
        return "pending"
    for s in ("error", "processing", "pending"):
        if s in statuses:
            return s
    return "completed"


class AnalysisService:
    def __init__(
        self, settings: Settings, matches: MatchRepository, videos: VideoRepository, runs: RunStore,
        launcher: PipelineLauncher, memory: FootballMemory, gpu_lock: threading.Lock | None = None,
    ) -> None:
        self.settings = settings
        self.matches = matches
        self.videos = videos
        self.runs = runs
        self.launcher = launcher
        self.memory = memory
        # Compartido con HighlightsService: el pipeline y el VLM no caben a la vez en la GPU.
        self.gpu_lock = gpu_lock
        # Llamado (en el hilo trabajador) cuando un análisis termina bien con
        # ``highlights_requested``; lo pone deps.build_services.
        self.on_completed: Callable[[AnalysisRun], None] | None = None
        self._queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._current: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # --- ciclo de vida ----------------------------------------------------- #

    def start_worker(self) -> None:
        self.recover()
        self._thread = threading.Thread(target=self._loop, name="analysis-worker", daemon=True)
        self._thread.start()

    def stop_worker(self) -> None:
        self._queue.put(None)
        with self._lock:
            proc = self._current
        if proc is not None and proc.poll() is None:
            proc.terminate()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def recover(self) -> None:
        """Al arrancar: lo que estaba en marcha se da por interrumpido (el
        subproceso ya no es nuestro) y lo pendiente vuelve a la cola."""
        for run in self.runs.list_all():
            if run.status == "processing":
                run.status, run.error, run.finished_at = "error", "interrupted by a backend restart; start the analysis again", utcnow_iso()
                self.runs.save(run)
            elif run.status == "pending":
                self._queue.put((run.match_id, run.part_id))

    # --- API -------------------------------------------------------------- #

    def start(self, match: MatchRecord, mode: AnalysisMode, highlights: bool = False) -> AnalysisStatusResponse:
        """Encola el análisis de las partes que no estén ya completadas.
        Devuelve también las partes ya completadas (la ruta encadena los
        highlights de esas directamente, porque aquí no se vuelven a ejecutar)."""
        current = [self.runs.get(match.id, p.id) for p in match.parts]
        if any(r is not None and r.status in ("processing", "pending") for r in current):
            raise ConflictError(f"match {match.id} already has an analysis in progress")
        for part in match.parts:
            run = self.runs.get(match.id, part.id)
            if run is not None and run.status == "completed" and run.mode == mode:
                continue
            if self.videos.get(part.video_id) is None:
                raise NotFoundError(f"video {part.video_id} of part {part.id} does not exist")
            self.runs.save(AnalysisRun(
                match_id=match.id, part_id=part.id, video_id=part.video_id, mode=mode, created_at=utcnow_iso(),
                highlights_requested=highlights,
            ))
            self._queue.put((match.id, part.id))
        return self.status(match)

    def completed_parts(self, match: MatchRecord) -> list[str]:
        return [p.id for p in match.parts if (r := self.runs.get(match.id, p.id)) is not None and r.status == "completed"]

    def status(self, match: MatchRecord) -> AnalysisStatusResponse:
        parts = []
        for part in match.parts:
            run = self.runs.get(match.id, part.id)
            if run is None:
                parts.append(PartAnalysisStatus(part_id=part.id, status="pending"))
                continue
            parts.append(PartAnalysisStatus(
                part_id=part.id, status=run.status, stage=self.stage_of(run),
                started_at=run.started_at, finished_at=run.finished_at, error=run.error,
            ))
        return AnalysisStatusResponse(match_id=match.id, status=derive_status([p.status for p in parts]), parts=parts)

    def stage_of(self, run: AnalysisRun) -> str | None:
        if run.status != "processing":
            return None
        if run.stage == "ingest":
            return "ingest"
        video = self.videos.get(run.video_id)
        if video is None:
            return None
        source = self.settings.videos_dir / video.id / video.stored_filename
        return self.launcher.detect_stage(
            source, self.memory.output_dir(run.match_id, run.part_id), run.mode,
            self.settings.skip_homography, self.settings.skip_classification,
        )

    # --- trabajador ------------------------------------------------------- #

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._execute(*item)
            except Exception:  # noqa: BLE001 - el hilo no debe morir nunca
                logger.exception("analysis worker failed on %s", item)

    def _execute(self, match_id: str, part_id: str) -> None:
        run = self.runs.get(match_id, part_id)
        if run is None or run.status != "pending":
            return
        video = self.videos.get(run.video_id)
        if video is None:
            self._finish(run, error=f"video {run.video_id} does not exist")
            return
        part_dir = self.runs.part_dir(match_id, part_id)
        output_dir = self.memory.output_dir(match_id, part_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        source = self.settings.videos_dir / video.id / video.stored_filename
        command = self.launcher.build_command(source, output_dir, run.mode)
        log_path = part_dir / "pipeline.log"

        run.status, run.started_at, run.command = "processing", utcnow_iso(), command
        run.stage, run.error, run.exit_code, run.finished_at = "pipeline", None, None, None
        run.pipeline_commit = self.launcher.pipeline_commit()
        self.runs.save(run)
        logger.info("analysis %s/%s: %s", match_id, part_id, " ".join(command))

        def _on_start(proc: subprocess.Popen) -> None:
            with self._lock:
                self._current = proc
            run.pid = proc.pid
            self.runs.save(run)

        try:
            with (self.gpu_lock or nullcontext()):
                exit_code = self.launcher.run(command, log_path, _on_start)
        except OSError as exc:
            self._finish(run, error=f"could not start the pipeline: {exc}")
            return
        finally:
            with self._lock:
                self._current = None

        run.exit_code = exit_code
        if exit_code != 0:
            self._finish(run, error=f"pipeline exited with code {exit_code}: {_log_tail(log_path)}")
            return
        run.stage = "ingest"
        self.runs.save(run)
        try:
            self.memory.ingest(run, video)
        except Exception as exc:  # noqa: BLE001 - cualquier fallo de ingesta es un análisis fallido
            self._finish(run, error=f"ingest failed: {exc}")
            return
        self._finish(run, error=None)

    def _finish(self, run: AnalysisRun, error: str | None) -> None:
        run.status = "error" if error else "completed"
        run.error, run.stage, run.pid, run.finished_at = error, None, None, utcnow_iso()
        self.runs.save(run)
        logger.info("analysis %s/%s finished: %s", run.match_id, run.part_id, run.status)
        if error is None and run.highlights_requested and self.on_completed is not None:
            try:
                self.on_completed(run)
            except Exception:  # noqa: BLE001 - los highlights no deben tumbar el análisis ya completado
                logger.exception("could not chain highlights for %s/%s", run.match_id, run.part_id)


def _log_tail(log_path: Path) -> str:
    try:
        data = log_path.read_bytes()[-_LOG_TAIL_BYTES:]
    except OSError:
        return "(no log)"
    text = data.decode("utf-8", errors="replace")
    lines = [l for l in text.splitlines() if l.strip()]
    for key in ("OutOfMemoryError", "CUDA out of memory"):
        if key in text:
            return "CUDA out of memory (the GPU is shared; retry when it is free). " + " | ".join(lines[-3:])
    return " | ".join(lines[-5:]) or "(empty log)"
