"""Generación de las mejores jugadas de una parte ya analizada, como
subproceso (``python -m analytics.highlights``) en una cola propia.

Comparte con ``AnalysisService`` el candado de la GPU: el VLM (Ollama) y el
pipeline no caben a la vez en la tarjeta, así que nunca corren juntos. El
resto del flujo calca al del análisis: ``highlights_run.json`` por parte,
log en ``highlights.log``, la etapa en curso la escribe el propio módulo en
``highlights/status.json``.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
from contextlib import nullcontext
from pathlib import Path

from app.core.config import Settings
from app.core.errors import ConflictError, NotFoundError, NotReadyError
from app.repositories.json_store import read_json, write_json_atomic
from app.repositories.records import RunStore
from app.schemas.common import utcnow_iso
from app.schemas.highlights import (
    HighlightClip, HighlightsResponse, HighlightsRun, HighlightsSummary,
)

logger = logging.getLogger(__name__)

HIGHLIGHTS_DIRNAME = "highlights"
_LOG_TAIL_BYTES = 3000


class HighlightsService:
    def __init__(self, settings: Settings, runs: RunStore, gpu_lock: threading.Lock | None = None) -> None:
        self.settings = settings
        self.runs = runs
        self.gpu_lock = gpu_lock
        self._queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._current: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # --- rutas ---------------------------------------------------------------- #

    def out_dir(self, match_id: str, part_id: str) -> Path:
        return self.runs.part_dir(match_id, part_id) / HIGHLIGHTS_DIRNAME

    def _record_path(self, match_id: str, part_id: str) -> Path:
        return self.runs.part_dir(match_id, part_id) / "highlights_run.json"

    def get_run(self, match_id: str, part_id: str) -> HighlightsRun | None:
        data = read_json(self._record_path(match_id, part_id))
        return HighlightsRun.model_validate(data) if data else None

    def _save(self, run: HighlightsRun) -> None:
        write_json_atomic(self._record_path(run.match_id, run.part_id), run.model_dump())

    def _list_runs(self) -> list[HighlightsRun]:
        out = []
        for path in self.runs.outputs_dir.glob("*/*/highlights_run.json"):
            data = read_json(path)
            if data:
                out.append(HighlightsRun.model_validate(data))
        return sorted(out, key=lambda r: r.created_at)

    # --- ciclo de vida -------------------------------------------------------- #

    def start_worker(self) -> None:
        self.recover()
        self._thread = threading.Thread(target=self._loop, name="highlights-worker", daemon=True)
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
        for run in self._list_runs():
            if run.status == "processing":
                run.status, run.error, run.finished_at = "error", "interrupted by a backend restart; generate again", utcnow_iso()
                self._save(run)
            elif run.status == "pending":
                self._queue.put((run.match_id, run.part_id))

    # --- API ------------------------------------------------------------------ #

    def start(self, match_id: str, part_id: str, force: bool = False) -> HighlightsResponse:
        analysis = self.runs.get(match_id, part_id)
        if analysis is None or analysis.status != "completed":
            status = analysis.status if analysis else "pending"
            raise NotReadyError(f"part {part_id} of match {match_id} has no completed analysis (status: {status})")
        current = self.get_run(match_id, part_id)
        if current is not None and current.status in ("processing", "pending"):
            raise ConflictError(f"highlights of {match_id}/{part_id} are already being generated")
        self._save(HighlightsRun(match_id=match_id, part_id=part_id, force=force, created_at=utcnow_iso()))
        self._queue.put((match_id, part_id))
        return self.get(match_id, part_id)

    def summary(self, match_id: str, part_id: str) -> HighlightsSummary | None:
        run = self.get_run(match_id, part_id)
        if run is None:
            return None
        report = read_json(self.out_dir(match_id, part_id) / "highlights.json") if run.status == "completed" else None
        return HighlightsSummary(
            status=run.status, stage=self._stage(run), error=run.error,
            n_clips=len(report.get("highlights", [])) if report else 0,
        )

    def get(self, match_id: str, part_id: str) -> HighlightsResponse:
        run = self.get_run(match_id, part_id)
        if run is None:
            raise NotFoundError(f"no highlights were generated for {match_id}/{part_id}")
        resp = HighlightsResponse(
            match_id=match_id, part_id=part_id, status=run.status, stage=self._stage(run), error=run.error,
            started_at=run.started_at, finished_at=run.finished_at,
        )
        if run.status != "completed":
            return resp
        report = read_json(self.out_dir(match_id, part_id) / "highlights.json")
        if not report:
            raise NotReadyError(f"highlights of {match_id}/{part_id} finished without a report")
        resp.generated_at = report.get("generated_at")
        vlm = report.get("vlm") or {}
        resp.vlm_model = vlm.get("model")
        resp.vlm_used = bool(vlm.get("used", False))
        resp.vlm_degraded = bool(vlm.get("degraded", False))
        resp.vjepa_used = bool((report.get("vjepa") or {}).get("used", False))
        scoreboard = report.get("scoreboard")
        if scoreboard is not None:
            resp.scoreboard_available = bool(scoreboard.get("available", False))
            resp.n_scoreboard_changes = len(scoreboard.get("changes") or [])
        resp.n_candidates = int(report.get("n_candidates", 0))
        base = f"{self.settings.public_base_url}/api/matches/{match_id}/parts/{part_id}/{HIGHLIGHTS_DIRNAME}"
        for h in report.get("highlights", []):
            resp.clips.append(HighlightClip(
                id=h["id"], rank=int(h["rank"]), label=h.get("label", "play"), title=h.get("title", ""),
                description=h.get("description", ""), start_s=h["start_s"], end_s=h["end_s"],
                duration_s=h["duration_s"], anchor_s=h["anchor_s"],
                team_cluster=h.get("team") if h.get("team", -1) >= 0 else None,
                final_score=h.get("final_score", 0.0), vlm_rank=h.get("vlm_rank"), goal_source=h.get("goal_source"),
                content_url=f"{base}/{h['id']}/content" if h.get("clip") else None,
            ))
        return resp

    def clip_path(self, match_id: str, part_id: str, clip_id: str) -> Path:
        run = self.get_run(match_id, part_id)
        if run is None or run.status != "completed":
            raise NotReadyError(f"highlights of {match_id}/{part_id} are not ready")
        report = read_json(self.out_dir(match_id, part_id) / "highlights.json") or {}
        for h in report.get("highlights", []):
            if h["id"] == clip_id and h.get("clip"):
                path = self.out_dir(match_id, part_id) / h["clip"]
                if path.exists():
                    return path
        raise NotFoundError(f"clip {clip_id} of {match_id}/{part_id} does not exist")

    def _stage(self, run: HighlightsRun) -> str | None:
        if run.status != "processing":
            return None
        status = read_json(self.out_dir(run.match_id, run.part_id) / "status.json") or {}
        stage = status.get("stage")
        if stage and status.get("total"):
            return f"{stage} {int(status.get('done', 0)) + 1}/{int(status['total'])}"
        return stage or "starting"

    # --- trabajador ----------------------------------------------------------- #

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._execute(*item)
            except Exception:  # noqa: BLE001 - el hilo no debe morir nunca
                logger.exception("highlights worker failed on %s", item)

    def build_command(self, match_id: str, part_id: str, force: bool) -> list[str]:
        cmd = [
            *self.settings.highlights_base_command(),
            "--backend-match", match_id, "--part", part_id, "--storage-dir", str(self.settings.storage_dir),
            "--out", str(self.out_dir(match_id, part_id)),
        ]
        if force:
            cmd.append("--force")
        return cmd

    def _execute(self, match_id: str, part_id: str) -> None:
        run = self.get_run(match_id, part_id)
        if run is None or run.status != "pending":
            return
        part_dir = self.runs.part_dir(match_id, part_id)
        # Clips y status.json de una generación anterior no deben confundirse con los nuevos.
        stale = self.out_dir(match_id, part_id) / "status.json"
        if stale.exists():
            stale.unlink()
        command = self.build_command(match_id, part_id, run.force)
        log_path = part_dir / "highlights.log"
        run.status, run.started_at, run.command = "processing", utcnow_iso(), command
        run.error, run.exit_code, run.finished_at = None, None, None
        self._save(run)
        logger.info("highlights %s/%s: %s", match_id, part_id, " ".join(command))

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        try:
            with (self.gpu_lock or nullcontext()):
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "ab") as log:
                    proc = subprocess.Popen(
                        command, cwd=str(self.settings.project_root), stdout=log, stderr=subprocess.STDOUT,
                        env=env, start_new_session=True,
                    )
                    with self._lock:
                        self._current = proc
                    run.pid = proc.pid
                    self._save(run)
                    exit_code = proc.wait()
        except OSError as exc:
            self._finish(run, error=f"could not start the highlights generator: {exc}")
            return
        finally:
            with self._lock:
                self._current = None
        run.exit_code = exit_code
        if exit_code != 0:
            self._finish(run, error=f"highlights generator exited with code {exit_code}: {_log_tail(log_path)}")
            return
        if not (self.out_dir(match_id, part_id) / "highlights.json").exists():
            self._finish(run, error="highlights generator finished without writing highlights.json")
            return
        self._finish(run, error=None)

    def _finish(self, run: HighlightsRun, error: str | None) -> None:
        run.status = "error" if error else "completed"
        run.error, run.pid, run.finished_at = error, None, utcnow_iso()
        self._save(run)
        logger.info("highlights %s/%s finished: %s", run.match_id, run.part_id, run.status)


def _log_tail(log_path: Path) -> str:
    try:
        data = log_path.read_bytes()[-_LOG_TAIL_BYTES:]
    except OSError:
        return "(no log)"
    lines = [l for l in data.decode("utf-8", errors="replace").splitlines() if l.strip() and not l.startswith("[")]
    return " | ".join(lines[-4:]) or "(empty log)"
