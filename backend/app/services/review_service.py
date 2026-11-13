"""Copia revisada del análisis de una parte: lo que el usuario corrige en el visor.

El pipeline deja ``memory/analysis.json`` y eso no se toca nunca. La primera corrección crea una copia en
``review/analysis.json`` y todas las siguientes se guardan en esa misma copia:

* cambiar la etiqueta, el ejecutor (``tracker_id``), el lado (local/visitante) o el instante de una acción de
  balón o de un evento;
* borrar lo que el pipeline detectó de más (queda en ``deleted_*`` para poder deshacer y para auditar);
* añadir lo que se le escapó.

Cada operación se apunta además en ``review/edits.jsonl`` (qué había y qué quedó). De la copia sale la hoja de
verdad ``tiempo,etiqueta,dorsal,equipo`` de un tramo (``export``), que es la que leen
``scripts/annotation_csv.py`` y ``scripts/evaluate_match.py --csv``: lo que no se ha tocado dentro del tramo
cuenta como correcto.
"""

from __future__ import annotations

import json
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path

from app.adapters.labels import REVIEW_LABELS, canonical_label
from app.core.config import Settings
from app.core.errors import InvalidInputError, MalformedOutputError, NotFoundError, NotReadyError
from app.repositories.json_store import read_json, write_json_atomic
from app.repositories.records import RunStore, VideoRepository
from app.schemas import (
    AnalysisRun, PartAnalysis, ReviewExport, ReviewFile, ReviewInfo, ReviewItemCreate, ReviewItemPatch,
    ReviewItemResult, ReviewedAnalysis, ReviewedBallAction, ReviewedEvent, VideoRecord,
)
from app.schemas.common import utcnow_iso

from .football_memory import FootballMemory

# Un evento de T-DEED es un tramo de ±1 s alrededor del pico: su instante es ``start_s + 1`` (mismo criterio que
# ``scripts/evaluate_match.py::event_instants``).
SPOT_EVENT_HALF_SPAN_S = 1.0
# En la hoja exportada, una acción y un evento de la misma jugada (p.ej. el saque de banda que T-DEED emite como
# acción y como evento) a menos de esta distancia son una sola fila.
EXPORT_DEDUP_S = 1.5
SHOT_LIKE = ("shot", "goal")
# Grafía de la hoja de anotación para las dos etiquetas largas de T-DEED.
CSV_LABELS = {"player_successful_tackle": "tackle", "ball_player_block": "block"}
CSV_HEADER = "tiempo,etiqueta,dorsal,equipo"


def event_instant(start_s: float, end_s: float) -> float:
    if end_s - start_s >= 2 * SPOT_EVENT_HALF_SPAN_S - 1e-6:
        return start_s + SPOT_EVENT_HALF_SPAN_S
    return (start_s + end_s) / 2


def kind_of(item_id: str) -> str:
    return "action" if item_id.startswith("act-") else "event"


def csv_clock(t_s: float) -> str:
    """Segundos de vídeo -> ``mm:ss.cc`` (``h:mm:ss.cc`` a partir de una hora), como en el reproductor."""
    t = round(max(0.0, t_s), 2)
    minutes, seconds = divmod(t, 60)
    minutes = int(minutes)
    if minutes >= 60:
        return f"{minutes // 60}:{minutes % 60:02d}:{seconds:05.2f}"
    return f"{minutes:02d}:{seconds:05.2f}"


@dataclass
class _Row:
    t_s: float
    label: str
    tracker_id: int | None
    side: str | None


class ReviewService:
    def __init__(self, settings: Settings, runs: RunStore, videos: VideoRepository, memory: FootballMemory) -> None:
        self.settings = settings
        self.runs = runs
        self.videos = videos
        self.memory = memory
        self._lock = threading.Lock()
        self._jerseys: dict[Path, tuple[float, dict[int, int | None]]] = {}

    # --- rutas ---------------------------------------------------------------- #

    def review_dir(self, match_id: str, part_id: str) -> Path:
        return self.runs.part_dir(match_id, part_id) / "review"

    def path(self, match_id: str, part_id: str) -> Path:
        return self.review_dir(match_id, part_id) / "analysis.json"

    # --- lectura --------------------------------------------------------------- #

    def get(self, match_id: str, part_id: str) -> ReviewedAnalysis:
        run, _ = self._context(match_id, part_id)
        with self._lock:
            file = self._load(match_id, part_id)
        if file is None:
            analysis = self.memory.get_part_analysis(match_id, part_id)
            return ReviewedAnalysis(**analysis.model_dump(), review=ReviewInfo())
        base = file.model_dump(include=set(PartAnalysis.model_fields))
        return ReviewedAnalysis(**base, review=self._info(file, run))

    # --- correcciones ---------------------------------------------------------- #

    def update(self, match_id: str, part_id: str, item_id: str, patch: ReviewItemPatch) -> ReviewItemResult:
        run, video = self._context(match_id, part_id)
        fields = patch.model_fields_set
        with self._lock:
            file = self._load_or_create(match_id, part_id, run)
            item = self._find(file, item_id)
            before = _summary(item)
            sides_before = dict(file.tracker_sides)
            if "label" in fields and patch.label is not None:
                self._set_label(item, patch.label)
            if "time_s" in fields and patch.time_s is not None:
                self._set_time(item, patch.time_s, video, file.frame_count)
            if "tracker_id" in fields:
                self._set_tracker(file, item, patch.tracker_id, match_id, part_id)
            if "side" in fields:
                item.side = patch.side
                if patch.side is not None and item.tracker_id is not None:
                    file.tracker_sides[item.tracker_id] = patch.side
            after = _summary(item)
            if after != before:
                if item.review != "added":
                    item.review = "edited"
                self._sort(file)
            if after != before or file.tracker_sides != sides_before:
                self._save(match_id, part_id, file)
                self._log(match_id, part_id, "update", item_id, before, _summary(item))
            return self._result(file, run, item, exists=self.path(match_id, part_id).exists())

    def add(self, match_id: str, part_id: str, data: ReviewItemCreate) -> ReviewItemResult:
        run, video = self._context(match_id, part_id)
        label = _review_label(data.label)
        with self._lock:
            file = self._load_or_create(match_id, part_id, run)
            item: ReviewedBallAction | ReviewedEvent
            if data.kind == "action":
                item = ReviewedBallAction(
                    id=f"act-new-{secrets.token_hex(4)}", frame_idx=0, timestamp_s=0.0, tracker_id=None,
                    team_cluster=None, label=label, confidence=1.0, probabilities={label: 1.0},
                    is_background=False, spot_score=None, review="added",
                )
                file.ball_actions.append(item)
            else:
                item = ReviewedEvent(
                    id=f"evt-new-{secrets.token_hex(4)}", label=label, start_s=0.0, end_s=2 * SPOT_EVENT_HALF_SPAN_S,
                    peak_confidence=1.0, mean_confidence=1.0, n_windows=1, review="added",
                )
                file.events.append(item)
            self._set_time(item, data.time_s, video, file.frame_count)
            self._set_tracker(file, item, data.tracker_id, match_id, part_id)
            item.side = data.side
            if data.side is not None and item.tracker_id is not None:
                file.tracker_sides[item.tracker_id] = data.side
            self._sort(file)
            self._save(match_id, part_id, file)
            self._log(match_id, part_id, "add", item.id, None, _summary(item))
            return self._result(file, run, item)

    def delete(self, match_id: str, part_id: str, item_id: str) -> ReviewItemResult:
        run, _ = self._context(match_id, part_id)
        with self._lock:
            file = self._load_or_create(match_id, part_id, run)
            item = self._find(file, item_id)
            if isinstance(item, ReviewedBallAction):
                file.ball_actions.remove(item)
                file.deleted_ball_actions.append(item)
            else:
                file.events.remove(item)
                file.deleted_events.append(item)
            self._save(match_id, part_id, file)
            self._log(match_id, part_id, "delete", item_id, _summary(item), None)
            return self._result(file, run, item, deleted=True)

    def restore(self, match_id: str, part_id: str, item_id: str) -> ReviewItemResult:
        """Deshace un borrado."""
        run, _ = self._context(match_id, part_id)
        with self._lock:
            file = self._load(match_id, part_id)
            deleted = [*file.deleted_ball_actions, *file.deleted_events] if file else []
            item = next((i for i in deleted if i.id == item_id), None)
            if file is None or item is None:
                raise NotFoundError(f"{item_id} is not among the deleted items of the reviewed copy")
            if isinstance(item, ReviewedBallAction):
                file.deleted_ball_actions.remove(item)
                file.ball_actions.append(item)
            else:
                file.deleted_events.remove(item)
                file.events.append(item)
            self._sort(file)
            self._save(match_id, part_id, file)
            self._log(match_id, part_id, "restore", item_id, None, _summary(item))
            return self._result(file, run, item)

    # --- hoja de verdad --------------------------------------------------------- #

    def export(self, match_id: str, part_id: str, from_s: float, to_s: float | None) -> ReviewExport:
        """Hoja ``tiempo,etiqueta,dorsal,equipo`` del tramo [from_s, to_s] de la copia revisada. Se devuelve y se
        deja también en ``review/`` para pasarla a ``scripts/evaluate_match.py --csv``."""
        _, video = self._context(match_id, part_id)
        end_s = video.duration_s if to_s is None else to_s
        if end_s <= from_s:
            raise InvalidInputError("to_s must be greater than from_s")
        with self._lock:
            file = self._load(match_id, part_id)
        if file is None:
            raise NotReadyError(f"part {part_id} of match {match_id} has no reviewed copy yet: correct something first")
        rows = [r for r in export_rows(file) if from_s <= r.t_s <= end_s]
        lines = [CSV_HEADER]
        for r in rows:
            label = CSV_LABELS.get(r.label, r.label.replace("_", " "))
            lines.append(f"{csv_clock(r.t_s)},{label},{r.tracker_id if r.tracker_id is not None else ''},{r.side or ''}")
        text = "\n".join(lines) + "\n"
        filename = f"ground_truth_{_clock_slug(from_s)}-{_clock_slug(end_s)}.csv"
        path = self.review_dir(match_id, part_id) / filename
        tmp = path.with_suffix(".csv.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        try:
            shown = str(path.resolve().relative_to(self.settings.project_root.resolve()))
        except ValueError:
            shown = str(path)
        return ReviewExport(from_s=from_s, to_s=end_s, n_rows=len(rows), filename=filename, path=shown, csv=text)

    # --- interno --------------------------------------------------------------- #

    def _context(self, match_id: str, part_id: str) -> tuple[AnalysisRun, VideoRecord]:
        run = self.runs.get(match_id, part_id)
        if run is None or run.status != "completed":
            status = run.status if run else "pending"
            raise NotReadyError(f"part {part_id} of match {match_id} has no completed analysis (status: {status})")
        video = self.videos.get(run.video_id)
        if video is None:
            raise NotReadyError(f"video {run.video_id} of part {part_id} is missing")
        return run, video

    def _load(self, match_id: str, part_id: str) -> ReviewFile | None:
        try:
            data = read_json(self.path(match_id, part_id))
        except ValueError as exc:
            raise MalformedOutputError(f"review/analysis.json is not valid JSON: {exc}") from exc
        return ReviewFile.model_validate(data) if data is not None else None

    def _load_or_create(self, match_id: str, part_id: str, run: AnalysisRun) -> ReviewFile:
        """La copia de disco o, si aún no hay, una recién copiada del análisis (se guarda con el primer cambio)."""
        file = self._load(match_id, part_id)
        if file is not None:
            return file
        analysis = self.memory.get_part_analysis(match_id, part_id)
        now = utcnow_iso()
        return ReviewFile(**analysis.model_dump(), created_at=now, updated_at=now, based_on_run=run.finished_at)

    def _save(self, match_id: str, part_id: str, file: ReviewFile) -> None:
        file.updated_at = utcnow_iso()
        write_json_atomic(self.path(match_id, part_id), file.model_dump())

    def _log(self, match_id: str, part_id: str, op: str, item_id: str, before: dict | None, after: dict | None) -> None:
        entry = {"at": utcnow_iso(), "op": op, "id": item_id, "before": before, "after": after}
        with open(self.review_dir(match_id, part_id) / "edits.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _find(file: ReviewFile, item_id: str) -> ReviewedBallAction | ReviewedEvent:
        items = file.ball_actions if kind_of(item_id) == "action" else file.events
        for item in items:
            if item.id == item_id:
                return item
        raise NotFoundError(f"{item_id} is not in the reviewed copy (deleted, or from another analysis)")

    @staticmethod
    def _sort(file: ReviewFile) -> None:
        file.ball_actions.sort(key=lambda a: a.frame_idx)
        file.events.sort(key=lambda e: e.start_s)

    @staticmethod
    def _set_label(item: ReviewedBallAction | ReviewedEvent, raw: str) -> None:
        label = _review_label(raw)
        if label == item.label:
            return
        item.label = label
        # Etiqueta puesta a mano: las confianzas del modelo ya no dicen nada de ella.
        if isinstance(item, ReviewedBallAction):
            item.confidence, item.probabilities, item.is_background = 1.0, {label: 1.0}, False
        else:
            item.peak_confidence = item.mean_confidence = 1.0
            item.estimated = False
            if label != "goal":
                item.goal_source = item.score = None

    @staticmethod
    def _set_time(item: ReviewedBallAction | ReviewedEvent, time_s: float, video: VideoRecord, frame_count: int) -> None:
        if isinstance(item, ReviewedBallAction):
            frame_idx = int(round(time_s * video.fps))
            if frame_count > 0:
                frame_idx = min(frame_idx, frame_count - 1)
            item.frame_idx = max(0, frame_idx)
            item.timestamp_s = round(item.frame_idx / video.fps, 3)
            return
        # Evento: se desplaza el tramo entero para que su instante (el pico) caiga en ``time_s``.
        span = item.end_s - item.start_s
        delta = time_s - event_instant(item.start_s, item.end_s)
        item.start_s = round(max(0.0, item.start_s + delta), 3)
        item.end_s = round(item.start_s + span, 3)

    def _set_tracker(
        self, file: ReviewFile, item: ReviewedBallAction | ReviewedEvent, tracker_id: int | None,
        match_id: str, part_id: str,
    ) -> None:
        if tracker_id == item.tracker_id:
            return
        item.tracker_id = tracker_id
        item.team_cluster = file.teams.assignments.get(tracker_id) if tracker_id is not None else None
        item.jersey = self._jersey_of(match_id, part_id, tracker_id)

    def _jersey_of(self, match_id: str, part_id: str, tracker_id: int | None) -> int | None:
        """Dorsal mayoritario del tracklet (``memory/tracklets.json``, cacheado mientras no cambie el fichero)."""
        if tracker_id is None:
            return None
        path = self.memory.memory_dir(match_id, part_id) / "tracklets.json"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        cached = self._jerseys.get(path)
        if cached is None or cached[0] != mtime:
            cached = (mtime, {int(t["tracker_id"]): t.get("jersey") for t in (read_json(path) or [])})
            self._jerseys[path] = cached
        return cached[1].get(tracker_id)

    def _info(self, file: ReviewFile, run: AnalysisRun, exists: bool = True) -> ReviewInfo:
        items = [*file.ball_actions, *file.events]
        deleted = [*file.deleted_ball_actions, *file.deleted_events]
        return ReviewInfo(
            exists=exists,
            created_at=file.created_at if exists else None,
            updated_at=file.updated_at if exists else None,
            n_edited=sum(1 for i in items if i.review == "edited"),
            n_added=sum(1 for i in items if i.review == "added"),
            n_deleted=sum(1 for i in deleted if i.review != "added"),
            stale=exists and file.based_on_run is not None and file.based_on_run != run.finished_at,
            tracker_sides=dict(file.tracker_sides),
        )

    def _result(
        self, file: ReviewFile, run: AnalysisRun, item: ReviewedBallAction | ReviewedEvent,
        deleted: bool = False, exists: bool = True,
    ) -> ReviewItemResult:
        is_action = isinstance(item, ReviewedBallAction)
        return ReviewItemResult(
            kind="action" if is_action else "event", id=item.id, deleted=deleted,
            action=item if is_action else None, event=None if is_action else item,
            review=self._info(file, run, exists),
        )


def _review_label(raw: str) -> str:
    label = canonical_label(raw)
    if label not in REVIEW_LABELS:
        raise InvalidInputError(f"unknown label '{raw}'; expected one of: {', '.join(REVIEW_LABELS)}")
    return label


def _summary(item: ReviewedBallAction | ReviewedEvent) -> dict:
    """Lo que puede cambiar una corrección (para comparar antes/después y para ``edits.jsonl``)."""
    if isinstance(item, ReviewedBallAction):
        return {"label": item.label, "frame_idx": item.frame_idx, "timestamp_s": item.timestamp_s,
                "tracker_id": item.tracker_id, "side": item.side}
    return {"label": item.label, "start_s": item.start_s, "end_s": item.end_s,
            "tracker_id": item.tracker_id, "side": item.side}


def _clock_slug(t_s: float) -> str:
    total = int(round(t_s))
    return f"{total // 60:02d}m{total % 60:02d}"


def _same_play(action_label: str, event_label: str) -> bool:
    return action_label == event_label or (action_label in SHOT_LIKE and event_label in SHOT_LIKE)


def export_rows(file: ReviewFile) -> list[_Row]:
    """Filas de la hoja, por tiempo. El lado de una fila es el suyo o, si no lo lleva, el que ya se indicó para su
    tracklet. Un evento con una acción de la misma jugada al lado no se repite: queda la fila de la acción (que es
    la que trae el instante exacto), con el ejecutor y el lado del evento si a ella le faltan."""
    sides = file.tracker_sides

    def side_of(item: ReviewedBallAction | ReviewedEvent) -> str | None:
        return item.side or (sides.get(item.tracker_id) if item.tracker_id is not None else None)

    actions = [_Row(a.timestamp_s, a.label, a.tracker_id, side_of(a)) for a in file.ball_actions if not a.is_background]
    rows = list(actions)
    used: set[int] = set()
    for e in file.events:
        t_s = event_instant(e.start_s, e.end_s)
        best: int | None = None
        for i, a in enumerate(actions):
            if i in used or abs(a.t_s - t_s) > EXPORT_DEDUP_S or not _same_play(a.label, e.label):
                continue
            if best is None or abs(a.t_s - t_s) < abs(actions[best].t_s - t_s):
                best = i
        if best is None:
            rows.append(_Row(round(t_s, 3), e.label, e.tracker_id, side_of(e)))
            continue
        used.add(best)
        twin = actions[best]
        if twin.tracker_id is None:
            twin.tracker_id = e.tracker_id
        if twin.side is None:
            twin.side = side_of(e)
        if e.label == "goal":
            twin.label = "goal"
    rows.sort(key=lambda r: r.t_s)
    return rows
