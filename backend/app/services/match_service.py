from __future__ import annotations

from app.core.errors import InvalidInputError, NotFoundError
from app.repositories.records import MatchRepository, RunStore
from app.schemas import Match, MatchCreate, MatchPart, MatchPartRecord, MatchRecord
from app.schemas.common import PART_LABELS, new_id, utcnow_iso

from .analysis_service import AnalysisService, derive_status
from .highlights_service import HighlightsService
from .video_service import VideoService


class MatchService:
    def __init__(
        self, matches: MatchRepository, runs: RunStore, videos: VideoService, analysis: AnalysisService,
        highlights: HighlightsService | None = None,
    ) -> None:
        self.matches = matches
        self.runs = runs
        self.videos = videos
        self.analysis = analysis
        self.highlights = highlights

    def create(self, data: MatchCreate) -> Match:
        for p in data.parts:
            if self.videos.repo.get(p.video_id) is None:
                raise InvalidInputError(f"video {p.video_id} does not exist")
        record = MatchRecord(
            id=new_id(),
            name=data.name.strip(),
            home_team=(data.home_team or "").strip() or None,
            away_team=(data.away_team or "").strip() or None,
            date=data.date,
            created_at=utcnow_iso(),
            parts=[MatchPartRecord(id=p.kind, kind=p.kind, label=PART_LABELS[p.kind], video_id=p.video_id) for p in data.parts],
        )
        self.matches.save(record)
        return self.to_public(record)

    def get_record(self, match_id: str) -> MatchRecord:
        record = self.matches.get(match_id)
        if record is None:
            raise NotFoundError(f"match {match_id} does not exist")
        return record

    def get(self, match_id: str) -> Match:
        return self.to_public(self.get_record(match_id))

    def list(self) -> list[Match]:
        return [self.to_public(r) for r in self.matches.list_all()]

    def to_public(self, record: MatchRecord) -> Match:
        status = self.analysis.status(record)
        by_part = {p.part_id: p for p in status.parts}
        parts: list[MatchPart] = []
        for p in record.parts:
            video = self.videos.repo.get(p.video_id)
            run = self.runs.get(record.id, p.id)
            st = by_part[p.id]
            parts.append(MatchPart(
                id=p.id, kind=p.kind, label=p.label, mode=run.mode if run else p.mode, status=st.status,
                stage=st.stage, error=st.error, video=self.videos.to_public(video) if video else None,
                highlights=self.highlights.summary(record.id, p.id) if self.highlights else None,
            ))
        errors = [p.error for p in parts if p.error]
        durations = [p.video.duration_s for p in parts if p.video]
        return Match(
            id=record.id, name=record.name, home_team=record.home_team, away_team=record.away_team, date=record.date,
            created_at=record.created_at, status=derive_status([p.status for p in parts]),
            error_message=errors[0] if errors else None, parts=parts,
            duration_s=round(sum(durations), 3) if durations else None,
        )
