from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import FileResponse

from app.api.deps import Services, get_services
from app.schemas import (
    AnalysisStartRequest, AnalysisStatusResponse, HighlightsResponse, HighlightsStartRequest, Match, MatchCreate,
    PartAnalysis, Tracklet, TrackingWindow,
)
from app.schemas.common import ID_PATTERN

router = APIRouter(prefix="/matches", tags=["matches"])

MatchId = Path(pattern=ID_PATTERN)
PartId = Path(pattern=r"^(full|first-half|second-half)$")
ClipId = Path(pattern=r"^hl-\d{2}$")


@router.get("", response_model=list[Match])
def list_matches(services: Services = Depends(get_services)) -> list[Match]:
    return services.matches.list()


@router.post("", response_model=Match, status_code=201)
def create_match(data: MatchCreate, services: Services = Depends(get_services)) -> Match:
    return services.matches.create(data)


@router.get("/{match_id}", response_model=Match)
def get_match(match_id: str = MatchId, services: Services = Depends(get_services)) -> Match:
    return services.matches.get(match_id)


# --- análisis ----------------------------------------------------------------- #

@router.post("/{match_id}/analysis", response_model=AnalysisStatusResponse, status_code=202)
def start_analysis(
    data: AnalysisStartRequest | None = None, match_id: str = MatchId, services: Services = Depends(get_services)
) -> AnalysisStatusResponse:
    record = services.matches.get_record(match_id)
    data = data or AnalysisStartRequest()
    already_done = services.analysis.completed_parts(record) if data.highlights else []
    status = services.analysis.start(record, data.mode, highlights=data.highlights)
    # Las partes ya analizadas no vuelven a pasar por el pipeline: sus highlights se encolan aquí.
    for part_id in already_done:
        current = services.highlights.get_run(match_id, part_id)
        if current is None or current.status not in ("processing", "pending"):
            services.highlights.start(match_id, part_id)
    return status


@router.get("/{match_id}/analysis/status", response_model=AnalysisStatusResponse)
def analysis_status(match_id: str = MatchId, services: Services = Depends(get_services)) -> AnalysisStatusResponse:
    return services.analysis.status(services.matches.get_record(match_id))


@router.get("/{match_id}/parts/{part_id}/analysis", response_model=PartAnalysis)
def part_analysis(
    match_id: str = MatchId, part_id: str = PartId, services: Services = Depends(get_services)
) -> PartAnalysis:
    services.matches.get_record(match_id)
    return services.memory.get_part_analysis(match_id, part_id)


@router.post("/{match_id}/parts/{part_id}/reingest", response_model=PartAnalysis)
def part_reingest(
    match_id: str = MatchId, part_id: str = PartId, services: Services = Depends(get_services)
) -> PartAnalysis:
    """Re-normaliza las salidas crudas de una parte completada (sin volver a correr el pipeline)."""
    services.matches.get_record(match_id)
    return services.memory.reingest(match_id, part_id)


@router.get("/{match_id}/parts/{part_id}/tracklets", response_model=list[Tracklet])
def part_tracklets(
    match_id: str = MatchId, part_id: str = PartId, services: Services = Depends(get_services)
) -> list[Tracklet]:
    services.matches.get_record(match_id)
    return services.memory.get_tracklets(match_id, part_id)


# --- mejores jugadas ---------------------------------------------------------- #

@router.post("/{match_id}/parts/{part_id}/highlights", response_model=HighlightsResponse, status_code=202)
def start_highlights(
    data: HighlightsStartRequest | None = None,
    match_id: str = MatchId, part_id: str = PartId, services: Services = Depends(get_services),
) -> HighlightsResponse:
    """Encola la generación de las mejores jugadas de una parte ya analizada
    (409 si ya está en marcha o si el análisis no ha terminado)."""
    services.matches.get_record(match_id)
    return services.highlights.start(match_id, part_id, force=(data or HighlightsStartRequest()).force)


@router.get("/{match_id}/parts/{part_id}/highlights", response_model=HighlightsResponse)
def get_highlights(
    match_id: str = MatchId, part_id: str = PartId, services: Services = Depends(get_services)
) -> HighlightsResponse:
    """Estado de la generación y, si ha terminado, la lista de clips."""
    services.matches.get_record(match_id)
    return services.highlights.get(match_id, part_id)


@router.get(
    "/{match_id}/parts/{part_id}/highlights/{clip_id}/content",
    response_class=FileResponse, responses={206: {"description": "Partial content"}},
)
def get_highlight_content(
    match_id: str = MatchId, part_id: str = PartId, clip_id: str = ClipId, services: Services = Depends(get_services)
) -> FileResponse:
    services.matches.get_record(match_id)
    path = services.highlights.clip_path(match_id, part_id, clip_id)
    return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"}, filename=f"{match_id}_{clip_id}.mp4")


@router.get("/{match_id}/parts/{part_id}/tracking", response_model=TrackingWindow)
def part_tracking(
    match_id: str = MatchId,
    part_id: str = PartId,
    from_s: float = Query(ge=0),
    to_s: float = Query(ge=0),
    services: Services = Depends(get_services),
) -> TrackingWindow:
    services.matches.get_record(match_id)
    return services.memory.get_tracking_window(match_id, part_id, from_s, to_s)
