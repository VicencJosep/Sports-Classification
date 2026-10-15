from fastapi import APIRouter, Depends, File, Path, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import Services, get_services
from app.schemas import Video
from app.schemas.common import ID_PATTERN

router = APIRouter(prefix="/videos", tags=["videos"])

VideoId = Path(pattern=ID_PATTERN)


@router.post("/upload", response_model=Video, status_code=201)
def upload_video(file: UploadFile = File(...), services: Services = Depends(get_services)) -> Video:
    record = services.videos.upload(file.filename or "", file.file)
    return services.videos.to_public(record)


@router.get("/{video_id}", response_model=Video)
def get_video(video_id: str = VideoId, services: Services = Depends(get_services)) -> Video:
    return services.videos.to_public(services.videos.get(video_id))


@router.get("/{video_id}/content", response_class=FileResponse, responses={206: {"description": "Partial content"}})
def get_video_content(video_id: str = VideoId, services: Services = Depends(get_services)) -> FileResponse:
    path, media_type = services.videos.content(video_id)
    # FileResponse de Starlette responde a cabeceras Range con 206 (reproducción en navegador).
    return FileResponse(path, media_type=media_type, headers={"Accept-Ranges": "bytes"})
