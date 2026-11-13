import re

from fastapi import APIRouter, Depends, File, Path, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response

from app.api.deps import Services, get_services
from app.schemas import Video
from app.schemas.common import ID_PATTERN

router = APIRouter(prefix="/videos", tags=["videos"])

VideoId = Path(pattern=ID_PATTERN)
_SINGLE_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


def parse_single_range(header: str, size: int) -> tuple[int, int] | None:
    """``bytes=a-b`` | ``bytes=a-`` | ``bytes=-n`` -> (primero, último) incluidos, dentro del fichero. ``None`` si no
    es un único rango satisfacible (varios rangos, sintaxis rara, fuera del fichero): eso lo responde Starlette."""
    match = _SINGLE_RANGE.match(header.strip().replace(" ", ""))
    if match is None or size <= 0:
        return None
    first, last = match.groups()
    if first == "":
        if last == "" or int(last) == 0:
            return None
        return max(0, size - int(last)), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last != "" else size - 1
    return (start, end) if start <= end else None


@router.post("/upload", response_model=Video, status_code=201)
def upload_video(file: UploadFile = File(...), services: Services = Depends(get_services)) -> Video:
    record = services.videos.upload(file.filename or "", file.file)
    return services.videos.to_public(record)


@router.get("/{video_id}", response_model=Video)
def get_video(video_id: str = VideoId, services: Services = Depends(get_services)) -> Video:
    return services.videos.to_public(services.videos.get(video_id))


@router.get("/{video_id}/content", response_class=FileResponse, responses={206: {"description": "Partial content"}})
def get_video_content(request: Request, video_id: str = VideoId, services: Services = Depends(get_services)) -> Response:
    path, media_type = services.videos.content(video_id)
    chunk = services.settings.video_range_chunk_bytes
    wanted = parse_single_range(request.headers.get("range", ""), path.stat().st_size) if chunk > 0 else None
    if wanted is not None:
        # Rango acotado: el reproductor recibe un trozo, la conexión se cierra y pide el siguiente cuando lo necesita.
        size = path.stat().st_size
        start, end = wanted[0], min(wanted[1], wanted[0] + chunk - 1)
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(end - start + 1)
        return Response(
            content=data, status_code=206, media_type=media_type,
            headers={"Content-Range": f"bytes {start}-{end}/{size}", "Accept-Ranges": "bytes"},
        )
    # Sin Range (o un Range que no es un tramo simple): FileResponse de Starlette, que también responde 206/416.
    return FileResponse(path, media_type=media_type, headers={"Accept-Ranges": "bytes"})


@router.get("/{video_id}/thumbnail", response_class=Response, responses={200: {"content": {"image/jpeg": {}}}})
def get_video_thumbnail(
    video_id: str = VideoId,
    t: float = Query(ge=0, description="Instante en segundos de vídeo (se redondea a décimas)"),
    h: int = Query(default=112, ge=36, le=360, description="Alto en píxeles"),
    services: Services = Depends(get_services),
) -> Response:
    """Un fotograma como JPEG pequeño (unos KB), para la tira de miniaturas del editor de clips. El vídeo no
    cambia nunca bajo el mismo id, así que el navegador puede guardarla."""
    data = services.videos.thumbnail(video_id, t, h)
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=604800, immutable"})
