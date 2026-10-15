"""Errores de la API: una jerarquía pequeña y un único formato de respuesta
``{"error": {"code": ..., "message": ...}}``."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


class ApiError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(ApiError):
    status_code = 404
    code = "not_found"


class ConflictError(ApiError):
    status_code = 409
    code = "conflict"


class InvalidInputError(ApiError):
    status_code = 422
    code = "invalid_input"


class UnsupportedMediaError(ApiError):
    status_code = 415
    code = "unsupported_media"


class PayloadTooLargeError(ApiError):
    status_code = 413
    code = "payload_too_large"


class NotReadyError(ApiError):
    """El recurso existe pero aún no tiene datos (análisis no terminado)."""

    status_code = 409
    code = "not_ready"


class MalformedOutputError(ApiError):
    """Una salida del pipeline no tiene la forma esperada."""

    status_code = 500
    code = "malformed_output"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_input", "message": "Request validation failed", "details": jsonable_encoder(exc.errors())}},
        )
