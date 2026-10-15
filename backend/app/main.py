"""Punto de entrada: ``uvicorn app.main:app`` (desde ``backend/``)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import build_services
from app.api.routes import health, matches, videos
from app.core.config import Settings, get_settings
from app.core.errors import install_error_handlers


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = build_services(settings)
        app.state.services.analysis.start_worker()
        try:
            yield
        finally:
            app.state.services.analysis.stop_worker()

    app = FastAPI(
        title="Football Intelligence API",
        description="Puente entre el pipeline de análisis de fútbol (Sports-Classification) y el frontend React.",
        version="0.1.0",
        lifespan=lifespan,
        # Bajo /api para que salgan por el mismo puerto que la API cuando el frontend (Vite) hace de proxy.
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
    )
    install_error_handlers(app)

    api = APIRouter(prefix="/api")
    api.include_router(health.router)
    api.include_router(videos.router)
    api.include_router(matches.router)
    app.include_router(api)
    return app


app = create_app()
