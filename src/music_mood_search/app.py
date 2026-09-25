import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request

from music_mood_search.api.health import router as health_router
from music_mood_search.config import get_settings
from music_mood_search.db import close_db_pool, create_db_pool
from music_mood_search.logging_config import configure_logging

logger = logging.getLogger("music_mood_search.app")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "Application starting",
        extra={"event": "application_startup"},
    )

    application.state.db_pool = await create_db_pool()

    logger.info(
        "Database pool created",
        extra={
            "event": "database_pool_created",
            "component": "postgres",
        },
    )

    try:
        yield
    finally:
        await close_db_pool(application.state.db_pool)

        logger.info(
            "Database pool closed",
            extra={
                "event": "database_pool_closed",
                "component": "postgres",
            },
        )


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    application = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def log_requests(request: Request, call_next):
        started_at = perf_counter()

        response = await call_next(request)

        duration_ms = (perf_counter() - started_at) * 1000

        logger.info(
            "HTTP request completed",
            extra={
                "event": "http_request",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )

        return response

    application.include_router(health_router)

    return application


app = create_app()
