import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from faststream.rabbit import RabbitBroker
from faststream.rabbit.opentelemetry import RabbitTelemetryMiddleware
from faststream.rabbit.schemas import Channel
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from music_mood_search.api.health import router as health_router
from music_mood_search.api.process import router as process_router
from music_mood_search.api.tasks import router as tasks_router
from music_mood_search.config import get_settings
from music_mood_search.db import close_db_pool, create_db_pool
from music_mood_search.logging_config import configure_logging
from music_mood_search.services.model import ModelService, load_model_on_startup
from music_mood_search.telemetry import (
    configure_telemetry,
    flush_telemetry,
    instruments,
    record_process_memory,
)

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

    # Модель загружается из MLflow один раз при старте.
    # /process дальше использует уже загруженный объект.
    settings = get_settings()
    model_service = ModelService(settings)
    await run_in_threadpool(load_model_on_startup, model_service, settings)
    application.state.model_service = model_service

    try:
        async with RabbitBroker(
            settings.rabbitmq_url,
            middlewares=(
                RabbitTelemetryMiddleware(tracer_provider=configure_telemetry()[0]),
            ),
            default_channel=Channel(
                publisher_confirms=True,
                on_return_raises=True,
            ),
        ) as broker:
            application.state.rabbit_broker = broker
            logger.info("RabbitMQ publisher connected")
            yield
    finally:
        await close_db_pool(application.state.db_pool)
        flush_telemetry()

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
    configure_telemetry()
    record_process_memory()

    application = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def log_requests(request: Request, call_next):
        started_at = perf_counter()

        response = await call_next(request)
        if (
            request.url.path in ("/healthz", "/api/v1/health")
            and response.status_code < 500
        ):
            return response

        duration_ms = (perf_counter() - started_at) * 1000
        route_obj = request.scope.get("route")
        route_name = getattr(route_obj, "path", "unmatched")
        metric_attrs = {"route": route_name, "status_code": str(response.status_code)}
        instruments()["http_requests"].add(1, metric_attrs)
        instruments()["http_duration"].record(duration_ms / 1000, metric_attrs)

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
    application.include_router(process_router)
    application.include_router(tasks_router)

    FastAPIInstrumentor.instrument_app(
        application,
        tracer_provider=configure_telemetry()[0],
        excluded_urls=r"/healthz,/api/v1/health",
    )
    return application


app = create_app()
