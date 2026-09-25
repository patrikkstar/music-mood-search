from importlib.metadata import version

from fastapi import APIRouter, Request, Response, status

from music_mood_search.schemas import (
    ComponentHealth,
    DetailedHealthResponse,
    HealthResponse,
    VersionResponse,
)
from music_mood_search.services.health import check_database_health

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/api/v1/version", response_model=VersionResponse)
async def get_version() -> VersionResponse:
    return VersionResponse(version=version("music-mood-search"))


@router.get("/api/v1/health", response_model=DetailedHealthResponse)
async def health(
    request: Request,
    response: Response,
) -> DetailedHealthResponse:
    database_health = await check_database_health(request.app.state.db_pool)

    database = ComponentHealth(**database_health)

    overall_status = "ok"

    if database.status != "ok":
        overall_status = "error"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return DetailedHealthResponse(
        status=overall_status,
        components={
            "database": database,
        },
    )
