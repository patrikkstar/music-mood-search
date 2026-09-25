from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from music_mood_search.api.health import router as health_router
from music_mood_search.config import get_settings
from music_mood_search.db import close_db_pool, create_db_pool


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    application.state.db_pool = await create_db_pool()

    try:
        yield
    finally:
        await close_db_pool(application.state.db_pool)


def create_app() -> FastAPI:
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
    )

    application.include_router(health_router)

    return application


app = create_app()
