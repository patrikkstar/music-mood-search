import asyncpg

from music_mood_search.config import get_settings


async def create_db_pool() -> asyncpg.Pool:
    settings = get_settings()

    return await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=0,
        max_size=5,
    )


async def close_db_pool(pool: asyncpg.Pool) -> None:
    await pool.close()
