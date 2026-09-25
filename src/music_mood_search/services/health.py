from time import perf_counter

import asyncpg


async def check_database_health(pool: asyncpg.Pool) -> dict[str, str | float]:
    started_at = perf_counter()

    try:
        async with pool.acquire() as connection:
            database_version = await connection.fetchval("SELECT version()")

        response_time_ms = (perf_counter() - started_at) * 1000

        return {
            "status": "ok",
            "version": database_version,
            "response_time_ms": round(response_time_ms, 2),
        }

    except Exception as exc:
        response_time_ms = (perf_counter() - started_at) * 1000

        return {
            "status": "error",
            "version": "unknown",
            "response_time_ms": round(response_time_ms, 2),
            "error": str(exc),
        }
