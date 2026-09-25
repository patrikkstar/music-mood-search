from importlib.metadata import version

from music_mood_search.app import app


async def test_healthz(client):
    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


async def test_version(client):
    response = await client.get("/api/v1/version")

    assert response.status_code == 200
    assert response.json() == {
        "version": version("music-mood-search"),
    }


async def test_health_success(client, monkeypatch):
    async def fake_check_database_health(pool):
        return {
            "status": "ok",
            "version": "PostgreSQL 17.11",
            "response_time_ms": 1.23,
        }

    monkeypatch.setattr(
        "music_mood_search.api.health.check_database_health",
        fake_check_database_health,
    )

    app.state.db_pool = object()

    response = await client.get("/api/v1/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["components"]["database"]["status"] == "ok"
    assert data["components"]["database"]["version"] == "PostgreSQL 17.11"
    assert data["components"]["database"]["response_time_ms"] == 1.23


async def test_health_database_error(client, monkeypatch):
    async def fake_check_database_health(pool):
        return {
            "status": "error",
            "version": "unknown",
            "response_time_ms": 5.0,
            "error": "Database unavailable",
        }

    monkeypatch.setattr(
        "music_mood_search.api.health.check_database_health",
        fake_check_database_health,
    )

    app.state.db_pool = object()

    response = await client.get("/api/v1/health")

    assert response.status_code == 503

    data = response.json()

    assert data["status"] == "error"
    assert data["components"]["database"]["status"] == "error"
    assert data["components"]["database"]["error"] == "Database unavailable"
