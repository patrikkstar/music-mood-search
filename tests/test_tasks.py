from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from music_mood_search.app import app

EMBEDDING = [0.0] * 1280


@pytest.fixture
def dependencies(monkeypatch):
    state = {"committed": False, "rolled_back": False}
    connection = SimpleNamespace(execute=AsyncMock())

    @asynccontextmanager
    async def transaction():
        try:
            yield
        except Exception:
            state["rolled_back"] = True
            raise
        else:
            state["committed"] = True

    @asynccontextmanager
    async def acquire():
        yield connection

    connection.transaction = transaction
    pool = SimpleNamespace(acquire=acquire)
    broker = SimpleNamespace(publish=AsyncMock(return_value=True))

    monkeypatch.setattr(app.state, "db_pool", pool, raising=False)
    monkeypatch.setattr(app.state, "rabbit_broker", broker, raising=False)

    return connection, broker, state


async def test_task_accepted(client, dependencies):
    connection, broker, state = dependencies

    response = await client.post(
        "/api/v1/tasks",
        json={"embeddings": [EMBEDDING]},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "PENDING"
    assert response.json()["task_id"]
    connection.execute.assert_awaited_once()
    broker.publish.assert_awaited_once()
    assert broker.publish.await_args.kwargs["queue"] == "music.tasks"
    assert state["committed"] is True


async def test_queue_overloaded(client, dependencies):
    connection, broker, state = dependencies
    broker.publish.return_value = False

    response = await client.post(
        "/api/v1/tasks",
        json={"embeddings": [EMBEDDING]},
    )

    assert response.status_code == 503
    assert int(response.headers["retry-after"]) >= 1
    assert state["rolled_back"] is True
    assert state["committed"] is False
    connection.execute.assert_awaited_once()


async def test_invalid_embedding(client, dependencies):
    connection, broker, _ = dependencies

    response = await client.post(
        "/api/v1/tasks",
        json={"embeddings": [[0.0]]},
    )

    assert response.status_code == 422
    connection.execute.assert_not_awaited()
    broker.publish.assert_not_awaited()
