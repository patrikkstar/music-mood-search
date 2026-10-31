from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from faststream.rabbit import TestRabbitBroker

from music_mood_search.messaging import worker


@pytest.fixture
def fake_dependencies(monkeypatch):
    pool = SimpleNamespace(
        fetchval=AsyncMock(return_value=None),
        execute=AsyncMock(),
    )

    model = SimpleNamespace(
        info=SimpleNamespace(version="1"),
        predict=Mock(return_value=[{"mood": [], "genre": []}]),
    )

    monkeypatch.setattr(worker, "db_pool", pool)
    monkeypatch.setattr(worker, "model_service", model)
    monkeypatch.setattr(worker.settings, "inference_delay_s", 0.0)

    return pool, model


@pytest.mark.asyncio
async def test_successful_task(fake_dependencies):
    pool, model = fake_dependencies
    task_id = uuid4()

    async with TestRabbitBroker(
        worker.broker,
        with_real=False,
        connect_only=False,
    ) as broker:
        await broker.publish(
            {
                "task_id": str(task_id),
                "embeddings": [[0.0] * worker.settings.embedding_dim],
            },
            queue="music.tasks",
        )

    model.predict.assert_called_once()
    pool.execute.assert_awaited_once()

    saved_args = pool.execute.await_args.args
    assert saved_args[1] == task_id
    assert saved_args[3] == "1"


@pytest.mark.asyncio
async def test_completed_task_is_not_processed_again(fake_dependencies):
    pool, model = fake_dependencies
    pool.fetchval.return_value = "COMPLETED"

    async with TestRabbitBroker(
        worker.broker,
        with_real=False,
        connect_only=False,
    ) as broker:
        await broker.publish(
            {
                "task_id": str(uuid4()),
                "embeddings": [[0.0] * worker.settings.embedding_dim],
            },
            queue="music.tasks",
        )

    model.predict.assert_not_called()
    pool.execute.assert_not_awaited()
