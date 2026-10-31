from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from music_mood_search.messaging import topology, worker


@pytest.fixture
def dependencies(monkeypatch):
    db = SimpleNamespace(
        fetchval=AsyncMock(return_value=None),
        execute=AsyncMock(),
    )
    model = SimpleNamespace(
        info=SimpleNamespace(version="1"),
        predict=Mock(return_value=[]),
    )
    monkeypatch.setattr(worker, "db_pool", db)
    monkeypatch.setattr(worker, "model_service", model)
    monkeypatch.setattr(worker.settings, "inference_delay_s", 0.0)
    monkeypatch.setattr(worker.asyncio, "sleep", AsyncMock())
    return db, model


def task(embedding=None):
    return {
        "task_id": str(uuid4()),
        "embeddings": [
            embedding
            if embedding is not None
            else [0.0] * worker.settings.embedding_dim
        ],
    }


@pytest.mark.asyncio
async def test_missing_database_requeues(monkeypatch):
    monkeypatch.setattr(worker, "db_pool", None)
    msg = AsyncMock()

    await worker.process_task(task(), msg)

    msg.nack.assert_awaited_once_with(requeue=True)


@pytest.mark.asyncio
async def test_invalid_message_goes_to_dlq(dependencies):
    msg = AsyncMock()

    await worker.process_task({"task_id": "invalid"}, msg)

    msg.reject.assert_awaited_once_with(requeue=False)


@pytest.mark.asyncio
async def test_previously_failed_task_is_rejected(dependencies):
    db, model = dependencies
    db.fetchval.return_value = "FAILED"
    msg = AsyncMock()

    await worker.process_task(task(), msg)

    msg.reject.assert_awaited_once_with(requeue=False)
    model.predict.assert_not_called()


@pytest.mark.asyncio
async def test_wrong_embedding_dimension_retries(dependencies):
    db, model = dependencies
    db.fetchval.side_effect = [None, 1]
    msg = AsyncMock()

    await worker.process_task(task([0.1, 0.2]), msg)

    msg.nack.assert_awaited_once_with(requeue=True)
    model.predict.assert_not_called()


@pytest.mark.asyncio
async def test_batch_limit_sends_to_dlq(dependencies, monkeypatch):
    db, model = dependencies
    db.fetchval.side_effect = [None, 3]
    monkeypatch.setattr(worker.settings, "max_batch_size", 1)
    msg = AsyncMock()

    payload = task()
    payload["embeddings"].append([0.0] * worker.settings.embedding_dim)

    await worker.process_task(payload, msg)

    msg.reject.assert_awaited_once_with(requeue=False)
    db.execute.assert_awaited_once()
    model.predict.assert_not_called()


@pytest.mark.asyncio
async def test_inference_failure_after_max_attempts(dependencies):
    db, model = dependencies
    db.fetchval.side_effect = [None, 3]
    model.predict.side_effect = RuntimeError("inference failed")
    msg = AsyncMock()

    await worker.process_task(task(), msg)

    msg.reject.assert_awaited_once_with(requeue=False)
    db.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_completed_during_error_handling_is_acked(dependencies):
    db, _ = dependencies
    db.fetchval.side_effect = [None, None]
    msg = AsyncMock()

    await worker.process_task(task([0.1]), msg)

    msg.ack.assert_awaited_once()
    msg.nack.assert_not_awaited()


@pytest.mark.asyncio
async def test_database_error_during_retry_requeues(dependencies):
    db, _ = dependencies
    db.fetchval.side_effect = [
        None,
        RuntimeError("database unavailable"),
    ]
    msg = AsyncMock()

    await worker.process_task(task([0.1]), msg)

    msg.nack.assert_awaited_once_with(requeue=True)


@pytest.mark.asyncio
async def test_worker_startup(monkeypatch):
    model = SimpleNamespace(
        load=Mock(
            return_value=SimpleNamespace(
                name="music-tagger",
                version="1",
                alias="champion",
            )
        )
    )
    pool = object()
    create_pool = AsyncMock(return_value=pool)

    monkeypatch.setattr(worker, "model_service", model)
    monkeypatch.setattr(worker, "create_db_pool", create_pool)
    monkeypatch.setattr(worker, "db_pool", None)

    await worker.startup()

    model.load.assert_called_once()
    create_pool.assert_awaited_once()
    assert worker.db_pool is pool


@pytest.mark.asyncio
async def test_worker_setup_queues(monkeypatch):
    declare = AsyncMock()
    monkeypatch.setattr(worker, "declare_topology", declare)

    await worker.setup_queues()

    declare.assert_awaited_once_with(
        worker.broker,
        worker.settings.rabbitmq_queue_limit,
    )


@pytest.mark.asyncio
async def test_worker_shutdown(monkeypatch):
    pool = object()
    close_pool = AsyncMock()
    monkeypatch.setattr(worker, "db_pool", pool)
    monkeypatch.setattr(worker, "close_db_pool", close_pool)

    await worker.shutdown()

    close_pool.assert_awaited_once_with(pool)


@pytest.mark.asyncio
async def test_topology_declaration():
    dead_queue = SimpleNamespace(bind=AsyncMock())
    exchange = object()
    broker = SimpleNamespace(
        declare_exchange=AsyncMock(return_value=exchange),
        declare_queue=AsyncMock(side_effect=[dead_queue, object()]),
    )

    await topology.declare_topology(broker, max_length=20)

    broker.declare_exchange.assert_awaited_once()
    assert broker.declare_queue.await_count == 2
    dead_queue.bind.assert_awaited_once_with(
        exchange,
        routing_key=topology.DLQ_ROUTING_KEY,
    )


def test_queue_limit_validation():
    with pytest.raises(ValueError, match="positive"):
        topology.make_task_queue(0)
