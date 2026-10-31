import asyncio
import json
from time import perf_counter
from uuid import UUID

import asyncpg
from faststream import AckPolicy, FastStream
from faststream.rabbit import RabbitBroker
from faststream.rabbit.annotations import RabbitMessage
from faststream.rabbit.schemas import Channel
from pydantic import BaseModel, Field, ValidationError

from music_mood_search.config import get_settings
from music_mood_search.db import close_db_pool, create_db_pool
from music_mood_search.messaging.topology import (
    declare_topology,
    make_task_queue,
)
from music_mood_search.services.model import ModelService

MAX_ATTEMPTS = 3

settings = get_settings()
model_service = ModelService(settings)
db_pool: asyncpg.Pool | None = None

broker = RabbitBroker(
    settings.rabbitmq_url,
    default_channel=Channel(prefetch_count=1),
    graceful_timeout=60.0,
)

app = FastStream(broker)


class InferenceTask(BaseModel):
    task_id: UUID
    embeddings: list[list[float]] = Field(min_length=1)


@app.on_startup
async def startup() -> None:
    global db_pool

    info = await asyncio.to_thread(model_service.load)
    print(
        f"Model loaded: {info.name} version={info.version} alias={info.alias}",
        flush=True,
    )

    db_pool = await create_db_pool()
    print("PostgreSQL connected.", flush=True)


@app.after_startup
async def setup_queues() -> None:
    await declare_topology(broker, settings.rabbitmq_queue_limit)
    print("RabbitMQ connected. Worker ready.", flush=True)


@broker.subscriber(
    make_task_queue(settings.rabbitmq_queue_limit),
    ack_policy=AckPolicy.MANUAL,
    no_reply=True,
)
async def process_task(
    payload: dict,
    msg: RabbitMessage,
) -> None:
    if db_pool is None:
        await msg.nack(requeue=True)
        return

    try:
        task = InferenceTask.model_validate(payload)
    except ValidationError as exc:
        print(f"Invalid message: {exc}")
        await msg.reject(requeue=False)
        return

    try:
        existing = await db_pool.fetchval(
            "SELECT status FROM inference_tasks WHERE id = $1",
            task.task_id,
        )

        if existing == "COMPLETED":
            await msg.ack()
            return

        if existing == "FAILED":
            await msg.reject(requeue=False)
            return

        if len(task.embeddings) > settings.max_batch_size:
            raise ValueError("Batch size exceeded")

        if any(
            len(embedding) != settings.embedding_dim for embedding in task.embeddings
        ):
            raise ValueError("Invalid embedding dimension")

        start = perf_counter()
        await asyncio.sleep(settings.inference_delay_s)

        predictions = await asyncio.to_thread(
            model_service.predict,
            task.embeddings,
        )

        duration_ms = (perf_counter() - start) * 1000

        await db_pool.execute(
            """
            INSERT INTO inference_tasks (
                id, status, result, model_version,
                inference_ms, attempts, updated_at, finished_at
            )
            VALUES (
                $1, 'COMPLETED', $2::jsonb, $3,
                $4, 1, NOW(), NOW()
            )
            ON CONFLICT (id) DO UPDATE SET
                status = 'COMPLETED',
                result = EXCLUDED.result,
                model_version = EXCLUDED.model_version,
                inference_ms = EXCLUDED.inference_ms,
                attempts = inference_tasks.attempts + 1,
                last_error = NULL,
                updated_at = NOW(),
                finished_at = NOW()
            WHERE inference_tasks.status <> 'COMPLETED'
            """,
            task.task_id,
            json.dumps(predictions),
            model_service.info.version,
            duration_ms,
        )

        await msg.ack()
        print(f"Task {task.task_id}: saved ({duration_ms:.1f} ms)")

    except Exception as exc:
        try:
            attempts = await db_pool.fetchval(
                """
                INSERT INTO inference_tasks (
                    id, status, attempts, last_error, updated_at
                )
                VALUES ($1, 'PENDING', 1, $2, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    attempts = inference_tasks.attempts + 1,
                    last_error = EXCLUDED.last_error,
                    updated_at = NOW()
                WHERE inference_tasks.status <> 'COMPLETED'
                RETURNING attempts
                """,
                task.task_id,
                str(exc)[:2000],
            )

            if attempts is None:
                await msg.ack()
                return

            if attempts >= MAX_ATTEMPTS:
                await db_pool.execute(
                    """
                    UPDATE inference_tasks
                    SET status = 'FAILED',
                        updated_at = NOW(),
                        finished_at = NOW()
                    WHERE id = $1 AND status <> 'COMPLETED'
                    """,
                    task.task_id,
                )

                print(f"Task {task.task_id}: FAILED after {attempts} attempts")
                await msg.reject(requeue=False)
            else:
                print(f"Task {task.task_id}: retry {attempts}/{MAX_ATTEMPTS}")
                await asyncio.sleep(1)
                await msg.nack(requeue=True)

        except Exception:
            # При недоступности БД не подтверждаем потерю задачи.
            await asyncio.sleep(1)
            await msg.nack(requeue=True)


@app.after_shutdown
async def shutdown() -> None:
    if db_pool is not None:
        await close_db_pool(db_pool)


if __name__ == "__main__":
    asyncio.run(app.run())
