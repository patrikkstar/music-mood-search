import asyncio
import json
import logging
from time import perf_counter
from uuid import UUID

import asyncpg
from faststream import AckPolicy, FastStream
from faststream.rabbit import RabbitBroker
from faststream.rabbit.annotations import RabbitMessage
from faststream.rabbit.opentelemetry import RabbitTelemetryMiddleware
from faststream.rabbit.schemas import Channel
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, Field, ValidationError

from music_mood_search.config import get_settings
from music_mood_search.db import close_db_pool, create_db_pool
from music_mood_search.logging_config import (
    configure_logging,
    task_context,
)
from music_mood_search.messaging.topology import (
    declare_topology,
    make_task_queue,
)
from music_mood_search.services.model import ModelService
from music_mood_search.telemetry import (
    configure_telemetry,
    flush_telemetry,
    instruments,
    record_process_memory,
    traced_db_execute,
    traced_inference,
)

logger = logging.getLogger("music_mood_search.messaging.worker")
MAX_ATTEMPTS = 3

settings = get_settings()
model_service = ModelService(settings)
db_pool: asyncpg.Pool | None = None

configure_telemetry()
broker = RabbitBroker(
    settings.rabbitmq_url,
    middlewares=(RabbitTelemetryMiddleware(tracer_provider=configure_telemetry()[0]),),
    default_channel=Channel(prefetch_count=1),
    graceful_timeout=60.0,
    logger=logger,
)

app = FastStream(broker, logger=logger)


class InferenceTask(BaseModel):
    task_id: UUID
    embeddings: list[list[float]] = Field(min_length=1)


@app.on_startup
async def startup() -> None:
    global db_pool

    configure_logging()

    logger.info(
        "Worker startup",
        extra={"event": "worker_startup"},
    )

    info = await asyncio.to_thread(model_service.load)

    logger.info(
        "Model loaded: %s",
        info.name,
        extra={
            "event": "model_loaded",
            "model_alias": info.alias,
            "model_version": info.version,
        },
    )

    db_pool = await create_db_pool()
    record_process_memory()

    logger.info(
        "PostgreSQL connected",
        extra={"event": "database_connected"},
    )


@app.after_startup
async def setup_queues() -> None:
    await declare_topology(
        broker,
        settings.rabbitmq_queue_limit,
    )

    logger.info(
        "RabbitMQ connected. Worker ready",
        extra={"event": "worker_ready"},
    )


@broker.subscriber(
    make_task_queue(settings.rabbitmq_queue_limit),
    ack_policy=AckPolicy.MANUAL,
    no_reply=True,
)
async def process_task(
    payload: dict,
    msg: RabbitMessage,
) -> None:
    raw_task_id = payload.get("task_id") if isinstance(payload, dict) else None
    raw_task_id = raw_task_id or getattr(msg, "message_id", None)

    with task_context(str(raw_task_id) if raw_task_id else "unknown"):
        with trace.get_tracer(__name__).start_as_current_span("process"):
            metric_tools = instruments()
            metric_tools["inflight"].add(1)
            raw_message = getattr(msg, "raw_message", None)
            if getattr(raw_message, "redelivered", getattr(msg, "redelivered", False)):
                metric_tools["redelivered"].add(1)
            try:
                await _process_task(payload, msg)
            finally:
                metric_tools["inflight"].add(-1)


async def _process_task(
    payload: dict,
    msg: RabbitMessage,
) -> None:
    if db_pool is None:
        logger.error(
            "Database pool unavailable",
            extra={"event": "database_unavailable"},
        )
        await msg.nack(requeue=True)
        return

    try:
        task = InferenceTask.model_validate(payload)
    except ValidationError:
        instruments()["processed"].add(1, {"result": "dlq"})
        logger.warning(
            "Invalid task message; sending to DLQ",
            extra={"event": "task_invalid_dlq"},
        )
        await msg.reject(requeue=False)
        return

    logger.info(
        "Task processing started",
        extra={"event": "task_started"},
    )

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
        predictions = await traced_inference(
            model_service.predict,
            task.embeddings,
            delay_s=settings.inference_delay_s,
            alias=getattr(model_service.info, "alias", "unknown"),
            version=model_service.info.version,
        )

        duration_ms = (perf_counter() - start) * 1000

        await traced_db_execute(
            db_pool,
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
        instruments()["processed"].add(1, {"result": "ok"})
        logger.info(
            "Task completed",
            extra={
                "event": "task_completed",
                "duration_ms": round(duration_ms, 1),
            },
        )

    except Exception as exc:
        span = trace.get_current_span()
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR))
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
            logger.error(
                "Task processing failed",
                exc_info=(type(exc), exc, exc.__traceback__),
                extra={
                    "event": "task_processing_error",
                    "attempt": attempts,
                },
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

                instruments()["processed"].add(1, {"result": "dlq"})
                logger.error(
                    "Task moved to DLQ",
                    extra={
                        "event": "task_dlq",
                        "attempt": attempts,
                    },
                )
                await msg.reject(requeue=False)
            else:
                instruments()["processed"].add(1, {"result": "error"})
                logger.warning(
                    "Task scheduled for retry",
                    extra={
                        "event": "task_retry",
                        "attempt": attempts,
                    },
                )
                await asyncio.sleep(1)
                await msg.nack(requeue=True)

        except Exception:
            # При недоступности БД не подтверждаем потерю задачи
            logger.exception(
                "Failed to update task state; requeueing",
                extra={"event": "task_state_update_error"},
            )
            await asyncio.sleep(1)
            await msg.nack(requeue=True)


@app.after_shutdown
async def shutdown() -> None:
    logger.info(
        "Worker shutdown started",
        extra={"event": "shutdown_started"},
    )

    if db_pool is not None:
        await close_db_pool(db_pool)
    flush_telemetry()

    logger.info(
        "Worker shutdown completed",
        extra={"event": "shutdown_completed"},
    )


if __name__ == "__main__":
    configure_logging()
    asyncio.run(app.run())
