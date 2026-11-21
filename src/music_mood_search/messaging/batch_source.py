import argparse
import asyncio
import json
import logging
from pathlib import Path
from time import perf_counter
from uuid import NAMESPACE_URL, uuid4, uuid5

import aio_pika
import pyarrow.parquet as pq
from aio_pika.exceptions import DeliveryError
from opentelemetry import propagate, trace

from music_mood_search.config import get_settings
from music_mood_search.logging_config import (
    configure_logging,
    task_context,
)
from music_mood_search.telemetry import (
    configure_telemetry,
    flush_telemetry,
    instruments,
    record_process_memory,
)

logger = logging.getLogger(__name__)


async def run_source(data_path: Path, limit: int) -> None:
    configure_logging()
    configure_telemetry()
    record_process_memory()

    settings = get_settings()
    columns = [f"e{i}" for i in range(settings.embedding_dim)]
    parquet = pq.ParquetFile(data_path)

    sent = 0
    rejected = 0
    total_wait_s = 0.0
    session_id = str(uuid4())
    started = perf_counter()

    logger.info(
        "Source started",
        extra={
            "event": "source_started",
            "batch_size": limit,
        },
    )

    connection = await aio_pika.connect_robust(settings.rabbitmq_url)

    async with connection:
        channel = await connection.channel(
            publisher_confirms=True,
            on_return_raises=True,
        )

        for index, batch in enumerate(
            parquet.iter_batches(
                batch_size=1,
                columns=columns,
            )
        ):
            if index >= limit:
                break

            embedding = [batch.column(j)[0].as_py() for j in range(len(columns))]

            # Идентификатор сохраняется при повторных попытках.
            task_id = str(uuid5(NAMESPACE_URL, f"{session_id}:{index}"))

            payload = json.dumps(
                {
                    "task_id": task_id,
                    "embeddings": [embedding],
                }
            ).encode()

            # Контекст устанавливается один раз на задачу.
            with task_context(task_id):
                with trace.get_tracer(__name__).start_as_current_span(
                    "source.publish", attributes={"messaging.message.id": task_id}
                ) as span:
                    headers = {}
                    propagate.inject(headers)
                    delay = 0.1

                    while True:
                        try:
                            await channel.default_exchange.publish(
                                aio_pika.Message(
                                    body=payload,
                                    delivery_mode=(aio_pika.DeliveryMode.PERSISTENT),
                                    message_id=task_id,
                                    correlation_id=task_id,
                                    headers=headers,
                                    content_type="application/json",
                                ),
                                routing_key="music.tasks",
                                mandatory=True,
                                timeout=10,
                            )

                            sent += 1
                            instruments()["published"].add(1, {"origin": "source"})

                            if sent % 10 == 0:
                                elapsed = perf_counter() - started

                                logger.info(
                                    "Source progress: %.2f msg/s",
                                    sent / max(elapsed, 1e-9),
                                    extra={
                                        "event": "source_progress",
                                        "sent": sent,
                                        "rejected": rejected,
                                    },
                                )

                            break

                        except DeliveryError:
                            rejected += 1
                            instruments()["rejected"].add(1, {"origin": "source"})
                            span.add_event(
                                "broker.publish.rejected", {"delay_s": delay}
                            )

                            logger.warning(
                                "Queue full; publication rejected",
                                extra={
                                    "event": "broker_rejected",
                                    "rejected": rejected,
                                    "delay_s": delay,
                                },
                            )

                            wait_start = perf_counter()
                            await asyncio.sleep(delay)
                            wait_elapsed = perf_counter() - wait_start
                            total_wait_s += wait_elapsed
                            instruments()["source_wait"].add(
                                wait_elapsed, {"origin": "source"}
                            )
                            span.add_event(
                                "backpressure.wait", {"duration_s": wait_elapsed}
                            )

                            delay = min(delay * 2, 1.0)

    elapsed = perf_counter() - started

    logger.info(
        "Source finished",
        extra={
            "event": "source_finished",
            "sent": sent,
            "rejected": rejected,
            "duration_ms": round(elapsed * 1000, 1),
            "total_wait_s": round(total_wait_s, 3),
        },
    )
    flush_telemetry()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("training/data/out/demo_embeddings.parquet"),
    )
    parser.add_argument("--limit", type=int, default=40)

    args = parser.parse_args()

    try:
        asyncio.run(run_source(args.data, args.limit))
    except Exception:
        logger.exception(
            "Source failed",
            extra={"event": "source_failed"},
        )
        raise SystemExit(1) from None
