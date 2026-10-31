import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from uuid import NAMESPACE_URL, uuid4, uuid5

import aio_pika
import pyarrow.parquet as pq
from aio_pika.exceptions import DeliveryError

from music_mood_search.config import get_settings


async def run_source(data_path: Path, limit: int) -> None:
    settings = get_settings()
    columns = [f"e{i}" for i in range(settings.embedding_dim)]
    parquet = pq.ParquetFile(data_path)

    sent = 0
    rejected = 0
    session_id = str(uuid4())
    started = perf_counter()

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

            # ID сохраняется при повторных попытках публикации.
            task_id = str(uuid5(NAMESPACE_URL, f"{session_id}:{index}"))

            payload = json.dumps(
                {
                    "task_id": task_id,
                    "embeddings": [embedding],
                }
            ).encode()

            delay = 0.1

            while True:
                try:
                    await channel.default_exchange.publish(
                        aio_pika.Message(
                            body=payload,
                            delivery_mode=(aio_pika.DeliveryMode.PERSISTENT),
                            message_id=task_id,
                            content_type="application/json",
                        ),
                        routing_key="music.tasks",
                        mandatory=True,
                        timeout=10,
                    )

                    sent += 1

                    if sent % 10 == 0:
                        elapsed = perf_counter() - started
                        print(
                            f"Sent={sent}, "
                            f"rejected={rejected}, "
                            f"rate={sent / elapsed:.2f} msg/s",
                            flush=True,
                        )
                    break

                except DeliveryError:
                    rejected += 1
                    print(
                        f"Queue full. Retrying {task_id} after {delay:.2f}s",
                        flush=True,
                    )

                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 1.0)

    elapsed = perf_counter() - started
    print(
        f"Finished: sent={sent}, "
        f"rejected={rejected}, "
        f"elapsed={elapsed:.1f}s, "
        f"rate={sent / elapsed:.2f} msg/s",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("training/data/out/demo_embeddings.parquet"),
    )
    parser.add_argument("--limit", type=int, default=40)

    args = parser.parse_args()

    asyncio.run(run_source(args.data, args.limit))
