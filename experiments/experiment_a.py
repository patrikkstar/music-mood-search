import argparse
import asyncio
import base64
import csv
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import quote, unquote, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

import aio_pika
import pyarrow.parquet as pq

from music_mood_search.config import get_settings
from music_mood_search.services.model import ModelService


def fetch_metrics(name, api_url, auth):
    def get(path):
        request = Request(
            api_url.rstrip("/") + path,
            headers={"Authorization": auth},
        )
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    queue = get(f"/api/queues/%2F/{quote(name, safe='')}")
    nodes = get("/api/nodes")
    memory = queue.get("memory")

    return {
        "ready": queue["messages_ready"],
        "unacked": queue["messages_unacknowledged"],
        "consumers": queue["consumers"],
        "queue_memory_bytes": memory if memory is not None else "",
        "broker_memory_mb": round(
            sum(node.get("mem_used", 0) for node in nodes) / 1024**2, 2
        ),
    }


async def run(args):
    settings = get_settings()
    name = f"music.tasks.unbounded.{uuid4().hex[:8]}"

    url = urlsplit(settings.rabbitmq_url)
    credentials = f"{unquote(url.username or '')}:{unquote(url.password or '')}"
    auth = "Basic " + base64.b64encode(credentials.encode()).decode()

    model = ModelService(settings)
    info = await asyncio.to_thread(model.load)
    print(f"Model loaded: {info.name} v{info.version}")

    completed = 0
    failed = 0
    durations = []
    finished = asyncio.Event()

    consumer_connection = await aio_pika.connect_robust(settings.rabbitmq_url)
    producer_connection = await aio_pika.connect_robust(settings.rabbitmq_url)

    async with consumer_connection, producer_connection:
        consumer = await consumer_connection.channel()
        await consumer.set_qos(prefetch_count=1)
        queue = await consumer.declare_queue(name, durable=False, auto_delete=True)

        async def handle(message):
            nonlocal completed, failed
            try:
                payload = json.loads(message.body)
                start = perf_counter()

                await asyncio.sleep(settings.inference_delay_s)
                await asyncio.to_thread(model.predict, payload["embeddings"])

                durations.append((perf_counter() - start) * 1000)
                await message.ack()
                completed += 1

            except Exception as error:
                failed += 1
                print(f"Inference failed: {error}", flush=True)
                await message.reject(requeue=False)

            if completed + failed == args.limit:
                finished.set()

        await queue.consume(handle)

        producer = await producer_connection.channel(
            publisher_confirms=True,
            on_return_raises=True,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        started = perf_counter()

        async def monitor():
            fields = [
                "elapsed_s",
                "ready",
                "unacked",
                "consumers",
                "queue_memory_bytes",
                "broker_memory_mb",
            ]

            with args.output.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()

                while perf_counter() - started < args.duration:
                    cycle = perf_counter()
                    try:
                        metrics = await asyncio.to_thread(
                            fetch_metrics, name, args.api_url, auth
                        )
                        writer.writerow(
                            {
                                "elapsed_s": round(perf_counter() - started, 2),
                                **metrics,
                            }
                        )
                        f.flush()
                    except (OSError, ValueError, KeyError) as error:
                        print(f"Monitor: {error}", flush=True)

                    await asyncio.sleep(
                        max(0, args.interval - (perf_counter() - cycle))
                    )

        monitor_task = asyncio.create_task(monitor())

        parquet = pq.ParquetFile(args.data)
        columns = [f"e{i}" for i in range(settings.embedding_dim)]

        sent = 0

        for batch in parquet.iter_batches(batch_size=1, columns=columns):
            if sent >= args.limit:
                break

            embedding = [batch.column(j)[0].as_py() for j in range(len(columns))]

            payload = json.dumps(
                {
                    "task_id": str(uuid4()),
                    "embeddings": [embedding],
                }
            ).encode()

            await producer.default_exchange.publish(
                aio_pika.Message(
                    body=payload,
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    content_type="application/json",
                ),
                routing_key=name,
                mandatory=True,
                timeout=10,
            )

            sent += 1
            if sent % 10 == 0:
                print(f"Sent={sent}", flush=True)

        publish_elapsed = perf_counter() - started

        try:
            await asyncio.wait_for(
                finished.wait(),
                timeout=max(1, args.duration - publish_elapsed),
            )
        except TimeoutError:
            print("Processing did not finish within the time limit")

        await monitor_task

    print(f"\nQueue: {name}")
    print(f"Published: {sent} in {publish_elapsed:.2f}s")
    print(f"Publish rate: {sent / publish_elapsed:.2f} msg/s")
    print(f"Completed: {completed}, failed: {failed}")
    if durations:
        print(f"Average inference: {sum(durations) / len(durations):.2f} ms")
    print(f"CSV: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("training/data/out/demo_embeddings.parquet"),
    )
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--duration", type=float, default=90)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--api-url", default="http://localhost:15672")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/a_unbounded.csv"),
    )
    asyncio.run(parser.parse_args() and run(parser.parse_args()))
