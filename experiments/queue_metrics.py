import argparse
import asyncio
import base64
import csv
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from music_mood_search.config import get_settings


def fetch_metrics(api_url: str, authorization: str) -> dict:
    def fetch(path: str):
        request = Request(
            api_url.rstrip("/") + path,
            headers={"Authorization": authorization},
        )
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    queue = fetch("/api/queues/%2F/music.tasks")
    nodes = fetch("/api/nodes")

    memory = queue.get("memory")
    broker_memory = [
        node["mem_used"]
        for node in nodes
        if isinstance(node.get("mem_used"), (int, float))
    ]

    return {
        "ready": queue["messages_ready"],
        "unacked": queue["messages_unacknowledged"],
        "consumers": queue["consumers"],
        "queue_memory_bytes": memory if memory is not None else "",
        "broker_memory_mb": (
            round(sum(broker_memory) / 1024**2, 2) if broker_memory else ""
        ),
    }


async def monitor(args: argparse.Namespace) -> None:
    settings = get_settings()
    url = urlsplit(settings.rabbitmq_url)

    credentials = f"{unquote(url.username or '')}:{unquote(url.password or '')}"
    authorization = "Basic " + base64.b64encode(credentials.encode()).decode()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "elapsed_s",
        "ready",
        "unacked",
        "consumers",
        "queue_memory_bytes",
        "broker_memory_mb",
    ]

    started = perf_counter()
    samples = 0

    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        while perf_counter() - started < args.duration:
            cycle_started = perf_counter()

            try:
                metrics = await asyncio.to_thread(
                    fetch_metrics, args.api_url, authorization
                )
                writer.writerow(
                    {
                        "elapsed_s": round(perf_counter() - started, 2),
                        **metrics,
                    }
                )
                file.flush()
                samples += 1
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(f"Metrics unavailable: {error}", flush=True)

            await asyncio.sleep(
                max(0, args.interval - (perf_counter() - cycle_started))
            )

    print(f"Saved {samples} samples to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--api-url", default="http://localhost:15672")
    asyncio.run(monitor(parser.parse_args()))


if __name__ == "__main__":
    main()
