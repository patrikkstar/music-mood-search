from faststream.rabbit import (
    ExchangeType,
    RabbitBroker,
    RabbitExchange,
    RabbitQueue,
)

TASK_QUEUE_NAME = "music.tasks"
DLQ_NAME = "music.tasks.dlq"
DLQ_ROUTING_KEY = "music.tasks.failed"

DLX = RabbitExchange(
    "music.tasks.dlx",
    type=ExchangeType.DIRECT,
    durable=True,
)

DLQ = RabbitQueue(
    DLQ_NAME,
    durable=True,
)


def make_task_queue(max_length: int = 20) -> RabbitQueue:
    if max_length < 1:
        raise ValueError("Queue limit must be positive")

    return RabbitQueue(
        TASK_QUEUE_NAME,
        durable=True,
        arguments={
            "x-max-length": max_length,
            "x-overflow": "reject-publish",
            "x-dead-letter-exchange": DLX.name,
            "x-dead-letter-routing-key": DLQ_ROUTING_KEY,
        },
    )


async def declare_topology(
    broker: RabbitBroker,
    max_length: int = 20,
) -> None:
    exchange = await broker.declare_exchange(DLX)
    dead_queue = await broker.declare_queue(DLQ)
    await dead_queue.bind(exchange, routing_key=DLQ_ROUTING_KEY)

    await broker.declare_queue(make_task_queue(max_length))
