import aio_pika
import pytest
from aio_pika.exceptions import DeliveryError
from testcontainers.community.rabbitmq import RabbitMqContainer

from music_mood_search.messaging.topology import make_task_queue


@pytest.fixture(scope="module")
def rabbitmq_url():
    with RabbitMqContainer(
        "rabbitmq:4.1.8-management",
        username="test",
        password="test",
    ) as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5672)
        yield f"amqp://test:test@{host}:{port}/"


@pytest.mark.asyncio
async def test_queue_limit_and_dead_letter(rabbitmq_url):
    connection = await aio_pika.connect_robust(rabbitmq_url)

    try:
        channel = await connection.channel(
            publisher_confirms=True,
            on_return_raises=True,
        )

        queue_config = make_task_queue(max_length=2)

        assert queue_config.arguments["x-max-length"] == 2
        assert queue_config.arguments["x-overflow"] == "reject-publish"

        dlx = await channel.declare_exchange(
            queue_config.arguments["x-dead-letter-exchange"],
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )

        dlq = await channel.declare_queue(
            "music.tasks.dlq",
            durable=True,
        )

        await dlq.bind(
            dlx,
            routing_key=queue_config.arguments["x-dead-letter-routing-key"],
        )

        queue = await channel.declare_queue(
            queue_config.name,
            durable=queue_config.durable,
            arguments=queue_config.arguments,
        )

        async def publish(body: bytes):
            return await channel.default_exchange.publish(
                aio_pika.Message(
                    body=body,
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=queue.name,
                mandatory=True,
                timeout=5.0,
            )

        # Лимит очереди — два сообщения.
        await publish(b"task-1")
        await publish(b"task-2")

        # Третья публикация должна быть отклонена RabbitMQ.
        with pytest.raises(DeliveryError):
            await publish(b"task-3")

        # Отвергаем первое сообщение — оно должно попасть в DLQ.
        first = await queue.get(timeout=5.0)
        assert first.body == b"task-1"
        await first.reject(requeue=False)

        dead = await dlq.get(timeout=5.0)
        assert dead.body == b"task-1"
        await dead.ack()

        # Второе сообщение не потерялось при переполнении.
        second = await queue.get(timeout=5.0)
        assert second.body == b"task-2"
        await second.ack()

    finally:
        await connection.close()
