import logging
from math import ceil
from typing import Literal
from uuid import UUID, uuid4

from aio_pika.exceptions import AMQPException, DeliveryError
from fastapi import APIRouter, HTTPException, Request, status
from faststream.rabbit import RabbitBroker
from opentelemetry import trace
from pydantic import BaseModel

from music_mood_search.config import get_settings
from music_mood_search.logging_config import task_context
from music_mood_search.schemas import ProcessRequest
from music_mood_search.telemetry import instruments

router = APIRouter(prefix="/api/v1", tags=["tasks"])
logger = logging.getLogger(__name__)


class TaskAcceptedResponse(BaseModel):
    task_id: UUID
    status: Literal["PENDING"] = "PENDING"


class PublishRejected(Exception):
    pass


@router.post(
    "/tasks",
    response_model=TaskAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_task(
    payload: ProcessRequest,
    request: Request,
) -> TaskAcceptedResponse:
    settings = get_settings()

    if len(payload.embeddings) > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail="Batch size exceeded",
        )

    if any(
        len(embedding) != settings.embedding_dim for embedding in payload.embeddings
    ):
        raise HTTPException(
            status_code=422,
            detail="Invalid embedding dimension",
        )

    # Два периода обработки с учётом измеренной
    # задержки около 1 секунды на задачу.
    retry_after = str(max(1, ceil(settings.inference_delay_s * 2)))

    broker: RabbitBroker | None = getattr(request.app.state, "rabbit_broker", None)
    pool = getattr(request.app.state, "db_pool", None)

    task_id = uuid4()

    with task_context(str(task_id)):
        if broker is None or pool is None:
            logger.error(
                "API dependencies unavailable",
                extra={
                    "event": "service_unavailable",
                    "status_code": 503,
                },
            )
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable",
                headers={"Retry-After": retry_after},
            )

        try:
            async with pool.acquire() as connection:
                async with connection.transaction():
                    with trace.get_tracer(__name__).start_as_current_span(
                        "db.enqueue_task"
                    ):
                        await connection.execute(
                            """
                            INSERT INTO inference_tasks (id, status)
                            VALUES ($1, 'PENDING')
                            """,
                            task_id,
                        )

                    with trace.get_tracer(__name__).start_as_current_span(
                        "rabbitmq.publish"
                    ):
                        confirmation = await broker.publish(
                            {
                                "task_id": str(task_id),
                                "embeddings": payload.embeddings,
                            },
                            queue="music.tasks",
                            persist=True,
                            mandatory=True,
                            message_id=str(task_id),
                            correlation_id=str(task_id),
                            timeout=1.0,
                        )

                    if confirmation is False:
                        raise PublishRejected("RabbitMQ rejected publication")

        except (
            TimeoutError,
            DeliveryError,
            PublishRejected,
            AMQPException,
            ConnectionError,
            OSError,
        ) as exc:
            instruments()["rejected"].add(1, {"origin": "api"})
            logger.warning(
                "Task publication failed: %s",
                type(exc).__name__,
                extra={
                    "event": "publish_rejected",
                    "status_code": 503,
                    "error": type(exc).__name__,
                },
            )
            raise HTTPException(
                status_code=503,
                detail="Queue overloaded or temporarily unavailable",
                headers={"Retry-After": retry_after},
            ) from None

        instruments()["published"].add(1, {"origin": "api"})
        logger.info(
            "Task accepted",
            extra={
                "event": "task_accepted",
                "status_code": 202,
                "batch_size": len(payload.embeddings),
            },
        )

        return TaskAcceptedResponse(task_id=task_id)
