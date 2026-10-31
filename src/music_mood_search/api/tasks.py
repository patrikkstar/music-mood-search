import logging
from math import ceil
from typing import Literal
from uuid import UUID, uuid4

from aio_pika.exceptions import AMQPException, DeliveryError
from fastapi import APIRouter, HTTPException, Request, status
from faststream.rabbit import RabbitBroker
from pydantic import BaseModel

from music_mood_search.config import get_settings
from music_mood_search.schemas import ProcessRequest

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

    if broker is None or pool is None:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable",
            headers={"Retry-After": retry_after},
        )

    task_id = uuid4()

    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO inference_tasks (id, status)
                    VALUES ($1, 'PENDING')
                    """,
                    task_id,
                )

                confirmation = await broker.publish(
                    {
                        "task_id": str(task_id),
                        "embeddings": payload.embeddings,
                    },
                    queue="music.tasks",
                    persist=True,
                    mandatory=True,
                    message_id=str(task_id),
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
    ):
        logger.warning(
            "Task publication rejected or unavailable",
            extra={"task_id": str(task_id)},
        )
        raise HTTPException(
            status_code=503,
            detail="Queue overloaded or temporarily unavailable",
            headers={"Retry-After": retry_after},
        ) from None

    return TaskAcceptedResponse(task_id=task_id)
