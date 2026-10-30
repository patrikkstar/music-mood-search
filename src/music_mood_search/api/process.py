import logging
from time import perf_counter

import numpy as np
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from music_mood_search.config import get_settings
from music_mood_search.schemas import (
    ModelInfoResponse,
    ProcessRequest,
    ProcessResponse,
    TrackPrediction,
)
from music_mood_search.services.model import ModelService

router = APIRouter(prefix="/api/v1")
logger = logging.getLogger("music_mood_search.process")


def _get_loaded_service(request: Request) -> ModelService:
    service: ModelService | None = getattr(request.app.state, "model_service", None)
    if service is None or not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )
    return service


def _model_info(service: ModelService) -> ModelInfoResponse:
    info = service.info
    return ModelInfoResponse(
        name=info.name,
        alias=info.alias,
        version=info.version,
        run_id=info.run_id,
        uri=info.uri,
        embedding=info.embedding,
    )


@router.get("/model", response_model=ModelInfoResponse)
async def get_model(request: Request) -> ModelInfoResponse:
    return _model_info(_get_loaded_service(request))


@router.post("/process", response_model=ProcessResponse)
async def process(payload: ProcessRequest, request: Request) -> ProcessResponse:
    settings = get_settings()
    service = _get_loaded_service(request)

    if len(payload.embeddings) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Batch size must be <= {settings.max_batch_size}",
        )
    bad = [
        i
        for i, emb in enumerate(payload.embeddings)
        if len(emb) != settings.embedding_dim
    ]
    if bad:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Each embedding must have {settings.embedding_dim} values, "
                f"wrong items: {bad[:10]}"
            ),
        )

    # NaN, Infinity и значения вне диапазона float32 модель обработать не может.
    # Проверяем здесь, а не в схеме: ошибка валидации pydantic вернула бы
    # исходное значение NaN в теле ответа, а его нельзя сериализовать в JSON.
    with np.errstate(over="ignore"):
        batch = np.asarray(payload.embeddings, dtype=np.float32)
    not_finite = np.flatnonzero(~np.isfinite(batch).all(axis=1)).tolist()
    if not_finite:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Embeddings must contain only finite float32 values, "
                f"wrong items: {not_finite[:10]}"
            ),
        )

    started_at = perf_counter()
    # Inference CPU-bound, поэтому выполняется в пуле потоков
    # и не блокирует event loop.
    raw = await run_in_threadpool(service.predict, batch)
    inference_ms = round((perf_counter() - started_at) * 1000, 2)

    info = _model_info(service)
    logger.info(
        "Inference completed",
        extra={
            "event": "inference",
            "model_version": info.version,
            "batch_size": len(payload.embeddings),
            "duration_ms": inference_ms,
        },
    )

    return ProcessResponse(
        model=info,
        predictions=[TrackPrediction(**item) for item in raw],
        inference_ms=inference_ms,
    )
