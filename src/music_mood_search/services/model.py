"""Загрузка модели из MLflow Model Registry и inference.

Модель загружается один раз при старте приложения (см. ``load_model_on_startup``).
Alias (например, ``champion``) резолвится в конкретный номер версии в момент
загрузки, и дальше приложение работает только с этой версией. Если alias
переназначат, работающее приложение этого не заметит до перезапуска.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.pyfunc import load_model as load_pyfunc_model

from music_mood_search.config import Settings

logger = logging.getLogger("music_mood_search.model")


class ModelNotLoadedError(RuntimeError):
    """Модель ещё не загружена или загрузка не удалась."""


@dataclass(frozen=True)
class ModelInfo:
    name: str
    alias: str
    version: str
    run_id: str | None
    uri: str
    loaded_at: float
    embedding: dict[str, Any] | None = None


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._info: ModelInfo | None = None
        self.last_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def info(self) -> ModelInfo:
        if self._info is None:
            raise ModelNotLoadedError("Model is not loaded")
        return self._info

    def load(self) -> ModelInfo:
        """Резолвит alias в версию и загружает именно эту версию."""
        settings = self._settings
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

        model_version = client.get_model_version_by_alias(
            settings.model_name, settings.model_alias
        )
        version_uri = f"models:/{settings.model_name}/{model_version.version}"
        model = load_pyfunc_model(version_uri)

        self._model = model
        self._info = ModelInfo(
            name=settings.model_name,
            alias=settings.model_alias,
            version=str(model_version.version),
            run_id=model_version.run_id,
            uri=version_uri,
            loaded_at=time.time(),
            embedding=_embedding_contract(model),
        )
        self.last_error = None
        return self._info

    def predict(
        self, embeddings: np.ndarray | list[list[float]]
    ) -> list[dict[str, Any]]:
        if self._model is None:
            raise ModelNotLoadedError("Model is not loaded")
        # Готовый массив float32 (его передаёт /process) используется как есть,
        # без копирования; список чисел превращается в массив.
        batch = np.asarray(embeddings, dtype=np.float32)
        return list(self._model.predict(batch))


def _embedding_contract(model: Any) -> dict[str, Any] | None:
    """Описание эмбеддинга из метаданных pyfunc-модели (если оно сохранено)."""
    metadata = getattr(getattr(model, "metadata", None), "metadata", None)
    if isinstance(metadata, dict) and isinstance(metadata.get("embedding"), dict):
        return metadata["embedding"]
    return None


def load_model_on_startup(
    service: ModelService,
    settings: Settings,
    sleep=time.sleep,
) -> bool:
    """Пытается загрузить модель несколько раз.

    Если не получилось, приложение всё равно стартует (healthz работает),
    а /process отвечает 503. Так сервис не уходит в бесконечный рестарт,
    пока в Model Registry ещё нет ни одной версии.
    """
    attempts = max(1, settings.model_load_retries)
    for attempt in range(1, attempts + 1):
        try:
            info = service.load()
        except Exception as exc:  # noqa: BLE001
            service.last_error = str(exc)
            logger.warning(
                "Model loading failed",
                extra={
                    "event": "model_load_failed",
                    "component": "mlflow",
                    "model_uri": settings.model_uri,
                    "attempt": attempt,
                    "error": str(exc),
                },
            )
            if attempt < attempts:
                sleep(settings.model_load_retry_delay_s)
            continue

        logger.info(
            "Model loaded",
            extra={
                "event": "model_loaded",
                "component": "mlflow",
                "model_uri": settings.model_uri,
                "model_version": info.version,
                "run_id": info.run_id,
            },
        )
        return True

    logger.error(
        "Model is not available, /process will return 503",
        extra={
            "event": "model_unavailable",
            "component": "mlflow",
            "model_uri": settings.model_uri,
        },
    )
    return False
