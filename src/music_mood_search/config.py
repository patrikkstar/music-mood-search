from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "MLOps Service"
    environment: str = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql://postgres:postgres@localhost:55432/postgres"

    # MLflow: модель выбирается через Model Registry по alias.
    # Чтобы сменить рабочую версию, достаточно переназначить alias
    # и перезапустить приложение, код при этом не меняется.
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "music-tagger"
    model_alias: str = "champion"
    model_load_retries: int = 3
    model_load_retry_delay_s: float = 5.0

    # Размерность эмбеддинга EffNet-Discogs.
    embedding_dim: int = 1280
    max_batch_size: int = 64

    # RabbitMQ (значения по умолчанию для локальной разработки).
    rabbitmq_url: str = "amqp://mood:mood_local_only@127.0.0.1:5672/"
    rabbitmq_queue_limit: int = 20
    inference_delay_s: float = 1.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def model_uri(self) -> str:
        return f"models:/{self.model_name}@{self.model_alias}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
