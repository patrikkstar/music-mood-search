import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

task_id_var: ContextVar[str | None] = ContextVar("task_id", default=None)

# Дополнительная защита от попадания паролей из DSN в логи
DSN_PASSWORD_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://[^:/@\s]+:)[^@\s/]+@")


def redact(value: str) -> str:
    return DSN_PASSWORD_RE.sub(r"\1***@", value)


@contextmanager
def task_context(task_id: str):
    token = task_id_var.set(task_id)
    try:
        yield
    finally:
        task_id_var.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "service": os.getenv("OTEL_SERVICE_NAME", "music-mood-search"),
            "env": os.getenv("ENVIRONMENT", "development"),
            "logger": record.name,
            "message": redact(record.getMessage()),
            "task_id": getattr(record, "task_id", None) or task_id_var.get(),
            "trace_id": None,
            "span_id": None,
        }

        # Если OpenTelemetry доступен, добавляем текущий
        # trace_id и span_id автоматически
        try:
            from opentelemetry import trace

            span_context = trace.get_current_span().get_span_context()
            if span_context.is_valid:
                log_data["trace_id"] = f"{span_context.trace_id:032x}"
                log_data["span_id"] = f"{span_context.span_id:016x}"
        except ImportError:
            pass

        extra_fields = (
            "event",
            "component",
            "method",
            "path",
            "status_code",
            "duration_ms",
            "total_wait_s",
            "model_uri",
            "model_version",
            "model_alias",
            "run_id",
            "attempt",
            "batch_size",
            "queue",
            "delay_s",
            "sent",
            "rejected",
            "error",
        )

        for field in extra_fields:
            if hasattr(record, field):
                value = getattr(record, field)
                if isinstance(value, str):
                    value = redact(value)
                log_data[field] = value

        if record.exc_info:
            log_data["exception"] = redact(self.formatException(record.exc_info))

        return json.dumps(
            log_data,
            ensure_ascii=False,
            default=str,
        )


def configure_logging(level: str | None = None) -> None:
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    # Единый формат для приложения и сторонних библиотек
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
    logging.captureWarnings(True)

    for name in (
        "music_mood_search",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "faststream",
        "mlflow",
    ):
        library_logger = logging.getLogger(name)
        library_logger.handlers.clear()
        library_logger.propagate = True

    # HTTP health-check не должен засорять INFO-логи
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
