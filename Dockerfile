FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src

RUN uv sync --frozen --no-dev

RUN useradd --create-home appuser

USER appuser

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "music_mood_search.app:app", "--host", "0.0.0.0", "--port", "8000"]
