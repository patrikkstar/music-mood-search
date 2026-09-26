# Music Mood Search

Асинхронный веб-сервис на FastAPI, разработанный в рамках курса MLOps.

В дальнейшем проект будет использоваться как основа для сервиса **Music Mood Search** — системы поиска музыкальных треков по текстовому описанию настроения, жанра и инструментов.

На текущем этапе реализована инфраструктурная часть проекта:

- асинхронное FastAPI-приложение;
- подключение к PostgreSQL;
- health-check эндпоинты;
- тестирование и измерение покрытия;
- статический анализ и форматирование;
- Docker и Docker Compose;
- pre-commit hooks;
- CI pipeline;
- CD pipeline;
- версионирование Docker-образов;
- публикация образов в GitHub Container Registry;
- структурированное JSON-логирование.

## Структура проекта

```text
music-mood-search/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── src/
│   └── music_mood_search/
│       ├── api/
│       │   ├── __init__.py
│       │   └── health.py
│       │
│       ├── services/
│       │   ├── __init__.py
│       │   └── health.py
│       │
│       ├── __init__.py
│       ├── app.py
│       ├── config.py
│       ├── db.py
│       ├── logging_config.py
│       └── schemas.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_health.py
│
├── .dockerignore
├── .env.example
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

## Используемые технологии

- Python 3.12
- FastAPI
- Uvicorn
- PostgreSQL 17
- asyncpg
- Pydantic Settings
- uv
- pytest
- pytest-asyncio
- pytest-cov
- HTTPX
- Ruff
- pre-commit
- Docker
- Docker Compose
- GitHub Actions
- GitHub Container Registry

## API

В приложении реализованы три основных эндпоинта.

### `GET /healthz`

Быстрая проверка того, что само приложение работает.


Эндпоинт намеренно не находится внутри `/api/v1`, так как используется не пользователями API, а инфраструктурой: Docker, оркестраторами и системами мониторинга.

### `GET /api/v1/version`

Возвращает текущую версию приложения.


Версия не хранится отдельно внутри endpoint. Она получается из metadata установленного Python-пакета, а источником версии является `pyproject.toml`.

### `GET /api/v1/health`

Выполняет end-to-end health-check сторонних компонентов приложения.

На текущем этапе проверяется PostgreSQL.

Эндпоинт возвращает:

- состояние компонента;
- версию PostgreSQL;
- время ответа;
- информацию об ошибке при недоступности компонента.

Если PostgreSQL недоступен, endpoint возвращает HTTP-код `503 Service Unavailable`.

## Конфигурация

Настройки приложения загружаются через `pydantic-settings`.

Для локального запуска необходимо создать `.env` на основе `.env.example`.

При работе приложения внутри Docker Compose используется внутренний адрес PostgreSQL:

```text
postgresql://postgres:postgres@db:5432/postgres
```



## Ruff

Для статического анализа и форматирования используется Ruff.

Проверка кода:

```bash
uv run ruff check .
```

Проверка форматирования:

```bash
uv run ruff format --check .
```

Автоматическое форматирование:

```bash
uv run ruff format .
```

В конфигурации Ruff используются правила для:

- стандартных ошибок Python;
- ошибок оформления;
- сортировки импортов;
- поиска распространённых потенциальных ошибок;
- использования современного синтаксиса Python.

## Тестирование

Запуск тестов:

```bash
uv run pytest -v
```

Запуск тестов с измерением покрытия:

```bash
uv run pytest --cov=music_mood_search --cov-report=term-missing
```

Проверяются следующие сценарии:

- успешный `/healthz`;
- успешный `/api/v1/version`;
- успешный `/api/v1/health`;
- ситуация, когда PostgreSQL недоступен.

Минимально допустимое покрытие:

```text
80%
```

Если coverage становится ниже 80%, проверка завершается ошибкой.

## Pre-commit

Для автоматических проверок перед каждым коммитом используется `pre-commit`.

Установка Git hook:

```bash
uv run pre-commit install
```

Ручной запуск всех проверок:

```bash
uv run pre-commit run --all-files
```

В pre-commit настроены:

- удаление trailing whitespace;
- проверка конца файлов;
- проверка YAML;
- проверка TOML;
- проверка больших файлов;
- проверка merge conflicts;
- проверка конфликтов регистра имён файлов;
- нормализация line endings;
- Ruff lint;
- Ruff format.

При выполнении:

```bash
git commit
```

pre-commit запускается автоматически.

## Docker

Сборка Docker-образа:

```bash
docker build -t music-mood-search:local .
```

Dockerfile использует:

- `python:3.12-slim`;
- установку зависимостей через `uv`;
- фиксированные зависимости из `uv.lock`;
- отдельные Docker layers для зависимостей и исходного кода;
- только production-зависимости;
- непривилегированного пользователя `appuser`.

Проверить пользователя контейнера:

Приложение внутри контейнера запускается не от `root`.

## Docker Compose

Для запуска всего приложения используется Docker Compose.

Запустить приложение и PostgreSQL:

```bash
docker compose up --build -d
```

Проверить состояние контейнеров:

```bash
docker compose ps
```

Оба сервиса должны иметь статус `healthy`:

```text
app    healthy
db     healthy
```


Для подключения к PostgreSQL с хостовой Windows-машины используется:

```text
localhost:55432
```

Docker Compose содержит:

- сервис приложения;
- сервис PostgreSQL;
- отдельную Docker network;
- persistent volume для PostgreSQL;
- healthcheck приложения;
- healthcheck базы данных;
- ограничения памяти и CPU;
- ротацию логов;
- restart policy.

Просмотр логов приложения:

```bash
docker compose logs app
```

Просмотр логов в реальном времени:

```bash
docker compose logs -f app
```

Остановка сервисов:

```bash
docker compose down
```

## Логирование

В приложении реализовано структурированное JSON-логирование.


Логируются:

- запуск приложения;
- создание пула соединений с PostgreSQL;
- HTTP-метод;
- URL запроса;
- HTTP status code;
- время выполнения запроса;
- завершение приложения;
- закрытие connection pool.

## CI

Continuous Integration реализован через GitHub Actions.

CI запускается:

- при push в `main`;
- при pull request в `main`.

Pipeline выполняет:

```text
Checkout
   ↓
Установка uv и Python
   ↓
Установка зависимостей
   ↓
Ruff lint
   ↓
Ruff format check
   ↓
pytest
   ↓
coverage check
```

Pipeline завершится ошибкой, если:

- Ruff обнаружит ошибку;
- код неправильно отформатирован;
- хотя бы один тест не пройдёт;
- coverage станет ниже 80%.

## CD

Continuous Delivery также реализован через GitHub Actions.

CD запускается только при push Git-тега формата:

```text
v*.*.*
```

Перед публикацией Docker-образа pipeline сравнивает версию Git-тега с версией приложения из `pyproject.toml`.

Например:

```text
pyproject.toml: 0.1.1
Git tag:        v0.1.1
```

Если версии не совпадают, CD завершается ошибкой.

Такой подход предотвращает случайную публикацию образа с неправильной версией.

## GitHub Container Registry

Docker-образы публикуются в:

```text
ghcr.io/patrikkstar/music-mood-search
```

Для релиза `v0.1.1` создаются следующие теги:

```text
0.1.1
0.1
latest
sha-1dc2e5d
```

Назначение тегов:

- `0.1.1` — конкретная версия приложения;
- `0.1` — последняя patch-версия ветки `0.1`;
- `latest` — последний опубликованный релиз;
- `sha-*` — образ, соответствующий конкретному Git-коммиту.

Скачать конкретную версию:

```bash
docker pull ghcr.io/patrikkstar/music-mood-search:0.1.1
```

## Git workflow

Полный процесс разработки выглядит следующим образом:

```text
изменение кода
      ↓
git commit
      ↓
pre-commit
      ↓
git push
      ↓
CI
      ↓
создание version tag
      ↓
CD
      ↓
GitHub Container Registry
```

Обычный push в `main` запускает CI, но не создаёт новый Docker-релиз.

Для выпуска новой версии необходимо явно создать Git-тег:

```bash
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

После этого автоматически запускается CD и публикуется новый Docker-образ.

## Версии

Текущие релизы:

```text
v0.1.0 — первая рабочая инфраструктурная версия
v0.1.1 — добавлено структурированное логирование
```

## Развитие Music Mood Search

Текущая версия проекта содержит инфраструктурную основу будущего ML-сервиса.

Планируемая функциональность Music Mood Search:

1. Извлечение музыкальных эмбеддингов с помощью Essentia EffNet-Discogs.
2. Обучение лёгких классификаторов для тегов:
   - mood;
   - genre;
   - instrument.
3. Использование MTG-Jamendo Dataset для обучения классификатора.
4. Использование Song Describer Dataset для построения поисковой базы.
5. Объединение предсказанных музыкальных тегов с текстовыми описаниями треков.
6. Индексация треков в Chroma.
7. Поиск треков по текстовому описанию пользователя.
8. Возможность загрузки собственного трека и добавления его в поисковую коллекцию.

ML-часть будет отделена от web-serving слоя, чтобы тяжёлые зависимости для обучения моделей не устанавливались в production-контейнер API.
