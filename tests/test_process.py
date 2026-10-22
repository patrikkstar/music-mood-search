from music_mood_search.app import app
from tests.fakes import PREDICTION, FakeModelService

EMBEDDING = [0.1] * 1280


async def test_process_returns_tags_and_model_version(client):
    service = FakeModelService()
    app.state.model_service = service

    response = await client.post(
        "/api/v1/process",
        json={"embeddings": [EMBEDDING, EMBEDDING]},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model"]["name"] == "music-tagger"
    assert data["model"]["version"] == "3"
    assert data["model"]["alias"] == "champion"
    assert data["predictions"] == [PREDICTION, PREDICTION]
    assert data["inference_ms"] >= 0
    assert len(service.calls) == 1


async def test_process_model_not_loaded(client):
    app.state.model_service = FakeModelService(loaded=False)

    response = await client.post("/api/v1/process", json={"embeddings": [EMBEDDING]})

    assert response.status_code == 503


async def test_process_without_model_service(client):
    if hasattr(app.state, "model_service"):
        del app.state.model_service

    response = await client.post("/api/v1/process", json={"embeddings": [EMBEDDING]})

    assert response.status_code == 503


async def test_process_wrong_embedding_size(client):
    app.state.model_service = FakeModelService()

    response = await client.post(
        "/api/v1/process",
        json={"embeddings": [EMBEDDING, [0.1] * 10]},
    )

    assert response.status_code == 422
    assert "wrong items: [1]" in response.json()["detail"]


async def test_process_empty_batch(client):
    app.state.model_service = FakeModelService()

    response = await client.post("/api/v1/process", json={"embeddings": []})

    assert response.status_code == 422


async def test_process_batch_too_large(client):
    app.state.model_service = FakeModelService()

    response = await client.post(
        "/api/v1/process",
        json={"embeddings": [EMBEDDING] * 65},
    )

    assert response.status_code == 422


async def test_get_model_info(client):
    app.state.model_service = FakeModelService()

    response = await client.get("/api/v1/model")

    assert response.status_code == 200
    assert response.json() == {
        "name": "music-tagger",
        "alias": "champion",
        "version": "3",
        "run_id": "run-123",
        "uri": "models:/music-tagger/3",
    }


async def test_get_model_info_not_loaded(client):
    app.state.model_service = FakeModelService(loaded=False)

    response = await client.get("/api/v1/model")

    assert response.status_code == 503
