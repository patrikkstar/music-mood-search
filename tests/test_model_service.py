from types import SimpleNamespace

import numpy as np
import pytest

from music_mood_search.config import Settings
from music_mood_search.services import model as model_module
from music_mood_search.services.health import check_model_health
from music_mood_search.services.model import (
    ModelNotLoadedError,
    ModelService,
    load_model_on_startup,
)


class FakeClient:
    def __init__(self, tracking_uri=None):
        self.tracking_uri = tracking_uri

    def get_model_version_by_alias(self, name, alias):
        assert (name, alias) == ("music-tagger", "champion")
        return SimpleNamespace(version="7", run_id="abc")


class FakePyfunc:
    def __init__(self):
        self.inputs = []

    def predict(self, batch):
        self.inputs.append(batch)
        return [{"genre": [], "instrument": [], "mood": []} for _ in batch]


@pytest.fixture
def settings():
    return Settings(
        mlflow_tracking_uri="http://mlflow:5000",
        model_load_retries=2,
        model_load_retry_delay_s=0,
    )


@pytest.fixture
def fake_mlflow(monkeypatch):
    pyfunc = FakePyfunc()
    loaded_uris = []

    def fake_load_model(uri):
        loaded_uris.append(uri)
        return pyfunc

    monkeypatch.setattr(model_module, "MlflowClient", FakeClient)
    monkeypatch.setattr(model_module.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(model_module, "load_pyfunc_model", fake_load_model)
    return SimpleNamespace(pyfunc=pyfunc, loaded_uris=loaded_uris)


def test_settings_model_uri(settings):
    assert settings.model_uri == "models:/music-tagger@champion"


def test_load_resolves_alias_to_version(settings, fake_mlflow):
    service = ModelService(settings)

    info = service.load()

    assert service.is_loaded
    assert info.version == "7"
    assert info.run_id == "abc"
    # загружается конкретная версия, а не alias
    assert fake_mlflow.loaded_uris == ["models:/music-tagger/7"]
    assert check_model_health(service)["version"] == "music-tagger v7 (@champion)"


def test_predict_converts_to_float32(settings, fake_mlflow):
    service = ModelService(settings)
    service.load()

    result = service.predict([[0.5] * 1280])

    assert result == [{"genre": [], "instrument": [], "mood": []}]
    batch = fake_mlflow.pyfunc.inputs[0]
    assert batch.dtype == np.float32
    assert batch.shape == (1, 1280)


def test_predict_uses_ready_float32_array_without_copy(settings, fake_mlflow):
    service = ModelService(settings)
    service.load()
    batch = np.zeros((2, 1280), dtype=np.float32)

    service.predict(batch)

    assert fake_mlflow.pyfunc.inputs[0] is batch


def test_predict_without_model_raises(settings):
    service = ModelService(settings)

    with pytest.raises(ModelNotLoadedError):
        service.predict([[0.0] * 1280])
    with pytest.raises(ModelNotLoadedError):
        _ = service.info


def test_load_on_startup_success(settings, fake_mlflow):
    service = ModelService(settings)

    assert load_model_on_startup(service, settings, sleep=lambda s: None) is True
    assert service.is_loaded


def test_load_on_startup_retries_and_gives_up(settings, monkeypatch):
    calls = []

    class BrokenClient(FakeClient):
        def get_model_version_by_alias(self, name, alias):
            calls.append(alias)
            raise RuntimeError("registry unavailable")

    monkeypatch.setattr(model_module, "MlflowClient", BrokenClient)
    monkeypatch.setattr(model_module.mlflow, "set_tracking_uri", lambda uri: None)
    sleeps = []

    service = ModelService(settings)
    loaded = load_model_on_startup(service, settings, sleep=sleeps.append)

    assert loaded is False
    assert not service.is_loaded
    assert len(calls) == 2
    assert sleeps == [0]
    assert service.last_error == "registry unavailable"
    health = check_model_health(service)
    assert health["status"] == "error"
    assert health["error"] == "registry unavailable"


def test_model_health_without_service():
    health = check_model_health(None)

    assert health["status"] == "error"
    assert health["error"] == "Model is not loaded"


def test_load_reads_embedding_contract_from_metadata(settings, fake_mlflow):
    contract = {"model": "discogs-effnet-bs64-1", "dim": 1280}
    fake_mlflow.pyfunc.metadata = SimpleNamespace(metadata={"embedding": contract})
    service = ModelService(settings)

    info = service.load()

    assert info.embedding == contract


def test_load_without_metadata_has_no_embedding_contract(settings, fake_mlflow):
    service = ModelService(settings)

    info = service.load()

    assert info.embedding is None
