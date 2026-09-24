import pytest

from src.config import Settings
from src.inference.contracts import RegistryDocument
from src.inference.errors import ArtifactMissingError, InferenceConfigurationError


def test_settings_load_artifact_paths_and_threshold(monkeypatch):
    monkeypatch.setenv("MODEL_REGISTRY_PATH", "custom/registry.json")
    monkeypatch.setenv("GRAPH_PATH", "custom/graph.pt")
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", "custom/model.pt")
    monkeypatch.setenv("FRAUD_THRESHOLD_OVERRIDE", "0.42")

    settings = Settings.from_env()

    assert settings.model_registry_path.is_absolute()
    assert settings.model_registry_path.name == "registry.json"
    assert settings.graph_path.name == "graph.pt"
    assert settings.model_checkpoint_path.name == "model.pt"
    assert settings.fraud_threshold_override == pytest.approx(0.42)


def test_settings_resolve_relative_registry_paths():
    settings = Settings.from_env()

    assert settings.resolve_path("models/checkpoint.pt") == (
        settings.root_dir / "models/checkpoint.pt"
    )


def test_settings_reject_invalid_threshold(monkeypatch):
    monkeypatch.setenv("FRAUD_THRESHOLD_OVERRIDE", "not-a-number")

    with pytest.raises(ValueError, match="FRAUD_THRESHOLD_OVERRIDE must be a number"):
        Settings.from_env()


def test_registry_contract_requires_production_checkpoint():
    with pytest.raises(ValueError):
        RegistryDocument.model_validate({"production_model": {"name": "GraphSAGE"}})


def test_artifact_missing_error_is_a_configuration_error():
    assert issubclass(ArtifactMissingError, InferenceConfigurationError)
