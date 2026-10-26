import os
from types import SimpleNamespace

import pytest
import requests

from camie_tagger import config
from camie_tagger.config import Settings
from camie_tagger.devices import DeviceSelection
from camie_tagger.state import StateStore


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DEFAULT_ENV_FILE", tmp_path / "missing.env")
    keys = set(os.environ) | {"CAMIE_THRESHOLD", "CAMIE_CATEGORY_THRESHOLDS"}
    for key in keys:
        if key.startswith(("CAMIE_", "IMMICH_", "SAUCENAO_")):
            monkeypatch.setenv(key, "")
            monkeypatch.delenv(key)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked_request(*args, **kwargs):
        raise AssertionError("Tests must not make network requests")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked_request)


@pytest.fixture
def settings(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    return Settings(
        scan_dirs=[images],
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        saucenao_api_key="test-key",
        tier0_interval=0,
    )


@pytest.fixture
def state(settings):
    return StateStore(settings.data_dir)


@pytest.fixture
def tagging(monkeypatch):
    from camie_tagger import pipeline

    context = SimpleNamespace(
        sidecars={},
        prediction={"character": [("new", 0.9)]},
        calls=[],
        fail_on=set(),
    )

    class FakeTagger:
        fingerprint = "test-model"

        def __init__(self, *args):
            pass

        def predict(self, image, threshold=0.5, category_thresholds=None):
            context.calls.append(image)
            return context.prediction

    def update(binary, image, add, remove):
        if image in context.fail_on:
            return "ERR:disk full"
        existing = context.sidecars.get(image, [])
        context.sidecars[image] = list(
            dict.fromkeys([tag for tag in existing if tag not in remove] + add)
        )
        return f"+{len(add)}-{len(remove)}"

    monkeypatch.setattr(pipeline, "CamieTagger", FakeTagger)
    monkeypatch.setattr(
        pipeline, "select_device", lambda *args: DeviceSelection(providers=["CPUExecutionProvider"])
    )
    monkeypatch.setattr(
        pipeline.sidecar,
        "scan_taglists",
        lambda binary, dirs: {
            image: tags
            for image, tags in context.sidecars.items()
            if any(image.is_relative_to(directory) for directory in dirs)
        },
    )
    monkeypatch.setattr(
        pipeline.sidecar, "read_taglist", lambda binary, image: context.sidecars.get(image, []).copy()
    )
    monkeypatch.setattr(
        pipeline.sidecar, "write_taglist", lambda binary, image, tags: update(binary, image, tags, [])
    )
    monkeypatch.setattr(pipeline.sidecar, "update_taglist", update)
    return context