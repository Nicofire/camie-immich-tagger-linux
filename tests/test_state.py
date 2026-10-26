import json
from pathlib import Path

import pytest

from camie_tagger import state as state_module
from camie_tagger.state import JsonFile, StateStore


def test_legacy_processed_record_has_no_assumed_tag_ownership(state):
    image = Path("legacy.jpg")
    state.replace_processed({str(image): {"tags": 3, "result": "+3"}})
    assert state.is_processed(image)
    assert state.managed_tags(image) == set()


def test_tag_ownership_survives_processing_and_reload(settings, state):
    image = settings.scan_dirs[0] / "image.jpg"
    state.update_managed_tags(image, ["character/model"], source="model")
    assert not state.is_processed(image)
    state.mark_processed(image, 1, "+1", profile={"model_fingerprint": "abc"})
    state.update_managed_tags(image, ["copyright/verified"], source="tier0")
    state.save_processed()

    reloaded = StateStore(settings.data_dir)
    assert reloaded.is_processed(image)
    assert reloaded.managed_tags(image, source="model") == {"character/model"}
    assert reloaded.managed_tags(image, source="tier0") == {"copyright/verified"}
    assert reloaded.processed[str(image)]["profile"]["model_fingerprint"] == "abc"


def test_invalid_ownership_is_not_trusted(state, settings):
    image = settings.scan_dirs[0] / "image.jpg"
    state.replace_processed({str(image): {"managed_tags": "character/manual"}})
    assert state.managed_tags(image) == set()


def test_atomic_save_failure_preserves_previous_document(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text('{"old":true}', encoding="utf-8")
    document = JsonFile(path, {})
    document.replace({"new": True})

    def failed_replace(*args):
        raise OSError("disk full")

    monkeypatch.setattr(state_module.os, "replace", failed_replace)
    with pytest.raises(OSError, match="disk full"):
        document.save()

    assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.glob("*.tmp")) == []