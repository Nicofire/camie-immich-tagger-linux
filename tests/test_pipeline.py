import pytest

from camie_tagger.config import ConfigError
from camie_tagger.pipeline import run_tagging
from camie_tagger.state import StateStore


def test_limited_all_run_preserves_prior_state(settings, state, tagging):
    previous = settings.scan_dirs[0] / "previous.jpg"
    state.mark_processed(previous, 2, "+2")
    state.save_processed()
    image = settings.scan_dirs[0] / "new.jpg"
    image.touch()

    run_tagging(settings, state, mode="all", limit=1)

    reloaded = StateStore(settings.data_dir)
    assert reloaded.is_processed(previous)
    assert reloaded.is_processed(image)


def test_selected_model_profile_and_inference_settings_are_used(settings, state, tagging, tmp_path):
    image = settings.scan_dirs[0] / "image.jpg"
    image.touch()
    settings.model_profile = tmp_path / "other-model.json"
    settings.model_dir = tmp_path / "other-model"
    settings.category_thresholds = {"character": 0.6}

    run_tagging(settings, state, mode="all")

    assert tagging.model_dir == settings.model_dir
    assert tagging.model_profile == settings.model_profile
    assert StateStore(settings.data_dir).processed[str(image)]["profile"] == {
        "model_fingerprint": "test-model",
        "preprocessing_version": "test-preprocessing",
        "model_profile": {"preset": "generic"},
        "thresholds": {"default": 0.5, "character": 0.6},
    }


def test_already_tagged_image_is_persisted(settings, state, tagging):
    image = settings.scan_dirs[0] / "image.jpg"
    image.touch()
    tagging.sidecars[image] = ["character/manual"]

    result = run_tagging(settings, state, mode="all")

    assert result.skipped == 1
    assert tagging.calls == []
    assert StateStore(settings.data_dir).is_processed(image)


def test_dry_run_does_not_write_state_or_sidecars(settings, state, tagging):
    image = settings.scan_dirs[0] / "image.jpg"
    image.touch()
    result = run_tagging(settings, state, mode="all", dry_run=True)

    assert result.tagged == 1
    assert not settings.data_dir.exists()
    assert tagging.sidecars == {}
    assert state.processed == {}


def test_failed_image_is_not_marked_processed(settings, state, tagging):
    image = settings.scan_dirs[0] / "image.jpg"
    image.touch()
    tagging.fail_on.add(image)

    result = run_tagging(settings, state, mode="all")

    assert result.failed == 1
    assert not StateStore(settings.data_dir).is_processed(image)


def test_reprocess_removes_only_model_owned_tags(settings, state, tagging):
    image = settings.scan_dirs[0] / "image.jpg"
    image.touch()
    tagging.sidecars[image] = ["character/old", "character/manual", "copyright/verified"]
    state.update_managed_tags(image, ["character/old"], source="model")
    state.update_managed_tags(image, ["copyright/verified"], source="tier0")
    state.mark_processed(image, 3, "+3")

    result = run_tagging(settings, state, mode="all", reprocess=True, allow_tag_removal=True)

    assert result.tagged == 1
    assert set(tagging.sidecars[image]) == {
        "character/manual", "copyright/verified", "character/new"
    }
    assert state.managed_tags(image, source="model") == {"character/new"}
    assert state.managed_tags(image, source="tier0") == {"copyright/verified"}
    assert state.processed[str(image)]["profile"]["model_fingerprint"] == "test-model"


def test_reprocess_legacy_tags_are_protected_and_second_run_is_idempotent(settings, state, tagging):
    image = settings.scan_dirs[0] / "image.jpg"
    image.touch()
    tagging.sidecars[image] = ["character/legacy"]
    state.mark_processed(image, 1, "+1")

    for _run in range(2):
        run_tagging(settings, state, mode="all", reprocess=True, allow_tag_removal=True)

    assert tagging.sidecars[image] == ["character/legacy", "character/new"]
    assert state.managed_tags(image) == {"character/new"}


def test_reprocess_dry_run_does_not_change_existing_state(settings, state, tagging):
    image = settings.scan_dirs[0] / "image.jpg"
    image.touch()
    tagging.sidecars[image] = ["character/manual"]
    state.mark_processed(image, 1, "+1")
    state.save_processed()
    before = (settings.data_dir / "processed.json").read_bytes()

    run_tagging(settings, state, mode="all", reprocess=True, dry_run=True)

    assert tagging.calls == [image]
    assert tagging.sidecars[image] == ["character/manual"]
    assert (settings.data_dir / "processed.json").read_bytes() == before


def test_reprocess_requires_explicit_nonincremental_mode(settings, state, tagging):
    with pytest.raises(ConfigError):
        run_tagging(settings, state, mode="recent", reprocess=True)


def test_narrowed_scan_preserves_records_from_other_directories(settings, state, tagging):
    previous = settings.scan_dirs[0] / "previous.jpg"
    previous.touch()
    state.mark_processed(previous, 1, "+1")
    narrowed = settings.scan_dirs[0] / "narrowed"
    narrowed.mkdir()
    image = narrowed / "new.jpg"
    image.touch()
    settings.scan_dirs = [narrowed]

    run_tagging(settings, state, mode="all")

    reloaded = StateStore(settings.data_dir)
    assert reloaded.is_processed(previous)
    assert reloaded.is_processed(image)


def test_already_tagged_dry_run_is_completely_read_only(settings, state, tagging):
    image = settings.scan_dirs[0] / "image.jpg"
    image.touch()
    tagging.sidecars[image] = ["character/manual"]
    run_tagging(settings, state, mode="all", dry_run=True)
    assert not settings.data_dir.exists()
    assert state.processed == {}


def test_provenance_only_record_becomes_processed_when_existing_tags_are_skipped(settings, state, tagging):
    image = settings.scan_dirs[0] / "image.jpg"
    image.touch()
    tagging.sidecars[image] = ["character/verified"]
    state.update_managed_tags(image, ["character/verified"], source="tier0")

    run_tagging(settings, state, mode="all")

    assert StateStore(settings.data_dir).is_processed(image)
    assert state.managed_tags(image, source="tier0") == {"character/verified"}


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_image_limit_is_rejected(settings, state, limit):
    with pytest.raises(ConfigError):
        run_tagging(settings, state, mode="all", limit=limit)