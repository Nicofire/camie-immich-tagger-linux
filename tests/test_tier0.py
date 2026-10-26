import pytest

from camie_tagger import tier0
from camie_tagger.config import ConfigError
from camie_tagger.state import StateStore


def test_failed_backfill_write_remains_retryable(settings, state, monkeypatch):
    image = settings.scan_dirs[0] / "image.jpg"
    image.touch()
    state.enqueue_tier0([image])
    monkeypatch.setattr(
        tier0,
        "_search_saucenao",
        lambda *args: {
            "header": {"status": 0, "long_remaining": 3},
            "results": [
                {"header": {"similarity": "99"}, "data": {"characters": "new"}}
            ],
        },
    )
    monkeypatch.setattr(tier0.sidecar, "write_taglist", lambda *args: "ERR:disk full")

    result = tier0.run_tier0(settings, state)

    assert result.failed == 1
    assert result.hits == 1
    assert str(image) not in state.tier0_progress


def test_verification_is_additive_by_default(settings, state, monkeypatch):
    image = settings.scan_dirs[0] / "image.jpg"
    existing = ["character/manual", "copyright/manual", "general/blue_hair"]
    monkeypatch.setattr(tier0.sidecar, "read_taglist", lambda *args: existing)
    writes = []
    monkeypatch.setattr(
        tier0.sidecar,
        "update_taglist",
        lambda binary, path, add, remove: writes.append((add, remove)) or "+1-0",
    )
    match = tier0.Match(99, "Danbooru", 123, ["character/new"], canonical=True)

    tier0._apply_verification(
        settings, state, tier0.Tier0Result(), image, match, True, 95, True
    )

    assert writes == [(["character/new"], [])]


def test_verification_read_failure_is_retryable(settings, state, monkeypatch):
    image = settings.scan_dirs[0] / "image.jpg"

    def unreadable(*args):
        raise tier0.sidecar.ExiftoolError("invalid sidecar")

    monkeypatch.setattr(tier0.sidecar, "read_taglist", unreadable)
    result = tier0.Tier0Result()
    match = tier0.Match(99, "Danbooru", 123, ["character/new"], canonical=True)
    tier0._apply_verification(settings, state, result, image, match, True, 95, True)

    assert result.failed == 1
    assert str(image) not in state.tier0_progress


def test_explicit_verification_removal_only_affects_owned_tags(settings, state, monkeypatch):
    image = settings.scan_dirs[0] / "image.jpg"
    state.update_managed_tags(image, ["character/old"], source="model")
    existing = ["character/old", "character/manual", "general/blue_hair"]
    monkeypatch.setattr(tier0.sidecar, "read_taglist", lambda *args: existing)
    writes = []
    monkeypatch.setattr(
        tier0.sidecar,
        "update_taglist",
        lambda binary, path, add, remove: writes.append((add, remove)) or "+1-1",
    )
    match = tier0.Match(99, "Danbooru", 123, ["character/new"], canonical=True)

    tier0._apply_verification(
        settings, state, tier0.Tier0Result(), image, match, True, 95, True,
        allow_tag_removal=True,
    )

    assert writes == [(["character/new"], ["character/old"])]
    assert state.managed_tags(image, source="tier0") == {"character/new"}


def test_verify_dry_run_does_not_persist_progress(settings, state, monkeypatch):
    image = settings.scan_dirs[0] / "image.jpg"
    image.touch()
    state.enqueue_tier0([image])
    monkeypatch.setattr(
        tier0, "_search_saucenao",
        lambda *args: {"header": {"status": 0, "long_remaining": 3}, "results": []},
    )
    tier0.run_tier0(settings, state, verify=True, confirm=False)
    assert not settings.data_dir.exists()


@pytest.mark.parametrize("similarity", ["NaN", "Infinity", "101", "-1", "invalid"])
def test_invalid_match_similarities_are_rejected(similarity):
    assert tier0._best_match(
        [{"header": {"similarity": similarity}, "data": {"characters": "alice"}}], 88
    ) is None


def test_nearby_danbooru_match_is_preferred():
    results = [
        {"header": {"similarity": "99"}, "data": {"characters": "alice"}},
        {"header": {"similarity": "97"}, "data": {"danbooru_id": 123}},
    ]
    assert tier0._best_match(results, 88).post_id == 123


@pytest.mark.parametrize("canonical,similarity", [(False, 99), (True, 94)])
def test_weak_match_never_removes_owned_tags(settings, state, monkeypatch, canonical, similarity):
    image = settings.scan_dirs[0] / "image.jpg"
    state.update_managed_tags(image, ["character/old"])
    monkeypatch.setattr(tier0.sidecar, "read_taglist", lambda *args: ["character/old"])
    writes = []
    monkeypatch.setattr(
        tier0.sidecar, "update_taglist",
        lambda binary, path, add, remove: writes.append(remove) or "+1-0",
    )
    match = tier0.Match(similarity, "source", None, ["character/new"], canonical=canonical)
    tier0._apply_verification(
        settings, state, tier0.Tier0Result(), image, match, True, 95, True,
        allow_tag_removal=True,
    )
    assert writes == [[]]


def test_successful_backfill_records_only_automatic_tag_origins(settings, state, monkeypatch):
    image = settings.scan_dirs[0] / "image.jpg"
    image.touch()
    state.enqueue_tier0([image])
    state.update_managed_tags(image, ["copyright/series"], source="model")
    existing = ["character/manual", "copyright/series"]
    match_tags = ["character/manual", "copyright/series", "character/new"]
    monkeypatch.setattr(tier0.sidecar, "read_taglist", lambda *args: existing.copy())
    monkeypatch.setattr(tier0.sidecar, "write_taglist", lambda *args: "+1")
    monkeypatch.setattr(tier0, "_fetch_danbooru_tags", lambda *args: match_tags)
    monkeypatch.setattr(
        tier0, "_search_saucenao",
        lambda *args: {
            "header": {"status": 0, "long_remaining": 3},
            "results": [{"header": {"similarity": "99"}, "data": {"danbooru_id": 123}}],
        },
    )

    result = tier0.run_tier0(settings, state)

    assert result.failed == 0
    assert result.hits == 1
    reloaded = StateStore(settings.data_dir)
    assert reloaded.managed_tags(image, source="tier0") == {"character/new", "copyright/series"}
    assert str(image) in reloaded.tier0_progress


def test_failed_verification_does_not_record_correction(settings, state, monkeypatch):
    image = settings.scan_dirs[0] / "image.jpg"
    state.update_managed_tags(image, ["character/old"])
    monkeypatch.setattr(tier0.sidecar, "read_taglist", lambda *args: ["character/old"])
    monkeypatch.setattr(tier0.sidecar, "update_taglist", lambda *args: "ERR:disk full")
    result = tier0.Tier0Result()
    tier0._apply_verification(
        settings, state, result, image,
        tier0.Match(99, "Danbooru", 123, ["character/new"], canonical=True),
        True, 95, True, allow_tag_removal=True,
    )
    assert result.failed == 1
    assert result.corrected == 0
    assert state.managed_tags(image) == {"character/old"}
    assert str(image) not in state.tier0_progress


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_similarity": float("nan")},
        {"min_similarity": -1},
        {"replace_min_similarity": float("nan")},
        {"replace_min_similarity": 101},
        {"limit": 0},
        {"limit": -1},
    ],
)
def test_invalid_search_limits_are_rejected_before_processing(settings, state, overrides):
    with pytest.raises(ConfigError):
        tier0.run_tier0(settings, state, **overrides)
    assert not settings.data_dir.exists()