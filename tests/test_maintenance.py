from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from camie_tagger import maintenance
from camie_tagger.config import ConfigError
from camie_tagger.immich import ImmichError


@pytest.fixture
def immich_api(settings, monkeypatch):
    settings.immich_url = "https://immich.example"
    settings.immich_api_key = "test-key"
    client = Mock(spec=maintenance.ImmichClient)
    client.tags.return_value = [
        {"id": "keep-character", "value": "character/alice"},
        {"id": "keep-copyright", "value": "copyright/series"},
        {"id": "keep-artist", "value": "artist/creator"},
        {"id": "keep-general", "value": "general/portrait"},
        {"id": "keep-rating", "value": "rating/safe"},
        {"id": "legacy", "value": "alice"},
        {"id": "foreign", "value": "other/tag"},
        {"id": "lookalike", "value": "characteristic/not-a-character"},
        {"id": "legacy-name", "name": "old-tag"},
    ]
    factory = Mock(return_value=client)
    monkeypatch.setattr(maintenance, "ImmichClient", factory)
    return SimpleNamespace(client=client, factory=factory)


@pytest.mark.parametrize("extension", [".jpg", ".jpeg", ".png", ".webp", ".bmp"])
@pytest.mark.parametrize("full_image_name", [False, True])
def test_existing_images_protect_both_sidecar_naming_formats(
    tmp_path, extension, full_image_name
):
    image = tmp_path / f"image{extension}"
    image.write_bytes(b"original image")
    name = image.name if full_image_name else image.stem
    sidecar = tmp_path / f"{name}.xmp"
    sidecar.write_text("metadata", encoding="utf-8")
    assert maintenance._is_orphan(sidecar) is False


@pytest.mark.parametrize("confirm", [False, True])
def test_orphan_cleanup_preserves_valid_files_and_respects_confirmation(settings, tmp_path, confirm):
    directory = settings.scan_dirs[0]
    image = directory / "existing.jpg"
    image.write_bytes(b"original image")
    valid = [directory / "existing.jpg.xmp", directory / "existing.xmp"]
    nested = directory / "nested"
    nested.mkdir()
    orphans = [directory / "missing.jpg.xmp", nested / "missing.xmp"]
    outside = tmp_path / "outside.jpg.xmp"
    unrelated = directory / "notes.txt"
    for path in valid + orphans + [outside, unrelated]:
        path.write_text("unchanged", encoding="utf-8")

    assert maintenance.cleanup_orphans(settings, confirm=confirm) == 0

    assert image.read_bytes() == b"original image"
    for path in valid + [outside, unrelated]:
        assert path.read_text(encoding="utf-8") == "unchanged"
    for path in orphans:
        assert path.exists() is not confirm
        if not confirm:
            assert path.read_text(encoding="utf-8") == "unchanged"
    assert not settings.data_dir.exists()


def test_orphan_cleanup_without_sidecars_is_a_noop(settings, caplog):
    caplog.set_level("INFO")
    assert maintenance.cleanup_orphans(settings, confirm=True) == 0
    assert "Found 0 orphan sidecars" in caplog.text


def test_orphan_cleanup_logs_deletion_error_and_continues(settings, monkeypatch, caplog):
    blocked = settings.scan_dirs[0] / "first.jpg.xmp"
    removable = settings.scan_dirs[0] / "second.jpg.xmp"
    blocked.touch()
    removable.touch()
    original_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("read-only sidecar")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    maintenance.cleanup_orphans(settings, confirm=True)
    assert blocked.exists()
    assert not removable.exists()
    assert "Could not delete" in caplog.text


def test_orphan_cleanup_requires_configured_directories(settings):
    settings.scan_dirs = []
    with pytest.raises(ConfigError, match="No scan directories configured"):
        maintenance.cleanup_orphans(settings, confirm=True)


@pytest.mark.parametrize("confirm", [False, True])
def test_tag_cleanup_keeps_namespaces_and_requires_confirmation(settings, immich_api, confirm):
    assert maintenance.cleanup_tags(settings, confirm=confirm) == 0
    immich_api.factory.assert_called_once_with(settings.immich_url, settings.immich_api_key)
    immich_api.client.tags.assert_called_once_with()
    expected = [call(tag) for tag in ["legacy", "foreign", "lookalike", "legacy-name"]]
    assert immich_api.client.delete_tag.call_args_list == (expected if confirm else [])


def test_tag_cleanup_does_not_delete_when_all_tags_are_in_allowed_namespaces(settings, immich_api):
    immich_api.client.tags.return_value = immich_api.client.tags.return_value[:5]
    assert maintenance.cleanup_tags(settings, confirm=True) == 0
    immich_api.client.delete_tag.assert_not_called()


@pytest.mark.parametrize("missing_field", ["immich_url", "immich_api_key"])
def test_tag_cleanup_requires_credentials_before_creating_client(settings, immich_api, missing_field):
    setattr(settings, missing_field, "")
    with pytest.raises(ConfigError):
        maintenance.cleanup_tags(settings, confirm=True)
    immich_api.factory.assert_not_called()


def test_tag_cleanup_stops_if_listing_tags_fails(settings, immich_api):
    immich_api.client.tags.side_effect = ImmichError("forbidden")
    assert maintenance.cleanup_tags(settings, confirm=True) == 1
    immich_api.client.delete_tag.assert_not_called()


def test_tag_cleanup_logs_failed_deletion_and_continues(settings, immich_api, caplog):
    immich_api.client.delete_tag.side_effect = [ImmichError("failed"), 204, 204, 204]
    maintenance.cleanup_tags(settings, confirm=True)
    assert immich_api.client.delete_tag.call_count == 4
    assert "Could not delete alice" in caplog.text


def test_tag_cleanup_handles_missing_tag_id_without_stopping(settings, immich_api, caplog):
    immich_api.client.tags.return_value = [
        {"value": "legacy-without-id"}, {"id": "valid-id", "value": "legacy"}
    ]
    maintenance.cleanup_tags(settings, confirm=True)
    immich_api.client.delete_tag.assert_called_once_with("valid-id")
    assert "Could not delete legacy-without-id" in caplog.text


def test_statistics_with_no_sidecars(settings, monkeypatch, caplog):
    monkeypatch.setattr(maintenance.sidecar, "scan_taglists", Mock(return_value={}))
    caplog.set_level("INFO")
    assert maintenance.show_stats(settings) == 0
    assert "No sidecars found" in caplog.text


def test_statistics_count_categories_and_top_tags(settings, monkeypatch, caplog):
    directory = settings.scan_dirs[0]
    taglists = {
        directory / "first.jpg": ["character/alice", "copyright/series", "artist/creator"],
        directory / "second.jpg": ["character/alice"],
        directory / "third.jpg": ["copyright/series"],
        directory / "fourth.jpg": ["copyright/original"],
        directory / "fifth.jpg": ["general/portrait"],
        directory / "sixth.jpg": [],
    }
    scan = Mock(return_value=taglists)
    monkeypatch.setattr(maintenance.sidecar, "scan_taglists", scan)
    caplog.set_level("INFO")
    assert maintenance.show_stats(settings) == 0
    scan.assert_called_once_with(settings.exiftool, settings.scan_dirs)
    assert "Sidecars: 6" in caplog.text
    assert "With a character tag: 2 (33.3%)" in caplog.text
    assert "No character but a real copyright (Tier 0 candidates): 1" in caplog.text
    assert "No character and no copyright: 2" in caplog.text
    assert "Top characters:" in caplog.text
    assert "Top copyrights:" in caplog.text
    assert "Top artists:" in caplog.text