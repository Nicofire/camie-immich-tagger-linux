import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from camie_tagger import sidecar


def test_exiftool_nonzero_exit_is_an_error(monkeypatch):
    monkeypatch.setattr(sidecar, "ensure_exiftool", lambda binary: binary)
    monkeypatch.setattr(
        sidecar.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="Permission denied"
        ),
    )
    with pytest.raises(sidecar.ExiftoolError, match="Permission denied"):
        sidecar.run_exiftool("exiftool", ["-ver"])


def test_invalid_sidecar_json_is_not_treated_as_empty():
    with pytest.raises(sidecar.ExiftoolError):
        sidecar._parse_taglist("not valid JSON")


def test_existing_sidecar_does_not_prove_update_succeeded(tmp_path, monkeypatch):
    image = tmp_path / "image.jpg"
    sidecar.sidecar_path(image).touch()
    monkeypatch.setattr(
        sidecar,
        "run_exiftool",
        lambda *args, **kwargs: (json.dumps([{"TagsList": ["manual/tag"]}]), ""),
    )
    assert sidecar.write_taglist("exiftool", image, ["character/new"]).startswith("ERR:")


@pytest.mark.parametrize("payload", ["[]", "{}", '[{"Error":"bad XMP"}]', '[{"TagsList":42}]'])
def test_unreadable_metadata_is_rejected(payload):
    with pytest.raises(sidecar.ExiftoolError):
        sidecar._parse_taglist(payload)


@pytest.fixture
def fake_exiftool(tmp_path, monkeypatch):
    image = tmp_path / "image with spaces.jpg"
    sidecar.sidecar_path(image).touch()
    store = SimpleNamespace(image=image, tags=["manual/tag"], calls=[])

    def run(binary, arguments, **kwargs):
        store.calls.append(arguments)
        if "-j" in arguments:
            return json.dumps([{"TagsList": store.tags}]), ""
        for argument in arguments:
            if argument.startswith(f"-{sidecar.TAGS_FIELD}+="):
                store.tags.append(argument.split("+=", 1)[1])
            elif argument.startswith(f"-{sidecar.TAGS_FIELD}-="):
                store.tags.remove(argument.split("-=", 1)[1])
        return "1 image files updated", ""

    monkeypatch.setattr(sidecar, "run_exiftool", run)
    return store


def test_merge_is_deduplicated_and_idempotent(fake_exiftool):
    store = fake_exiftool
    assert sidecar.write_taglist("exiftool", store.image, ["character/new"] * 2) == "+1"
    assert sidecar.write_taglist("exiftool", store.image, ["character/new"]) == "NO_CHANGE"
    assert store.tags == ["manual/tag", "character/new"]


def test_update_preserves_unrelated_tags(fake_exiftool):
    store = fake_exiftool
    store.tags.append("character/old")
    assert sidecar.update_taglist(
        "exiftool", store.image, ["character/new"], ["character/old"]
    ) == "+1-1"
    assert store.tags == ["manual/tag", "character/new"]


@pytest.mark.parametrize("argument", ["general/tag\n-overwrite_original", "bad\rpath.xmp"])
def test_argfile_line_breaks_are_rejected(argument):
    with pytest.raises(sidecar.ExiftoolError, match="line breaks"):
        sidecar.run_exiftool("exiftool", [argument])


def test_argfile_is_removed_after_subprocess_error(tmp_path, monkeypatch):
    monkeypatch.setattr(sidecar, "ensure_exiftool", lambda binary: binary)
    paths = []

    def fail(arguments, **kwargs):
        paths.append(Path(arguments[-1]))
        assert paths[0].read_text(encoding="utf-8") == "-ver\n"
        raise sidecar.subprocess.TimeoutExpired(arguments, 1)

    monkeypatch.setattr(sidecar.subprocess, "run", fail)
    with pytest.raises(sidecar.ExiftoolError, match="timed out"):
        sidecar.run_exiftool("exiftool", ["-ver"])
    assert not paths[0].exists()


def test_invalid_directory_scan_is_not_treated_as_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(sidecar, "run_exiftool", lambda *args, **kwargs: ("invalid", ""))
    with pytest.raises(sidecar.ExiftoolError):
        sidecar.scan_directory_taglists("exiftool", tmp_path)


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("exiftool") is None, reason="exiftool is not installed")
def test_real_exiftool_roundtrip_preserves_image_and_manual_tags(tmp_path):
    image = tmp_path / "image with spaces.jpg"
    Image.new("RGB", (8, 4), "red").save(image)
    original_hash = hashlib.sha256(image.read_bytes()).digest()

    assert sidecar.write_taglist("exiftool", image, ["manual/tag"]) == "+1"
    assert sidecar.write_taglist("exiftool", image, ["character/old"]) == "+1"
    assert sidecar.write_taglist("exiftool", image, ["character/old"]) == "NO_CHANGE"
    assert sidecar.update_taglist(
        "exiftool", image, ["character/new"], ["character/old"]
    ) == "+1-1"
    assert set(sidecar.read_taglist("exiftool", image)) == {"manual/tag", "character/new"}
    assert sidecar.scan_directory_taglists("exiftool", tmp_path)[image] == ["manual/tag", "character/new"]
    assert hashlib.sha256(image.read_bytes()).digest() == original_hash


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("exiftool") is None, reason="exiftool is not installed")
def test_real_exiftool_invalid_sidecar_is_not_overwritten(tmp_path):
    image = tmp_path / "image.jpg"
    xmp = sidecar.sidecar_path(image)
    xmp.write_bytes(b"not XMP")

    assert sidecar.write_taglist("exiftool", image, ["character/new"]).startswith("ERR:")
    assert xmp.read_bytes() == b"not XMP"


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("exiftool") is None, reason="exiftool is not installed")
def test_real_exiftool_scan_without_sidecars_is_empty(tmp_path):
    Image.new("RGB", (8, 4), "red").save(tmp_path / "image.jpg")
    assert sidecar.scan_directory_taglists("exiftool", tmp_path) == {}