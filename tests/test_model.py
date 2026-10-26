import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from camie_tagger.devices import DeviceSelection
from camie_tagger.model import CamieTagger, IMAGENET_MEAN, IMAGENET_STD, ModelError, _sigmoid


@pytest.fixture
def tagger():
    return CamieTagger.__new__(CamieTagger)


def test_sigmoid_handles_extreme_finite_logits():
    with np.errstate(over="raise", invalid="raise"):
        actual = _sigmoid(np.array([-1e6, 0, 1e6], dtype=np.float32))
    np.testing.assert_allclose(actual, [0, 0.5, 1])


@pytest.mark.parametrize(
    "size", [(8, 4), (4, 8), (8, 8), (1, 4096), (4096, 1), (49, 37), (37, 49)]
)
def test_preprocess_preserves_aspect_ratio_and_padding(tmp_path, tagger, size):
    image = tmp_path / "image.png"
    Image.new("RGB", size, (255, 0, 0)).save(image)

    actual = tagger.preprocess(image)

    assert actual.shape == (1, 3, 512, 512)
    assert actual.dtype == np.float32
    red = (np.array([1, 0, 0], dtype=np.float32) - IMAGENET_MEAN) / IMAGENET_STD
    width = max(1, int(512 * size[0] / max(size)))
    height = max(1, int(512 * size[1] / max(size)))
    np.testing.assert_allclose(actual[0, :, (512 - height) // 2, (512 - width) // 2], red)
    red_pixels = np.all(np.isclose(actual[0], red[:, None, None]), axis=0)
    assert np.count_nonzero(red_pixels) == width * height
    if size[0] != size[1]:
        padding = (np.array([124, 116, 104], dtype=np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        np.testing.assert_allclose(actual[0, :, 0, 0], padding)


def test_preprocess_applies_exif_orientation(tmp_path, tagger):
    original = Image.new("RGB", (8, 4), "red")
    original.paste("blue", (0, 0, 4, 4))
    exif = Image.Exif()
    exif[274] = 6
    oriented = tmp_path / "oriented.png"
    rotated = tmp_path / "rotated.png"
    original.save(oriented, exif=exif)
    original.transpose(Image.Transpose.ROTATE_270).save(rotated)

    np.testing.assert_array_equal(tagger.preprocess(oriented), tagger.preprocess(rotated))


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_preprocess_accepts_non_rgb_images(tmp_path, tagger, mode):
    image = tmp_path / "image.png"
    Image.new(mode, (3, 5)).save(image)
    assert tagger.preprocess(image).shape == (1, 3, 512, 512)


def configure_session(tagger, logits):
    tagger.idx_to_tag = {"0": "alice", "1": "blue_hair"}
    tagger.tag_to_category = {"alice": "character", "blue_hair": "general"}
    tagger.input_name = "input"
    tagger.session = SimpleNamespace(run=lambda *args: [np.asarray(logits, dtype=np.float32)])
    tagger.preprocess = lambda path: np.zeros((1, 3, 512, 512), dtype=np.float32)


@pytest.mark.parametrize("logits", [[[0]], [[0, 0, 0]], [[np.nan, 0]], [[np.inf, 0]]])
def test_predict_rejects_invalid_model_outputs(tagger, logits):
    configure_session(tagger, logits)
    with pytest.raises(ModelError):
        tagger.predict(None, threshold=1)


def test_predict_uses_inclusive_threshold(tagger):
    configure_session(tagger, [[0, -1]])
    assert tagger.predict(None, threshold=0.5) == {"character": [("alice", 0.5)]}


def test_category_threshold_can_be_lower_than_global_threshold(tagger):
    configure_session(tagger, [[0, 0]])
    assert tagger.predict(None, threshold=0.8, category_thresholds={"character": 0.4}) == {
        "character": [("alice", 0.5)]
    }


def test_model_fingerprint_includes_metadata_and_is_cached(tmp_path, tagger):
    weights = tmp_path / "model.onnx"
    metadata = tmp_path / "metadata.json"
    weights.write_bytes(b"weights")
    metadata.write_bytes(b"metadata")
    tagger._model_paths = (weights, metadata)

    expected = hashlib.sha256(b"weightsmetadata").hexdigest()
    assert tagger.fingerprint == expected
    metadata.unlink()
    assert tagger.fingerprint == expected


@pytest.fixture
def model_files(tmp_path, monkeypatch):
    metadata = {
        "dataset_info": {
            "total_tags": 2,
            "tag_mapping": {
                "idx_to_tag": {"0": "alice", "1": "blue_hair"},
                "tag_to_category": {"alice": "character", "blue_hair": "general"},
            },
        }
    }
    (tmp_path / "model.onnx").write_bytes(b"test weights")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="input")],
        get_outputs=lambda: [SimpleNamespace(name="refined_predictions")],
        get_providers=lambda: ["CPUExecutionProvider"],
    )
    runtime = SimpleNamespace(
        SessionOptions=SimpleNamespace,
        InferenceSession=lambda *args, **kwargs: session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    return tmp_path, metadata_path, session


@pytest.mark.parametrize("damage", ["json", "missing", "noncontiguous", "missing_category", "count"])
def test_invalid_metadata_fails_at_model_load(model_files, damage):
    directory, metadata_path, _session = model_files
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if damage == "missing":
        metadata = {}
    elif damage == "noncontiguous":
        metadata["dataset_info"]["tag_mapping"]["idx_to_tag"] = {"1": "alice", "2": "blue_hair"}
    elif damage == "missing_category":
        metadata["dataset_info"]["tag_mapping"]["tag_to_category"] = {}
    elif damage == "count":
        metadata["dataset_info"]["total_tags"] = 3
    metadata_path.write_text("invalid" if damage == "json" else json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ModelError):
        CamieTagger(directory, DeviceSelection(providers=["CPUExecutionProvider"]))


def test_missing_refined_output_is_a_model_error(model_files):
    directory, _metadata_path, session = model_files
    session.get_outputs = lambda: [SimpleNamespace(name="other_output")]
    with pytest.raises(ModelError, match="refined_predictions"):
        CamieTagger(directory, DeviceSelection(providers=["CPUExecutionProvider"]))


def test_valid_model_metadata_loads(model_files):
    directory, _metadata_path, session = model_files
    loaded = CamieTagger(directory, DeviceSelection(providers=["CPUExecutionProvider"]))
    assert loaded.session is session
    assert loaded.input_name == "input"


def test_missing_onnxruntime_has_clear_model_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ModelError, match="onnxruntime"):
        CamieTagger(tmp_path, DeviceSelection())