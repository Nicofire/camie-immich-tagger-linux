import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageOps

from camie_tagger.devices import DeviceSelection
from camie_tagger.model import CamieTagger, OnnxTagger, load_labels
from camie_tagger.model_profile import (
    ModelError, ModelProfile, find_model_files, load_model_profile,
)

CPU = DeviceSelection(providers=["CPUExecutionProvider"])
GENERIC = {
    "preset": "generic",
    "model": "model.onnx",
    "labels": "labels.json",
    "layout": "NCHW",
    "color_order": "RGB",
    "resize": "stretch",
    "scale": 255,
    "mean": [0, 0, 0],
    "std": [1, 1, 1],
    "activation": "none",
}


def write_profile(directory, *, filename="profile.json", **overrides):
    path = directory / filename
    path.write_text(json.dumps({**GENERIC, **overrides}), encoding="utf-8")
    return path


@pytest.fixture
def model_files(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"weights")
    (tmp_path / "labels.json").write_text(
        '[{"name":"alice","category":"character"},{"name":"blue_hair","category":"general"}]',
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def runtime(monkeypatch):
    image_input = SimpleNamespace(name="image", shape=[1, 3, 4, 6], type="tensor(float)")
    output = SimpleNamespace(name="scores", shape=[1, 2], type="tensor(float)")
    context = SimpleNamespace(
        inputs=[image_input], outputs=[output],
        scores=np.array([[0.8, 0.2]], dtype=np.float32), calls=[],
    )

    def run(names, feed):
        context.calls.append((names, feed))
        return [context.scores]

    context.session = SimpleNamespace(
        get_inputs=lambda: context.inputs,
        get_outputs=lambda: context.outputs,
        get_providers=lambda: ["CPUExecutionProvider"],
        run=run,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        SessionOptions=SimpleNamespace,
        InferenceSession=lambda *args, **kwargs: context.session,
    ))
    return context


def test_legacy_camie_defaults_and_import_remain_available():
    assert CamieTagger is OnnxTagger
    assert load_model_profile() == ModelProfile()
    assert load_model_profile().activation == "sigmoid"
    assert load_model_profile().size == (512, 512)


@pytest.mark.parametrize("preset", ["camie", "wd", "generic"])
def test_documented_profile_templates_are_valid(preset):
    directory = Path(__file__).resolve().parents[1] / "docs" / "model-profiles"
    profile = load_model_profile(directory / f"{preset}.json")
    assert profile.preset == preset
    assert profile.model_path.parent == directory
    assert profile.labels_path.parent == directory


def test_wd_preset_uses_raw_bgr_and_probabilities(tmp_path):
    path = tmp_path / "wd.json"
    path.write_text(
        '{"preset":"wd","model":"model.onnx","labels":"selected_tags.csv"}', encoding="utf-8"
    )
    profile = load_model_profile(path)
    assert (profile.layout, profile.color_order, profile.scale) == ("NHWC", "BGR", 1)
    assert (profile.resize, profile.interpolation) == ("pad_square", "bicubic")
    assert profile.activation == "none"
    assert profile.output_name is None
    assert profile.size is None
    assert profile.alpha_background == (255, 255, 255)
    assert profile.category_map["4"] == "character"
    assert profile.model_path == tmp_path / "model.onnx"
    assert profile.labels_path == tmp_path / "selected_tags.csv"


def test_absolute_profile_paths_and_external_data_are_preserved(tmp_path):
    weights = tmp_path / "weights"
    path = write_profile(
        tmp_path, model=str(tmp_path / "absolute.onnx"),
        labels=str(tmp_path / "absolute.txt"), external_data=[str(weights)],
    )
    profile = load_model_profile(path)
    assert profile.model_path == tmp_path / "absolute.onnx"
    assert profile.labels_path == tmp_path / "absolute.txt"
    assert profile.external_data == (weights,)


@pytest.mark.parametrize("key, value", [
    ("preset", "unknown"), ("preset", None), ("model", ""), ("labels", 42),
    ("layout", "auto"), ("color_order", "CMYK"), ("resize", "guess"),
    ("interpolation", "invalid"), ("activation", "auto"), ("input_name", ""),
    ("output_name", []), ("size", [0, 4]), ("size", [4]), ("size", [True, 4]),
    ("size", [4.5, 6]), ("scale", 0), ("scale", -1), ("scale", True),
    ("scale", float("nan")), ("scale", float("inf")), ("scale", "255"),
    ("mean", [0, 0]), ("mean", [0, False, 0]), ("mean", [0, float("inf"), 0]),
    ("std", [1, 0, 1]), ("std", [1, -1, 1]), ("std", [1, float("nan"), 1]),
    ("pad_color", [255, 256, 255]), ("pad_color", [True, 0, 0]),
    ("alpha_background", [0]), ("default_category", "species"),
    ("category_map", []), ("category_map", {"animal": "species"}),
    ("category_map", {"": "general"}), ("external_data", "weights"),
    ("external_data", [None]), ("typo", True),
])
def test_invalid_profile_options_fail_explicitly(tmp_path, key, value):
    path = write_profile(tmp_path, **{key: value})
    with pytest.raises(ModelError, match="Invalid model profile"):
        load_model_profile(path)


@pytest.mark.parametrize("contents", ["invalid", "[]", "null", '{"preset":"generic"}'])
def test_malformed_and_incomplete_profiles_fail(tmp_path, contents):
    path = tmp_path / "profile.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ModelError):
        load_model_profile(path)


def test_missing_profile_is_not_replaced_with_camie_defaults(tmp_path):
    with pytest.raises(ModelError, match="Invalid model profile"):
        load_model_profile(tmp_path / "missing.json")


def test_generic_preprocessing_must_be_explicit(tmp_path):
    data = {**GENERIC}
    del data["activation"]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ModelError, match="activation"):
        load_model_profile(path)


def test_profile_paths_do_not_depend_on_model_dir(model_files):
    path = write_profile(model_files)
    profile = load_model_profile(path)
    assert find_model_files(model_files / "nonexistent", profile) == (
        model_files / "model.onnx", model_files / "labels.json"
    )


@pytest.mark.parametrize("missing", ["model.onnx", "labels.json", "weights.data"])
def test_missing_profile_files_are_rejected(model_files, missing):
    extra = model_files / "weights.data"
    extra.write_bytes(b"data")
    path = write_profile(model_files, external_data=["weights.data"])
    (model_files / missing).unlink()
    with pytest.raises(ModelError, match="file not found"):
        find_model_files(model_files, load_model_profile(path))


@pytest.mark.parametrize("extra", ["second.onnx", "other-metadata.json"])
def test_legacy_discovery_rejects_ambiguous_files(model_files, extra):
    (model_files / "metadata.json").write_text('["tag"]', encoding="utf-8")
    (model_files / extra).write_bytes(b"ambiguous")
    with pytest.raises(ModelError, match="Multiple.*--model-profile"):
        find_model_files(model_files)


@pytest.mark.parametrize("filename, contents, expected", [
    ("labels.txt", "\ufeffalice\nblue_hair\n", {"0": "alice", "1": "blue_hair"}),
    ("labels.json", '["alice", "blue_hair"]', {"0": "alice", "1": "blue_hair"}),
    ("labels.json", '{"1":"blue_hair","0":"alice"}', {"0": "alice", "1": "blue_hair"}),
    ("labels.json", '{"id2label":{"1":"blue_hair","0":"alice"}}', {"0": "alice", "1": "blue_hair"}),
    ("labels.csv", 'id,name\n99,"alice, example"\n2,blue_hair\n', {"0": "alice, example", "1": "blue_hair"}),
])
def test_label_formats_preserve_output_order(tmp_path, filename, contents, expected):
    path = tmp_path / filename
    path.write_text(contents, encoding="utf-8")
    names, categories = load_labels(path, ModelProfile())
    assert names == expected
    assert categories == {name: "general" for name in expected.values()}


def test_json_categories_can_be_remapped(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(
        '[{"name":"cat","category":"species"},{"name":"alice","category":4}]', encoding="utf-8"
    )
    profile = replace(ModelProfile(), category_map={"species": "general", "4": "character"})
    assert load_labels(path, profile) == (
        {"0": "cat", "1": "alice"}, {"cat": "general", "alice": "character"}
    )


def test_wd_csv_uses_row_order_not_danbooru_ids(tmp_path):
    path = tmp_path / "selected_tags.csv"
    path.write_text(
        "\ufefftag_id,name,category,count\r\n"
        "999,general,9,10\r\n4,blue_hair,0,20\r\n1,alice,4,30\r\n"
        "6,illustrator,1,1\r\n2,series,3,1\r\n3,metadata,5,1\r\n",
        encoding="utf-8",
    )
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(
        '{"preset":"wd","model":"model.onnx","labels":"selected_tags.csv"}', encoding="utf-8"
    )
    names, categories = load_labels(path, load_model_profile(profile_file))
    assert list(names.values()) == ["general", "blue_hair", "alice", "illustrator", "series", "metadata"]
    assert list(categories.values()) == ["rating", "general", "character", "artist", "copyright", "meta"]


@pytest.mark.parametrize("filename, contents", [
    ("labels.txt", ""), ("labels.txt", "cat\n\nbird\n"), ("labels.txt", "cat\ncat\n"),
    ("labels.json", "{}"), ("labels.json", "[]"), ("labels.json", "null"),
    ("labels.json", '{"1":"cat"}'), ("labels.json", '["cat",1]'),
    ("labels.json", '[{"name":"cat","category":"unknown"}]'),
    ("labels.json", '[{"name":"cat","category":null}]'),
    ("labels.json", '[{"name":"cat","category":true}]'),
    ("labels.json", '[{"name":"cat"},{"name":" cat "}]'),
    ("labels.json", '{"idx_to_tag":{"0":"cat"},"tag_to_category":[]}'),
    ("labels.json", '{"idx_to_tag":{"0":"cat"},"tag_to_category":null}'),
    ("labels.json", '{"dataset_info":null}'),
    ("labels.json", '{"dataset_info":{"tag_mapping":{"idx_to_tag":{"0":"cat"},"tag_to_category":null}}}'),
    ("labels.json", '{"dataset_info":{"total_tags":null}}'),
    ("labels.csv", "id,tag\n1,cat\n"),
    ("labels.csv", "name,category\ncat,invalid\n"),
    ("labels.csv", "name,category\ncat,general,extra\n"),
    ("labels.csv", "name,name\ncat,bird\n"),
    ("labels.xml", "<tag>cat</tag>"),
])
def test_bad_labels_are_not_silently_accepted(tmp_path, filename, contents):
    path = tmp_path / filename
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ModelError, match="Invalid model labels"):
        load_labels(path, ModelProfile())


def test_wd_requires_category_column(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("name\nalice\n", encoding="utf-8")
    with pytest.raises(ModelError, match="category"):
        load_labels(labels, replace(ModelProfile(), preset="wd"))


@pytest.mark.parametrize("layout, shape", [
    ("NCHW", [1, 3, 4, 6]), ("NHWC", [1, 4, 6, 3]),
    ("CHW", [3, 4, 6]), ("HWC", [4, 6, 3]),
])
def test_image_layout_and_size_are_taken_from_profile_and_graph(model_files, runtime, layout, shape):
    runtime.inputs[0].shape = shape
    path = write_profile(model_files, layout=layout)
    tagger = OnnxTagger(model_files, CPU, path)
    image = model_files / "red.png"
    Image.new("RGB", (8, 2), "red").save(image)
    tensor = tagger.preprocess(image)
    assert tagger.image_size == (6, 4)
    assert list(tensor.shape) == shape
    assert tensor.flags.c_contiguous
    axis = layout.index("C")
    np.testing.assert_array_equal(np.take(tensor, 0, axis=axis), 1)
    np.testing.assert_array_equal(np.take(tensor, 1, axis=axis), 0)


@pytest.mark.parametrize("dtype, expected", [
    ("tensor(float)", np.float32), ("tensor(float16)", np.float16),
    ("tensor(double)", np.float64), ("tensor(uint8)", np.uint8), ("tensor(int8)", np.int8),
])
def test_input_dtype_is_taken_from_onnx(model_files, runtime, dtype, expected):
    runtime.inputs[0].type = dtype
    path = write_profile(
        model_files, scale=1, mean=[128, 128, 128] if expected == np.int8 else [0, 0, 0]
    )
    tagger = OnnxTagger(model_files, CPU, path)
    image = model_files / "image.png"
    Image.new("RGB", (6, 4), (255, 0, 20)).save(image)
    tensor = tagger.preprocess(image)
    assert tensor.dtype == expected
    np.testing.assert_array_equal(
        tensor[0, :, 0, 0], [127, -128, -108] if expected == np.int8 else [255, 0, 20]
    )


def test_grayscale_input(model_files, runtime):
    runtime.inputs[0].shape = [1, 1, 4, 6]
    path = write_profile(model_files, color_order="GRAY", mean=[0], std=[1])
    tagger = OnnxTagger(model_files, CPU, path)
    image = model_files / "gray.png"
    Image.new("L", (6, 4), 128).save(image)
    tensor = tagger.preprocess(image)
    assert tensor.shape == (1, 1, 4, 6)
    np.testing.assert_allclose(tensor, 128 / 255)


def test_dynamic_dimensions_require_explicit_size(model_files, runtime):
    runtime.inputs[0].shape = ["batch", 3, "height", None]
    with pytest.raises(ModelError, match="Dynamic"):
        OnnxTagger(model_files, CPU, write_profile(model_files))
    tagger = OnnxTagger(model_files, CPU, write_profile(model_files, size=[9, 7]))
    assert tagger.image_size == (9, 7)


@pytest.mark.parametrize("shape, message", [
    ([2, 3, 4, 6], "batch size"), ([1, 4, 4, 6], "channels"),
    ([3, 4, 6], "layout"), ([1, 4, 6, 3], "channels"),
])
def test_incompatible_inputs_fail_before_inference(model_files, runtime, shape, message):
    runtime.inputs[0].shape = shape
    with pytest.raises(ModelError, match=message):
        OnnxTagger(model_files, CPU, write_profile(model_files))


@pytest.mark.parametrize("problem, message", [
    ("size", "conflicts"), ("input_name", "expected input"), ("input_type", "input type"),
    ("multi_input", "exactly one"), ("missing_input", "exactly one"),
    ("multi_output", "output_name"), ("output_name", "expected output"),
    ("output_count", "labels"), ("output_rank", "score vector"),
    ("output_type", "floating-point"), ("output_batch", "one output batch"),
    ("pad_square", "square input"),
])
def test_invalid_io_contracts_fail_at_load(model_files, runtime, problem, message):
    overrides = {}
    if problem == "size":
        overrides["size"] = [5, 4]
    elif problem == "input_name":
        overrides["input_name"] = "wrong"
    elif problem == "input_type":
        runtime.inputs[0].type = "tensor(string)"
    elif problem == "multi_input":
        runtime.inputs.append(runtime.inputs[0])
    elif problem == "missing_input":
        runtime.inputs = []
    elif problem == "multi_output":
        runtime.outputs.append(SimpleNamespace(name="extra", shape=[1, 2], type="tensor(float)"))
    elif problem == "output_name":
        overrides["output_name"] = "missing"
    elif problem == "output_count":
        runtime.outputs[0].shape = [1, 3]
    elif problem == "output_rank":
        runtime.outputs[0].shape = [1, 2, 4]
    elif problem == "output_type":
        runtime.outputs[0].type = "tensor(int64)"
    elif problem == "output_batch":
        runtime.outputs[0].shape = [2, 2]
    elif problem == "pad_square":
        overrides["resize"] = "pad_square"
    with pytest.raises(ModelError, match=message):
        OnnxTagger(model_files, CPU, write_profile(model_files, **overrides))


def test_named_output_is_used_even_when_it_is_not_first(model_files, runtime):
    runtime.outputs.insert(0, SimpleNamespace(name="embeddings", shape=[1, 99], type="tensor(float)"))
    path = write_profile(model_files, input_name="image", output_name="scores")
    tagger = OnnxTagger(model_files, CPU, path)
    image = model_files / "image.png"
    Image.new("RGB", (6, 4), "red").save(image)
    result = tagger.predict(image, 0.7)
    assert list(result) == ["character"]
    assert result["character"][0][0] == "alice"
    assert runtime.calls[0][0] == ["scores"]
    assert list(runtime.calls[0][1]) == ["image"]


@pytest.mark.parametrize("activation, scores, expected", [
    ("none", [[0.4, 0.2]], [0.4, 0.2]),
    ("sigmoid", [[0, -1000000]], [0.5, 0]),
    ("softmax", [[1000000, 1000000]], [0.5, 0.5]),
    ("softmax", [[1e308, -1e308]], [1, 0]),
])
def test_output_activation_is_explicit_and_numerically_stable(
    model_files, runtime, activation, scores, expected
):
    runtime.scores = np.array(scores, dtype=np.float64)
    tagger = OnnxTagger(model_files, CPU, write_profile(model_files, activation=activation))
    image = model_files / "image.png"
    Image.new("RGB", (6, 4)).save(image)
    with np.errstate(over="raise", invalid="raise"):
        prediction = tagger.predict(image, 0)
    actual = [prediction["character"][0][1], prediction["general"][0][1]]
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("scores", [
    [[1.01, 0]], [[-0.01, 0]], [[np.nan, 0]], [[np.inf, 0]],
    [[0.8]], [[0.8, 0.2], [0.8, 0.2]], np.array([[1, 0]], dtype=np.int64),
])
def test_invalid_runtime_scores_are_errors(model_files, runtime, scores):
    runtime.scores = np.asarray(scores)
    tagger = OnnxTagger(model_files, CPU, write_profile(model_files))
    image = model_files / "image.png"
    Image.new("RGB", (6, 4)).save(image)
    with pytest.raises(ModelError):
        tagger.predict(image)


def test_unbatched_scores_and_per_category_thresholds(model_files, runtime):
    runtime.outputs[0].shape = [2]
    runtime.scores = np.array([0.5, 0.7], dtype=np.float32)
    tagger = OnnxTagger(model_files, CPU, write_profile(model_files))
    image = model_files / "image.png"
    Image.new("RGB", (6, 4)).save(image)
    assert tagger.predict(image, 0.8, {"character": 0.5}) == {"character": [("alice", 0.5)]}


def test_bgr_channel_order_precedes_normalization(model_files, runtime):
    path = write_profile(
        model_files, color_order="BGR", scale=1, mean=[1, 2, 3], std=[1, 2, 3]
    )
    tagger = OnnxTagger(model_files, CPU, path)
    image = model_files / "image.png"
    Image.new("RGB", (6, 4), (12, 20, 30)).save(image)
    np.testing.assert_array_equal(tagger.preprocess(image)[0, :, 0, 0], [29, 9, 3])


def test_wd_padding_resizes_after_compositing_and_padding(model_files, runtime):
    runtime.inputs[0].shape = [1, 6, 6, 3]
    (model_files / "labels.csv").write_text("name,category\nalice,4\nblue_hair,0\n", encoding="utf-8")
    profile_path = model_files / "wd.json"
    profile_path.write_text(
        '{"preset":"wd","model":"model.onnx","labels":"labels.csv"}', encoding="utf-8"
    )
    tagger = OnnxTagger(model_files, CPU, profile_path)
    image = model_files / "image.png"
    Image.new("RGBA", (3, 1), (255, 0, 0, 128)).save(image)
    expected_image = ImageOps.expand(
        Image.new("RGB", (3, 1), (255, 127, 127)), border=(0, 1, 0, 1), fill="white"
    ).resize((6, 6), Image.Resampling.BICUBIC)
    expected = np.asarray(expected_image, dtype=np.float32)[:, :, ::-1][None, ...]
    np.testing.assert_array_equal(tagger.preprocess(image), expected)


@pytest.mark.parametrize("resize, center, edge", [
    ("stretch", (0, 1, 0), (1, 0, 0)),
    ("center_crop", (0, 1, 0), (0, 1, 0)),
    ("letterbox", (0, 1, 0), (1, 1, 1)),
])
def test_resize_modes_have_distinct_geometry(model_files, runtime, resize, center, edge):
    runtime.inputs[0].shape = [1, 3, 6, 6]
    path = write_profile(model_files, resize=resize, interpolation="nearest")
    tagger = OnnxTagger(model_files, CPU, path)
    source = Image.new("RGB", (18, 6), "green")
    source.paste((0, 255, 0), (6, 0, 12, 6))
    source.paste("red", (0, 0, 6, 6))
    source.paste("blue", (12, 0, 18, 6))
    image = model_files / "image.png"
    source.save(image)
    tensor = tagger.preprocess(image)
    np.testing.assert_array_equal(tensor[0, :, 3, 3], center)
    np.testing.assert_array_equal(tensor[0, :, 0, 0], edge)


@pytest.mark.parametrize("dtype, mean, message", [
    ("tensor(uint8)", [1, 1, 1], "input range"),
    ("tensor(int8)", [0, 0, 0], "input range"),
    ("tensor(float16)", [-100000, 0, 0], "overflow"),
])
def test_input_conversion_does_not_wrap_or_overflow(model_files, runtime, dtype, mean, message):
    runtime.inputs[0].type = dtype
    path = write_profile(model_files, scale=1, mean=mean)
    tagger = OnnxTagger(model_files, CPU, path)
    image = model_files / "image.png"
    Image.new("RGB", (6, 4), (255, 0, 0)).save(image)
    with pytest.raises(ModelError, match=message):
        tagger.preprocess(image)


@pytest.mark.parametrize("component", ["model", "labels", "normalization", "activation", "external"])
def test_fingerprint_covers_all_model_inputs(model_files, runtime, component):
    external = model_files / "weights.data"
    external.write_bytes(b"external")
    path = write_profile(model_files, external_data=["weights.data"])
    before = OnnxTagger(model_files, CPU, path).fingerprint
    overrides = {"external_data": ["weights.data"]}
    if component == "model":
        (model_files / "model.onnx").write_bytes(b"changed weights")
    elif component == "labels":
        (model_files / "labels.json").write_text('["new", "labels"]', encoding="utf-8")
    elif component == "normalization":
        overrides["scale"] = 1
    elif component == "activation":
        overrides["activation"] = "sigmoid"
    else:
        external.write_bytes(b"changed external data")
    path = write_profile(model_files, **overrides)
    assert OnnxTagger(model_files, CPU, path).fingerprint != before


def test_fingerprint_does_not_depend_on_model_location(model_files, runtime):
    before = OnnxTagger(model_files, CPU, write_profile(model_files)).fingerprint
    moved = model_files / "moved"
    moved.mkdir()
    for name in ("model.onnx", "labels.json"):
        (moved / name).write_bytes((model_files / name).read_bytes())
    assert OnnxTagger(moved, CPU, write_profile(moved)).fingerprint == before
