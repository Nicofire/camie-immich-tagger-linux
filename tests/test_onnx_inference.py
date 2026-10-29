"""Inference against tiny, generated ONNX graphs; no downloaded model weights."""

import json

import numpy as np
import pytest
from PIL import Image

from camie_tagger import pipeline
from camie_tagger.devices import DeviceSelection
from camie_tagger.model import ModelError, OnnxTagger
from camie_tagger.state import StateStore

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytestmark = pytest.mark.onnx

CPU = DeviceSelection(providers=["CPUExecutionProvider"])


def build_model(
    directory, *, filename="model.onnx", layout="NCHW", dtype="FLOAT",
    size=(6, 4), dynamic=False, output_name="scores", external=False, divisor=1.0,
):
    helper = onnx.helper
    proto = onnx.TensorProto
    dimensions = {"N": 1, "C": 3, "H": size[1], "W": size[0]}
    if dynamic:
        dimensions.update(N="batch", H="height", W="width")
    image = helper.make_tensor_value_info("pixels", getattr(proto, dtype), [dimensions[c] for c in layout])
    output_shape = [1, 3] if "N" in layout else [3]
    output = helper.make_tensor_value_info(output_name, proto.FLOAT, output_shape)
    initializers = [
        onnx.numpy_helper.from_array(np.array([divisor], dtype=np.float32), "divisor")
    ]
    nodes = [helper.make_node("Cast", ["pixels"], ["float_pixels"], to=proto.FLOAT)]
    source = "float_pixels"
    if dtype == "INT8":
        initializers.append(onnx.numpy_helper.from_array(np.array([128], dtype=np.float32), "offset"))
        nodes.append(helper.make_node("Add", [source, "offset"], ["unshifted"]))
        source = "unshifted"
    nodes.extend([
        helper.make_node(
            "ReduceMean", [source], ["averages"],
            axes=[layout.index("H"), layout.index("W")], keepdims=0,
        ),
        helper.make_node("Div", ["averages", "divisor"], [output_name]),
    ])
    graph = helper.make_graph(nodes, "channel_tagger", [image], [output], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = directory / filename
    if external:
        onnx.save_model(
            model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
            location=f"{filename}.data", size_threshold=0,
        )
    else:
        onnx.save_model(model, str(path))
    return path


def write_profile(directory, model, **overrides):
    labels = directory / f"{model.stem}-labels.json"
    labels.write_text('["red","green","blue"]', encoding="utf-8")
    profile = directory / f"{model.stem}-profile.json"
    data = {
        "preset": "generic", "model": model.name, "labels": labels.name,
        "layout": "NCHW", "color_order": "RGB", "resize": "stretch",
        "scale": 255, "mean": [0, 0, 0], "std": [1, 1, 1], "activation": "none",
        **overrides,
    }
    profile.write_text(json.dumps(data), encoding="utf-8")
    return profile


@pytest.mark.parametrize("layout, dtype, dynamic, activation, expected", [
    ("NCHW", "FLOAT", False, "none", 1.0),
    ("NHWC", "FLOAT", False, "none", 1.0),
    ("CHW", "FLOAT", False, "none", 1.0),
    ("HWC", "FLOAT", False, "none", 1.0),
    ("NCHW", "FLOAT16", False, "none", 1.0),
    ("NHWC", "DOUBLE", False, "none", 1.0),
    ("NHWC", "UINT8", False, "none", 1.0),
    ("NCHW", "INT8", False, "none", 1.0),
    ("NHWC", "FLOAT", True, "none", 1.0),
    ("NCHW", "FLOAT", False, "sigmoid", 0.7310585786),
    ("NHWC", "FLOAT", False, "softmax", 0.5761168848),
])
def test_real_onnx_inference(tmp_path, layout, dtype, dynamic, activation, expected):
    integer = dtype in {"UINT8", "INT8"}
    model = build_model(
        tmp_path, layout=layout, dtype=dtype, dynamic=dynamic, divisor=255.0 if integer else 1.0
    )
    overrides = {
        "layout": layout, "activation": activation, "scale": 1 if integer else 255,
        "mean": [128, 128, 128] if dtype == "INT8" else [0, 0, 0],
    }
    if dynamic:
        overrides["size"] = [6, 4]
    profile = write_profile(tmp_path, model, **overrides)
    image = tmp_path / "image.png"
    Image.new("RGB", (9, 7), "red").save(image)
    tagger = OnnxTagger(tmp_path, CPU, profile)

    result = tagger.predict(image, 0.6 if activation == "sigmoid" else 0.5)

    assert list(result) == ["general"]
    assert len(result["general"]) == 1
    assert result["general"][0][0] == "red"
    assert result["general"][0][1] == pytest.approx(expected)


def test_real_camie_defaults_remain_usable(tmp_path):
    build_model(tmp_path, size=(512, 512), output_name="refined_predictions")
    metadata = {
        "dataset_info": {
            "total_tags": 3,
            "tag_mapping": {
                "idx_to_tag": {"0": "red", "1": "green", "2": "blue"},
                "tag_to_category": {"red": "general", "green": "general", "blue": "general"},
            },
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    image = tmp_path / "red.png"
    Image.new("RGB", (8, 8), "red").save(image)
    tagger = OnnxTagger(tmp_path, CPU)
    result = tagger.predict(image)
    assert list(result) == ["general"]
    assert [name for name, _ in result["general"]] == ["red"]
    assert result["general"][0][1] > 0.9


def test_real_wd_profile_uses_bgr_csv_and_existing_probabilities(tmp_path):
    model = build_model(tmp_path, layout="NHWC", size=(4, 4), divisor=255.0)
    (tmp_path / "selected_tags.csv").write_text(
        "name,category\nblue,9\ngreen,0\nred,4\n", encoding="utf-8"
    )
    profile = tmp_path / "wd.json"
    profile.write_text(json.dumps({
        "preset": "wd", "model": model.name, "labels": "selected_tags.csv",
    }), encoding="utf-8")
    image = tmp_path / "red.png"
    Image.new("RGB", (8, 8), "red").save(image)
    assert OnnxTagger(tmp_path, CPU, profile).predict(image, 0.9) == {
        "character": [("red", 1.0)]
    }


def test_real_external_weights_are_loaded_and_fingerprinted(tmp_path):
    model = build_model(tmp_path, external=True)
    profile = write_profile(tmp_path, model, external_data=[f"{model.name}.data"])
    image = tmp_path / "red.png"
    Image.new("RGB", (6, 4), "red").save(image)
    first = OnnxTagger(tmp_path, CPU, profile)
    assert first.predict(image) == {"general": [("red", 1.0)]}
    fingerprint = first.fingerprint
    model_bytes = model.read_bytes()
    del first
    (tmp_path / f"{model.name}.data").write_bytes(np.array([2.0], dtype=np.float32).tobytes())
    second = OnnxTagger(tmp_path, CPU, profile)
    assert model.read_bytes() == model_bytes
    assert second.fingerprint != fingerprint
    assert second.predict(image) == {"general": [("red", 0.5)]}


def test_real_label_count_mismatch_fails_before_tagging(tmp_path):
    model = build_model(tmp_path)
    profile = write_profile(tmp_path, model)
    (tmp_path / "model-labels.json").write_text('["wrong"]', encoding="utf-8")
    with pytest.raises(ModelError, match="3 scores.*1 tags"):
        OnnxTagger(tmp_path, CPU, profile)


def test_model_switching_respects_reprocessing_and_tag_ownership(
    settings, state, tagging, tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "OnnxTagger", OnnxTagger)
    first = build_model(tmp_path, filename="first.onnx")
    settings.model_profile = write_profile(tmp_path, first)
    image = settings.scan_dirs[0] / "red.png"
    Image.new("RGB", (6, 4), "red").save(image)
    tagging.sidecars[image] = ["manual/favourite"]

    preview = pipeline.run_tagging(settings, state, mode="all", dry_run=True)
    assert preview.tagged == 1
    assert state.processed == {}
    assert tagging.sidecars[image] == ["manual/favourite"]
    assert pipeline.run_tagging(settings, state, mode="all").tagged == 1
    assert tagging.sidecars[image] == ["manual/favourite", "general/red"]
    first_fingerprint = state.processed[str(image)]["profile"]["model_fingerprint"]

    second = build_model(tmp_path, filename="second.onnx", layout="NHWC")
    settings.model_profile = write_profile(tmp_path, second, layout="NHWC")
    (tmp_path / "second-labels.json").write_text(
        '[{"name":"scarlet","category":"character"},"green","blue"]', encoding="utf-8"
    )
    assert pipeline.run_tagging(settings, state, mode="all").skipped == 1
    assert tagging.sidecars[image] == ["manual/favourite", "general/red"]

    result = pipeline.run_tagging(
        settings, state, mode="all", reprocess=True, allow_tag_removal=True
    )
    assert result.tagged == 1
    assert tagging.sidecars[image] == ["manual/favourite", "character/scarlet"]
    stored = StateStore(settings.data_dir).processed[str(image)]["profile"]
    assert stored["model_fingerprint"] != first_fingerprint
    assert stored["model_profile"]["layout"] == "NHWC"
    assert stored["model_profile"]["input_name"] == "pixels"
