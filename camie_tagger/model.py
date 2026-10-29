"""Profile-driven ONNX image tagging."""

from __future__ import annotations

import csv
import hashlib
import json
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import validate_category_thresholds, validate_threshold
from .devices import DeviceSelection
from .logging_setup import get_logger
from .model_profile import (
    MODEL_CATEGORIES,
    ModelError,
    ModelProfile,
    find_model_files,
    load_model_profile,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 512
PREPROCESSING_VERSION = "onnx-profile-exif-v1"
PAD_COLOR = (124, 116, 104)
OUTPUT_NAME = "refined_predictions"

RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}
INPUT_DTYPES = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(double)": np.dtype(np.float64),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(int8)": np.dtype(np.int8),
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    result = np.empty_like(x, dtype=np.result_type(x.dtype, np.float32))
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    negative_exp = np.exp(x[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def load_labels(path: Path, profile: ModelProfile) -> tuple[dict[str, str], dict[str, str]]:
    try:
        categories = None
        suffix = path.suffix.lower()
        if suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                required = {"name", "category"} if profile.preset == "wd" else {"name"}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f"CSV labels require columns: {', '.join(sorted(required))}")
                if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
                    raise ValueError("CSV label headers must be unique")
                entries = list(reader)
                if any(None in entry for entry in entries):
                    raise ValueError("CSV label row has more values than the header")
        elif suffix == ".txt":
            entries = path.read_text(encoding="utf-8-sig").splitlines()
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            total_tags = None
            if isinstance(data, dict) and "dataset_info" in data:
                dataset = data["dataset_info"]
                total_tags = dataset.get("total_tags")
                if "total_tags" in dataset and type(total_tags) is not int:
                    raise ValueError("total_tags must be an integer")
                data = dataset["tag_mapping"]
                categories = data["tag_to_category"]
                if not isinstance(categories, dict):
                    raise ValueError("tag_to_category must be an object")
                data = data["idx_to_tag"]
            elif isinstance(data, dict) and ("idx_to_tag" in data or "id2label" in data):
                if "tag_to_category" in data:
                    categories = data["tag_to_category"]
                    if not isinstance(categories, dict):
                        raise ValueError("tag_to_category must be an object")
                data = data["idx_to_tag"] if "idx_to_tag" in data else data["id2label"]
            if isinstance(data, dict):
                if set(data) != {str(index) for index in range(len(data))}:
                    raise ValueError("tag indices must be contiguous, starting at zero")
                entries = [data[str(index)] for index in range(len(data))]
            elif isinstance(data, list):
                entries = data
            else:
                raise ValueError("JSON labels must be an array or an indexed tag mapping")
            if total_tags is not None and (type(total_tags) is not int or total_tags != len(entries)):
                raise ValueError("total_tags does not match the tag mapping")
        else:
            raise ValueError("labels must be a .json, .csv or .txt file")
        if not entries:
            raise ValueError("the label list is empty")

        idx_to_tag: dict[str, str] = {}
        tag_to_category: dict[str, str] = {}
        for index, entry in enumerate(entries):
            name = entry.get("name") if isinstance(entry, dict) else entry
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"label {index} must have a nonempty name")
            name = name.strip()
            if name in tag_to_category:
                raise ValueError(f"duplicate label: {name!r}")
            if categories is not None:
                category = categories.get(name)
            elif isinstance(entry, dict):
                category = entry.get("category", profile.default_category)
            else:
                category = profile.default_category
            if isinstance(category, int) and not isinstance(category, bool):
                category = str(category)
            if not isinstance(category, str) or not category.strip():
                raise ValueError(f"missing category for tag {name!r}")
            category = category.strip()
            category = profile.category_map.get(category, category)
            if category not in MODEL_CATEGORIES:
                raise ValueError(
                    f"unsupported category {category!r} for {name!r}; configure category_map"
                )
            idx_to_tag[str(index)] = name
            tag_to_category[name] = category
        return idx_to_tag, tag_to_category
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError, csv.Error) as exc:
        raise ModelError(f"Invalid model labels {path}: {exc}") from exc


class OnnxTagger:
    def __init__(
        self, model_dir: Path, selection: DeviceSelection, model_profile: Path | None = None
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelError("onnxruntime is not installed; run install.sh to select a runtime") from exc

        log = get_logger()
        self.profile = load_model_profile(model_profile)
        onnx_path, labels_path = find_model_files(model_dir, self.profile)
        self._model_paths = (onnx_path, labels_path, *self.profile.external_data)
        self.idx_to_tag, self.tag_to_category = load_labels(labels_path, self.profile)

        options = ort.SessionOptions()
        options.log_severity_level = 3  # Suppress routine provider warnings.

        try:
            self.session = ort.InferenceSession(
                str(onnx_path), sess_options=options, providers=selection.providers
            )
        except Exception as exc:
            raise ModelError(f"Could not load model {onnx_path.name}: {exc}") from exc

        self._configure_io()
        active = self.session.get_providers()
        log.info("Model loaded: %s", onnx_path.name)
        log.info(
            "Profile: %s; input=%s %s %s %s; output=%s (%s); labels=%d",
            self.profile.preset, self.input_name, self.profile.layout, self.image_size,
            self.input_dtype.name, self.output_name, self.profile.activation, len(self.idx_to_tag),
        )
        log.info("Execution providers: %s", ", ".join(active))

        if selection.using_gpu and not any(p != "CPUExecutionProvider" for p in active):
            log.warning(
                "Requested %s acceleration but the session runs on CPU only.",
                selection.vendor.upper(),
            )

    def _configure_io(self) -> None:
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ModelError("Tagging requires exactly one image input; multi-input models need an adapter")
        image_input = inputs[0]
        if self.profile.input_name is not None and image_input.name != self.profile.input_name:
            raise ModelError(f"Model does not expose the expected input {self.profile.input_name!r}")
        self.input_name = image_input.name
        if image_input.type not in INPUT_DTYPES:
            raise ModelError(f"Unsupported image input type: {image_input.type}")
        self.input_dtype = INPUT_DTYPES[image_input.type]
        shape = image_input.shape
        layout = self.profile.layout
        if len(shape) != len(layout):
            raise ModelError(f"Input shape {shape} does not match layout {layout}")
        if "N" in layout and isinstance(shape[0], int) and shape[0] != 1:
            raise ModelError("Only batch size 1 is supported")
        channels = 1 if self.profile.color_order == "GRAY" else 3
        channel_dim = shape[layout.index("C")]
        if isinstance(channel_dim, int) and channel_dim != channels:
            raise ModelError(f"Input has {channel_dim} channels, but {self.profile.color_order} requires {channels}")
        dimensions = (shape[layout.index("W")], shape[layout.index("H")])
        resolved: list[int] = []
        for index, dimension in enumerate(dimensions):
            configured = self.profile.size[index] if self.profile.size is not None else None
            if isinstance(dimension, int) and dimension > 0:
                if configured is not None and configured != dimension:
                    raise ModelError(f"Profile size {self.profile.size} conflicts with input shape {shape}")
                resolved.append(dimension)
            elif configured is not None:
                resolved.append(configured)
            else:
                raise ModelError("Dynamic image dimensions require size: [width, height] in the profile")
        self.image_size = (resolved[0], resolved[1])
        if self.profile.resize == "pad_square" and self.image_size[0] != self.image_size[1]:
            raise ModelError("pad_square requires a square input size")

        outputs = self.session.get_outputs()
        if self.profile.output_name is None:
            if len(outputs) != 1:
                raise ModelError("Models with multiple outputs require output_name in the profile")
            output = outputs[0]
        else:
            matching = [item for item in outputs if item.name == self.profile.output_name]
            if not matching:
                raise ModelError(f"Model does not expose the expected output {self.profile.output_name!r}")
            output = matching[0]
        self.output_name = output.name
        if output.type not in {"tensor(float)", "tensor(float16)", "tensor(double)"}:
            raise ModelError(f"Model scores must be floating-point, got {output.type}")
        output_shape = output.shape
        if len(output_shape) not in (1, 2):
            raise ModelError(f"Expected a tag score vector [tags] or [1, tags], got {output_shape}")
        if len(output_shape) == 2 and isinstance(output_shape[0], int) and output_shape[0] != 1:
            raise ModelError(f"Expected one output batch, got {output_shape}")
        if isinstance(output_shape[-1], int) and output_shape[-1] != len(self.idx_to_tag):
            raise ModelError(
                f"Model output has {output_shape[-1]} scores, but labels contain {len(self.idx_to_tag)} tags"
            )

    @property
    def inference_settings(self) -> dict[str, object]:
        return {
            **self.profile.inference_settings(),
            "size": self.image_size, "input_dtype": self.input_dtype.name,
            "input_name": self.input_name, "output_name": self.output_name,
        }

    @property
    def inference_profile(self) -> dict[str, object]:
        return {
            "model_fingerprint": self.fingerprint,
            "preprocessing_version": PREPROCESSING_VERSION,
            "model_profile": self.inference_settings,
        }

    @cached_property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(PREPROCESSING_VERSION.encode("ascii"))
        digest.update(json.dumps(self.inference_settings, sort_keys=True, allow_nan=False).encode("utf-8"))
        try:
            for path in self._model_paths:
                digest.update(path.stat().st_size.to_bytes(8, "big"))
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
        except OSError as exc:
            raise ModelError(f"Could not fingerprint model files: {exc}") from exc
        return digest.hexdigest()

    def preprocess(self, image_path: Path) -> np.ndarray:
        profile = self.profile
        resample = RESAMPLING[profile.interpolation]
        target_width, target_height = self.image_size
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            if profile.alpha_background is not None:
                background = Image.new("RGBA", img.size, (*profile.alpha_background, 255))
                img = Image.alpha_composite(background, img.convert("RGBA"))
            img = img.convert("RGB")
            width, height = img.size
            if profile.resize == "stretch":
                img = img.resize(self.image_size, resample)
            elif profile.resize == "center_crop":
                img = ImageOps.fit(img, self.image_size, method=resample)
            elif profile.resize == "pad_square":
                side = max(width, height)
                padded = Image.new("RGB", (side, side), profile.pad_color)
                padded.paste(img, ((side - width) // 2, (side - height) // 2))
                img = padded.resize(self.image_size, resample)
            else:
                if width * target_height > height * target_width:
                    resized = (target_width, max(1, height * target_width // width))
                else:
                    resized = (max(1, width * target_height // height), target_height)
                img = img.resize(resized, resample)
                padded = Image.new("RGB", self.image_size, profile.pad_color)
                padded.paste(img, ((target_width - resized[0]) // 2, (target_height - resized[1]) // 2))
                img = padded
            if profile.color_order == "GRAY":
                img = img.convert("L")
            working_dtype = np.float64 if self.input_dtype == np.dtype(np.float64) else np.float32
            array = np.asarray(img, dtype=working_dtype)
        if profile.color_order == "GRAY":
            array = array[:, :, None]
        elif profile.color_order == "BGR":
            array = array[:, :, ::-1]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            array = (array / profile.scale - np.asarray(profile.mean, dtype=working_dtype)) / np.asarray(
                profile.std, dtype=working_dtype
            )
        if not np.all(np.isfinite(array)):
            raise ModelError("Image normalization produced nonfinite values; check scale, mean and std")
        if np.issubdtype(self.input_dtype, np.integer):
            array = np.rint(array)
            bounds = np.iinfo(self.input_dtype)
            if np.any(array < bounds.min) or np.any(array > bounds.max):
                raise ModelError(f"Normalized pixels are outside the {self.input_dtype.name} input range")
        if profile.layout in {"NCHW", "CHW"}:
            array = array.transpose(2, 0, 1)
        if profile.layout.startswith("N"):
            array = array[None, ...]
        with np.errstate(over="ignore", invalid="ignore"):
            tensor = np.ascontiguousarray(array, dtype=self.input_dtype)
        if not np.all(np.isfinite(tensor)):
            raise ModelError(f"Normalized pixels overflow the {self.input_dtype.name} input type")
        return tensor

    def predict(
        self,
        image_path: Path,
        threshold: float = 0.5,
        category_thresholds: dict[str, float] | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        threshold = validate_threshold(threshold)
        category_thresholds = validate_category_thresholds(category_thresholds or {})
        tensor = self.preprocess(image_path)
        scores = np.asarray(self.session.run([self.output_name], {self.input_name: tensor})[0])
        count = len(self.idx_to_tag)
        if scores.shape not in {(count,), (1, count)}:
            raise ModelError(f"Model output shape {scores.shape} must be ({count},) or (1, {count})")
        if not np.issubdtype(scores.dtype, np.floating) or not np.all(np.isfinite(scores)):
            raise ModelError("Model output must contain finite floating-point scores")
        scores = scores.reshape(count).astype(np.result_type(scores.dtype, np.float32), copy=False)
        if self.profile.activation == "sigmoid":
            probabilities = _sigmoid(scores)
        elif self.profile.activation == "softmax":
            scores = scores.astype(np.float64)
            with np.errstate(over="ignore"):
                exponents = np.exp(scores - scores.max())
            probabilities = exponents / exponents.sum()
        else:
            if np.any(scores < 0) or np.any(scores > 1):
                raise ModelError("Model probabilities must be between 0 and 1; check activation in the profile")
            probabilities = scores

        result: dict[str, list[tuple[str, float]]] = {}
        minimum = min([threshold, *category_thresholds.values()])
        for index in np.where(probabilities >= minimum)[0]:
            name = self.idx_to_tag[str(int(index))]
            category = self.tag_to_category[name]
            if probabilities[index] < category_thresholds.get(category, threshold):
                continue
            result.setdefault(category, []).append((name, float(probabilities[index])))
        for tags in result.values():
            tags.sort(key=lambda item: item[1], reverse=True)
        return result


CamieTagger = OnnxTagger
