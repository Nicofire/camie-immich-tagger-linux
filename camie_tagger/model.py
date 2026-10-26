"""camie-tagger-v2 ONNX inference."""

from __future__ import annotations

import hashlib
import json
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import validate_category_thresholds, validate_threshold
from .devices import DeviceSelection
from .logging_setup import get_logger

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 512
PREPROCESSING_VERSION = "letterbox-lanczos-exif-v1"
PAD_COLOR = (124, 116, 104)
OUTPUT_NAME = "refined_predictions"

# Pillow >= 9.1 exposes the enum; older releases keep the constant on the module.
RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


class ModelError(RuntimeError):
    """Raised when the model files are missing or unusable."""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    result = np.empty_like(x, dtype=np.result_type(x.dtype, np.float32))
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    negative_exp = np.exp(x[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def find_model_files(model_dir: Path) -> tuple[Path, Path]:
    if not model_dir.is_dir():
        raise ModelError(
            f"Model directory not found: {model_dir}\n"
            "Download the model as described in the README."
        )
    onnx_files = sorted(model_dir.rglob("*.onnx"))
    meta_files = sorted(model_dir.rglob("*metadata*.json"))
    if not onnx_files:
        raise ModelError(f"No .onnx file found under {model_dir}")
    if not meta_files:
        raise ModelError(f"No *metadata*.json file found under {model_dir}")
    return onnx_files[0], meta_files[0]


class CamieTagger:
    def __init__(self, model_dir: Path, selection: DeviceSelection):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelError("onnxruntime is not installed; run install.sh to select a runtime") from exc

        log = get_logger()
        onnx_path, meta_path = find_model_files(model_dir)
        self._model_paths = (onnx_path, meta_path)

        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            dataset_info = metadata["dataset_info"]
            tag_mapping = dataset_info["tag_mapping"]
            idx_to_tag = tag_mapping["idx_to_tag"]
            tag_to_category = tag_mapping["tag_to_category"]
            if not isinstance(idx_to_tag, dict) or not isinstance(tag_to_category, dict):
                raise ValueError("tag mappings must be objects")
            if not idx_to_tag or set(idx_to_tag) != {str(index) for index in range(len(idx_to_tag))}:
                raise ValueError("tag indices must be contiguous, starting at zero")
            if dataset_info.get("total_tags", len(idx_to_tag)) != len(idx_to_tag):
                raise ValueError("total_tags does not match the tag mapping")
            for name in idx_to_tag.values():
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("tag names must be nonempty strings")
                category = tag_to_category.get(name)
                if not isinstance(category, str) or not category.strip():
                    raise ValueError(f"missing category for tag {name!r}")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ModelError(f"Invalid model metadata {meta_path.name}: {exc}") from exc
        self.idx_to_tag: dict[str, str] = idx_to_tag
        self.tag_to_category: dict[str, str] = tag_to_category

        options = ort.SessionOptions()
        options.log_severity_level = 3  # Suppress routine provider warnings.

        try:
            self.session = ort.InferenceSession(
                str(onnx_path), sess_options=options, providers=selection.providers
            )
        except Exception as exc:
            raise ModelError(f"Could not load model {onnx_path.name}: {exc}") from exc

        if OUTPUT_NAME not in {output.name for output in self.session.get_outputs()}:
            raise ModelError(f"Model does not expose the expected output {OUTPUT_NAME!r}")
        self.input_name = self.session.get_inputs()[0].name
        active = self.session.get_providers()
        log.info("Model loaded: %s", onnx_path.name)
        log.info("Execution providers: %s", ", ".join(active))

        if selection.using_gpu and not any(p != "CPUExecutionProvider" for p in active):
            log.warning(
                "Requested %s acceleration but the session runs on CPU only.",
                selection.vendor.upper(),
            )

    @cached_property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for path in self._model_paths:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()

    def preprocess(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            width, height = img.size
            if width > height:
                resized = (IMAGE_SIZE, max(1, height * IMAGE_SIZE // width))
            else:
                resized = (max(1, width * IMAGE_SIZE // height), IMAGE_SIZE)
            img = img.resize(resized, RESAMPLE)
            padded = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), PAD_COLOR)
            padded.paste(img, ((IMAGE_SIZE - resized[0]) // 2, (IMAGE_SIZE - resized[1]) // 2))
            array = np.asarray(padded, dtype=np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        array = array.transpose(2, 0, 1)  # HWC -> CHW
        return array[None, ...].astype(np.float32)  # -> NCHW

    def predict(
        self,
        image_path: Path,
        threshold: float = 0.5,
        category_thresholds: dict[str, float] | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        threshold = validate_threshold(threshold)
        category_thresholds = validate_category_thresholds(category_thresholds or {})
        tensor = self.preprocess(image_path)
        logits = np.asarray(self.session.run([OUTPUT_NAME], {self.input_name: tensor})[0])
        expected_shape = (1, len(self.idx_to_tag))
        if logits.shape != expected_shape:
            raise ModelError(f"Model output shape {logits.shape} does not match {expected_shape}")
        if not np.issubdtype(logits.dtype, np.floating) or not np.all(np.isfinite(logits)):
            raise ModelError("Model output must contain finite floating-point logits")
        probabilities = _sigmoid(logits[0])

        result: dict[str, list[tuple[str, float]]] = {}
        minimum = min([threshold, *category_thresholds.values()])
        for index in np.where(probabilities >= minimum)[0]:
            name = self.idx_to_tag[str(int(index))]
            category = self.tag_to_category.get(name, "unknown")
            if probabilities[index] < category_thresholds.get(category, threshold):
                continue
            result.setdefault(category, []).append((name, float(probabilities[index])))
        for tags in result.values():
            tags.sort(key=lambda item: item[1], reverse=True)
        return result
