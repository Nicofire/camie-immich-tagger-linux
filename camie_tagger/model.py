"""camie-tagger-v2 ONNX inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .devices import DeviceSelection
from .logging_setup import get_logger

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 512
OUTPUT_NAME = "refined_predictions"

# Pillow >= 9.1 exposes the enum; older releases keep the constant on the module.
RESAMPLE = getattr(getattr(Image, "Resampling", Image), "BICUBIC")


class ModelError(RuntimeError):
    """Raised when the model files are missing or unusable."""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Branch on sign to avoid overflow in exp for large negative values.
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


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
        import onnxruntime as ort

        log = get_logger()
        onnx_path, meta_path = find_model_files(model_dir)

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        tag_mapping = metadata["dataset_info"]["tag_mapping"]
        self.idx_to_tag: dict[str, str] = tag_mapping["idx_to_tag"]
        self.tag_to_category: dict[str, str] = tag_mapping["tag_to_category"]

        options = ort.SessionOptions()
        options.log_severity_level = 3  # Suppress routine provider warnings.

        try:
            self.session = ort.InferenceSession(
                str(onnx_path), sess_options=options, providers=selection.providers
            )
        except Exception as exc:
            raise ModelError(f"Could not load model {onnx_path.name}: {exc}") from exc

        self.input_name = self.session.get_inputs()[0].name
        active = self.session.get_providers()
        log.info("Model loaded: %s", onnx_path.name)
        log.info("Execution providers: %s", ", ".join(active))

        if selection.using_gpu and not any(p != "CPUExecutionProvider" for p in active):
            log.warning(
                "Requested %s acceleration but the session runs on CPU only.",
                selection.vendor.upper(),
            )

    def preprocess(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), RESAMPLE)
            array = np.asarray(img, dtype=np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        array = array.transpose(2, 0, 1)  # HWC -> CHW
        return array[None, ...].astype(np.float32)  # -> NCHW

    def predict(self, image_path: Path, threshold: float = 0.5) -> dict[str, list[tuple[str, float]]]:
        tensor = self.preprocess(image_path)
        logits = self.session.run([OUTPUT_NAME], {self.input_name: tensor})[0][0]
        probabilities = _sigmoid(logits)

        result: dict[str, list[tuple[str, float]]] = {}
        for index in np.where(probabilities >= threshold)[0]:
            name = self.idx_to_tag[str(int(index))]
            category = self.tag_to_category.get(name, "unknown")
            result.setdefault(category, []).append((name, float(probabilities[index])))
        for tags in result.values():
            tags.sort(key=lambda item: item[1], reverse=True)
        return result
