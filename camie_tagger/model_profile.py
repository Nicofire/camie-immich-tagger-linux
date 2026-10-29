"""Validated, portable ONNX image-tagger profiles."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .tags import CATEGORIES_TO_WRITE

MODEL_CATEGORIES = frozenset((*CATEGORIES_TO_WRITE, "meta", "year"))
WD_CATEGORIES = {
    "0": "general", "1": "artist", "3": "copyright",
    "4": "character", "5": "meta", "9": "rating",
}


class ModelError(RuntimeError):
    """Raised when a model, vocabulary or inference profile is unusable."""


@dataclass(frozen=True)
class ModelProfile:
    preset: str = "camie"
    model_path: Path | None = None
    labels_path: Path | None = None
    external_data: tuple[Path, ...] = ()
    input_name: str | None = None
    output_name: str | None = "refined_predictions"
    layout: str = "NCHW"
    size: tuple[int, int] | None = (512, 512)
    color_order: str = "RGB"
    resize: str = "letterbox"
    interpolation: str = "lanczos"
    scale: float = 255.0
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    pad_color: tuple[int, int, int] = (124, 116, 104)
    alpha_background: tuple[int, int, int] | None = None
    activation: str = "sigmoid"
    default_category: str = "general"
    category_map: dict[str, str] = field(default_factory=dict)

    def inference_settings(self) -> dict[str, object]:
        values = asdict(self)
        for key in ("model_path", "labels_path", "external_data"):
            del values[key]
        return values


def _choice(value: object, key: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{key} must be one of {', '.join(choices)}")
    return value


def _number(value: object, key: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{key} must be a finite {'positive ' if positive else ''}number")
    return number


def _channels(value: object, key: str, count: int, *, positive: bool = False) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{key} must contain {count} channel values")
    return tuple(_number(item, key, positive=positive) for item in value)


def _color(value: object, key: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple)) or len(value) != 3
        or any(type(item) is not int or not 0 <= item <= 255 for item in value)
    ):
        raise ValueError(f"{key} must contain three RGB integers between 0 and 255")
    return value[0], value[1], value[2]


def _path(value: object, key: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty file path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _optional_name(value: object, key: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{key} must be a nonempty string or null")
    return value


def load_model_profile(path: Path | None = None) -> ModelProfile:
    if path is None:
        return ModelProfile()
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("the profile must be a JSON object")
        allowed = set(ModelProfile.__dataclass_fields__) - {"model_path", "labels_path"}
        unknown = set(data) - allowed - {"model", "labels"}
        if unknown:
            raise ValueError(f"unknown profile option(s): {', '.join(sorted(unknown))}")
        preset = _choice(data.get("preset"), "preset", ("camie", "wd", "generic"))
        base = ModelProfile()
        if preset == "wd":
            base = replace(
                base, preset=preset, output_name=None, layout="NHWC", size=None,
                color_order="BGR", resize="pad_square", interpolation="bicubic",
                scale=1.0, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0),
                pad_color=(255, 255, 255), alpha_background=(255, 255, 255),
                activation="none", category_map=WD_CATEGORIES.copy(),
            )
        elif preset == "generic":
            required = {"layout", "color_order", "resize", "scale", "mean", "std", "activation"}
            missing = required - set(data)
            if missing:
                raise ValueError(f"generic profiles require: {', '.join(sorted(missing))}")
            base = replace(
                base, preset=preset, output_name=None, size=None,
                interpolation="bilinear", pad_color=(255, 255, 255),
                alpha_background=(255, 255, 255),
            )

        model_path = _path(data.get("model"), "model", path.parent)
        labels_path = _path(data.get("labels"), "labels", path.parent)
        extra = data.get("external_data", [])
        if not isinstance(extra, list):
            raise ValueError("external_data must be an array of file paths")
        external_data = tuple(_path(item, "external_data", path.parent) for item in extra)

        size = data.get("size", base.size)
        if size is not None:
            if (
                not isinstance(size, (list, tuple)) or len(size) != 2
                or any(type(item) is not int or item <= 0 for item in size)
            ):
                raise ValueError("size must be [width, height] with positive integers, or null")
            size = (size[0], size[1])
        color_order = _choice(
            data.get("color_order", base.color_order), "color_order", ("RGB", "BGR", "GRAY")
        )
        channels = 1 if color_order == "GRAY" else 3
        category_map = data.get("category_map", {})
        if not isinstance(category_map, dict) or any(
            not isinstance(key, str) or not key.strip()
            or not isinstance(value, str) or value not in MODEL_CATEGORIES
            for key, value in category_map.items()
        ):
            raise ValueError("category_map must map category strings to supported tag categories")
        alpha_background = data.get("alpha_background", base.alpha_background)
        return ModelProfile(
            preset=preset, model_path=model_path, labels_path=labels_path,
            external_data=external_data,
            input_name=_optional_name(data.get("input_name", base.input_name), "input_name"),
            output_name=_optional_name(data.get("output_name", base.output_name), "output_name"),
            layout=_choice(data.get("layout", base.layout), "layout", ("NCHW", "NHWC", "CHW", "HWC")),
            size=size, color_order=color_order,
            resize=_choice(
                data.get("resize", base.resize), "resize",
                ("letterbox", "pad_square", "stretch", "center_crop"),
            ),
            interpolation=_choice(
                data.get("interpolation", base.interpolation), "interpolation",
                ("nearest", "bilinear", "bicubic", "lanczos"),
            ),
            scale=_number(data.get("scale", base.scale), "scale", positive=True),
            mean=_channels(data.get("mean", base.mean), "mean", channels),
            std=_channels(data.get("std", base.std), "std", channels, positive=True),
            pad_color=_color(data.get("pad_color", base.pad_color), "pad_color"),
            alpha_background=(
                None if alpha_background is None else _color(alpha_background, "alpha_background")
            ),
            activation=_choice(
                data.get("activation", base.activation), "activation", ("none", "sigmoid", "softmax")
            ),
            default_category=_choice(
                data.get("default_category", base.default_category),
                "default_category", tuple(sorted(MODEL_CATEGORIES)),
            ),
            category_map={**base.category_map, **category_map},
        )
    except (OSError, UnicodeError, ValueError, OverflowError) as exc:
        raise ModelError(f"Invalid model profile {path}: {exc}") from exc


def find_model_files(model_dir: Path, profile: ModelProfile | None = None) -> tuple[Path, Path]:
    profile = profile or ModelProfile()
    if profile.model_path is not None and profile.labels_path is not None:
        for path in (profile.model_path, profile.labels_path, *profile.external_data):
            if not path.is_file():
                raise ModelError(f"Model profile file not found: {path}")
        return profile.model_path, profile.labels_path
    if not model_dir.is_dir():
        raise ModelError(
            f"Model directory not found: {model_dir}\n"
            "Download the default Camie model or select --model-profile as described in the README."
        )
    onnx_files = sorted(model_dir.rglob("*.onnx"))
    meta_files = sorted(model_dir.rglob("*metadata*.json"))
    for files, description in ((onnx_files, ".onnx"), (meta_files, "*metadata*.json")):
        if not files:
            raise ModelError(f"No {description} file found under {model_dir}")
        if len(files) > 1:
            raise ModelError(
                f"Multiple {description} files found under {model_dir}; "
                "select the exact model and labels with --model-profile"
            )
    return onnx_files[0], meta_files[0]
