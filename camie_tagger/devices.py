"""GPU detection and ONNX Runtime execution provider selection.

Vendor detection reads the PCI vendor ID exposed by the DRM subsystem, which works
inside an LXC as long as /dev/dri has been passed through to the container.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

DEVICE_CHOICES = ("auto", "intel", "nvidia", "amd", "cpu")

PCI_VENDORS = {
    "0x8086": "intel",
    "0x1002": "amd",
    "0x10de": "nvidia",
}

# Preference order when several GPUs are present.
VENDOR_PRIORITY = ("nvidia", "intel", "amd")

VENDOR_PROVIDERS = {
    "nvidia": ("CUDAExecutionProvider",),
    "intel": ("OpenVINOExecutionProvider",),
    "amd": ("ROCMExecutionProvider", "MIGraphXExecutionProvider"),
}

CPU_PROVIDER = "CPUExecutionProvider"

VENDOR_HINTS = {
    "nvidia": "Install the CUDA build: pip install onnxruntime-gpu",
    "intel": "Install the OpenVINO build: pip install onnxruntime-openvino",
    "amd": (
        "ONNX Runtime has no prebuilt AMD GPU wheel on PyPI. AMD integrated GPUs are "
        "not supported; discrete cards require a ROCm build compiled from source."
    ),
}


class DeviceError(RuntimeError):
    """Raised when the requested accelerator cannot be used."""


@dataclass
class DeviceSelection:
    providers: list[object] = field(default_factory=list)
    vendor: str = "cpu"
    using_gpu: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def provider_names(self) -> list[str]:
        return [p[0] if isinstance(p, tuple) else p for p in self.providers]

    def describe(self) -> str:
        target = self.vendor.upper() if self.using_gpu else "CPU"
        return f"{target} via {', '.join(self.provider_names)}"


def render_nodes() -> list[Path]:
    return [Path(p) for p in sorted(glob.glob("/dev/dri/renderD*"))]


def detect_vendors() -> list[str]:
    """Return GPU vendors present on this machine, most preferred first."""
    found: set[str] = set()
    for vendor_file in sorted(glob.glob("/sys/class/drm/card*/device/vendor")):
        try:
            vendor_id = Path(vendor_file).read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        vendor = PCI_VENDORS.get(vendor_id)
        if vendor:
            found.add(vendor)
    return [v for v in VENDOR_PRIORITY if v in found]


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on install variant
        raise DeviceError(
            "onnxruntime is not installed. Run install.sh, or install a variant manually "
            "(onnxruntime-openvino, onnxruntime-gpu or onnxruntime)."
        ) from exc
    return list(ort.get_available_providers())


def _provider_entry(name: str) -> object:
    if name == "OpenVINOExecutionProvider":
        # AUTO lets OpenVINO pick GPU and fall back internally on unsupported operators.
        return (name, {"device_type": "AUTO:GPU,CPU", "precision": "FP16"})
    return name


def _first_supported(vendor: str, installed: list[str]) -> str | None:
    for provider in VENDOR_PROVIDERS.get(vendor, ()):
        if provider in installed:
            return provider
    return None


def select_device(requested: str = "auto", fail_on_cpu_fallback: bool = False) -> DeviceSelection:
    """Resolve the execution providers to use for inference."""
    requested = (requested or "auto").lower()
    if requested not in DEVICE_CHOICES:
        raise DeviceError(
            f"Unknown device {requested!r}. Choose one of: {', '.join(DEVICE_CHOICES)}"
        )

    selection = DeviceSelection()

    if requested == "cpu":
        selection.providers = [CPU_PROVIDER]
        selection.notes.append("CPU execution requested explicitly.")
        return selection

    installed = available_providers()
    present = detect_vendors()
    nodes = render_nodes()

    candidates = present if requested == "auto" else [requested]
    if requested != "auto" and requested not in present:
        selection.notes.append(
            f"No {requested.upper()} GPU found in /sys/class/drm; trying anyway."
        )

    for vendor in candidates:
        provider = _first_supported(vendor, installed)
        if not provider:
            hint = VENDOR_HINTS.get(vendor, "")
            selection.notes.append(
                f"{vendor.upper()} GPU detected but no matching execution provider is "
                f"installed. {hint}".strip()
            )
            continue
        if vendor in {"intel", "amd"} and not nodes:
            selection.notes.append(
                f"{vendor.upper()} GPU detected but /dev/dri/renderD* is missing. "
                "Pass the render node through to the LXC and add the user to the "
                "'render' group."
            )
            continue
        selection.providers = [_provider_entry(provider), CPU_PROVIDER]
        selection.vendor = vendor
        selection.using_gpu = True
        return selection

    if not present:
        selection.notes.append("No supported GPU found in /sys/class/drm.")

    if fail_on_cpu_fallback:
        raise DeviceError(
            "GPU acceleration unavailable and --fail-on-cpu-fallback was set. "
            + " ".join(selection.notes)
        )

    selection.providers = [CPU_PROVIDER]
    selection.notes.append("Falling back to CPU. Tagging will be significantly slower.")
    return selection
