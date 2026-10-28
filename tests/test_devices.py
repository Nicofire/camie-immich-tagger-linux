import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from camie_tagger import devices


@pytest.fixture
def hardware(monkeypatch):
    hardware = SimpleNamespace(providers=[devices.CPU_PROVIDER], vendors=[], nodes=[])
    monkeypatch.setattr(devices, "available_providers", lambda: hardware.providers)
    monkeypatch.setattr(devices, "detect_vendors", lambda: hardware.vendors)
    monkeypatch.setattr(devices, "render_nodes", lambda: hardware.nodes)
    return hardware


def test_detect_vendors_deduplicates_and_orders_known_pci_ids(tmp_path, monkeypatch):
    files = []
    for index, value in enumerate(["0x8086\n", "0X10DE", "0x1002", "0x8086", "0xffff"]):
        path = tmp_path / str(index)
        path.write_text(value, encoding="utf-8")
        files.append(str(path))
    files.append(str(tmp_path / "missing"))
    scan = Mock(return_value=files)
    monkeypatch.setattr(devices.glob, "glob", scan)
    assert devices.detect_vendors() == ["nvidia", "intel", "amd"]
    scan.assert_called_once_with("/sys/class/drm/card*/device/vendor")


def test_detect_vendors_without_drm_devices(monkeypatch):
    monkeypatch.setattr(devices.glob, "glob", lambda pattern: [])
    assert devices.detect_vendors() == []


def test_render_nodes_are_sorted_paths(monkeypatch):
    scan = Mock(return_value=["/dev/dri/renderD129", "/dev/dri/renderD128"])
    monkeypatch.setattr(devices.glob, "glob", scan)
    assert devices.render_nodes() == [Path("/dev/dri/renderD128"), Path("/dev/dri/renderD129")]
    scan.assert_called_once_with("/dev/dri/renderD*")


def test_available_providers_uses_installed_runtime(monkeypatch):
    runtime = SimpleNamespace(get_available_providers=lambda: [devices.CPU_PROVIDER])
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    assert devices.available_providers() == [devices.CPU_PROVIDER]


def test_missing_onnxruntime_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(devices.DeviceError, match="onnxruntime is not installed"):
        devices.available_providers()


def test_explicit_cpu_does_not_probe_runtime_or_hardware(monkeypatch):
    for name in ("available_providers", "detect_vendors", "render_nodes"):
        monkeypatch.setattr(devices, name, Mock(side_effect=AssertionError("Unexpected probe")))
    selection = devices.select_device("CPU", fail_on_cpu_fallback=True)
    assert selection.providers == [devices.CPU_PROVIDER]
    assert selection.vendor == "cpu"
    assert selection.using_gpu is False
    assert selection.describe() == "CPU via CPUExecutionProvider"


def test_intel_uses_openvino_with_gpu_cpu_and_fp16_options(hardware):
    hardware.vendors = ["intel"]
    hardware.providers.append("OpenVINOExecutionProvider")
    hardware.nodes = [Path("/dev/dri/renderD128")]
    selection = devices.select_device("intel")
    assert selection.providers == [
        ("OpenVINOExecutionProvider", {"device_type": "AUTO:GPU,CPU", "precision": "FP16"}),
        devices.CPU_PROVIDER,
    ]
    assert selection.vendor == "intel"
    assert selection.using_gpu is True
    assert selection.provider_names == ["OpenVINOExecutionProvider", devices.CPU_PROVIDER]
    assert selection.describe() == "INTEL via OpenVINOExecutionProvider, CPUExecutionProvider"


@pytest.mark.parametrize("providers, expected", [
    (["ROCMExecutionProvider", "MIGraphXExecutionProvider"], "ROCMExecutionProvider"),
    (["MIGraphXExecutionProvider"], "MIGraphXExecutionProvider"),
])
def test_amd_selects_preferred_available_provider(hardware, providers, expected):
    hardware.vendors = ["amd"]
    hardware.providers += providers
    hardware.nodes = [Path("/dev/dri/renderD128")]
    selection = devices.select_device("amd")
    assert selection.providers == [expected, devices.CPU_PROVIDER]
    assert selection.using_gpu is True


def test_auto_prefers_nvidia_without_requiring_drm_render_nodes(hardware):
    hardware.vendors = ["nvidia", "intel"]
    hardware.providers += ["CUDAExecutionProvider", "OpenVINOExecutionProvider"]
    selection = devices.select_device()
    assert selection.vendor == "nvidia"
    assert selection.providers == ["CUDAExecutionProvider", devices.CPU_PROVIDER]


def test_auto_tries_next_vendor_if_preferred_provider_is_missing(hardware):
    hardware.vendors = ["nvidia", "intel"]
    hardware.providers.append("OpenVINOExecutionProvider")
    hardware.nodes = [Path("/dev/dri/renderD128")]
    selection = devices.select_device()
    assert selection.vendor == "intel"
    assert any("no matching execution provider" in note for note in selection.notes)


@pytest.mark.parametrize("vendor, provider", [
    ("intel", "OpenVINOExecutionProvider"),
    ("amd", "ROCMExecutionProvider"),
])
def test_missing_render_node_falls_back_to_cpu(hardware, vendor, provider):
    hardware.vendors = [vendor]
    hardware.providers.append(provider)
    selection = devices.select_device(vendor)
    assert selection.providers == [devices.CPU_PROVIDER]
    assert selection.using_gpu is False
    assert any("renderD* is missing" in note for note in selection.notes)
    with pytest.raises(devices.DeviceError, match="--fail-on-cpu-fallback"):
        devices.select_device(vendor, fail_on_cpu_fallback=True)


@pytest.mark.parametrize("requested", ["auto", "intel", "nvidia", "amd"])
def test_unavailable_gpu_falls_back_or_fails_when_requested(hardware, requested):
    selection = devices.select_device(requested)
    assert selection.providers == [devices.CPU_PROVIDER]
    assert selection.using_gpu is False
    assert any("Falling back to CPU" in note for note in selection.notes)
    with pytest.raises(devices.DeviceError, match="GPU acceleration unavailable"):
        devices.select_device(requested, fail_on_cpu_fallback=True)


def test_explicit_vendor_can_be_used_without_pci_detection(hardware):
    hardware.providers.append("CUDAExecutionProvider")
    selection = devices.select_device("nvidia")
    assert selection.using_gpu is True
    assert selection.vendor == "nvidia"
    assert any("trying anyway" in note for note in selection.notes)


def test_invalid_device_is_rejected_before_probing(monkeypatch):
    probe = Mock(side_effect=AssertionError("Unexpected runtime probe"))
    monkeypatch.setattr(devices, "available_providers", probe)
    with pytest.raises(devices.DeviceError, match="Unknown device"):
        devices.select_device("unsupported")
    probe.assert_not_called()