import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from camie_tagger import doctor
from camie_tagger.devices import DeviceError, DeviceSelection
from camie_tagger.immich import ImmichError
from camie_tagger.model import ModelError
from camie_tagger.sidecar import ExiftoolError


@pytest.fixture
def healthy_checks(monkeypatch):
    checks = {}
    for name in (
        "_check_onnxruntime", "_check_device", "_check_exiftool",
        "_check_model", "_check_scan_dirs", "_check_immich",
    ):
        check = Mock(return_value=0)
        monkeypatch.setattr(doctor, name, check)
        checks[name] = check
    return checks


@pytest.fixture
def immich_client(settings, monkeypatch):
    settings.immich_url = "https://immich.example"
    settings.immich_api_key = "test-key"
    settings.library_ids = ["library-id"]
    client = Mock(spec=doctor.ImmichClient)
    client.libraries.return_value = [
        {"id": "library-id", "name": "Art", "assetCount": 3, "importPaths": ["/images"]}
    ]
    monkeypatch.setattr(doctor, "ImmichClient", Mock(return_value=client))
    return client


def test_doctor_succeeds_when_all_checks_pass(settings, healthy_checks):
    assert doctor.run_doctor(settings) == 0
    for check in healthy_checks.values():
        check.assert_called_once()


@pytest.mark.parametrize("failed_check", [
    "_check_onnxruntime", "_check_device", "_check_exiftool",
    "_check_model", "_check_scan_dirs", "_check_immich",
])
def test_doctor_returns_failure_and_still_runs_all_checks(settings, healthy_checks, failed_check):
    healthy_checks[failed_check].return_value = 1
    assert doctor.run_doctor(settings) == 1
    for check in healthy_checks.values():
        check.assert_called_once()


def test_doctor_reports_total_problem_count(settings, healthy_checks, caplog):
    healthy_checks["_check_scan_dirs"].return_value = 2
    healthy_checks["_check_model"].return_value = 1
    assert doctor.run_doctor(settings) == 1
    assert "3 problem(s) found" in caplog.text


def test_missing_onnxruntime_is_a_diagnostic_problem(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    log = Mock()
    assert doctor._check_onnxruntime(log) == 1
    log.error.assert_called_once()


def test_onnxruntime_version_and_providers_are_reported(monkeypatch):
    runtime = SimpleNamespace(
        __version__="test-version",
        get_available_providers=lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    log = Mock()
    assert doctor._check_onnxruntime(log) == 0
    log.info.assert_any_call("onnxruntime: %s", "test-version")
    log.info.assert_any_call("Available providers: %s", "CPUExecutionProvider")


@pytest.mark.parametrize("failed", [False, True])
def test_device_check_reports_selection_or_error(settings, monkeypatch, failed):
    monkeypatch.setattr(doctor.devices, "render_nodes", lambda: [])
    monkeypatch.setattr(doctor.devices, "detect_vendors", lambda: [])
    selection = Mock(return_value=DeviceSelection(
        providers=["CPUExecutionProvider"], notes=["Using CPU fallback"]
    ))
    if failed:
        selection.side_effect = DeviceError("GPU unavailable")
    monkeypatch.setattr(doctor.devices, "select_device", selection)
    settings.fail_on_cpu_fallback = True
    log = Mock()
    assert doctor._check_device(log, settings) == int(failed)
    selection.assert_called_once_with(settings.device, True)
    if failed:
        log.error.assert_called_once()
    else:
        log.warning.assert_any_call("Using CPU fallback")
        log.info.assert_any_call("Selected device: %s", "CPU via CPUExecutionProvider")


@pytest.mark.parametrize("error", [ExiftoolError("bad executable"), FileNotFoundError("missing")])
def test_missing_or_broken_exiftool_is_a_problem(settings, monkeypatch, error):
    monkeypatch.setattr(doctor.sidecar, "exiftool_version", Mock(side_effect=error))
    assert doctor._check_exiftool(Mock(), settings) == 1


def test_exiftool_version_is_reported(settings, monkeypatch):
    version = Mock(return_value="13.00")
    monkeypatch.setattr(doctor.sidecar, "exiftool_version", version)
    log = Mock()
    assert doctor._check_exiftool(log, settings) == 0
    version.assert_called_once_with(settings.exiftool)
    log.info.assert_called_once_with("exiftool: %s", "13.00")


def test_missing_model_is_a_problem(settings, monkeypatch):
    finder = Mock(side_effect=ModelError("Model files missing"))
    monkeypatch.setattr(doctor, "find_model_files", finder)
    assert doctor._check_model(Mock(), settings) == 1
    finder.assert_called_once_with(settings.model_dir)


def test_model_file_details_are_reported_without_loading_weights(settings, tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    metadata = tmp_path / "metadata.json"
    model.write_bytes(b"placeholder")
    metadata.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(doctor, "find_model_files", Mock(return_value=(model, metadata)))
    log = Mock()
    assert doctor._check_model(log, settings) == 0
    log.info.assert_any_call("Metadata: %s", "metadata.json")


@pytest.mark.parametrize("names, expected", [
    ([], 1), (["images"], 0), (["images", "missing"], 1), (["missing", "other"], 2),
])
def test_scan_directory_check_counts_missing_directories(settings, tmp_path, names, expected):
    settings.scan_dirs = [tmp_path / name for name in names]
    assert doctor._check_scan_dirs(Mock(), settings) == expected


@pytest.mark.parametrize("missing_field", ["immich_url", "immich_api_key"])
def test_unconfigured_immich_is_skipped(settings, immich_client, missing_field):
    setattr(settings, missing_field, "")
    assert doctor._check_immich(Mock(), settings) == 0
    doctor.ImmichClient.assert_not_called()


def test_configured_immich_and_library_are_checked(settings, immich_client):
    assert doctor._check_immich(Mock(), settings) == 0
    doctor.ImmichClient.assert_called_once_with(settings.immich_url, settings.immich_api_key)
    immich_client.ping.assert_called_once_with()
    immich_client.libraries.assert_called_once_with()


@pytest.mark.parametrize("endpoint", ["ping", "libraries"])
def test_immich_connection_and_permission_failures_are_problems(settings, immich_client, endpoint):
    getattr(immich_client, endpoint).side_effect = ImmichError("Unavailable or forbidden")
    assert doctor._check_immich(Mock(), settings) == 1
    if endpoint == "ping":
        immich_client.libraries.assert_not_called()


def test_unknown_immich_library_ids_are_counted(settings, immich_client):
    settings.library_ids = ["library-id", "missing-first", "missing-second"]
    assert doctor._check_immich(Mock(), settings) == 2


def test_empty_library_configuration_is_a_warning_not_an_error(settings, immich_client):
    settings.library_ids = []
    log = Mock()
    assert doctor._check_immich(log, settings) == 0
    log.warning.assert_called_once()