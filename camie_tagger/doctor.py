"""Environment diagnostics."""

from __future__ import annotations

import platform
import sys

from . import devices, sidecar
from .config import Settings
from .immich import ImmichClient, ImmichError
from .logging_setup import get_logger
from .model import ModelError, find_model_files


def run_doctor(settings: Settings) -> int:
    log = get_logger()
    problems = 0

    log.info("Python: %s (%s)", platform.python_version(), sys.executable)
    log.info("Config file: %s", settings.env_file or "none (using defaults)")

    problems += _check_onnxruntime(log)
    problems += _check_device(log, settings)
    problems += _check_exiftool(log, settings)
    problems += _check_model(log, settings)
    problems += _check_scan_dirs(log, settings)
    problems += _check_immich(log, settings)

    if problems:
        log.error("%d problem(s) found.", problems)
        return 1
    log.info("All checks passed.")
    return 0


def _check_onnxruntime(log) -> int:
    try:
        import onnxruntime as ort
    except ImportError:
        log.error("onnxruntime is not installed. Run install.sh.")
        return 1
    log.info("onnxruntime: %s", ort.__version__)
    log.info("Available providers: %s", ", ".join(ort.get_available_providers()))
    return 0


def _check_device(log, settings: Settings) -> int:
    nodes = devices.render_nodes()
    log.info("Render nodes: %s", ", ".join(str(n) for n in nodes) or "none")
    if not nodes:
        log.warning(
            "No /dev/dri render node. In an LXC the host must pass it through for "
            "Intel or AMD GPU acceleration."
        )
    vendors = devices.detect_vendors()
    log.info("Detected GPU vendors: %s", ", ".join(vendors) or "none")

    try:
        selection = devices.select_device(settings.device, settings.fail_on_cpu_fallback)
    except devices.DeviceError as exc:
        log.error("Device selection failed: %s", exc)
        return 1
    for note in selection.notes:
        log.warning(note)
    log.info("Selected device: %s", selection.describe())
    return 0


def _check_exiftool(log, settings: Settings) -> int:
    try:
        version = sidecar.exiftool_version(settings.exiftool)
    except (sidecar.ExiftoolError, OSError) as exc:
        log.error("exiftool: %s", exc)
        return 1
    log.info("exiftool: %s", version)
    return 0


def _check_model(log, settings: Settings) -> int:
    try:
        onnx_path, meta_path = find_model_files(settings.model_dir)
    except ModelError as exc:
        log.error("%s", exc)
        return 1
    log.info("Model: %s (%.0f MB)", onnx_path.name, onnx_path.stat().st_size / 1e6)
    log.info("Metadata: %s", meta_path.name)
    return 0


def _check_scan_dirs(log, settings: Settings) -> int:
    if not settings.scan_dirs:
        log.error("CAMIE_SCAN_DIRS is not set.")
        return 1
    problems = 0
    for directory in settings.scan_dirs:
        if directory.is_dir():
            log.info("Scan directory: %s", directory)
        else:
            log.error("Scan directory missing: %s", directory)
            problems += 1
    return problems


def _check_immich(log, settings: Settings) -> int:
    if not settings.immich_url or not settings.immich_api_key:
        log.warning("Immich URL or API key not configured; skipping Immich checks.")
        return 0

    client = ImmichClient(settings.immich_url, settings.immich_api_key)
    try:
        client.ping()
        log.info("Immich reachable: %s", settings.immich_url)
    except ImmichError as exc:
        log.error("Immich unreachable: %s", exc)
        return 1

    try:
        libraries = client.libraries()
    except ImmichError as exc:
        log.error("Could not list libraries (check the API key permissions): %s", exc)
        return 1

    known = {lib.get("id"): lib for lib in libraries}
    log.info("Immich libraries:")
    for library in libraries:
        log.info(
            "  %s  name=%s assets=%s paths=%s",
            library.get("id"),
            library.get("name"),
            library.get("assetCount"),
            ", ".join(library.get("importPaths") or []),
        )

    problems = 0
    if not settings.library_ids:
        log.warning("IMMICH_LIBRARY_IDS is empty; library scans will be skipped.")
    for library_id in settings.library_ids:
        if library_id in known:
            log.info("Configured library %s found.", library_id)
        else:
            log.error(
                "Configured library %s does not exist. Use the UUID, not the name.",
                library_id,
            )
            problems += 1
    return problems
