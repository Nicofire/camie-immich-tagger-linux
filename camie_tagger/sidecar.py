"""XMP sidecar reading and writing via exiftool.

Immich reads tags from XMP-digiKam:TagsList (not dc:Subject), and expects the sidecar
to keep the image's full name, e.g. photo.jpg -> photo.jpg.xmp.

Arguments are passed through a UTF-8 argfile rather than argv. A full run can append
thousands of tags, which would otherwise exceed the system argument length limit.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .logging_setup import get_logger

TAGS_FIELD = "XMP-digiKam:TagsList"
DEFAULT_TIMEOUT = 120
SCAN_TIMEOUT = 1800


class ExiftoolError(RuntimeError):
    """Raised when exiftool is missing or fails."""


def sidecar_path(image_path: Path) -> Path:
    return Path(f"{image_path}.xmp")


def image_path_for_sidecar(sidecar: Path) -> Path:
    return Path(str(sidecar).removesuffix(".xmp"))


def ensure_exiftool(exiftool: str) -> str:
    resolved = shutil.which(exiftool) or (exiftool if Path(exiftool).is_file() else None)
    if not resolved:
        raise ExiftoolError(
            f"exiftool not found: {exiftool}\n"
            "Install it with: sudo apt-get install -y libimage-exiftool-perl"
        )
    return resolved


def exiftool_version(exiftool: str) -> str:
    binary = ensure_exiftool(exiftool)
    result = subprocess.run(
        [binary, "-ver"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result.stdout.strip()


def run_exiftool(
    exiftool: str, arg_lines: list[str], timeout: int = DEFAULT_TIMEOUT
) -> tuple[str, str]:
    binary = ensure_exiftool(exiftool)
    handle, argfile = tempfile.mkstemp(suffix=".args", prefix="camie-")
    os.close(handle)
    try:
        Path(argfile).write_text("\n".join(arg_lines) + "\n", encoding="utf-8")
        result = subprocess.run(
            [binary, "-charset", "utf8", "-charset", "filename=UTF8", "-@", argfile],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired as exc:
        raise ExiftoolError(f"exiftool timed out after {timeout}s") from exc
    finally:
        try:
            os.unlink(argfile)
        except OSError:
            pass


def read_taglist(exiftool: str, image_path: Path) -> list[str]:
    sidecar = sidecar_path(image_path)
    if not sidecar.exists():
        return []
    stdout, _ = run_exiftool(exiftool, ["-j", f"-{TAGS_FIELD}", str(sidecar)])
    return _parse_taglist(stdout)


def _parse_taglist(stdout: str) -> list[str]:
    try:
        entries = json.loads(stdout)
    except (ValueError, IndexError):
        return []
    if not entries:
        return []
    return _coerce_taglist(entries[0].get("TagsList"))


def _coerce_taglist(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(tag) for tag in value]


def write_taglist(exiftool: str, image_path: Path, taglist: list[str]) -> str:
    """Merge taglist into the sidecar. Returns '+N', 'NO_CHANGE' or 'ERR:...'.

    Only tags that are not already present are appended, so manual tags added in
    Immich survive and repeated runs are idempotent.
    """
    sidecar = sidecar_path(image_path)
    existing = set(read_taglist(exiftool, image_path))
    new_tags = [tag for tag in taglist if tag and tag not in existing]
    if not new_tags:
        return "NO_CHANGE"

    if sidecar.exists():
        arg_lines = ["-overwrite_original"]
        arg_lines += [f"-{TAGS_FIELD}+={tag}" for tag in new_tags]
        arg_lines.append(str(sidecar))
    else:
        arg_lines = [f"-{TAGS_FIELD}+={tag}" for tag in new_tags]
        arg_lines += ["-o", str(sidecar)]

    try:
        stdout, stderr = run_exiftool(exiftool, arg_lines)
    except ExiftoolError as exc:
        return f"ERR:{exc}"

    if not sidecar.exists():
        detail = (stderr or stdout).strip().replace("\n", " ")
        return f"ERR:{detail[:200]}"
    return f"+{len(new_tags)}"


def scan_directory_taglists(exiftool: str, directory: Path) -> dict[Path, list[str]]:
    """Read every sidecar under a directory in a single exiftool call."""
    log = get_logger()
    stdout, stderr = run_exiftool(
        exiftool,
        ["-j", f"-{TAGS_FIELD}", "-r", "-ext", "xmp", str(directory)],
        timeout=SCAN_TIMEOUT,
    )
    try:
        entries = json.loads(stdout) if stdout.strip() else []
    except ValueError:
        log.warning("Could not parse sidecar scan for %s: %s", directory, stderr.strip()[:200])
        return {}

    mapping: dict[Path, list[str]] = {}
    for entry in entries:
        source = entry.get("SourceFile")
        if not source:
            continue
        tags = _coerce_taglist(entry.get("TagsList"))
        mapping[image_path_for_sidecar(Path(source))] = tags
    return mapping


def scan_taglists(exiftool: str, directories: list[Path]) -> dict[Path, list[str]]:
    combined: dict[Path, list[str]] = {}
    for directory in directories:
        combined.update(scan_directory_taglists(exiftool, directory))
    return combined
