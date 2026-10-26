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
    if any("\n" in argument or "\r" in argument for argument in arg_lines):
        raise ExiftoolError("exiftool arguments must not contain line breaks")
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
        if result.returncode:
            detail = (result.stderr or result.stdout or "no diagnostic").strip()
            raise ExiftoolError(f"exiftool exited with {result.returncode}: {detail[:200]}")
        return result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired as exc:
        raise ExiftoolError(f"exiftool timed out after {timeout}s") from exc
    except OSError as exc:
        raise ExiftoolError(f"Could not run exiftool: {exc}") from exc
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


def _parse_entries(stdout: str) -> list[dict]:
    try:
        entries = json.loads(stdout)
    except ValueError as exc:
        raise ExiftoolError("Invalid JSON from exiftool") from exc
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ExiftoolError("Expected a JSON array of metadata objects from exiftool")
    for entry in entries:
        if "Error" in entry:
            raise ExiftoolError(f"Could not read sidecar: {entry['Error']}")
    return entries


def _parse_taglist(stdout: str) -> list[str]:
    entries = _parse_entries(stdout)
    if len(entries) != 1:
        raise ExiftoolError("Expected metadata for exactly one sidecar")
    return _coerce_taglist(entries[0].get("TagsList"))


def _coerce_taglist(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list) and all(isinstance(tag, str) for tag in value):
        return value
    raise ExiftoolError("Invalid TagsList value in sidecar metadata")


def _verify_write(
    exiftool: str, image_path: Path, expected: set[str], removed: set[str]
) -> None:
    actual = set(read_taglist(exiftool, image_path))
    if not expected.issubset(actual) or removed.intersection(actual):
        raise ExiftoolError("Sidecar tags do not match the requested update")


def write_taglist(exiftool: str, image_path: Path, taglist: list[str]) -> str:
    """Merge taglist into the sidecar. Returns '+N', 'NO_CHANGE' or 'ERR:...'.

    Only tags that are not already present are appended, so manual tags added in
    Immich survive and repeated runs are idempotent.
    """
    try:
        sidecar = sidecar_path(image_path)
        existing = set(read_taglist(exiftool, image_path))
        new_tags = [tag for tag in dict.fromkeys(taglist) if tag and tag not in existing]
        if not new_tags:
            return "NO_CHANGE"

        if sidecar.exists():
            arg_lines = ["-overwrite_original"]
            arg_lines += [f"-{TAGS_FIELD}+={tag}" for tag in new_tags]
            arg_lines.append(str(sidecar))
        else:
            arg_lines = [f"-{TAGS_FIELD}+={tag}" for tag in new_tags]
            arg_lines += ["-o", str(sidecar)]

        run_exiftool(exiftool, arg_lines)
        _verify_write(exiftool, image_path, existing.union(new_tags), set())
    except ExiftoolError as exc:
        return f"ERR:{exc}"
    return f"+{len(new_tags)}"


def update_taglist(
    exiftool: str, image_path: Path, add: list[str], remove: list[str]
) -> str:
    """Add and remove tags in one exiftool call. Returns '+N-M', 'NO_CHANGE' or 'ERR:...'.

    Unlike write_taglist this can delete values, so callers must restrict `remove` to
    tags they are authoritative for.
    """
    try:
        sidecar_file = sidecar_path(image_path)
        existing = set(read_taglist(exiftool, image_path))
        new_tags = [tag for tag in dict.fromkeys(add) if tag and tag not in existing]
        drop_tags = [tag for tag in dict.fromkeys(remove) if tag and tag in existing]

        if not new_tags and not drop_tags:
            return "NO_CHANGE"
        if not sidecar_file.exists():
            return write_taglist(exiftool, image_path, new_tags)

        arg_lines = ["-overwrite_original"]
        arg_lines += [f"-{TAGS_FIELD}-={tag}" for tag in drop_tags]
        arg_lines += [f"-{TAGS_FIELD}+={tag}" for tag in new_tags]
        arg_lines.append(str(sidecar_file))

        run_exiftool(exiftool, arg_lines)
        _verify_write(
            exiftool, image_path, existing.difference(drop_tags).union(new_tags), set(drop_tags)
        )
    except ExiftoolError as exc:
        return f"ERR:{exc}"
    return f"+{len(new_tags)}-{len(drop_tags)}"


def scan_directory_taglists(exiftool: str, directory: Path) -> dict[Path, list[str]]:
    """Read every sidecar under a directory in a single exiftool call."""
    stdout, _ = run_exiftool(
        exiftool,
        ["-j", f"-{TAGS_FIELD}", "-r", "-ext", "xmp", str(directory)],
        timeout=SCAN_TIMEOUT,
    )
    entries = _parse_entries(stdout) if stdout.strip() else []

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
