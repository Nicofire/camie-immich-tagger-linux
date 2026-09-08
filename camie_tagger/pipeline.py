"""Image discovery and the tagging run."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from . import sidecar
from .config import IMAGE_EXTENSIONS, Settings
from .devices import DeviceSelection, select_device
from .logging_setup import get_logger
from .model import CamieTagger
from .tags import build_taglist, has_camie_tags

TEST_LIMIT = 20
PROGRESS_EVERY = 50


@dataclass
class RunResult:
    total: int = 0
    tagged: int = 0
    skipped: int = 0
    failed: int = 0
    elapsed: float = 0.0

    def summary(self) -> str:
        return (
            f"tagged={self.tagged} skipped={self.skipped} failed={self.failed} "
            f"of {self.total} in {self.elapsed:.0f}s"
        )


def collect_images(scan_dirs: list[Path]) -> list[Path]:
    images: list[Path] = []
    for directory in scan_dirs:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)
    return images


def run_tagging(
    settings: Settings,
    state,
    mode: str = "recent",
    limit: int | None = None,
    threshold: float | None = None,
    dry_run: bool = False,
) -> RunResult:
    log = get_logger()
    scan_dirs = settings.require_scan_dirs()
    threshold = threshold if threshold is not None else settings.threshold

    log.info("Mode=%s threshold=%.2f%s", mode, threshold, " (dry run)" if dry_run else "")

    images = collect_images(scan_dirs)
    log.info("Found %d images", len(images))

    if mode == "recent":
        images = [p for p in images if not state.is_processed(p)]
        log.info("%d images are not in the processed list", len(images))

    # One exiftool call per directory instead of one per image.
    log.info("Reading existing sidecars...")
    existing = sidecar.scan_taglists(settings.exiftool, scan_dirs)
    already_tagged = {path for path, tags in existing.items() if has_camie_tags(tags)}
    log.info("%d images already carry tags from this tool", len(already_tagged))

    if mode == "test":
        images = images[: (limit or TEST_LIMIT)]
    elif limit:
        images = images[:limit]

    result = RunResult(total=len(images))
    if not images:
        log.info("Nothing to do.")
        return result

    pending = [p for p in images if p not in already_tagged]
    result.skipped = len(images) - len(pending)

    if not pending:
        log.info("All selected images are already tagged.")
        if mode == "all":
            _rebuild_processed(state, images)
        return result

    selection = select_device(settings.device, settings.fail_on_cpu_fallback)
    _log_selection(selection)
    tagger = CamieTagger(settings.model_dir, selection)

    started = time.time()
    processed_now: list[Path] = list(set(images) - set(pending))

    for index, image in enumerate(pending, start=1):
        try:
            prediction = tagger.predict(image, threshold)
            taglist = build_taglist(prediction)
            if dry_run:
                log.info("[dry run] %s -> %d tags", image.name, len(taglist))
                result.tagged += 1
                continue

            outcome = sidecar.write_taglist(settings.exiftool, image, taglist)
            if outcome.startswith("ERR:"):
                result.failed += 1
                log.error("%s: %s", image, outcome[4:])
            else:
                result.tagged += 1
                processed_now.append(image)
                state.mark_processed(image, len(taglist), outcome)
        except Exception as exc:
            result.failed += 1
            log.error("%s: %s", image, exc)

        if index % PROGRESS_EVERY == 0 or index == len(pending):
            rate = index / max(time.time() - started, 1e-6)
            remaining = (len(pending) - index) / rate if rate else 0
            log.info(
                "%d/%d tagged=%d failed=%d (%.1f img/s, %.0fs left)",
                index,
                len(pending),
                result.tagged,
                result.failed,
                rate,
                remaining,
            )

    result.elapsed = time.time() - started

    if not dry_run:
        if mode == "all":
            _rebuild_processed(state, processed_now)
        state.save_processed()

    log.info("Done: %s", result.summary())
    return result


def _rebuild_processed(state, images: list[Path]) -> None:
    """After a full run the processed list is authoritative, so rewrite it."""
    existing = state.processed
    rebuilt = {}
    for image in images:
        key = str(image)
        rebuilt[key] = existing.get(key, {"tags": 0, "result": "PRESENT"})
    state.replace_processed(rebuilt)


def _log_selection(selection: DeviceSelection) -> None:
    log = get_logger()
    for note in selection.notes:
        if selection.using_gpu:
            log.info(note)
        else:
            log.warning(note)
    log.info("Inference device: %s", selection.describe())
