"""Image discovery and the tagging run."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from . import sidecar
from .config import ConfigError, IMAGE_EXTENSIONS, Settings, validate_category_thresholds, validate_threshold
from .devices import DeviceSelection, select_device
from .logging_setup import get_logger
from .model import PREPROCESSING_VERSION, CamieTagger
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
    reprocess: bool = False,
    allow_tag_removal: bool = False,
) -> RunResult:
    log = get_logger()
    if reprocess and mode not in {"all", "test"}:
        raise ConfigError("--reprocess requires --mode all or --mode test")
    if allow_tag_removal and not reprocess:
        raise ConfigError("Tag removal requires --reprocess")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ConfigError("Image limit must be a positive integer")
    scan_dirs = settings.require_scan_dirs()
    threshold = validate_threshold(threshold if threshold is not None else settings.threshold)
    category_thresholds = validate_category_thresholds(settings.category_thresholds)

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

    pending = images if reprocess else [p for p in images if p not in already_tagged]
    result.skipped = len(images) - len(pending)

    if not pending:
        log.info("All selected images are already tagged.")
        if not dry_run:
            _rebuild_processed(state, images)
            state.save_processed()
        return result

    selection = select_device(settings.device, settings.fail_on_cpu_fallback)
    _log_selection(selection)
    tagger = CamieTagger(settings.model_dir, selection)
    profile = {
        "model_fingerprint": tagger.fingerprint,
        "preprocessing_version": PREPROCESSING_VERSION,
        "thresholds": {"default": threshold, **category_thresholds},
    }

    started = time.time()
    processed_now: list[Path] = list(set(images) - set(pending))

    for index, image in enumerate(pending, start=1):
        try:
            prediction = tagger.predict(image, threshold, category_thresholds)
            taglist = build_taglist(prediction)
            before = set(existing.get(image, []))
            owned = state.managed_tags(image)
            new_tags = [tag for tag in taglist if tag not in before]
            to_remove = (
                sorted(state.managed_tags(image, source="model").intersection(before).difference(taglist))
                if allow_tag_removal else []
            )
            if dry_run:
                log.info(
                    "[dry run] %s -> %d predicted tags, add=%s remove=%s",
                    image.name, len(taglist), new_tags, to_remove,
                )
                result.tagged += 1
                continue

            if to_remove:
                outcome = sidecar.update_taglist(settings.exiftool, image, new_tags, to_remove)
            else:
                outcome = sidecar.write_taglist(settings.exiftool, image, taglist)
            if outcome.startswith("ERR:"):
                result.failed += 1
                log.error("%s: %s", image, outcome[4:])
            else:
                result.tagged += 1
                processed_now.append(image)
                state.update_managed_tags(
                    image, new_tags, sorted(owned.difference(before).union(to_remove)), source="model"
                )
                state.mark_processed(image, len(taglist), outcome, profile=profile)
        except Exception as exc:
            result.failed += 1
            log.error("%s: %s", image, exc)

        if index % PROGRESS_EVERY == 0 or index == len(pending):
            if not dry_run:
                state.save_processed()
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
        _rebuild_processed(state, processed_now)
        state.save_processed()

    log.info("Done: %s", result.summary())
    return result


def _rebuild_processed(state, images: list[Path]) -> None:
    """Record selected images without discarding records outside this run."""
    for image in images:
        if not state.is_processed(image):
            state.mark_processed(image, 0, "PRESENT")


def _log_selection(selection: DeviceSelection) -> None:
    log = get_logger()
    for note in selection.notes:
        if selection.using_gpu:
            log.info(note)
        else:
            log.warning(note)
    log.info("Inference device: %s", selection.describe())
