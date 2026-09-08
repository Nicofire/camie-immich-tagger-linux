"""Statistics and cleanup commands."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import sidecar
from .config import IMAGE_EXTENSIONS, Settings
from .immich import ImmichClient, ImmichError
from .logging_setup import get_logger
from .tags import CAMIE_ROOTS, tags_in_namespace


def show_stats(settings: Settings) -> int:
    log = get_logger()
    scan_dirs = settings.require_scan_dirs()
    taglists = sidecar.scan_taglists(settings.exiftool, scan_dirs)

    total = len(taglists)
    if not total:
        log.info("No sidecars found.")
        return 0

    with_character = 0
    without_character_real_copyright = 0
    without_anything = 0
    characters: Counter[str] = Counter()
    copyrights: Counter[str] = Counter()
    artists: Counter[str] = Counter()

    for tags in taglists.values():
        found_characters = tags_in_namespace(tags, "character")
        found_copyrights = tags_in_namespace(tags, "copyright")
        characters.update(found_characters)
        copyrights.update(found_copyrights)
        artists.update(tags_in_namespace(tags, "artist"))

        if found_characters:
            with_character += 1
        elif any(tag != "copyright/original" for tag in found_copyrights):
            without_character_real_copyright += 1
        elif not found_copyrights:
            without_anything += 1

    log.info("Sidecars: %d", total)
    log.info(
        "With a character tag: %d (%.1f%%)", with_character, 100 * with_character / total
    )
    log.info(
        "No character but a real copyright (Tier 0 candidates): %d",
        without_character_real_copyright,
    )
    log.info("No character and no copyright: %d", without_anything)

    _log_top(log, "characters", characters)
    _log_top(log, "copyrights", copyrights)
    _log_top(log, "artists", artists)
    return 0


def _log_top(log, label: str, counter: Counter[str], limit: int = 15) -> None:
    if not counter:
        return
    log.info("Top %s:", label)
    for name, count in counter.most_common(limit):
        log.info("  %5d  %s", count, name)


def cleanup_orphans(settings: Settings, confirm: bool = False) -> int:
    """Remove .xmp files whose image no longer exists."""
    log = get_logger()
    scan_dirs = settings.require_scan_dirs()

    orphans: list[Path] = []
    for directory in scan_dirs:
        for candidate in sorted(directory.rglob("*.xmp")):
            if _is_orphan(candidate):
                orphans.append(candidate)

    log.info("Found %d orphan sidecars", len(orphans))
    for path in orphans[:40]:
        log.info("  %s", path)
    if len(orphans) > 40:
        log.info("  ... and %d more", len(orphans) - 40)

    if not orphans:
        return 0
    if not confirm:
        log.info("Dry run. Re-run with --confirm to delete these files.")
        return 0

    deleted = 0
    for path in orphans:
        try:
            path.unlink()
            deleted += 1
        except OSError as exc:
            log.error("Could not delete %s: %s", path, exc)
    log.info("Deleted %d sidecars", deleted)
    return 0


def _is_orphan(sidecar_file: Path) -> bool:
    base = sidecar.image_path_for_sidecar(sidecar_file)
    if base.suffix.lower() in IMAGE_EXTENSIONS:
        return not base.exists()
    # Sidecars named photo.xmp rather than photo.jpg.xmp
    return not any(base.with_suffix(ext).exists() for ext in IMAGE_EXTENSIONS)


def cleanup_tags(settings: Settings, confirm: bool = False) -> int:
    """Delete Immich tags outside this tool's namespaces.

    Anything whose first path segment is not one of character, copyright, artist,
    general or rating is removed. This includes legacy flat tags and the zh/ Chinese
    tags written by earlier versions.
    """
    log = get_logger()
    settings.require_immich()
    client = ImmichClient(settings.immich_url, settings.immich_api_key)

    try:
        tags = client.tags()
    except ImmichError as exc:
        log.error("Could not list tags: %s", exc)
        return 1

    keep, remove = [], []
    for tag in tags:
        value = tag.get("value") or tag.get("name") or ""
        root = value.split("/", 1)[0]
        (keep if root in CAMIE_ROOTS else remove).append(tag)

    log.info("Tags: %d total, %d kept, %d to delete", len(tags), len(keep), len(remove))
    for tag in remove[:40]:
        log.info("  - %s", tag.get("value"))
    if len(remove) > 40:
        log.info("  ... and %d more", len(remove) - 40)

    if not remove:
        return 0
    if not confirm:
        log.info("Dry run. Re-run with --confirm to delete these tags.")
        return 0

    deleted = failed = 0
    for tag in remove:
        try:
            client.delete_tag(tag["id"])
            deleted += 1
        except (ImmichError, KeyError) as exc:
            failed += 1
            log.error("Could not delete %s: %s", tag.get("value"), exc)
    log.info("Deleted %d tags, %d failures", deleted, failed)
    return 0
