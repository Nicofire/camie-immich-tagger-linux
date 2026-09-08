"""Tier 0: SauceNAO reverse search with Danbooru tag backfill.

Tier 1 (the local model) recognises most content but misses specific characters that
are outside its training set. Tier 0 looks those up by reverse image search. The free
SauceNAO tier is rate limited, so this is a slow, resumable background job.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from . import sidecar
from .config import Settings
from .logging_setup import get_logger
from .tags import normalize_tag, tags_in_namespace

SAUCENAO_URL = "https://saucenao.com/search.php"
DANBOORU_POST_URL = "https://danbooru.donmai.us/posts/{post_id}.json"
USER_AGENT = "camie-immich-tagger/1.0 (personal library)"
DANBOORU_POST_PATTERN = re.compile(r"donmai\.us/posts/(\d+)")

SAUCENAO_TIMEOUT = 40
DANBOORU_TIMEOUT = 20
# Stop before the daily allowance is exhausted so the next run still has headroom.
LONG_REMAINING_MARGIN = 3

DANBOORU_TAG_FIELDS = {
    "tag_string_character": "character",
    "tag_string_copyright": "copyright",
    "tag_string_artist": "artist",
}

# Namespaces Danbooru is authoritative for. Verification never touches general/ or
# rating/, which the local model owns and Danbooru results do not describe.
VERIFIABLE_NAMESPACES = ("character", "copyright", "artist")

# Deleting a tag needs more confidence than adding one.
DEFAULT_REPLACE_MIN_SIMILARITY = 95.0


@dataclass
class Tier0Result:
    processed: int = 0
    hits: int = 0
    misses: int = 0
    failed: int = 0
    corrected: int = 0

    def summary(self) -> str:
        return (
            f"processed={self.processed} hits={self.hits} "
            f"misses={self.misses} corrected={self.corrected} failed={self.failed}"
        )


def plan_changes(existing: list[str], danbooru: list[str]) -> tuple[list[str], list[str]]:
    """Return (tags to add, tags to remove) for a verified image."""
    danbooru_set = set(danbooru)
    existing_set = set(existing)
    add = [tag for tag in danbooru if tag not in existing_set]
    remove = [
        tag
        for tag in existing
        if tag.split("/", 1)[0] in VERIFIABLE_NAMESPACES and tag not in danbooru_set
    ]
    return add, remove


def build_queue(settings: Settings, state, include_tagged: bool = False) -> int:
    """Queue images that have a real copyright tag but no character tag.

    With include_tagged the character rule is dropped, so images that already have
    character tags are queued too and can be checked against Danbooru.
    """
    log = get_logger()
    scan_dirs = settings.require_scan_dirs()

    stale = state.prune_queue(scan_dirs)
    if stale:
        log.info("Dropped %d queued images outside the scan directories", stale)

    log.info("Scanning sidecars to build the Tier 0 queue...")
    taglists = sidecar.scan_taglists(settings.exiftool, scan_dirs)

    processed = set(state.tier0_progress)
    candidates: list[Path] = []

    for image, tags in taglists.items():
        if str(image) in processed:
            continue
        characters = tags_in_namespace(tags, "character")
        if characters and not include_tagged:
            continue
        copyrights = tags_in_namespace(tags, "copyright")
        # 'copyright/original' means no source work, so reverse search cannot help.
        real_copyright = any(tag != "copyright/original" for tag in copyrights)
        if real_copyright or (include_tagged and characters):
            candidates.append(image)

    added = state.enqueue_tier0(candidates)
    state.save_queue()
    log.info("Queued %d new images (queue size: %d)", added, len(state.tier0_queue))
    return added


def _search_saucenao(session: requests.Session, api_key: str, image: Path) -> dict:
    with image.open("rb") as handle:
        response = session.post(
            SAUCENAO_URL,
            params={
                "api_key": api_key,
                "output_type": "2",
                "numres": "8",
                "db": "999",
            },
            files={"file": (image.name, handle)},
            timeout=SAUCENAO_TIMEOUT,
        )
    response.raise_for_status()
    return response.json()


def _best_match(results: list[dict], min_similarity: float) -> tuple[float, int] | None:
    for entry in results or []:
        header = entry.get("header", {})
        data = entry.get("data", {})
        try:
            similarity = float(header.get("similarity", 0))
        except (TypeError, ValueError):
            continue
        if similarity < min_similarity:
            continue

        post_id = data.get("danbooru_id")
        if post_id is None:
            for url in data.get("ext_urls", []) or []:
                match = DANBOORU_POST_PATTERN.search(str(url))
                if match:
                    post_id = match.group(1)
                    break
        if post_id is None:
            continue
        try:
            return similarity, int(post_id)
        except (TypeError, ValueError):
            continue
    return None


def _fetch_danbooru_tags(session: requests.Session, post_id: int) -> list[str]:
    response = session.get(
        DANBOORU_POST_URL.format(post_id=post_id),
        headers={"User-Agent": USER_AGENT},
        timeout=DANBOORU_TIMEOUT,
    )
    response.raise_for_status()
    post = response.json()

    taglist: list[str] = []
    for field, namespace in DANBOORU_TAG_FIELDS.items():
        for tag in (post.get(field) or "").split():
            taglist.append(f"{namespace}/{normalize_tag(tag)}")
    return taglist


def _apply_verification(
    settings: Settings,
    state,
    result: Tier0Result,
    image: Path,
    taglist: list[str],
    similarity: float,
    confirm: bool,
    replace_min_similarity: float,
    record_progress: bool,
) -> None:
    """Compare the sidecar against Danbooru, then add missing and drop wrong tags."""
    log = get_logger()
    existing = sidecar.read_taglist(settings.exiftool, image)
    to_add, to_remove = plan_changes(existing, taglist)

    if to_remove and similarity < replace_min_similarity:
        log.info(
            "%s: %.0f%% is below the %.0f%% replace threshold, keeping %d existing tag(s)",
            image.name,
            similarity,
            replace_min_similarity,
            len(to_remove),
        )
        to_remove = []

    if not to_add and not to_remove:
        log.info("ok %.0f%% %s (tags already match)", similarity, image.name)
        if record_progress:
            state.record_tier0(image, f"verified:{similarity:.0f}%")
        return

    if to_remove:
        result.corrected += 1
        log.info("%s: replacing %s", image.name, ", ".join(to_remove))
    if to_add:
        log.info("%s: adding %s", image.name, ", ".join(to_add))

    if not confirm:
        log.info(
            "[dry run] %s would gain %d and lose %d tag(s)",
            image.name,
            len(to_add),
            len(to_remove),
        )
        return

    outcome = sidecar.update_taglist(settings.exiftool, image, to_add, to_remove)
    if outcome.startswith("ERR:"):
        result.failed += 1
        log.error("%s: %s", image, outcome[4:])
        return
    log.info("hit %.0f%% %s [%s]", similarity, image.name, outcome)
    if record_progress:
        state.record_tier0(
            image, f"corrected:{similarity:.0f}%:+{len(to_add)}-{len(to_remove)}"
        )


def run_tier0(
    settings: Settings,
    state,
    limit: int | None = None,
    min_similarity: float | None = None,
    verify: bool = False,
    confirm: bool = False,
    replace_min_similarity: float = DEFAULT_REPLACE_MIN_SIMILARITY,
) -> Tier0Result:
    log = get_logger()
    settings.require_saucenao()

    min_similarity = (
        min_similarity if min_similarity is not None else settings.tier0_min_similarity
    )
    cap = limit or settings.tier0_daily_cap

    progress = state.tier0_progress
    pending = [
        Path(path)
        for path in state.tier0_queue
        if path not in progress and Path(path).is_file()
    ]

    result = Tier0Result()
    if not pending:
        log.info("Tier 0 queue is empty; nothing to look up.")
        return result

    log.info(
        "Tier 0: %d queued, processing up to %d at %.0fs intervals (min similarity %.0f%%)",
        len(pending),
        cap,
        settings.tier0_interval,
        min_similarity,
    )
    if verify:
        log.info(
            "Verify mode: existing %s tags are compared against Danbooru; "
            "replacements need at least %.0f%% similarity.",
            "/".join(VERIFIABLE_NAMESPACES),
            replace_min_similarity,
        )
        if not confirm:
            log.warning(
                "Dry run: nothing is written and progress is NOT recorded, so these "
                "searches will run again. Add --confirm to apply the changes."
            )

    # In a verify dry run the results are not persisted, so the queue stays intact.
    record_progress = confirm or not verify

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    try:
        for image in pending[:cap]:
            try:
                payload = _search_saucenao(session, settings.saucenao_api_key, image)
            except requests.RequestException as exc:
                log.error("SauceNAO request failed for %s: %s", image.name, exc)
                result.failed += 1
                break

            header = payload.get("header", {})
            if header.get("status", -1) != 0:
                log.error("SauceNAO returned status %s; stopping.", header.get("status"))
                break

            result.processed += 1
            match = _best_match(payload.get("results", []), min_similarity)

            if match is None:
                result.misses += 1
                if record_progress:
                    state.record_tier0(image, "miss")
                log.info("miss %s", image.name)
            else:
                similarity, post_id = match
                try:
                    taglist = _fetch_danbooru_tags(session, post_id)
                except requests.RequestException as exc:
                    result.failed += 1
                    if record_progress:
                        state.record_tier0(image, f"danbooru_error:{str(exc)[:60]}")
                    log.error("Danbooru lookup failed for post %s: %s", post_id, exc)
                    taglist = []

                if taglist:
                    result.hits += 1
                    if verify:
                        _apply_verification(
                            settings,
                            state,
                            result,
                            image,
                            taglist,
                            similarity,
                            confirm,
                            replace_min_similarity,
                            record_progress,
                        )
                    else:
                        outcome = sidecar.write_taglist(settings.exiftool, image, taglist)
                        characters = tags_in_namespace(taglist, "character")
                        if record_progress:
                            state.record_tier0(
                                image, f"hit:{similarity:.0f}%:{','.join(characters)}"
                            )
                        log.info(
                            "hit %.0f%% %s -> %s [%s]",
                            similarity,
                            image.name,
                            ", ".join(characters) or "no character tag",
                            outcome,
                        )

            state.save_progress()

            long_remaining = header.get("long_remaining")
            if isinstance(long_remaining, int) and long_remaining <= LONG_REMAINING_MARGIN:
                log.warning(
                    "SauceNAO daily allowance nearly exhausted (%s left); stopping.",
                    long_remaining,
                )
                break

            time.sleep(settings.tier0_interval)
    except KeyboardInterrupt:
        log.warning("Interrupted; progress has been saved.")
    finally:
        state.save_progress()

    log.info("Tier 0 done: %s", result.summary())
    return result
