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


@dataclass
class Tier0Result:
    processed: int = 0
    hits: int = 0
    misses: int = 0
    failed: int = 0

    def summary(self) -> str:
        return (
            f"processed={self.processed} hits={self.hits} "
            f"misses={self.misses} failed={self.failed}"
        )


def build_queue(settings: Settings, state) -> int:
    """Queue images that have a real copyright tag but no character tag."""
    log = get_logger()
    scan_dirs = settings.require_scan_dirs()

    log.info("Scanning sidecars to find images without a character tag...")
    taglists = sidecar.scan_taglists(settings.exiftool, scan_dirs)

    processed = set(state.tier0_progress)
    candidates: list[Path] = []

    for image, tags in taglists.items():
        if str(image) in processed:
            continue
        if tags_in_namespace(tags, "character"):
            continue
        copyrights = tags_in_namespace(tags, "copyright")
        # 'copyright/original' means no source work, so reverse search cannot help.
        if any(tag != "copyright/original" for tag in copyrights):
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


def run_tier0(
    settings: Settings,
    state,
    limit: int | None = None,
    min_similarity: float | None = None,
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
                state.record_tier0(image, "miss")
                log.info("miss %s", image.name)
            else:
                similarity, post_id = match
                try:
                    taglist = _fetch_danbooru_tags(session, post_id)
                except requests.RequestException as exc:
                    result.failed += 1
                    state.record_tier0(image, f"danbooru_error:{str(exc)[:60]}")
                    log.error("Danbooru lookup failed for post %s: %s", post_id, exc)
                    taglist = []

                if taglist:
                    outcome = sidecar.write_taglist(settings.exiftool, image, taglist)
                    characters = tags_in_namespace(taglist, "character")
                    result.hits += 1
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
