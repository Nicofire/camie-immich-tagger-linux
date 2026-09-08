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

# SauceNAO already searches every index (db=999). These fields are returned by the
# booru indexes (Danbooru, Gelbooru, Konachan, yande.re, e621), so a match outside
# Danbooru is still usable even though the tags are less precise.
SAUCENAO_TAG_FIELDS = {
    "characters": "character",
    "material": "copyright",
    "creator": "artist",
}

# Indexes without a 'creator' field still name the author, e.g. Pixiv and Kemono.
SAUCENAO_ARTIST_FALLBACKS = ("member_name", "user_name", "author_name")

# Prefer a Danbooru result over a slightly better non-Danbooru one, because only
# Danbooru tags are canonical enough to replace existing tags.
DANBOORU_PREFERENCE_MARGIN = 3.0

MAX_SOURCE_TAG_LENGTH = 100

# Namespaces Danbooru is authoritative for. Verification never touches general/ or
# rating/, which the local model owns and Danbooru results do not describe.
VERIFIABLE_NAMESPACES = ("character", "copyright", "artist")

# Deleting a tag needs more confidence than adding one.
DEFAULT_REPLACE_MIN_SIMILARITY = 95.0


@dataclass
class Match:
    similarity: float
    index_name: str
    post_id: int | None
    tags: list[str]
    # Danbooru API tags are canonical; SauceNAO metadata is only trusted enough to add.
    canonical: bool = False

    @property
    def source(self) -> str:
        return "danbooru" if self.canonical else (self.index_name or "saucenao")


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


def build_queue(
    settings: Settings, state, include_tagged: bool = False, prune: bool = True
) -> int:
    """Queue images that have a real copyright tag but no character tag.

    With include_tagged the character rule is dropped, so images that already have
    character tags are queued too and can be checked against Danbooru. Pruning is
    skipped when the scan directories were narrowed for a single run.
    """
    log = get_logger()
    scan_dirs = settings.require_scan_dirs()

    if prune:
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


def _normalize_source_tag(value: str) -> str:
    text = str(value).strip().lower()
    if not text or "http" in text or len(text) > MAX_SOURCE_TAG_LENGTH:
        return ""
    return normalize_tag(re.sub(r"\s+", "_", text))


def _split_values(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return re.split(r"[,\n]", str(raw))


def _tags_from_result(data: dict) -> list[str]:
    """Build tags from SauceNAO's own metadata, for matches outside Danbooru."""
    tags: list[str] = []
    seen: set[str] = set()

    for field, namespace in SAUCENAO_TAG_FIELDS.items():
        for value in _split_values(data.get(field) or ""):
            name = _normalize_source_tag(value)
            tag = f"{namespace}/{name}"
            if name and tag not in seen:
                seen.add(tag)
                tags.append(tag)

    if not data.get("creator"):
        for field in SAUCENAO_ARTIST_FALLBACKS:
            for value in _split_values(data.get(field) or ""):
                name = _normalize_source_tag(value)
                tag = f"artist/{name}"
                if name and tag not in seen:
                    seen.add(tag)
                    tags.append(tag)

    return tags


def _index_label(name: object) -> str:
    """Turn 'Index #26: - Konachan.com - 1.jpg' into 'Konachan.com'."""
    parts = [part.strip() for part in str(name).split(" - ") if part.strip()]
    for part in parts[1:]:
        if not part.lower().startswith("index #"):
            return part[:40]
    return (parts[0] if parts else "saucenao")[:40]


def _danbooru_post_id(data: dict) -> int | None:
    post_id = data.get("danbooru_id")
    if post_id is None:
        for url in data.get("ext_urls", []) or []:
            found = DANBOORU_POST_PATTERN.search(str(url))
            if found:
                post_id = found.group(1)
                break
    try:
        return int(post_id) if post_id is not None else None
    except (TypeError, ValueError):
        return None


def _best_match(
    results: list[dict], min_similarity: float, danbooru_only: bool = False
) -> Match | None:
    candidates: list[Match] = []

    for entry in results or []:
        header = entry.get("header", {})
        data = entry.get("data", {})
        try:
            similarity = float(header.get("similarity", 0))
        except (TypeError, ValueError):
            continue
        if similarity < min_similarity:
            continue

        post_id = _danbooru_post_id(data)
        if danbooru_only and post_id is None:
            continue
        tags = _tags_from_result(data)
        if post_id is None and not tags:
            continue
        candidates.append(
            Match(similarity, _index_label(header.get("index_name", "")), post_id, tags)
        )

    if not candidates:
        return None

    candidates.sort(key=lambda m: m.similarity, reverse=True)
    best = candidates[0]
    if best.post_id is None:
        for candidate in candidates:
            if (
                candidate.post_id is not None
                and candidate.similarity >= best.similarity - DANBOORU_PREFERENCE_MARGIN
            ):
                return candidate
    return best


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
    match: Match,
    confirm: bool,
    replace_min_similarity: float,
    record_progress: bool,
) -> None:
    """Compare the sidecar against the match, then add missing and drop wrong tags."""
    log = get_logger()
    similarity = match.similarity
    existing = sidecar.read_taglist(settings.exiftool, image)
    to_add, to_remove = plan_changes(existing, match.tags)

    if to_remove and not match.canonical:
        log.info(
            "%s: matched on %s rather than Danbooru, keeping %d existing tag(s)",
            image.name,
            match.source,
            len(to_remove),
        )
        to_remove = []
    elif to_remove and similarity < replace_min_similarity:
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
    log.info("hit %.0f%% %s via %s [%s]", similarity, image.name, match.source, outcome)
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
    danbooru_only: bool = False,
) -> Tier0Result:
    log = get_logger()
    settings.require_saucenao()
    scan_dirs = settings.require_scan_dirs()

    min_similarity = (
        min_similarity if min_similarity is not None else settings.tier0_min_similarity
    )
    cap = limit or settings.tier0_daily_cap

    progress = state.tier0_progress
    # The queue may span several directories, so honour the configured scan dirs here too.
    pending = [
        path
        for path in (Path(entry) for entry in state.tier0_queue)
        if str(path) not in progress
        and path.is_file()
        and any(path.is_relative_to(directory) for directory in scan_dirs)
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
    if danbooru_only:
        log.info("Only Danbooru matches are accepted.")

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
            match = _best_match(
                payload.get("results", []), min_similarity, danbooru_only
            )

            if match is None:
                result.misses += 1
                top = (payload.get("results") or [{}])[0].get("header", {})
                log.info(
                    "miss %s (best %s%% on %s)",
                    image.name,
                    top.get("similarity", "?"),
                    _index_label(top.get("index_name", "unknown")),
                )
                if record_progress:
                    state.record_tier0(image, "miss")
            else:
                if match.post_id is not None:
                    try:
                        danbooru_tags = _fetch_danbooru_tags(session, match.post_id)
                        if danbooru_tags:
                            match.tags = danbooru_tags
                            match.canonical = True
                    except requests.RequestException as exc:
                        log.warning(
                            "Danbooru lookup failed for post %s, falling back to %s: %s",
                            match.post_id,
                            match.source,
                            exc,
                        )

                if match.tags:
                    result.hits += 1
                    if verify:
                        _apply_verification(
                            settings,
                            state,
                            result,
                            image,
                            match,
                            confirm,
                            replace_min_similarity,
                            record_progress,
                        )
                    else:
                        outcome = sidecar.write_taglist(
                            settings.exiftool, image, match.tags
                        )
                        characters = tags_in_namespace(match.tags, "character")
                        if record_progress:
                            state.record_tier0(
                                image,
                                f"hit:{match.similarity:.0f}%:{','.join(characters)}",
                            )
                        log.info(
                            "hit %.0f%% %s via %s -> %s [%s]",
                            match.similarity,
                            image.name,
                            match.source,
                            ", ".join(characters) or "no character tag",
                            outcome,
                        )
                else:
                    result.misses += 1
                    if record_progress:
                        state.record_tier0(image, "miss:no_tags")
                    log.info("miss %s (match had no usable tags)", image.name)

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
