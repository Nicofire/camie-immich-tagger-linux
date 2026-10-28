# Command reference

[Home](../README.md) | [Setup](setup.md) | [Operations](operations.md) |
[Development](development.md)

## Help and common options

```bash
camie-tagger
camie-tagger --help
camie-tagger run --help
```

No arguments and `--help` show the same overview without loading configuration or
starting work. Options go **after the subcommand**. The module entry point is
equivalent: `python -m camie_tagger COMMAND [OPTIONS]` in the installed environment.

| Common option | Effect |
| --- | --- |
| `--config PATH` | Select a different `.env` file. |
| `--scan-dir PATH` | Override image directories; repeat for several. |
| `--device auto\|intel\|nvidia\|amd\|cpu` | Choose an inference device. |
| `--fail-on-cpu-fallback` | Reject a CPU-only provider fallback. |
| `--log-level DEBUG\|INFO\|WARNING\|ERROR` | Set log verbosity. |
| `--log-file PATH` | Override the log file location. |
| `--immich-url URL` | Override the Immich base URL. |
| `--immich-api-key KEY` | Override the Immich key; prefer `.env`. |
| `--saucenao-api-key KEY` | Override the SauceNAO key; prefer `.env`. |
| `-h`, `--help` | Show help for the selected command. |

These options are accepted by every subcommand, but only relevant settings affect
each operation. `--scan-dir` scopes local image processing, **not** Immich's
global Sidecar queue or server-wide `cleanup-tags`.

```bash
camie-tagger run --mode test --dry-run --scan-dir /mnt/media/new-import
camie-tagger stats --scan-dir /mnt/media/series-a --scan-dir /mnt/media/series-b
```

## Local tagging

`run` recursively discovers JPG/JPEG, PNG, WebP and BMP images and writes
`<image-filename>.xmp` alongside each image. It uses `XMP-digiKam:TagsList` with
the roots `character/`, `copyright/`, `artist/`, `general/` and `rating/`.
Unrelated sidecar tags are merged, not intentionally discarded. Original image
bytes are not changed.

| Run option | Effect |
| --- | --- |
| `--mode test\|recent\|all` | Sample, unprocessed paths, or all paths. |
| `--limit N` | Limit the selected image paths. |
| `--threshold VALUE` | Global confidence threshold from 0 to 1. |
| `--category-threshold CATEGORY=VALUE` | Override a category; repeatable. |
| `--dry-run` | Predict without XMP/state writes or follow-up API actions. |
| `--reprocess` | Include tagged images; only with `all` or `test`. |
| `--allow-tag-removal` | Permit removal of obsolete model-owned tags. |
| `--confirm` | Confirm requested removals during reprocessing. |
| `--tier0` | Run reverse-search backfill after local tagging. |
| `--immich-scan` | Request Immich scans after processing. |

### Choosing a mode

- **`recent` (default):** uses the stored processed paths. It is not a date or
  file-content filter; editing an image at the same path does not retrigger tagging.
- **`all`:** considers every discovered path, but still skips images with existing
  tool-namespace tags unless `--reprocess` is supplied.
- **`test`:** selects up to 20 paths by default. It **writes sidecars** unless you
  add `--dry-run`; it is not inherently a preview.

Limits apply before the already-tagged filter, so `--limit 20` need not produce
20 new predictions. Dry runs can still write logs, but not sidecars or processing
state; combined `--tier0` and `--immich-scan` are suppressed in a local dry run.
On a combined run, `--limit` limits local tagging, not the later Tier 0 searches.
For independent limits, invoke the stages separately.

```bash
camie-tagger run --mode test --limit 5 --dry-run
camie-tagger run --mode all --immich-scan
camie-tagger run --mode recent --tier0 --immich-scan
```

Follow-up stages are invoked in order, but are not a transaction. Reported failures
in one stage do not necessarily prevent subsequent stages from being attempted.

### Thresholds and quality

The default global threshold is `0.5`, with no category overrides. Categories are
`character`, `copyright`, `artist`, `general` and `rating`. Values must be finite
numbers between 0 and 1 inclusive. A category threshold takes precedence over the
global fallback even if it is lower.

```bash
camie-tagger run --mode test --dry-run \
  --category-threshold character=0.6 \
  --category-threshold copyright=0.6
```

CLI overrides merge by category with `CAMIE_CATEGORY_THRESHOLDS`; categories not
mentioned remain configured. Lower thresholds can improve recall but introduce
false positives; higher values can miss valid tags. Scores are not calibrated
guarantees of correctness, and these example thresholds are not universal presets.
Changing a threshold, model or preprocessing version does **not** automatically
reprocess previously tagged images.

### Reprocessing and removals

Start with a small additive preview, then apply additions:

```bash
camie-tagger run --mode all --reprocess --limit 20 --dry-run
camie-tagger run --mode all --reprocess --limit 20
```

To remove obsolete automatic tags, use the separate opt-in and review first:

```bash
camie-tagger run --mode all --reprocess --allow-tag-removal --limit 20 --dry-run
camie-tagger run --mode all --reprocess --allow-tag-removal --limit 20 --confirm
```

Only tags recorded as written by the **local model** are eligible for local removal.
Untracked/manual tags and Tier 0-owned tags remain protected. Legacy tags are not
adopted just because they use one of the tool's namespaces.

The tool cannot distinguish a tracked automatic tag from the identical string
later reused manually. Keep removals disabled if that distinction matters. Back
up sidecars and state before bulk changes. Local sidecar safeguards are not a
promise that [Immich will preserve database-only tags](operations.md#immich-import).

## Tier 0 reverse search

This optional stage needs `SAUCENAO_API_KEY` and **uploads selected images to
SauceNAO**. It normally queues images with a real copyright/series tag, excluding
`copyright/original`, and no character tag. Previously recorded hits **and misses**
are skipped on later runs. It is not an unlimited retry of every unresolved image.

```bash
camie-tagger tier0 --enqueue-only
camie-tagger tier0 --limit 20
camie-tagger tier0 --min-similarity 92 --danbooru-only
```

Normal `tier0` runs **write additions without `--confirm`**. The preview behavior
below applies specifically to `--verify`, not to ordinary backfill.

| Tier 0 option | Effect |
| --- | --- |
| `--limit N` | Maximum searches this run. |
| `--min-similarity PCT` | Minimum accepted similarity; default `88`. |
| `--enqueue-only` | Save an updated queue, without searching. |
| `--no-enqueue` | Search using the existing queue. |
| `--verify` | Also consider images that already have character tags. |
| `--confirm` | Apply a verification preview. |
| `--replace-min-similarity PCT` | Removal similarity; default `95`. |
| `--danbooru-only` | Accept only matches linked to a Danbooru post. |
| `--allow-tag-removal` | Opt into tracked automatic removals with `--verify`. |

The default search limit is 100 per run and the default interval is 18 seconds.
The API's remaining quota is checked too. Account limits can vary: verify your
SauceNAO plan rather than assuming a guaranteed daily allowance.

### Sources and scoped queues

SauceNAO searches its available indexes (`db=999`). Matches linked to Danbooru use
canonical tags retrieved through the Danbooru API. Other matches can contribute
character, copyright or artist names supplied by SauceNAO; author-only indexes
may contribute only artist tags. Nearby scores favor Danbooru matches.

`--scan-dir` limits which local images are queued and searched. Queue entries for
other directories are retained. Results describe search matches: a `hit` can still
have a failed sidecar write, so inspect `failed` as well. Failed writes are not
recorded as completed and remain retryable.

### Verifying existing tags

```bash
camie-tagger tier0 --verify --limit 10
camie-tagger tier0 --verify --confirm --limit 10
```

Without `--confirm`, verification previews leave sidecars, progress and the queue
unchanged, unless `--enqueue-only` explicitly requests queue persistence. They
**still upload images and consume quota**. Confirming can search the same images
again, so keep previews small. Already completed progress entries remain excluded.

`--confirm` alone adds missing tags, without deleting existing ones. To preview
and then permit removals:

```bash
camie-tagger tier0 --verify --allow-tag-removal --limit 10
camie-tagger tier0 --verify --allow-tag-removal --confirm --limit 10
```

Removal requires a tracked automatic origin, a canonical Danbooru match, and the
replacement similarity threshold (default 95%). It affects only `character/`,
`copyright/` and `artist/`; general/rating and untracked tags are retained.
Non-Danbooru matches can add tags but cannot authorize removals.

Tier 0 does not trigger Immich scans on its own. Run `immich-scan` afterwards or
use the combined `run --tier0 --immich-scan` workflow.

## Diagnostics, scans and cleanup

```bash
camie-tagger doctor
camie-tagger stats
camie-tagger immich-scan
camie-tagger cleanup-orphans
camie-tagger cleanup-tags
```

`doctor` checks configuration, providers, model files, ExifTool, scan directories
and Immich connectivity/library IDs. It does not certify prediction accuracy or
prove that a full inference and Immich import will succeed. `stats` reports XMP
coverage and common character, copyright and artist tags, **not accuracy**.

Both cleanup commands preview by default; `--confirm` enables deletion:

- **`cleanup-orphans`:** deletes XMP sidecars without corresponding images, including
  supported older naming conventions. First verify that storage is mounted and
  readable; an unavailable image can otherwise look deleted.
- **`cleanup-tags`:** deletes Immich tags whose first path segment is outside
  `character`, `copyright`, `artist`, `general`, `rating`. This is server-wide API
  cleanup, can delete manually created tags and is **not scoped by `--scan-dir`**.

Neither destructive cleanup belongs in a default unattended schedule. Review
previews and backups before deliberately adding `--confirm`.
