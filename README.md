# camie-immich-tagger (Linux)

Local anime/illustration auto-tagging for [Immich](https://immich.app/).

Images are tagged on your own machine with `camie-tagger-v2` (ONNX), written to XMP
sidecars next to the originals, and picked up by Immich as a browsable, searchable tag
tree. Characters the model cannot recognise can optionally be filled in through a
SauceNAO reverse search.

Designed to run inside the Immich LXC: installed from git into `/opt`, configured with a
single `.env` file, driven by one command, and scheduled with cron.

> This tool tags images by content. The tag vocabulary comes from a Danbooru-trained
> model and may include mature descriptors. Use it on your own library at your discretion.

---

## How it works

```
images ──> camie-tagger-v2 (ONNX, GPU) ──> hierarchical tags
                                              │
                                              ▼
                              photo.jpg.xmp  (XMP-digiKam:TagsList)
                                              │
                                              ▼
                              Immich library scan + sidecar import
```

Tags are written under five namespaces: `character/`, `copyright/`, `artist/`,
`general/` and `rating/`.

Key properties:

- **Non-destructive.** Only `.xmp` sidecars are written; your images are never modified.
- **Idempotent.** Only tags that are not already present get appended, so tags you added
  by hand in Immich survive re-runs.
- **Incremental.** A processed-list makes the nightly run touch only genuinely new images.

---

## Requirements

- Debian or Ubuntu (the Immich LXC works out of the box)
- Python 3.10 or newer
- About 1 GB of disk space for the model
- A running Immich instance with an **External Library**
- Optional: a GPU (see below)

---

## Installation

```bash
sudo git clone https://github.com/Nicofire/camie-immich-tagger-linux /opt/camie-immich-tagger-linux
cd /opt/camie-immich-tagger-linux
sudo ./install.sh
```

The installer detects your GPU, installs the matching ONNX Runtime build, installs
`exiftool` and the Python package into a virtual environment, downloads the model, and
creates a `.env` file with mode `600`.

Options:

| Option | Effect |
| --- | --- |
| `--device nvidia\|intel\|amd\|cpu` | Skip hardware detection and force an accelerator |
| `--skip-model` | Do not download the model |
| `--skip-apt` | Do not install system packages |

For convenience, put the command on your `PATH`:

```bash
sudo ln -s /opt/camie-immich-tagger/camie-immich-tagger-linux/venv/bin/camie-tagger /usr/local/bin/camie-tagger
```

---

## GPU support

ONNX Runtime needs a hardware-specific build, which `install.sh` selects for you:

| Hardware | Package | Status |
| --- | --- | --- |
| NVIDIA | `onnxruntime-gpu` | Fast. Needs the NVIDIA driver and CUDA libraries. |
| Intel iGPU / Arc / NPU | `onnxruntime-openvino` | Works well, uses `/dev/dri`. |
| AMD | `onnxruntime` | **CPU only.** See below. |
| No GPU | `onnxruntime` | Works, but noticeably slower. |

**AMD:** ONNX Runtime publishes no AMD GPU wheel on PyPI. Integrated AMD GPUs are not
supported at all; discrete cards require a ROCm build compiled from source. On AMD
hardware this tool runs on the CPU.

Select or verify the device at runtime:

```bash
camie-tagger doctor                       # show detected hardware and the chosen provider
camie-tagger run --device cpu             # force CPU
camie-tagger run --fail-on-cpu-fallback   # abort instead of silently using the CPU
```

`--fail-on-cpu-fallback` is worth adding to cron jobs, so a broken driver produces a
failure instead of a run that quietly takes twenty times longer.

### GPU access inside an LXC

An Intel or AMD GPU is only visible to the container if the render node is passed
through. On the Proxmox host, add to the container config:

```
lxc.cgroup2.devices.allow: c 226:* rwm
lxc.mount.entry: /dev/dri dev/dri none bind,optional,create=dir
```

Then make sure the user running the tool is in the `render` and `video` groups:

```bash
sudo usermod -aG render,video "$USER"
```

Verify with `ls -l /dev/dri` inside the container; `camie-tagger doctor` reports this too.

---

## Downloading the model manually

`install.sh` does this for you. To do it by hand, place both files in the directory that
`CAMIE_MODEL_DIR` points at (default `models/`):

```bash
cd /opt/camie-immich-tagger/camie-immich-tagger-linux
mkdir -p models
BASE=https://huggingface.co/Camais03/camie-tagger-v2/resolve/main
curl -fL --progress-bar -o models/camie-tagger-v2.onnx          "$BASE/camie-tagger-v2.onnx?download=true"
curl -fL --progress-bar -o models/camie-tagger-v2-metadata.json "$BASE/camie-tagger-v2-metadata.json?download=true"
```

Alternatively, with the Hugging Face CLI:

```bash
pip install huggingface-hub
hf download Camais03/camie-tagger-v2 \
    camie-tagger-v2.onnx camie-tagger-v2-metadata.json \
    --local-dir models
```

The `.onnx` file is about 789 MB and the metadata about 8 MB.

---

## Configuration

Settings are read from `.env` next to the package. Every value can be overridden on the
command line, and real environment variables take precedence over the file.

Precedence, highest first: **command line option → environment variable → `.env` → default**

| Key | Meaning |
| --- | --- |
| `CAMIE_SCAN_DIRS` | Colon-separated directories to scan, as seen by *this* machine |
| `CAMIE_MODEL_DIR` | Where the `.onnx` and metadata files live (default `models/`) |
| `CAMIE_EXIFTOOL` | exiftool binary (default: whatever is on `PATH`) |
| `CAMIE_DATA_DIR` | JSON run state (default `data/`) |
| `CAMIE_LOG_DIR`, `CAMIE_LOG_LEVEL` | Log destination and verbosity |
| `CAMIE_LOG_MAX_MB`, `CAMIE_LOG_BACKUPS` | Log rotation size and number of kept files |
| `IMMICH_URL` | Base URL, for example `http://localhost:2283` |
| `IMMICH_API_KEY` | Immich API key |
| `IMMICH_LIBRARY_IDS` | Comma-separated External Library **UUIDs** |
| `CAMIE_THRESHOLD` | Confidence threshold, `0.0`–`1.0` (default `0.5`) |
| `CAMIE_CATEGORY_THRESHOLDS` | JSON category overrides, e.g. `{"character":0.6,"copyright":0.6}` (default `{}`) |
| `CAMIE_DEVICE` | `auto`, `intel`, `nvidia`, `amd` or `cpu` |
| `SAUCENAO_API_KEY` | Optional, only for the Tier 0 backfill |
| `CAMIE_TIER0_MIN_SIMILARITY` | Minimum match similarity, default `88` |
| `CAMIE_TIER0_DAILY_CAP` | Maximum searches per run, default `100` |
| `CAMIE_TIER0_INTERVAL` | Seconds between searches, default `18` |

### Paths: this machine vs. Immich

`CAMIE_SCAN_DIRS` uses the paths where **this tool** can read the files. Immich has its
own view of the same storage through its `importPaths`. The two do not need to match, but
they must point at the same physical files.

### Finding the library UUID

`IMMICH_LIBRARY_IDS` needs the UUID, not the library's display name:

```bash
curl -H "x-api-key: YOUR_KEY" "http://localhost:2283/api/libraries"
```

```json
[{ "id": "74683eef-8eaa-4b66-9fbf-61294c36ad02", "name": "media", "importPaths": ["/mnt/media"] }]
```

Use the `id` value.

### API key permissions

The key needs to trigger an external library scan (`POST /api/libraries/{id}/scan`) and
the sidecar job (`PUT /api/jobs/sidecar`). `cleanup-tags` additionally needs to read and
delete tags. If a scoped key returns `403`, use a key with full access.

### A note on secrets

`--immich-api-key` and `--saucenao-api-key` exist for convenience, but arguments are
visible to other users through `ps`. On a shared machine prefer `.env` (mode `600`) or an
environment variable. Keys are redacted from all log output.

---

## Usage

One command with subcommands:

```bash
camie-tagger <command> [options]
```

### Check the installation

```bash
camie-tagger doctor
```

Reports Python, ONNX Runtime, the detected GPU and chosen provider, exiftool, the model
files, the scan directories, and whether Immich is reachable and the configured library
UUIDs exist.

### Tagging

```bash
# Small sample, writes nothing - use this first
camie-tagger run --mode test --limit 5 --dry-run

# Small sample, writes sidecars
camie-tagger run --mode test

# Whole library (first run; slow) and hand it to Immich afterwards
camie-tagger run --mode all --immich-scan

# Daily incremental: only images not processed before
camie-tagger run --mode recent --immich-scan

# Everything in one go, including the SauceNAO backfill
camie-tagger run --mode recent --tier0 --immich-scan
```

| `run` option | Meaning |
| --- | --- |
| `--mode test\|recent\|all` | Sample / only new images / the whole library |
| `--limit N` | Process at most N images |
| `--threshold 0.45` | Override the confidence threshold |
| `--category-threshold character=0.6` | Override one category; repeatable |
| `--dry-run` | Predict and report, write nothing |
| `--reprocess` | Predict already tagged images too; requires `--mode all` or `test` |
| `--allow-tag-removal` | Permit removal of obsolete model-owned tags during reprocessing |
| `--confirm` | Confirm removals; otherwise use `--dry-run` to preview them |
| `--tier0` | Run the SauceNAO backfill afterwards |
| `--immich-scan` | Trigger the Immich library and sidecar scan at the end |

Images that already carry tags from this tool are skipped, so re-running is cheap.

### Improving tag quality

Preprocessing preserves the aspect ratio, uses Lanczos resizing and centered padding,
and applies EXIF orientation before inference. The input remains ImageNet-normalized
RGB at 512x512, matching Camie's reference inference geometry.

Separate thresholds can be set for `character`, `copyright`, `artist`, `general` and
`rating`. A category override takes precedence over the global fallback, even if its
value is lower. Values must be finite numbers from 0 to 1, inclusive. Defaults remain
unchanged: global `0.5`, no category overrides.

```bash
# Illustrative values, not a claim that these thresholds fit every collection
camie-tagger run --mode test --dry-run \
  --category-threshold character=0.6 --category-threshold copyright=0.6
```

CLI category overrides merge with `CAMIE_CATEGORY_THRESHOLDS`, retaining categories
not mentioned on the command line. For each category: CLI overrides environment, which
overrides `.env`. Lower values find more tags but can introduce false positives; higher
values can miss valid characters. Scores are not calibrated guarantees of correctness.

For an existing library, start with an additive reprocessing preview:

```bash
camie-tagger run --mode all --reprocess --limit 20 --dry-run
camie-tagger run --mode all --reprocess --limit 20

# Preview obsolete automatic tags, then explicitly confirm their removal
camie-tagger run --mode all --reprocess --allow-tag-removal --limit 20 --dry-run
camie-tagger run --mode all --reprocess --allow-tag-removal --limit 20 --confirm
```

Reprocessing is additive unless removal is explicitly enabled. Only tags recorded as
written by the local model can be removed by local reprocessing. Untracked/manual tags
and tags confirmed by Tier 0 remain protected. Older sidecars are not automatically
adopted as model-owned tags; existing mistakes in them need manual review.

An identical tag value later reused manually cannot be distinguished from the original
automatic tag. Leave removal disabled when that distinction matters. Review a small
preview and retain backups before enabling removals.

To evaluate quality, use a representative set of your own images, including uncommon
characters and multiple aspect ratios. Tune on one subset and evaluate character/series
precision and recall on a separate held-out subset. Change preprocessing and thresholds
separately when comparing versions. Measure time per image and memory on the actual
Intel iGPU before considering a larger model. No accuracy gain or faster inference is
implied by the unit tests; these checks still require labeled images and real hardware.

### Tier 0: SauceNAO backfill

Optional. Finds images that have a real copyright tag but no character tag, and looks
them up by reverse image search to recover specific character names.

This optional step uploads the image to SauceNAO. Local model inference does not
upload images.

```bash
camie-tagger tier0                      # update the queue, then search
camie-tagger tier0 --enqueue-only       # only refresh the queue
camie-tagger tier0 --limit 20           # fewer searches this run
camie-tagger tier0 --min-similarity 92  # stricter matching
```

The free SauceNAO tier allows roughly 100 searches per day, so this deliberately runs
slowly, honours the remaining daily quota, and resumes where it left off.

By default Tier 0 only *adds* tags, and only looks at images that have a real copyright
tag but no character tag.

### Limiting a run to one folder

`--scan-dir` overrides `CAMIE_SCAN_DIRS` for a single run and works with every command.
Repeat it for several folders:

```bash
camie-tagger tier0 --scan-dir /mnt/media/series-a
camie-tagger tier0 --scan-dir /mnt/media/series-a --scan-dir /mnt/media/series-b
camie-tagger run --mode all --scan-dir /mnt/media/new-import
camie-tagger stats --scan-dir /mnt/media/series-a
```

Only images inside those directories are queued and searched. Entries already queued for
other folders stay in the queue untouched, and the queue pruner is skipped, so narrowing
the scope never discards work.

### Match sources

SauceNAO searches every index it has (`db=999`), not just Danbooru. When a match has a
Danbooru post, its canonical tags are fetched from the Danbooru API. Otherwise the
character, copyright and artist names SauceNAO itself reports are used, which covers
Gelbooru, Konachan, yande.re, e621 and similar indexes. Indexes that only name an author,
such as Pixiv and Kemono, contribute an `artist/` tag.

If two matches are close in similarity, the Danbooru one wins, because only Danbooru
tags are precise enough to replace existing tags.

```bash
camie-tagger tier0 --danbooru-only       # stricter: skip everything except Danbooru
camie-tagger tier0 --min-similarity 80   # more hits, higher risk of a wrong match
```

A miss now reports the best score that was available, so you can tell a genuinely
unindexed image from a threshold that is set too high:

```
miss 4q7sy82m73x51.jpg (best 62% on Pixiv)
```

### Verifying and correcting existing tags

The local model sometimes assigns the wrong character or copyright. `--verify` also
queues images that already have character tags, compares them against the matching
Danbooru post and adds missing tags. Removal of tracked automatic tags can be enabled
separately.

```bash
camie-tagger tier0 --verify --limit 10             # preview additions, writes nothing
camie-tagger tier0 --verify --confirm --limit 10   # apply additions only

# Removal needs a separate explicit opt-in, and only affects tracked automatic tags
camie-tagger tier0 --verify --allow-tag-removal --limit 10
camie-tagger tier0 --verify --allow-tag-removal --confirm --limit 10
```

Safety rules:

- **Dry run by default.** Without `--confirm` nothing is written.
- **Additive by default.** `--confirm` alone no longer permits deletions. Removing tags
  also requires `--allow-tag-removal` and a recorded automatic origin.
- **Only `character/`, `copyright/` and `artist/` tags can be replaced.** Your
  `general/` and `rating/` tags are never deleted, because a Danbooru post does not
  describe them.
- **Only Danbooru matches may replace anything.** A hit from Konachan, yande.re, Pixiv or
  any other index can add tags but never delete them.
- **Replacing needs more confidence than adding.** A tag is only deleted at a similarity
  of 95% or higher (`--replace-min-similarity`). Below that, missing tags are still added
  and the existing ones are kept.
- Untracked/manual tags are protected in all namespaces. Legacy tags are not assumed
  to be automatic just because their namespace matches.

In a `--verify` dry run, progress is deliberately not recorded, so the same images are
searched again when you re-run with `--confirm`. Use `--limit` to keep that cheap.
The preview also leaves the queue and processed-state files unchanged. Explicit
`--enqueue-only` still persists the queue.

Failed sidecar writes are reported as failures and are not recorded as completed
searches, so a later run can retry them. `hits` counts usable search matches, including
matches whose subsequent write fails; check `failed` as well.

### Immich, statistics and cleanup

```bash
camie-tagger immich-scan                # library scan + sidecar import
camie-tagger stats                      # tag coverage, top characters/copyrights/artists

camie-tagger cleanup-orphans            # list .xmp files whose image is gone
camie-tagger cleanup-orphans --confirm  # delete them

camie-tagger cleanup-tags               # list Immich tags outside the five namespaces
camie-tagger cleanup-tags --confirm     # delete them
```

Both cleanup commands are dry-run by default and only act with `--confirm`.

`cleanup-tags` removes every Immich tag whose first segment is not `character`,
`copyright`, `artist`, `general` or `rating`. That includes the `zh/` Chinese tags written
by older versions of this project.

---

## Scheduling with cron

See [docs/cron.example](docs/cron.example). A typical setup:

```cron
CAMIE=/opt/camie-immich-tagger/camie-immich-tagger-linux/venv/bin/camie-tagger

0 3 * * * $CAMIE run --mode recent --immich-scan
0 4 * * * $CAMIE tier0
0 5 * * 0 $CAMIE cleanup-orphans --confirm
```

Install with `crontab -e`. Cron runs with a minimal environment, so always use the full
path to the `camie-tagger` binary inside the virtual environment. The tool writes its own
rotating log, so no output redirection is required.

---

## Logs and state

| Location | Contents |
| --- | --- |
| `logs/camie-tagger.log` | Rotating log, 10 MB per file, 5 kept |
| `data/processed.json` | Processed images, automatic tag origins and inference profiles |
| `data/tier0_queue.json` | Images waiting for reverse search |
| `data/tier0_progress.json` | Reverse search results, used to resume |

Errors go to the log rather than to separate text files. Successful writes are verified
by reading the tags back. Unreadable sidecars and failed directory scans stop safe
processing rather than being treated as empty metadata. Commands return a nonzero exit
code for processing failures, including partial failures.

Limited runs and narrowed scan directories preserve records outside the selection.
The state records a model/metadata fingerprint, preprocessing version and thresholds
for new inference runs. Keep this state: deleting it loses automatic-tag ownership and
does not force existing sidecars to be tagged again. Use `--reprocess` instead.

---

## Development and tests

```bash
python -m pip install -e ".[test]"
python -m pytest -q -m "not integration"
python -m pytest -m "not integration" --cov=camie_tagger --cov-branch --cov-report=term-missing
```

Unit tests use generated images, temporary state and fake inference sessions. They
block network requests and do not require ONNX Runtime, model weights, a GPU, API
keys or a running Immich instance.

They also cover simulated GPU/provider selection and CPU fallback, Immich API
requests and failures, diagnostic checks and exit codes, and maintenance previews,
selective cleanup and statistics. The existing CI unit jobs automatically include
these tests on both Python versions.

For real XMP roundtrips, install exiftool and run:

```bash
sudo apt-get install -y libimage-exiftool-perl
python -m pytest -q -m integration
```

Integration tests skip when exiftool is absent. They check sidecar creation, merging,
idempotency, selective removal, manual-tag preservation and unchanged original image
bytes. GitHub Actions runs unit tests on Python 3.10 to 3.14, plus a Linux job with
real exiftool. Actual ONNX/OpenVINO inference and Immich import still need a smoke test
on the target system with copies of a few images.

---

## Troubleshooting

**Tags do not show up in Immich.** Immich reads `XMP-digiKam:TagsList`, which is what this
tool writes. New sidecars need the *sidecar discovery* job, not just metadata extraction:
run `camie-tagger immich-scan`, or start the job from Administration → Jobs.

**Library `assetCount` stays 0.** Immich cannot see the files. Check the library's
`importPaths` and the container mount; this is independent of `CAMIE_SCAN_DIRS`.

**`403` from Immich.** The API key lacks permission for library scan or job control.

**It runs on the CPU although a GPU is present.** Run `camie-tagger doctor`. Usual causes
are a missing `/dev/dri` passthrough, the user not being in the `render` group, or the
wrong ONNX Runtime build. Reinstall with `sudo ./install.sh --device intel`.

**`exiftool not found`.** `sudo apt-get install -y libimage-exiftool-perl`.

**`install.sh: cannot execute`.** The file has Windows line endings:
`sed -i 's/\r$//' install.sh`.

---

## Differences from the Windows version

- One `camie-tagger` command instead of eight loose scripts, with a single set of options
- `.env` configuration instead of an editable `config.py` containing secrets
- Rotating logs instead of assorted `.txt` files; state consolidated in `data/`
- Runtime GPU detection for Intel, AMD and NVIDIA rather than a CUDA-only path
- The `zh/` Chinese tag branch and its translation table have been removed
- All code, comments, logs and documentation are in English

---

## License and model attribution

The code in this repository is MIT licensed; see [../LICENSE](../LICENSE).

No model weights are bundled or distributed. The tool loads
[camie-tagger-v2](https://huggingface.co/Camais03/camie-tagger-v2) by **Camais03**, which
you download yourself:

- Model `Camais03/camie-tagger-v2`, licensed under **GPL-3.0**
- Trained on the `p1atdev/danbooru-2024` dataset

Please review and comply with the model's GPL-3.0 terms. This project's code only calls
the ONNX file at runtime and incorporates no GPL-licensed source.
