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
sudo git clone https://github.com/PlanetMeow/camie-immich-tagger.git /opt/camie-immich-tagger
cd /opt/camie-immich-tagger/camie-immich-tagger-linux
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
| `--dry-run` | Predict and report, write nothing |
| `--tier0` | Run the SauceNAO backfill afterwards |
| `--immich-scan` | Trigger the Immich library and sidecar scan at the end |

Images that already carry tags from this tool are skipped, so re-running is cheap.

### Tier 0: SauceNAO backfill

Optional. Finds images that have a real copyright tag but no character tag, and looks
them up by reverse image search to recover specific character names.

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
Danbooru post, adds what is missing and replaces what is wrong.

```bash
camie-tagger tier0 --verify --limit 10             # preview, writes nothing
camie-tagger tier0 --verify --confirm --limit 10   # apply the changes
```

Safety rules:

- **Dry run by default.** Without `--confirm` nothing is written.
- **Only `character/`, `copyright/` and `artist/` tags can be replaced.** Your
  `general/` and `rating/` tags are never deleted, because a Danbooru post does not
  describe them.
- **Only Danbooru matches may replace anything.** A hit from Konachan, yande.re, Pixiv or
  any other index can add tags but never delete them.
- **Replacing needs more confidence than adding.** A tag is only deleted at a similarity
  of 95% or higher (`--replace-min-similarity`). Below that, missing tags are still added
  and the existing ones are kept.
- Manual tags outside those three namespaces are untouched.

In a `--verify` dry run, progress is deliberately not recorded, so the same images are
searched again when you re-run with `--confirm`. Use `--limit` to keep that cheap.

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
| `data/processed.json` | Images already tagged |
| `data/tier0_queue.json` | Images waiting for reverse search |
| `data/tier0_progress.json` | Reverse search results, used to resume |

Errors go to the log rather than to separate text files. To re-tag everything from
scratch, delete `data/processed.json` and run `--mode all`.

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
