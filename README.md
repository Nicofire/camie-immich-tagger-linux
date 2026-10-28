# Camie Immich Tagger

[![Tests](https://github.com/Nicofire/camie-immich-tagger-linux/actions/workflows/tests.yml/badge.svg)](https://github.com/Nicofire/camie-immich-tagger-linux/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Platform](https://img.shields.io/badge/Platform-Linux-22863A)

**Local anime, manga illustration and fanart tagging for Immich external libraries.**

Run Camie on your own machine, write searchable tags to XMP sidecars, and let
[Immich](https://immich.app/) pick them up. Your original image files stay unchanged.
Optional SauceNAO reverse search can help fill in missing character names.

[Quick start](#quick-start) | [Commands](#commands) |
[Automation](#automation) | [Documentation](#documentation)

## What it does

- **Local inference:** use Camie v2 on CPU, Intel OpenVINO or NVIDIA CUDA.
- **Organized tags:** `character/`, `copyright/`, `artist/`, `general/`, `rating/`.
- **Sidecar workflow:** writes `image.jpg.xmp`, not the original image.
- **Repeatable runs:** skips processed images and adds tags by default.
- **Optional backfill:** search SauceNAO and use Danbooru metadata where available.
- **One CLI:** diagnose, preview, tag, inspect statistics and trigger Immich scans.

```text
Your images -> local Camie inference -> XMP sidecars -> searchable Immich tags
```

> [!IMPORTANT]
> Model predictions can be wrong and the vocabulary can include mature descriptors.
> Start with a preview. SauceNAO is optional and **uploads selected images**, even
> during a Tier 0 verification preview. Local model inference does not upload images.

## Quick start

You need Debian/Ubuntu, Python 3.10+, and images accessible to both this tool and
an Immich external library. Allow about 1 GB for model files, plus space for the
Python environment. A GPU is optional; the installer uses CPU on AMD systems.
See [setup and GPU access](docs/setup.md) for LXC, permissions and device details.

### 1. Install

These commands assume a **root shell on your Linux server or container** and use
`/opt/camie-immich-tagger-linux`. Run the tool and its cron jobs as an account that
can read the configuration and images and write sidecars, logs and state.

```bash
apt-get update
apt-get install -y git
git clone https://github.com/Nicofire/camie-immich-tagger-linux /opt/camie-immich-tagger-linux
cd /opt/camie-immich-tagger-linux
bash install.sh
```

The installer installs dependencies, downloads the model, creates the configuration
and registers `camie-tagger` in `/usr/local/bin` without overwriting another command.
The virtual environment does not need to be activated.

### 2. Configure

```bash
nano /opt/camie-immich-tagger-linux/.env
```

Set these values using the [example configuration](.env.example):

```dotenv
CAMIE_SCAN_DIRS=/mnt/media
IMMICH_URL=http://localhost:2283
IMMICH_API_KEY=YOUR_IMMICH_API_KEY
IMMICH_LIBRARY_IDS=YOUR_EXTERNAL_LIBRARY_UUID
```

Use the image paths visible to **the tagger**, not necessarily Immich's container
paths. The library ID is a UUID, not its name.
[Configuration and library IDs](docs/setup.md#configuration) covers the details.

### 3. Preview, then tag

```bash
camie-tagger doctor
camie-tagger run --mode test --limit 5 --dry-run
```

The preview performs inference but changes no sidecars or processing state.
Check the predictions, then run:

```bash
camie-tagger run --mode all --immich-scan
```

This writes tags for eligible images and asks Immich to scan in the background.
Already tagged images are skipped unless you explicitly use `--reprocess`.

> [!NOTE]
> Immich's Sidecar job is server-wide. Metadata import can replace tags assigned
> only in Immich, so test and back up first. See [import behavior](docs/operations.md#immich-import).

## Commands

Run `camie-tagger` for an overview or `camie-tagger COMMAND --help` for all options.

| Command | Purpose |
| --- | --- |
| `doctor` | Check dependencies, devices, configuration and Immich. |
| `run` | Predict tags locally and write XMP sidecars. |
| `tier0` | Optional reverse-search backfill using SauceNAO. |
| `immich-scan` | Request library scans and Sidecar processing in Immich. |
| `stats` | Show tag coverage and common characters, series and artists. |
| `cleanup-orphans` | Preview sidecars whose image no longer exists. |
| `cleanup-tags` | Preview Immich tags outside the five tool namespaces. |

Everyday incremental run:

```bash
camie-tagger run --mode recent --immich-scan
```

Optional reverse search, followed by an Immich refresh after a successful run:

```bash
camie-tagger tier0 --limit 20 && camie-tagger immich-scan
```

> [!WARNING]
> Cleanup only deletes with `--confirm`. `cleanup-tags` can delete **manual Immich
> tags**, too; it is not limited by `--scan-dir`. Keep the processing state and
> review [safe reprocessing and cleanup](docs/usage.md) before enabling removals.

## Automation

Run `crontab -e` for the same account used above. This example tags new images at
03:00 in the server's timezone; `flock` prevents overlapping scheduled runs.

```cron
CAMIE=/opt/camie-immich-tagger-linux/venv/bin/camie-tagger
LOCK=/run/lock/camie-tagger.lock

0 3 * * * /usr/bin/flock -n "$LOCK" "$CAMIE" run --mode recent --immich-scan
```

Use absolute paths and verify the image storage is mounted. A busy lock skips that
scheduled run. There is no built-in daemon or process lock for manual invocations.
[Operations](docs/operations.md) and the [cron examples](docs/cron.example) cover
Tier 0 scheduling, permissions, logs and cleanup precautions.

## Documentation

| Guide | Contents |
| --- | --- |
| [Setup](docs/setup.md) | Configuration, GPU/LXC access, model files, permissions. |
| [Usage](docs/usage.md) | All CLI options, thresholds, reprocessing, Tier 0 safety. |
| [Operations](docs/operations.md) | Cron, Immich behavior, state, updates, troubleshooting. |
| [Development](docs/development.md) | Tests, CI, contribution checks and validation limits. |

## Credits and licensing

Project code is declared **MIT** in the [package metadata](pyproject.toml).
The separately downloaded [Camie v2 model](https://huggingface.co/Camais03/camie-tagger-v2)
by **Camais03** is licensed **GPL-3.0** and was trained on
`p1atdev/danbooru-2024`. No model weights are bundled. Review the model's license
before use or redistribution; the project and model have separate licensing terms.
