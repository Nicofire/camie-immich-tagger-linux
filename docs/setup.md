# Setup and configuration

[Home](../README.md) | [Usage](usage.md) | [Operations](operations.md) |
[Development](development.md)

## Installation choices

Start with the [quick start](../README.md#quick-start). Examples use
`/opt/camie-immich-tagger-linux`; substitute your checkout path consistently.
The installer targets Debian/Ubuntu, not native Windows. Other distributions need
equivalent system packages and a manually selected ONNX Runtime build.

| Installer option | Effect |
| --- | --- |
| `--device nvidia\|intel\|amd\|cpu` | Choose the runtime installation profile. |
| `--skip-model` | Leave model downloading to you. |
| `--skip-apt` | Skip system package installation; dependencies must exist. |
| `--help` | Show installer help without installing anything. |

For example, from the checkout:

```bash
bash install.sh --device intel
```

The installer uses root directly or `sudo` for system changes. The account running
it must be able to write to the checkout. It installs ExifTool and the selected
runtime, creates `venv/`, installs the package in editable mode, downloads model
files to `models/`, and creates `.env` if absent. Existing `.env` contents stay
intact and its permissions are set to `600`.

Installing only the Python package does **not** install ONNX Runtime: its CPU,
OpenVINO and CUDA distributions are selected separately by the installer.

### The camie-tagger command

`/usr/local/bin/camie-tagger` links to the executable inside the project's virtual
environment. Activation is unnecessary, but the checkout must stay in place.
An existing file, directory or foreign symlink is preserved with a warning.

If the short command is unavailable:

```bash
command -v camie-tagger
/opt/camie-immich-tagger-linux/venv/bin/camie-tagger doctor
```

Check that `/usr/local/bin` is on `PATH`. To register an existing installation
manually, run from its checkout with appropriate system permissions:

```bash
sudo ln -sT -- "$PWD/venv/bin/camie-tagger" /usr/local/bin/camie-tagger
```

This intentionally has no force option: inspect conflicts rather than replacing
another installation. Cron can always use the full virtual-environment path.

## GPU and container access

| Hardware | Installer runtime | Requirements |
| --- | --- | --- |
| CPU | `onnxruntime` | No accelerator required. |
| Intel GPU | `onnxruntime-openvino` | Render node and Intel drivers. |
| NVIDIA | `onnxruntime-gpu` | Compatible NVIDIA driver and CUDA libraries. |
| AMD | `onnxruntime` | CPU with the standard installer. |

The installer does not configure host GPU passthrough or a complete NVIDIA/CUDA
stack. A custom AMD ROCm/MIGraphX runtime is outside the standard installation;
do not assume that detecting an AMD GPU enables acceleration.

```bash
camie-tagger doctor
camie-tagger run --device cpu
camie-tagger run --device intel --fail-on-cpu-fallback
```

Runtime choices are `auto`, `intel`, `nvidia`, `amd`, `cpu`. Auto detection prefers
NVIDIA, then Intel, then AMD when matching hardware and providers are available.
The device option selects a provider; it does not install drivers or runtime wheels.

`--fail-on-cpu-fallback` rejects selection of a CPU-only fallback. It is not a
guarantee that every operator runs on the GPU: OpenVINO is configured with
`AUTO:GPU,CPU`, and execution providers can use CPU support internally.
NPU execution is not configured. Benchmark on your actual machine; no throughput
or Intel-iGPU memory target is guaranteed.

### Intel GPU inside Proxmox LXC

The container needs access to `/dev/dri/renderD*`. A common host-side container
configuration is shown below; adapt it to your device and security requirements:

```ini
lxc.cgroup2.devices.allow: c 226:* rwm
lxc.mount.entry: /dev/dri dev/dri none bind,optional,create=dir
```

After applying the configuration and restarting the container, check inside it:

```bash
ls -l /dev/dri
id
camie-tagger doctor
```

The runtime account needs access to the actual render node, usually through
the `render` and/or `video` groups. Add it to the required groups and start a
new login session. Unprivileged containers may also require group-ID mapping
between host and container; matching group names alone is not enough.

## Configuration

Use [the example configuration](../.env.example) as the complete starting point.
By default, `.env` is read from the **checkout root**, not the current directory.
An alternative file can be selected after the subcommand:

```bash
camie-tagger run --config /etc/camie-tagger.env --dry-run --mode test
```

Precedence is **CLI option > environment variable > `.env` > default** where a
CLI option exists. Not every setting has a command-line equivalent.
Use absolute paths for custom locations, particularly under cron. The default
model, data and log directories are anchored to the checkout.

| Setting | Default / purpose |
| --- | --- |
| `CAMIE_SCAN_DIRS` | Image directories, separated by colons. |
| `CAMIE_MODEL_DIR` | `models/`; default Camie model and metadata directory. |
| `CAMIE_MODEL_PROFILE` | Unset; optional JSON profile selecting another ONNX model, labels and inference settings. |
| `CAMIE_EXIFTOOL` | `exiftool`; executable name or path. |
| `CAMIE_DATA_DIR` | `data/`; persistent processing state. |
| `CAMIE_LOG_DIR` | `logs/`; rotating log directory. |
| `CAMIE_LOG_LEVEL` | `INFO`. |
| `CAMIE_LOG_MAX_MB` | `10`; rotation size in MB. |
| `CAMIE_LOG_BACKUPS` | `5`; retained rotated files. |
| `IMMICH_URL` | Immich base URL, e.g. `http://localhost:2283`. |
| `IMMICH_API_KEY` | Required for Immich API operations. |
| `IMMICH_LIBRARY_IDS` | External-library UUIDs, separated by commas. |
| `CAMIE_THRESHOLD` | `0.5`; global confidence threshold, from 0 to 1. |
| `CAMIE_CATEGORY_THRESHOLDS` | `{}`; JSON category overrides. |
| `CAMIE_DEVICE` | `auto`; inference provider selection. |
| `SAUCENAO_API_KEY` | Optional; required for reverse searches. |
| `CAMIE_TIER0_MIN_SIMILARITY` | `88`; minimum search similarity percent. |
| `CAMIE_TIER0_DAILY_CAP` | `100`; default maximum searches **per run**. |
| `CAMIE_TIER0_INTERVAL` | `18`; seconds between searches. |

Despite its historical name, `CAMIE_TIER0_DAILY_CAP` is not a persistent daily
counter. SauceNAO's reported remaining quota is also checked; repeated runs can
still consume more than the configured per-run cap over a day.
`--fail-on-cpu-fallback` is a CLI flag, not a loaded `CAMIE_*` environment setting.

Example with multiple directories and category thresholds:

```dotenv
CAMIE_SCAN_DIRS=/mnt/media/anime:/mnt/media/fanart
CAMIE_CATEGORY_THRESHOLDS={"character":0.6,"copyright":0.6}
IMMICH_LIBRARY_IDS=FIRST_LIBRARY_UUID,SECOND_LIBRARY_UUID
```

These threshold values are examples, not measured recommendations.

### Paths and library IDs

The tagger and Immich must access the **same underlying images and sidecars**, but
their container mount paths can differ. `CAMIE_SCAN_DIRS` is the tagger's view;
Immich's external-library import paths are its view. Include only intended scan
directories, and verify that **all** expected mounts exist before running: when
some configured directories are missing, existing ones can still be processed.

Use external-library UUIDs, not display names. Retrieve them through an API client
using `GET /api/libraries`; take each library's `id` value. The response also lists
`importPaths`, which helps diagnose mismatched mounts.

### Permissions and secrets

Grant the API key only the access needed for the commands you use:

| Operation | API access |
| --- | --- |
| Diagnostics | `GET /api/server/ping`, `GET /api/libraries`. |
| Library scans | `POST /api/libraries/{id}/scan`. |
| Sidecar processing | `PUT /api/jobs/sidecar`. |
| Tag cleanup | `GET /api/tags`, `DELETE /api/tags/{id}`. |

Immich permission names and admin requirements depend on the server version.
A `403` means checking the endpoint's required permission and account access,
not automatically switching to a full-access key.

Keep `.env` readable only by the runtime account (`chmod 600`). A root-owned
configuration will not automatically be readable by a dedicated service account.
That account also needs read access to images and model files and write access to
image directories for sidecars, plus data and log directories.

Prefer `.env` or protected environment variables to CLI key arguments. Arguments
can appear in process listings and shell history. Logs redact configured secrets,
but do not publish configuration files or unreviewed debug output.

## Model selection and manual download

To supply a different ONNX tagger, install with `--skip-model` and select a
`CAMIE_MODEL_PROFILE`. The [model guide](models.md) covers compatible models,
Camie/WD/generic templates, file formats and safe switching. Profile paths can
also be selected with `--model-profile PATH` on `run` and `doctor`.

The installer downloads both default **Camie** files. For an offline Camie deployment, place them
in `CAMIE_MODEL_DIR` or download from the checkout as follows:

```bash
cd /opt/camie-immich-tagger-linux
mkdir -p models
BASE=https://huggingface.co/Camais03/camie-tagger-v2/resolve/main
curl -fL -o models/camie-tagger-v2.onnx "$BASE/camie-tagger-v2.onnx?download=true"
curl -fL -o models/camie-tagger-v2-metadata.json "$BASE/camie-tagger-v2-metadata.json?download=true"
```

The ONNX file is approximately 789 MB and the metadata approximately 8 MB.
Keep the model and metadata from the same release together. See the
[model page](https://huggingface.co/Camais03/camie-tagger-v2) for licensing and
upstream information; weights are not part of this repository.
