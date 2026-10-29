# Development and validation

[Home](../README.md) | [Setup](setup.md) | [Usage](usage.md) |
[Operations](operations.md)

## Test environment

From a checkout, use a separate virtual environment for development:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q -m "not integration and not onnx"
```

Unit tests use temporary images/state and fake inference sessions. They block
network requests and do not require ONNX Runtime, model weights, a GPU, API keys
or a running Immich instance. Installer tests exercise command registration in
temporary paths rather than performing a system installation.

Coverage with branch reporting:

```bash
python -m pytest -m "not integration and not onnx" \
  --cov=camie_tagger --cov-branch --cov-report=term-missing
```

For inference against tiny generated ONNX graphs on CPU:

```bash
python -m pip install -e ".[test,test-onnx]"
python -m pytest -q -m onnx
```

Use this extra only in a development environment: it installs the CPU runtime,
not the hardware-specific runtime chosen for production. The ONNX tests exercise
real input types/layouts, dynamic dimensions, activations, Camie/WD profiles,
external weights and model switching through the tagging pipeline. They need no
downloaded weights, private images or network services. They skip if the optional
ONNX dependencies are absent.

For real ExifTool sidecar roundtrips on Debian/Ubuntu:

```bash
sudo apt-get install -y libimage-exiftool-perl
python -m pytest -q -m integration
```

These tests skip if ExifTool is absent. They check creation, merging, idempotency,
selective removal, manual-sidecar-tag preservation and unchanged original image
bytes. POSIX-specific symlink tests can skip on Windows; the deployment target is
Linux.

## Continuous integration

The [test workflow](../.github/workflows/tests.yml) runs unit tests and branch
coverage on **Python 3.10, 3.11, 3.12, 3.13 and 3.14** under Ubuntu 24.04. A separate
Python 3.12 job installs real ExifTool and runs the integration tests.
Another Python 3.12 job installs `test-onnx` and runs the real ONNX fixture tests.

Tests cover preprocessing/output contracts, thresholds, provenance, reprocessing,
sidecar safety, Tier 0 state/failures, CLI help and validation, provider selection,
Immich requests, diagnostics and maintenance. They do not measure tag accuracy,
validate real GPU throughput or perform a live Immich import.

## Target-system checks

Before a production rollout, use copies of a few representative images:

1. Run `doctor` as the intended runtime account.
2. Preview local predictions with `run --mode test --limit 5 --dry-run`.
3. Write sidecars for a small directory without requesting an Immich scan yet.
4. Inspect the XMP and check original image bytes remain unchanged.
5. Test an Immich import after considering its global Sidecar-job scope.
6. Measure inference time and peak memory on the actual device.

Use the [operations guide](operations.md#immich-import) for import behavior and
database-only tag caveats. Live deployment checks require model files, runtime
dependencies, appropriate storage access and a server; unit tests replace none
of those prerequisites.

## Assessing tag quality

The default Camie preprocessing applies EXIF orientation, converts to RGB, preserves aspect ratio,
uses Lanczos resizing with centered padding, and produces ImageNet-normalized
512x512 NCHW float32 input. Its expected output is `refined_predictions`,
interpreted as sigmoid logits. Other models use their selected
[profile](models.md); preprocessing version `onnx-profile-exif-v1` and the
effective settings are recorded with every processed image. Profile changes
are included in the fingerprint, even when weights and labels stay unchanged.

For threshold/model comparisons, label a representative image set including rare
characters, multiple aspect ratios and difficult illustrations. Tune on one subset
and evaluate character/series precision and recall on a separate held-out subset.
Change preprocessing and thresholds separately, and record the inference profile.
A passing test suite is not evidence of improved recognition accuracy.

## Before contributing

Keep changes focused and add regression coverage near the behavior being changed.
Run the relevant tests and real-ExifTool checks when modifying sidecar behavior.
For documentation, check Markdown formatting, local links and CLI examples against
`camie-tagger COMMAND --help`. Never commit model weights, processing state,
private images or credentials.
