# Interchangeable ONNX models

[Home](../README.md) | [Setup](setup.md) | [Usage](usage.md) |
[Development](development.md)

The tagger is not tied to Camie weights. A **model profile** specifies the ONNX
file, matching vocabulary, image preprocessing and interpretation of its scores.
No code changes are needed when switching between compatible image taggers.
The command and existing `CAMIE_*` environment names remain unchanged.

## Compatibility

Supported contracts:

- Exactly one image input: `NCHW`, `NHWC`, `CHW` or `HWC`, RGB/BGR or grayscale.
- Static image sizes, or dynamic dimensions with an explicit profile size.
  Batched inputs run with batch size 1.
- Input types: float32, float16, float64, uint8 and int8, taken from the graph.
- A selected floating-point output shaped `[tags]` or `[1, tags]`, with one
  score per label. Select the output by name if the graph has multiple outputs.
- Already-normalized probabilities, sigmoid logits or softmax logits.

An `.onnx` extension alone does **not** make a model a tagger. Embeddings,
object detection, segmentation, multiple required inputs and other output
structures need model-specific adapters and are rejected rather than guessed.
Your ONNX Runtime/provider must also support the model's operators and opset.
Integer outputs need dequantization in the exported graph; the tagger does not
guess output quantization parameters.

Preprocessing and label order are not reliably encoded in ONNX. Always use
settings and labels from the **same model release** and follow its publisher's
inference instructions. These profiles do not certify every model in a family
or guarantee recognition accuracy.

## Select or switch a model

1. Download the ONNX weights and corresponding labels from the model publisher.
   Keep any external tensor-data files in the locations expected by the graph.
2. Put each model in its own directory, for example `models/wd/`.
3. Copy an appropriate template into that directory and adjust its filenames
   and settings:

   ```bash
   cp docs/model-profiles/wd.json models/wd/profile.json
   ```

4. Validate the files and preview a small sample:

   ```bash
   camie-tagger doctor --model-profile "$PWD/models/wd/profile.json"
   camie-tagger run --model-profile "$PWD/models/wd/profile.json" \
     --mode test --limit 5 --dry-run
   ```

5. To persist the selection, set the absolute profile path in `.env`:

   ```dotenv
   CAMIE_MODEL_PROFILE=/opt/camie-immich-tagger-linux/models/wd/profile.json
   ```

CLI `--model-profile` overrides the environment and `.env`. Paths **inside** a
profile are relative to that profile's directory, or may be absolute. They do
not depend on the working directory or `CAMIE_MODEL_DIR`.

Without a profile, the original Camie behavior remains: find one `.onnx` and one
`*metadata*.json` recursively under `CAMIE_MODEL_DIR`. Ambiguous directories now
fail with an instruction to select explicit files instead of choosing the first
alphabetically. `--model-dir PATH` overrides this legacy directory, not the file
paths in a selected profile.

Changing the model does not automatically reprocess existing images. To compare
a different model on already-tagged images:

```bash
camie-tagger run --model-profile "$PWD/models/wd/profile.json" \
  --mode all --reprocess --limit 20 --dry-run
```

Remove `--dry-run` to add predictions. Removing obsolete tags still requires the
separate [reprocessing/removal opt-in](usage.md#reprocessing-and-removals).
Manual and Tier 0-owned tags retain their existing protections.
To switch back while another profile is configured, select an explicit
[Camie profile](model-profiles/camie.json), or unset `CAMIE_MODEL_PROFILE` in both
the environment and `.env` to restore legacy discovery.

`doctor` validates the profile and labels without loading weights. Input/output
shape and type validation happens when `run` loads the model for pending images.
A run that skips every image does not load it; use `--reprocess` for a meaningful
preview of a replacement model.

## Presets and templates

| Preset | Template | Defaults |
| --- | --- | --- |
| `camie` | [camie.json](model-profiles/camie.json) | RGB, 512x512 NCHW, Lanczos letterbox, ImageNet normalization, `refined_predictions` logits with sigmoid. |
| `wd` | [wd.json](model-profiles/wd.json) | NHWC BGR, raw 0-255 pixels, white alpha background and square padding **before** bicubic resizing, probability output. Size comes from ONNX. |
| `generic` | [generic.json](model-profiles/generic.json) | Explicit preprocessing and activation; size and input dtype come from ONNX. |

The WD preset follows the common
[SmilingWolf inference conventions](https://huggingface.co/spaces/SmilingWolf/wd-tagger).
Check individual exports; override fields if their requirements differ.
For your own model, start from `generic.json`. Its example assumes RGB NCHW,
stretch resizing, pixels divided by 255, and an output already containing
probabilities. These are **example settings, not automatic model detection**.

## Profile reference

Profiles are JSON objects. Unknown options, invalid values and missing files
produce errors. `preset`, `model` and `labels` are required. Generic profiles
also require `layout`, `color_order`, `resize`, `scale`, `mean`, `std` and
`activation`, so critical conventions cannot be silently inherited from Camie.
Preset fields can be overridden.

| Field | Values / meaning |
| --- | --- |
| `preset` | `camie`, `wd`, `generic`. |
| `model` | ONNX file path. |
| `labels` | Matching `.json`, `.csv` or `.txt` vocabulary path. |
| `external_data` | Optional array of external weight-file paths to include in validation and the fingerprint. |
| `input_name` | Optional exact image input name; does not enable extra inputs. |
| `output_name` | Exact tag output name. `null` selects the sole output; required when there are several. Camie defaults to `refined_predictions`. |
| `layout` | `NCHW`, `NHWC`, `CHW`, `HWC`; not inferred from ambiguous dimensions. |
| `size` | `[width, height]`; must match static dimensions. `null` infers from ONNX. Dynamic dimensions require explicit values. Camie defaults to `[512, 512]`. |
| `color_order` | `RGB`, `BGR`, `GRAY`. |
| `resize` | `stretch`, `letterbox`, `pad_square`, `center_crop`. |
| `interpolation` | `nearest`, `bilinear`, `bicubic`, `lanczos`. Generic default: `bilinear`. |
| `scale` | Positive divisor of pixel values, e.g. `255` for 0-1 or `1` for 0-255. |
| `mean`, `std` | Per-channel arrays in the configured color order; one value for `GRAY`, three otherwise. Standard deviations must be positive. |
| `pad_color` | RGB integers 0-255. Camie: `[124,116,104]`; WD/generic: white. |
| `alpha_background` | RGB color for transparency compositing. WD/generic: white. `null` discards alpha, preserving Camie's existing behavior. |
| `activation` | `none`: probabilities already in 0-1; `sigmoid`: independent multi-label logits; `softmax`: mutually exclusive class logits. Never inferred from sample scores. |
| `default_category` | Namespace for uncategorized labels; default `general`. |
| `category_map` | Map source category names/codes to tool categories. Extends/overrides the preset mapping. |

EXIF orientation is applied first. `letterbox` preserves aspect ratio, resizes
to fit and centers on a padded target. `pad_square` pads to the longest original
edge, then resizes, and requires a square target. `center_crop` uses a centered
aspect-ratio crop fitted to the target; it is not an arbitrary sequence of
model-specific resize/crop operations.

After resizing and channel conversion, normalization is:

```text
tensor = (pixel / scale - mean) / std
```

The result is laid out and converted to the graph's input type. Integer inputs
are rounded to nearest-even; out-of-range values fail instead of wrapping.
Set the affine normalization to the export's quantization convention when using
integer image inputs. Nonfinite tensors and scores are rejected.

Global and category thresholds keep working for every profile. Scores must pass
the applicable threshold inclusively. A default of `0.5` may be too high for
softmax classifiers with many classes; tune thresholds using a labeled sample.

### External weights and provenance

ONNX Runtime resolves external tensor files from the graph, not from
`external_data`. List **all** of those files in `external_data` as well, so
changes to external weights are reflected in the fingerprint. Example:

```json
{
  "preset": "wd",
  "model": "model.onnx",
  "labels": "selected_tags.csv",
  "external_data": ["model.onnx.data"]
}
```

Each successful image record stores the effective profile (including resolved
size, input dtype and input/output names), preprocessing version and thresholds.
The fingerprint covers the ONNX file, labels, declared external files and
effective inference settings. Moving an unchanged model does not change it.
Replace files between runs, not while a process is using them.

## Label formats

The label order **must match the output vector**. The graph's tag count is
checked at load when static, and the actual count is checked on every prediction.
Duplicate/empty names, unknown categories and inconsistent mappings are errors.

- **Text:** UTF-8, one tag per line. A trailing newline is fine; empty internal
  lines are not skipped. All tags use `default_category`.
- **CSV:** a `name` header and optional `category`. The WD preset requires both.
  Other columns such as Danbooru `tag_id` and `count` are ignored. **Row order**,
  not tag IDs or names, defines the output indices.
- **JSON:** an array of names or objects with `name` and optional `category`;
  a zero-based contiguous index object such as `{"0":"cat","1":"dog"}`;
  an `id2label` object; or `idx_to_tag` with optional `tag_to_category`.
  The original nested Camie `dataset_info.tag_mapping` metadata also works.
  When a category mapping is supplied, it must cover every label.

For example:

```json
[
  {"name": "alice", "category": "character"},
  {"name": "example_series", "category": "copyright"},
  {"name": "blue_hair", "category": "general"}
]
```

Output namespaces remain `character/`, `copyright/`, `artist/`, `general/` and
`rating/`, keeping statistics, cleanup and tag ownership compatible.
`meta` and `year` remain excluded from sidecar tags. Use `category_map`, for
example `{"species":"general"}`, to translate custom categories rather than
creating new namespace roots.

The WD preset maps numeric categories `0` to general, `1` to artist, `3` to
copyright, `4` to character, `5` to meta and `9` to rating. Numeric JSON categories
are converted to string keys before applying this mapping.
