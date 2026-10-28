# Operations

[Home](../README.md) | [Setup](setup.md) | [Usage](usage.md) |
[Development](development.md)

## Scheduling

There is no background daemon. Use `crontab -e` for the runtime account and adapt
the [cron example file](cron.example). User crontabs do not include a username
column; `/etc/cron.d` entries do.

```cron
CAMIE=/opt/camie-immich-tagger-linux/venv/bin/camie-tagger
LOCK=/run/lock/camie-tagger.lock

0 3 * * * /usr/bin/flock -n "$LOCK" "$CAMIE" run --mode recent --immich-scan
```

This uses the server's timezone and absolute paths, so virtual-environment
activation and a particular working directory are unnecessary. Configuration
still comes from the checkout's `.env`, unless `--config` selects another file.

### Locks and runtime accounts

`flock` is external scheduling protection, not an application feature. On Debian/
Ubuntu it is supplied by `util-linux`; verify `/usr/bin/flock` exists. The account
must be able to create and open the lock file. The examples suit the root setup
in the README. For a dedicated account, choose a protected, writable lock location
or pre-create the file with the correct ownership and permissions.

Use the **same lock path** for all jobs touching the same images/state. `-n` skips
a run immediately when another holds the lock. Separate timed jobs do not wait
for each other merely because their start times differ. Manual commands must
also use that lock if they can overlap:

```bash
flock -n /run/lock/camie-tagger.lock \
  /opt/camie-immich-tagger-linux/venv/bin/camie-tagger run --mode recent --immich-scan
```

The lock protects the tagger process only, not Immich's asynchronous server jobs.
Before enabling cron, test the exact command as its account and verify **every**
expected image mount, `.env` access, sidecar write permissions and state/log access.
For removable/network storage, use your deployment's mount/readiness checks; do
not schedule destructive cleanup when a mount can disappear.

### Optional Tier 0 schedule

To finish local tagging, then reverse search, then request an Immich refresh in
one process, replace the daily entry rather than adding a second competing one:

```cron
0 3 * * * /usr/bin/flock -n "$LOCK" "$CAMIE" run --mode recent --tier0 --immich-scan
```

This uploads eligible images and uses the configured SauceNAO quota/interval.
At 100 searches and an 18-second interval, allow roughly 30 minutes just for
spacing, plus inference and network time. `run --limit` does not cap that backfill.
For a separate Tier 0 job, use an explicit limit, then request `immich-scan`
after success. Use the same lock for the entire sequence.

Rotating application logs need no shell redirection. Cron output/mail is separate:
configure monitoring for failed/skipped runs and inspect processing summaries.
Do not routinely discard all stderr or schedule confirmed cleanup commands.

## Immich import

The tagger writes `XMP-digiKam:TagsList` to adjacent `<image-filename>.xmp` files.
Immich must be able to read those sidecars through its external-library mounts.
The tagger's local `--scan-dir` and Immich's library scope are different controls.

`camie-tagger immich-scan` and `run --immich-scan` submit:

1. `POST /api/libraries/{id}/scan` for each `IMMICH_LIBRARY_IDS` UUID.
2. `PUT /api/jobs/sidecar` with `{"command":"start","force":true}`.

The second request is **global**, not restricted to those library IDs or a local
scan directory. Missing library IDs skip only the first step; the Sidecar request
can still run. API acceptance is not completion: watch Immich's jobs and logs.
The library scan is not awaited before the Sidecar request.

### Version-specific behavior

For **Immich v3.3.5**, the forced Sidecar job checks all assets and chains metadata
extraction, including assets whose linked sidecar path is unchanged. Upstream
continues that follow-up even when the path check reports `Skipped`. Thus an
existing linked XMP can be reread; the command is not just discovery of new files.
If an initially unlinked/new asset is missed while a library scan is still running,
wait for the library scan to complete and request another scan.

The legacy jobs endpoint has been deprecated since v2.4 but remains available in
v3.3.5. Future Immich versions may change endpoints or import behavior. The tool
does not offer a discovery-only/refresh mode selector. Relevant versioned upstream
implementations are
[job.service.ts](https://github.com/immich-app/immich/blob/v3.3.5/server/src/services/job.service.ts)
and
[metadata.service.ts](https://github.com/immich-app/immich/blob/v3.3.5/server/src/services/metadata.service.ts).

> [!WARNING]
> Immich metadata synchronization can replace an asset's tag associations from
> imported metadata. Tags assigned only in the Immich database are not guaranteed
> to survive. Additive local sidecar handling is not a database-tag backup.

Back up Immich and sidecars, and test with a small representative library before
large imports or upstream upgrades. A small **local** test can still submit a
global Sidecar job when `--immich-scan` is included.

## Logs and state

Default paths are beneath the checkout; configuration can relocate them.

| File | Contents |
| --- | --- |
| `logs/camie-tagger.log` | Rotating log, 10 MB per file, 5 backups. |
| `data/processed.json` | Processed paths, owned tags and inference profiles. |
| `data/tier0_queue.json` | Pending reverse searches. |
| `data/tier0_progress.json` | Completed search results, including misses. |

Keep state together with sidecar backups. Deleting it loses automatic-tag
provenance and does not force retagging of existing XMP. Use `--reprocess` instead.
The stored inference profile records model/metadata fingerprints, preprocessing
version and thresholds; a changed profile does not automatically schedule work.
Limited runs and narrowed directories preserve unrelated records.

Successful sidecar writes are read back for verification. Unreadable XMP is not
treated as empty metadata. Tagging/Tier 0 processing failures return a nonzero
status, including reported partial failures; some maintenance per-item failures
are logged without a nonzero final status. Inspect summaries/logs as well as exit
codes. There is no built-in cross-process lock or general automatic retry service.

## Updating

Back up `.env`, state and sidecars first, and pause scheduled jobs. Review changes
before updating, especially around preprocessing, ownership tracking and Immich
compatibility. For a clean git checkout:

```bash
cd /opt/camie-immich-tagger-linux
git pull --ff-only
venv/bin/python -m pip install -e .
venv/bin/camie-tagger doctor
venv/bin/camie-tagger run --mode test --limit 5 --dry-run
```

This refreshes package dependencies but does not intentionally replace the model
or hardware-specific ONNX Runtime distribution. Follow release-specific runtime
instructions when needed. Test a small reprocessing preview separately if you
want old images evaluated with the new inference settings.

## Troubleshooting

| Symptom | First checks |
| --- | --- |
| Command not found | Check PATH or use the full `venv/bin/` path. |
| Configuration cannot be read | Check `.env` path, owner and runtime account. |
| Missing model / runtime | Run `doctor`; check both model files and runtime. |
| ExifTool not found | Check `libimage-exiftool-perl` and `CAMIE_EXIFTOOL`. |
| GPU present, CPU selected | Check providers, drivers and render-node access. |
| Immich `403` | Check required API permissions and admin/account restrictions. |
| No assets in a library | Check Immich import paths and mounts independently. |
| Tags missing in Immich | Check sidecar access, job completion and failures. |
| Old image unchanged | `recent` is path-based; preview `all --reprocess`. |
| Tier 0 skips an image | Check eligibility, completed progress and similarity. |

For GPU issues, see [GPU/container setup](setup.md#gpu-and-container-access).
An available provider alone does not prove a successful real-model inference.
For XMP problems, inspect a sidecar without changing it:

```bash
exiftool -j -XMP-digiKam:TagsList /mnt/media/example.jpg.xmp
```

If a downloaded installer has Windows line endings, normalize that file before
running it on Linux:

```bash
sed -i 's/\r$//' install.sh
```

Share the command, version, sanitized error and relevant diagnostics when reporting
issues. Do not include API keys, private images or your complete `.env`.
