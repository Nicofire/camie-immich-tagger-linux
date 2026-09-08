"""Command line interface."""

from __future__ import annotations

import argparse
import sys

from .config import ConfigError, Settings, load_settings
from .devices import DEVICE_CHOICES, DeviceError
from .doctor import run_doctor
from .immich import ImmichClient, ImmichError, trigger_scan
from .logging_setup import get_logger, redactor, setup_logging
from .maintenance import cleanup_orphans, cleanup_tags, show_stats
from .model import ModelError
from .pipeline import run_tagging
from .sidecar import ExiftoolError
from .state import StateStore
from .tier0 import build_queue, run_tier0

DESCRIPTION = """\
Local anime/illustration auto-tagging for Immich.

Tags images with camie-tagger-v2, writes hierarchical XMP sidecars, and asks Immich
to pick them up. Configuration is read from .env; command line options win over it.
"""

EPILOG = """\
examples:
  camie-tagger doctor
  camie-tagger run --mode test --limit 5 --dry-run
  camie-tagger run --mode all
  camie-tagger run --mode recent --immich-scan
  camie-tagger run --mode recent --tier0 --immich-scan
  camie-tagger cleanup-tags --confirm

Passing secrets on the command line makes them visible to other users via `ps`.
Prefer .env (chmod 600) or environment variables on shared machines.
"""


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    group = common.add_argument_group("global options")
    group.add_argument("--config", metavar="PATH", help="path to the .env file")
    group.add_argument("--immich-url", metavar="URL")
    group.add_argument("--immich-api-key", metavar="KEY")
    group.add_argument("--saucenao-api-key", metavar="KEY")
    group.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        help="inference device (default: auto)",
    )
    group.add_argument(
        "--fail-on-cpu-fallback",
        action="store_true",
        default=None,
        help="abort instead of running on the CPU when no GPU is usable",
    )
    group.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="console and file log level",
    )
    group.add_argument("--log-file", metavar="PATH", help="override the log file path")

    parser = argparse.ArgumentParser(
        prog="camie-tagger",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run", parents=[common], help="tag images and write sidecars"
    )
    run.add_argument(
        "--mode",
        choices=("test", "recent", "all"),
        default="recent",
        help="test: a small sample; recent: only new images; all: the whole library",
    )
    run.add_argument("--limit", type=int, help="maximum number of images to process")
    run.add_argument("--threshold", type=float, help="confidence threshold (0-1)")
    run.add_argument(
        "--tier0", action="store_true", help="run the SauceNAO backfill afterwards"
    )
    run.add_argument(
        "--immich-scan",
        action="store_true",
        help="ask Immich to rescan and import the sidecars when finished",
    )
    run.add_argument(
        "--dry-run", action="store_true", help="predict tags but do not write anything"
    )

    tier0 = subparsers.add_parser(
        "tier0", parents=[common], help="SauceNAO reverse search backfill"
    )
    tier0.add_argument("--limit", type=int, help="maximum number of searches this run")
    tier0.add_argument(
        "--min-similarity", type=float, help="minimum SauceNAO similarity percentage"
    )
    tier0.add_argument(
        "--enqueue-only", action="store_true", help="update the queue and stop"
    )
    tier0.add_argument(
        "--no-enqueue", action="store_true", help="use the existing queue as-is"
    )

    subparsers.add_parser(
        "immich-scan", parents=[common], help="trigger an Immich library and sidecar scan"
    )
    subparsers.add_parser("stats", parents=[common], help="tag coverage statistics")

    orphans = subparsers.add_parser(
        "cleanup-orphans", parents=[common], help="delete sidecars whose image is gone"
    )
    orphans.add_argument("--confirm", action="store_true", help="actually delete")

    old_tags = subparsers.add_parser(
        "cleanup-tags",
        parents=[common],
        help="delete Immich tags outside this tool's namespaces",
    )
    old_tags.add_argument("--confirm", action="store_true", help="actually delete")

    subparsers.add_parser("doctor", parents=[common], help="check the installation")

    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = load_settings(
        args.config,
        immich_url=args.immich_url,
        immich_api_key=args.immich_api_key,
        saucenao_api_key=args.saucenao_api_key,
        device=args.device,
        fail_on_cpu_fallback=args.fail_on_cpu_fallback,
        log_level=args.log_level,
        log_file=args.log_file,
    )
    redactor.add(*settings.secrets())
    setup_logging(
        level=settings.log_level,
        log_file=settings.resolved_log_file,
        max_bytes=settings.log_max_bytes,
        backups=settings.log_backups,
    )
    return settings


def _immich_scan(settings: Settings) -> int:
    settings.require_immich()
    client = ImmichClient(settings.immich_url, settings.immich_api_key)
    return 0 if trigger_scan(client, settings.library_ids) else 1


def _cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    state = StateStore(settings.data_dir)
    result = run_tagging(
        settings,
        state,
        mode=args.mode,
        limit=args.limit,
        threshold=args.threshold,
        dry_run=args.dry_run,
    )

    exit_code = 0
    if args.tier0 and not args.dry_run:
        build_queue(settings, state)
        run_tier0(settings, state)
    if args.immich_scan and not args.dry_run:
        exit_code = _immich_scan(settings)

    return 1 if result.failed and result.tagged == 0 else exit_code


def _cmd_tier0(settings: Settings, args: argparse.Namespace) -> int:
    state = StateStore(settings.data_dir)
    if not args.no_enqueue:
        build_queue(settings, state)
    if args.enqueue_only:
        return 0
    run_tier0(settings, state, limit=args.limit, min_similarity=args.min_similarity)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = _settings_from_args(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    log = get_logger()
    try:
        if args.command == "run":
            return _cmd_run(settings, args)
        if args.command == "tier0":
            return _cmd_tier0(settings, args)
        if args.command == "immich-scan":
            return _immich_scan(settings)
        if args.command == "stats":
            return show_stats(settings)
        if args.command == "cleanup-orphans":
            return cleanup_orphans(settings, args.confirm)
        if args.command == "cleanup-tags":
            return cleanup_tags(settings, args.confirm)
        if args.command == "doctor":
            return run_doctor(settings)
    except KeyboardInterrupt:
        log.warning("Interrupted.")
        return 130
    except (ConfigError, DeviceError, ExiftoolError, ModelError, ImmichError) as exc:
        log.error("%s", exc)
        return 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
