"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, Settings, load_settings, validate_threshold
from .devices import DEVICE_CHOICES, DeviceError
from .doctor import run_doctor
from .immich import ImmichClient, ImmichError, trigger_scan
from .logging_setup import get_logger, redactor, setup_logging
from .maintenance import cleanup_orphans, cleanup_tags, show_stats
from .model import ModelError
from .pipeline import run_tagging
from .sidecar import ExiftoolError
from .state import StateStore
from .tier0 import DEFAULT_REPLACE_MIN_SIMILARITY, build_queue, run_tier0

DESCRIPTION = """\
Tag images locally for Immich with interchangeable ONNX models.
Write hierarchical XMP sidecars without changing the original images.
"""

EPILOG = (
    "Quick start:\n"
    "    camie-tagger doctor\n"
    "    camie-tagger run --mode test --limit 5 --dry-run\n"
    "    camie-tagger run --mode recent --immich-scan\n"
    "    camie-tagger tier0 --limit 20\n"
    "\n"
    "More help:\n"
    "    camie-tagger COMMAND --help\n"
    "    camie-tagger run --help\n"
    "\n"
    "Options follow the command. Settings come from .env; command line options win.\n"
    "\n"
    "Passing secrets on the command line makes them visible to other users via `ps`.\n"
    "Prefer .env (chmod 600) or environment variables on shared machines.\n"
)


class CommandHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _format_action(self, action: argparse.Action) -> str:
        if isinstance(action, argparse._SubParsersAction):
            return "".join(self._format_action(command) for command in action._get_subactions())
        return super()._format_action(action)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    group = common.add_argument_group("global options")
    group.add_argument("--config", metavar="PATH", help="path to the .env file")
    group.add_argument(
        "--model-dir", metavar="PATH",
        help="default Camie model directory; overrides CAMIE_MODEL_DIR",
    )
    group.add_argument(
        "--model-profile", metavar="PATH",
        help="ONNX model profile JSON; overrides CAMIE_MODEL_PROFILE",
    )
    group.add_argument(
        "--scan-dir",
        action="append",
        metavar="PATH",
        help="limit this run to one directory; repeatable, overrides CAMIE_SCAN_DIRS",
    )
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
        usage="%(prog)s COMMAND [OPTIONS]",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=CommandHelpFormatter,
        add_help=False,
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, title="Commands", metavar="COMMAND", prog=parser.prog,
    )
    parser.add_argument_group("Options").add_argument(
        "-h", "--help", action="help", help="show this help message and exit",
    )

    run = subparsers.add_parser(
        "run", parents=[common], help="tag images locally and write XMP sidecars"
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
        "--category-threshold",
        action="append",
        metavar="CATEGORY=VALUE",
        help="override one category threshold; repeatable",
    )
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
    run.add_argument("--reprocess", action="store_true", help="also predict already tagged images (all/test)")
    run.add_argument(
        "--allow-tag-removal", action="store_true",
        help="allow reprocessing to remove obsolete model-owned tags; requires --confirm or --dry-run",
    )
    run.add_argument("--confirm", action="store_true", help="confirm tag removals during reprocessing")

    tier0 = subparsers.add_parser(
        "tier0", parents=[common], help="look up missing tags via SauceNAO (uploads images)"
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
    tier0.add_argument(
        "--verify",
        action="store_true",
        help="also check images that already have character tags, and correct wrong ones",
    )
    tier0.add_argument(
        "--confirm",
        action="store_true",
        help="apply the changes found by --verify (without it, nothing is written)",
    )
    tier0.add_argument(
        "--replace-min-similarity",
        type=float,
        default=DEFAULT_REPLACE_MIN_SIMILARITY,
        metavar="PCT",
        help="similarity required before an existing tag is replaced "
        f"(default: {DEFAULT_REPLACE_MIN_SIMILARITY:.0f})",
    )
    tier0.add_argument(
        "--danbooru-only",
        action="store_true",
        help="ignore matches that have no Danbooru post (fewer hits, better tags)",
    )
    tier0.add_argument(
        "--allow-tag-removal", action="store_true",
        help="allow --verify to remove tracked automatic tags; untracked tags stay protected",
    )

    subparsers.add_parser(
        "immich-scan", parents=[common], help="trigger an Immich library and sidecar scan"
    )
    subparsers.add_parser("stats", parents=[common], help="tag coverage statistics")

    orphans = subparsers.add_parser(
        "cleanup-orphans", parents=[common], help="preview orphan sidecars; --confirm deletes them"
    )
    orphans.add_argument("--confirm", action="store_true", help="actually delete")

    old_tags = subparsers.add_parser(
        "cleanup-tags",
        parents=[common],
        help="preview Immich tags outside namespaces; --confirm deletes",
    )
    old_tags.add_argument("--confirm", action="store_true", help="actually delete")

    subparsers.add_parser("doctor", parents=[common], help="check the installation")

    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    scan_dirs = [Path(p).expanduser() for p in args.scan_dir] if args.scan_dir else None
    category_thresholds: dict[str, float] = {}
    for entry in getattr(args, "category_threshold", None) or []:
        category, separator, value = entry.partition("=")
        if not separator:
            raise ConfigError("--category-threshold requires CATEGORY=VALUE")
        category_thresholds[category.strip()] = validate_threshold(value, entry)
    settings = load_settings(
        args.config,
        scan_dirs=scan_dirs,
        model_dir=args.model_dir,
        model_profile=args.model_profile,
        immich_url=args.immich_url,
        immich_api_key=args.immich_api_key,
        saucenao_api_key=args.saucenao_api_key,
        device=args.device,
        fail_on_cpu_fallback=args.fail_on_cpu_fallback,
        log_level=args.log_level,
        log_file=args.log_file,
        threshold=getattr(args, "threshold", None),
        category_thresholds=category_thresholds or None,
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
    if args.allow_tag_removal and not args.reprocess:
        raise ConfigError("--allow-tag-removal requires --reprocess")
    if args.allow_tag_removal and not (args.confirm or args.dry_run):
        raise ConfigError("Preview removals with --dry-run, then apply with --confirm")
    state = StateStore(settings.data_dir)
    result = run_tagging(
        settings,
        state,
        mode=args.mode,
        limit=args.limit,
        threshold=args.threshold,
        dry_run=args.dry_run,
        reprocess=args.reprocess,
        allow_tag_removal=args.allow_tag_removal,
    )

    exit_code = 0
    failed = bool(result.failed)
    if args.tier0 and not args.dry_run:
        build_queue(settings, state)
        tier0_result = run_tier0(settings, state)
        failed = failed or bool(tier0_result.failed)
    if args.immich_scan and not args.dry_run:
        exit_code = _immich_scan(settings)

    return 1 if failed else exit_code


def _cmd_tier0(settings: Settings, args: argparse.Namespace) -> int:
    if args.allow_tag_removal and not args.verify:
        raise ConfigError("--allow-tag-removal requires --verify")
    state = StateStore(settings.data_dir)
    if not args.no_enqueue:
        # Narrowing the directories is temporary, so it must not discard the rest of the queue.
        build_queue(
            settings, state, include_tagged=args.verify, prune=not args.scan_dir,
            persist=args.enqueue_only or not args.verify or args.confirm,
        )
    if args.enqueue_only:
        return 0
    result = run_tier0(
        settings,
        state,
        limit=args.limit,
        min_similarity=args.min_similarity,
        verify=args.verify,
        confirm=args.confirm,
        replace_min_similarity=args.replace_min_similarity,
        danbooru_only=args.danbooru_only,
        allow_tag_removal=args.allow_tag_removal,
    )
    return 1 if result.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)

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
