import pytest

from camie_tagger import cli
from camie_tagger.config import ConfigError
from camie_tagger.pipeline import RunResult
from camie_tagger.tier0 import Tier0Result


def test_cli_category_thresholds_override_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMIE_CATEGORY_THRESHOLDS", '{"character":0.7,"copyright":0.6}')
    monkeypatch.setattr(cli, "setup_logging", lambda **kwargs: None)
    args = cli.build_parser().parse_args(
        ["run", "--threshold", "0.8", "--category-threshold", "character=0.4"]
    )
    settings = cli._settings_from_args(args)
    assert settings.threshold == 0.8
    assert settings.category_thresholds == {"character": 0.4, "copyright": 0.6}


def test_cli_global_threshold_is_validated(monkeypatch):
    monkeypatch.setattr(cli, "setup_logging", lambda **kwargs: None)
    args = cli.build_parser().parse_args(["run", "--threshold", "nan"])
    with pytest.raises(ConfigError):
        cli._settings_from_args(args)


def test_partial_tagging_failure_sets_exit_code(settings, monkeypatch):
    monkeypatch.setattr(cli, "run_tagging", lambda *args, **kwargs: RunResult(tagged=1, failed=1))
    args = cli.build_parser().parse_args(["run"])
    assert cli._cmd_run(settings, args) == 1


def test_tier0_failure_sets_exit_code(settings, monkeypatch):
    monkeypatch.setattr(cli, "run_tier0", lambda *args, **kwargs: Tier0Result(failed=1))
    args = cli.build_parser().parse_args(["tier0", "--no-enqueue"])
    assert cli._cmd_tier0(settings, args) == 1


def test_combined_run_reports_tier0_failure(settings, monkeypatch):
    monkeypatch.setattr(cli, "run_tagging", lambda *args, **kwargs: RunResult(tagged=1))
    monkeypatch.setattr(cli, "build_queue", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "run_tier0", lambda *args, **kwargs: Tier0Result(failed=1))
    args = cli.build_parser().parse_args(["run", "--tier0"])
    assert cli._cmd_run(settings, args) == 1


def test_run_tag_removal_needs_explicit_confirmation(settings):
    args = cli.build_parser().parse_args(["run", "--mode", "all", "--reprocess", "--allow-tag-removal"])
    with pytest.raises(ConfigError):
        cli._cmd_run(settings, args)


def test_reprocess_flags_are_forwarded(settings, monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli, "run_tagging",
        lambda *args, **kwargs: calls.append(kwargs) or RunResult(),
    )
    args = cli.build_parser().parse_args(
        ["run", "--mode", "all", "--reprocess", "--allow-tag-removal", "--dry-run"]
    )
    assert cli._cmd_run(settings, args) == 0
    assert calls[0]["reprocess"] is True
    assert calls[0]["allow_tag_removal"] is True


def test_verification_preview_does_not_save_enqueued_work(settings, tagging, monkeypatch):
    image = settings.scan_dirs[0] / "image.jpg"
    image.touch()
    tagging.sidecars[image] = ["copyright/series"]
    monkeypatch.setattr(cli, "run_tier0", lambda *args, **kwargs: Tier0Result())
    args = cli.build_parser().parse_args(["tier0", "--verify"])
    assert cli._cmd_tier0(settings, args) == 0
    assert not settings.data_dir.exists()