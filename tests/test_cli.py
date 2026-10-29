import pytest

from camie_tagger import cli
from camie_tagger.config import ConfigError
from camie_tagger.pipeline import RunResult
from camie_tagger.tier0 import Tier0Result


COMMANDS = ("run", "tier0", "immich-scan", "stats", "cleanup-orphans", "cleanup-tags", "doctor")


@pytest.mark.parametrize("arguments", [[], None])
def test_no_arguments_show_help_without_loading_configuration(arguments, monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "argv", ["camie-tagger"])
    monkeypatch.setattr(
        cli, "_settings_from_args", lambda args: pytest.fail("Help must not load configuration")
    )
    assert cli.main(arguments) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert "Commands:" in output.out
    for command in COMMANDS:
        assert command in output.out


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_root_help_has_compact_usage_and_command_list(flag, monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "_settings_from_args", lambda args: pytest.fail("Help must not load configuration")
    )
    with pytest.raises(SystemExit) as raised:
        cli.main([flag])
    assert raised.value.code == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out.startswith("usage: camie-tagger COMMAND [OPTIONS]\n")
    assert "Commands:\n" in output.out
    assert "positional arguments:" not in output.out
    assert "{run,tier0" not in output.out
    assert "camie-tagger COMMAND --help" in output.out
    examples = [line for line in output.out.splitlines() if line.lstrip().startswith("camie-tagger ")]
    assert examples
    assert all(line == "    " + line.lstrip() for line in examples)
    for command in COMMANDS:
        assert any(line.startswith(f"  {command} ") for line in output.out.splitlines())


def test_default_output_matches_explicit_help(capsys):
    assert cli.main([]) == 0
    default_output = capsys.readouterr()
    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])
    assert raised.value.code == 0
    assert capsys.readouterr() == default_output


@pytest.mark.parametrize("arguments", [["unknown-command"], ["--invalid-option"]])
def test_invalid_arguments_still_report_usage_errors(arguments, capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)
    assert raised.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("command", COMMANDS)
def test_subcommand_help_remains_available(command, capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main([command, "--help"])
    assert raised.value.code == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out.startswith(f"usage: camie-tagger {command} ")
    assert "--config PATH" in output.out
    assert "--scan-dir PATH" in output.out
    assert "--model-profile PATH" in output.out
    assert "--model-dir PATH" in output.out


@pytest.mark.parametrize("command", ["run", "doctor"])
def test_cli_model_profile_overrides_environment(command, tmp_path, monkeypatch):
    monkeypatch.setenv("CAMIE_MODEL_PROFILE", str(tmp_path / "environment.json"))
    monkeypatch.setattr(cli, "setup_logging", lambda **kwargs: None)
    profile = tmp_path / "selected.json"
    directory = tmp_path / "selected-model"
    args = cli.build_parser().parse_args(
        [command, "--model-profile", str(profile), "--model-dir", str(directory)]
    )
    settings = cli._settings_from_args(args)
    assert settings.model_profile == profile
    assert settings.model_dir == directory


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