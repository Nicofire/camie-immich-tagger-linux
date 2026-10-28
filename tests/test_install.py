import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


INSTALLER = Path(__file__).resolve().parents[1] / "install.sh"
POSIX_SYMLINKS = pytest.mark.skipif(sys.platform == "win32", reason="Requires POSIX symlinks")


@pytest.fixture
def bash():
    if sys.platform == "win32":
        candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        executable = str(candidate) if candidate.is_file() else None
    else:
        executable = shutil.which("bash")
    if executable is None:
        pytest.skip("Bash is not installed")
    return executable


@pytest.fixture
def register(bash, tmp_path):
    source = INSTALLER.read_text(encoding="utf-8")
    body = source.split("register_command() {\n", 1)[1].split("\n}\n", 1)[0]
    setup = """
set -euo pipefail
SUDO=""
info() { printf 'INFO: %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*"; }
mkdir -p "project with spaces/venv/bin"
target="$PWD/project with spaces/venv/bin/camie-tagger"
link="$PWD/bin/camie-tagger"
touch "$target"
"""

    def run(script):
        result = subprocess.run(
            [bash, "-c", setup + "\nregister_command() {\n" + body + "\n}\n" + script],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    return run


def test_installer_has_valid_bash_syntax(bash):
    result = subprocess.run(
        [bash, "-n", INSTALLER.as_posix()], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


@POSIX_SYMLINKS
def test_command_registration_creates_link_and_is_idempotent(register):
    output = register("""
PATH="$PWD/bin:$PATH"
register_command "$target" "$link"
test -L "$link"
test "$(readlink "$link")" = "$target"
register_command "$target" "$link"
test "$link" -ef "$target"
""")
    assert "Registered command:" in output
    assert "Command already registered:" in output
    assert "WARNING:" not in output


@pytest.mark.parametrize("setup, check", [
    ('printf sentinel > "$link"', 'test "$(cat "$link")" = sentinel'),
    ('mkdir "$link"', 'test -d "$link"'),
    pytest.param(
        'touch "$PWD/other"; ln -s "$PWD/other" "$link"',
        'test "$(readlink "$link")" = "$PWD/other"',
        marks=POSIX_SYMLINKS,
        id="foreign-symlink",
    ),
    pytest.param(
        'ln -s "$PWD/missing" "$link"',
        'test "$(readlink "$link")" = "$PWD/missing"',
        marks=POSIX_SYMLINKS,
        id="broken-symlink",
    ),
])
def test_registration_preserves_existing_entries(register, setup, check):
    output = register('mkdir -p "$PWD/bin"\n' + setup + '\nregister_command "$target" "$link"\n' + check)
    assert "already exists; leaving it unchanged" in output


@pytest.mark.parametrize("failed_command", ["mkdir", "ln"])
def test_registration_failure_warns_instead_of_aborting(register, failed_command):
    output = register(f"""
{failed_command}() {{ return 1; }}
register_command "$target" "$link"
test ! -e "$link"
printf 'continued\n'
""")
    assert "Could not register" in output
    assert "continued" in output


def test_registration_uses_non_overwriting_link_arguments_and_sudo(register):
    output = register("""
SUDO=sudo
sudo() { printf 'sudo: %s\n' "$1"; "$@"; }
ln() {
    test "$#" -eq 4 &&
    test "$1" = -sT &&
    test "$2" = -- &&
    test "$3" = "$target" &&
    test "$4" = "$link"
}
PATH="$PWD/bin:$PATH"
register_command "$target" "$link"
""")
    assert "sudo: mkdir" in output
    assert "sudo: ln" in output
    assert "Could not register" not in output


def test_registration_warns_if_destination_is_not_on_path(register):
    output = register("""
ln() { return 0; }
register_command "$target" "$link"
""")
    assert "is not on PATH" in output