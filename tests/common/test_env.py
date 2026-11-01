"""Reading secrets from .env, without ever showing them."""

import os

import pytest

from common.env import load_env, require


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("HEARKEN_TEST_A", "HEARKEN_TEST_B", "HEARKEN_TEST_C", "HEARKEN_TEST_D"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_names_and_values_are_read_with_quotes_and_comments_handled(tmp_path, clean_env):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n\nHEARKEN_TEST_A=plain\nHEARKEN_TEST_B = \"quoted value\"\nHEARKEN_TEST_C='single'\nnot a setting\n",
        encoding="utf-8",
    )
    load_env(env)
    assert os.environ["HEARKEN_TEST_A"] == "plain"
    assert os.environ["HEARKEN_TEST_B"] == "quoted value"
    assert os.environ["HEARKEN_TEST_C"] == "single"


def test_a_value_set_outside_the_file_wins(tmp_path, clean_env):
    clean_env.setenv("HEARKEN_TEST_A", "from the shell")
    env = tmp_path / ".env"
    env.write_text("HEARKEN_TEST_A=from the file\n", encoding="utf-8")
    load_env(env)
    assert os.environ["HEARKEN_TEST_A"] == "from the shell"


def test_a_missing_file_is_fine_and_a_missing_key_names_only_the_variable(tmp_path, clean_env):
    load_env(tmp_path / "no-such-file")
    with pytest.raises(RuntimeError, match="HEARKEN_TEST_D is not set"):
        require("HEARKEN_TEST_D")


def test_the_real_env_file_is_ignored_by_git():
    """The key must never reach GitHub."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["git", "check-ignore", ".env"], cwd=root, capture_output=True, text=True)
    assert result.stdout.strip() == ".env"
