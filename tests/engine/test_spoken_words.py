"""Spoken words: the engine's loader and its load-time checks."""

import pytest

from engine.contract.spoken import SpokenWords
from packs.toy.pack import ToyPack

GOOD = """
status:
  ready:
    hinglish: "Taiyaar hai."
    english: "It's ready."
"""


def write(tmp_path, text):
    path = tmp_path / "spoken.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_a_phrase_in_each_language(tmp_path):
    words = SpokenWords.from_yaml(write(tmp_path, GOOD))
    assert words.phrase("status", "ready", "hinglish") == "Taiyaar hai."
    assert words.phrase("status", "ready", "english") == "It's ready."
    assert words.values("status") == {"ready"}


def test_a_missing_phrase_fails_loudly(tmp_path):
    words = SpokenWords.from_yaml(write(tmp_path, GOOD))
    with pytest.raises(KeyError, match="status.gone"):
        words.phrase("status", "gone", "hinglish")


def test_an_unknown_language_is_refused(tmp_path):
    words = SpokenWords.from_yaml(write(tmp_path, GOOD))
    with pytest.raises(ValueError, match="unknown language"):
        words.phrase("status", "ready", "tamil")


def test_a_missing_language_fails_when_the_file_loads(tmp_path):
    text = 'status:\n  ready:\n    hinglish: "Taiyaar hai."\n'
    with pytest.raises(ValueError, match="status.ready needs exactly"):
        SpokenWords.from_yaml(write(tmp_path, text))


def test_hindi_script_fails_when_the_file_loads(tmp_path):
    """Part 2 decision 12: English letters everywhere."""
    text = 'status:\n  ready:\n    hinglish: "तैयार है"\n    english: "It\'s ready."\n'
    with pytest.raises(ValueError, match="Hindi script"):
        SpokenWords.from_yaml(write(tmp_path, text))


def test_a_pack_without_spoken_words_fails_loudly_when_asked():
    words = ToyPack().spoken_words()
    assert words.values("status") == set()
    with pytest.raises(KeyError):
        words.phrase("status", "anything", "english")
