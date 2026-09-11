"""Normalisation tests. No dataset needed - these run today."""

from cryptic_star.text import (
    answer_letters,
    enum_signature,
    enumeration_of,
    extract_definitions,
    norm_clue,
    sanitize_for_t5,
    spaced_letters,
    split_trailing_enumeration,
    strip_definition_markers,
)


def test_definition_markers_are_extracted_then_stripped():
    clue = "{Bird of prey} caught in the middle of a sequence"
    assert extract_definitions(clue) == ["Bird of prey"]
    assert strip_definition_markers(clue) == "Bird of prey caught in the middle of a sequence"


def test_multiple_definition_spans():
    assert extract_definitions("{Fast} runner is {quick}") == ["Fast", "quick"]


def test_no_braces_survive_sanitisation():
    # T5's tokenizer has no token for { or }, so they must be gone before
    # any text reaches the model.
    out = sanitize_for_t5("anagram of {LISTEN} -> [SILENT] <yes>")
    for ch in "{}[]<>|":
        assert ch not in out


def test_trailing_enumeration_split():
    assert split_trailing_enumeration("Some clue here (3,4)") == ("Some clue here", "3,4")
    assert split_trailing_enumeration("No enumeration") == ("No enumeration", None)


def test_clue_join_key_ignores_punctuation_case_and_markers():
    a = "{Bird of prey}, caught out! (5)"
    b = "Bird of prey caught out (5)"
    assert norm_clue(a) == norm_clue(b)


def test_answer_letters_and_enumeration():
    assert answer_letters("pipe dream") == "PIPEDREAM"
    assert answer_letters("well-known") == "WELLKNOWN"
    assert enumeration_of("pipe dream") == "4,5"
    assert enumeration_of("well-known") == "4-5"


def test_enum_signature_ignores_space_vs_hyphen():
    # The two datasets disagree on this; word lengths are the real signal.
    assert enum_signature("4,5") == enum_signature("4-5") == (4, 5)
    assert enum_signature("") == ()
    assert enum_signature(None) == ()


def test_spaced_letters():
    assert spaced_letters("pipe dream") == "P I P E D R E A M"
