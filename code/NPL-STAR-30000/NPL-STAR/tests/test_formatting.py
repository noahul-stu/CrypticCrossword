"""Prompt/target templates, and the hint-leak guard.

If `test_hint_never_reaches_the_training_source` fails, the model is being
trained with the answer in its input and every result is meaningless.
"""

import pytest

from cryptic_star.config import FormatCfg
from cryptic_star.data.cryptonite import Clue
from cryptic_star.data.formatting import (
    HINT_KEY,
    example_from_trace,
    make_example,
    render_source,
    render_target,
)
from cryptic_star.rationalize import assert_no_hint_leak

FMT = FormatCfg()
CLUE = Clue(
    clue="Bird of prey caught out",
    answer="EAGLE",
    enumeration="5",
    definition="Bird of prey",
    wordplay="EAGLE hidden in the surface",
    clue_id="t1",
)


def test_source_has_clue_enumeration_and_letter_count():
    src = render_source(CLUE, FMT)
    assert "Bird of prey caught out" in src
    assert "enumeration: 5" in src
    assert "letters: 5" in src
    assert HINT_KEY not in src


def test_source_never_contains_the_answer():
    assert "EAGLE" not in render_source(CLUE, FMT)


def test_target_carries_definition_wordplay_and_spaced_answer():
    tgt = render_target(CLUE, FMT)
    assert "definition: Bird of prey" in tgt
    assert "wordplay: EAGLE hidden in the surface" in tgt
    assert "answer: E A G L E" in tgt


def test_unspaced_answer_option():
    tgt = render_target(CLUE, FormatCfg(spaced_answer=False))
    assert "answer: EAGLE" in tgt


def test_missing_rationale_becomes_unknown_not_empty():
    bare = Clue(clue="X", answer="EAGLE", enumeration="5")
    tgt = render_target(bare, FMT)
    assert "wordplay: unknown" in tgt


def test_hinted_source_contains_the_answer_but_the_example_does_not():
    hinted = render_source(CLUE, FMT, hint=CLUE.answer)
    assert HINT_KEY in hinted and "E A G L E" in hinted
    # The training row rebuilt from an accepted hinted trace must be hint-free.
    row = example_from_trace(CLUE, render_target(CLUE, FMT), FMT)
    assert HINT_KEY not in row["source"]


def test_hint_never_reaches_the_training_source():
    good = [make_example(CLUE, FMT)]
    assert_no_hint_leak(good)

    leaked = [{"source": render_source(CLUE, FMT, hint="EAGLE"), "target": "x"}]
    with pytest.raises(AssertionError):
        assert_no_hint_leak(leaked)


def test_field_separator_is_stripped_from_values():
    messy = Clue(
        clue="Clue with ; a separator",
        answer="EAGLE",
        enumeration="5",
        wordplay="first part ; second part",
    )
    tgt = render_target(messy, FMT)
    # Exactly the separators the parser expects: definition | wordplay | answer.
    assert tgt.count(";") == 2
