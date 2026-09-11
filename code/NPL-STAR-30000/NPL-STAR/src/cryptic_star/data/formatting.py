"""Prompt and target templates.

One place decides what the model reads and writes, because four stages
(warm-start, generation, rationalisation, evaluation) have to agree exactly.

Source (encoder input):

    solve the cryptic clue. clue: Bird of prey ; enumeration: 5 ; letters: 5

Target (decoder output):

    definition: Bird of prey ; wordplay: EAGLE hidden in ... ; answer: E A G L E

The rationalisation pass appends `; hint: E A G L E` to the *source*. When an
accepted rationalised trace is added to the training set, the hint is removed -
that is the whole trick in STaR: the model learns to produce the reasoning
without ever having been given the answer.

No `{ } [ ] < > |` anywhere: the T5 tokenizer has no tokens for them.
"""

from __future__ import annotations

from ..config import FormatCfg
from ..text import (
    FIELD_SEP,
    answer_letters,
    collapse_ws,
    sanitize_field,
    sanitize_for_t5,
    spaced_letters,
)
from .cryptonite import Clue

PREFIX = "solve the cryptic clue."
DEF_KEY = "definition:"
WORDPLAY_KEY = "wordplay:"
ANSWER_KEY = "answer:"
HINT_KEY = "hint:"


def render_answer(answer: str, fmt: FormatCfg) -> str:
    return spaced_letters(answer) if fmt.spaced_answer else answer_letters(answer)


def render_source(clue: Clue, fmt: FormatCfg, hint: str | None = None) -> str:
    parts = [f"{PREFIX} clue: {sanitize_field(clue.clue)}"]
    if clue.enumeration:
        parts.append(f"enumeration: {sanitize_field(clue.enumeration)}")
    if fmt.include_letter_count:
        parts.append(f"letters: {len(clue.letters)}")
    if hint:
        parts.append(f"{HINT_KEY} {render_answer(hint, fmt)}")
    return collapse_ws(f" {FIELD_SEP} ".join(parts))


def render_target(
    clue: Clue,
    fmt: FormatCfg,
    wordplay: str | None = None,
    definition: str | None = None,
) -> str:
    """Build the target string from a rationale plus the gold answer."""
    parts = []
    if fmt.include_definition:
        d = sanitize_field(definition if definition is not None else clue.definition)
        parts.append(f"{DEF_KEY} {d}" if d else f"{DEF_KEY} unknown")
    w = sanitize_field(wordplay if wordplay is not None else clue.wordplay)
    parts.append(f"{WORDPLAY_KEY} {w}" if w else f"{WORDPLAY_KEY} unknown")
    parts.append(f"{ANSWER_KEY} {render_answer(clue.answer, fmt)}")
    return collapse_ws(f" {FIELD_SEP} ".join(parts))


def make_example(clue: Clue, fmt: FormatCfg, hint: str | None = None) -> dict:
    """A single seq2seq training row."""
    return {
        "clue_id": clue.clue_id,
        "source": render_source(clue, fmt, hint=hint),
        "target": render_target(clue, fmt),
        "answer": clue.answer,
        "letters": clue.letters,
    }


def example_from_trace(clue: Clue, trace: str, fmt: FormatCfg) -> dict:
    """Turn an *accepted self-generated trace* into a training row.

    `trace` is the model's own output; it is re-emitted verbatim as the target
    (after sanitising) and the source is rebuilt **without** any hint.
    """
    return {
        "clue_id": clue.clue_id,
        "source": render_source(clue, fmt, hint=None),
        "target": collapse_ws(sanitize_for_t5(trace)),
        "answer": clue.answer,
        "letters": clue.letters,
    }
