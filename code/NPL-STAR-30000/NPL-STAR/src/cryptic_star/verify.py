"""Automatic labelling of generated traces - the `1/0` in the proposal.

STaR's filter is "did the trace end at the gold answer". Cryptic crosswords let
us add one free, strictly-correct extra check: the enumeration. A trace whose
answer has the wrong number of letters is wrong regardless of its reasoning, and
rejecting it early keeps obvious garbage out of the fine-tuning set.

Rejection reasons are recorded rather than collapsed into a boolean, because the
discriminator export needs to tell "confidently wrong answer" (a good hard
negative) apart from "unparseable output" (just noise).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .data.cryptonite import Clue
from .data.formatting import ANSWER_KEY, DEF_KEY, WORDPLAY_KEY
from .text import FIELD_SEP, answer_letters, collapse_ws, enum_signature

# The model is asked to end with "answer: ...". Take the LAST occurrence: some
# traces mention "answer:" mid-reasoning, and the final one is the commitment.
_ANSWER_RE = re.compile(rf"{ANSWER_KEY}\s*(.+?)\s*$", re.I | re.S)
_ANY_ANSWER_RE = re.compile(rf"{ANSWER_KEY}\s*([^;]+)", re.I)


class Reason:
    OK = "ok"
    NO_ANSWER_FIELD = "no_answer_field"
    EMPTY_ANSWER = "empty_answer"
    WRONG_ANSWER = "wrong_answer"
    WRONG_LENGTH = "wrong_length"
    NO_REASONING = "no_reasoning"
    DEGENERATE = "degenerate"
    # Matches one of the clue's documented alternative answers rather than the
    # gold one. Its own reason so it is neither counted as correct (which would
    # break comparability with the published baseline) nor mixed in with the
    # genuinely wrong answers the discriminator trains on.
    ALT_ANSWER = "alt_answer"


@dataclass
class Verdict:
    label: int
    reason: str
    predicted: str  # letters-only prediction, "" if unparseable
    trace: str
    wordplay: str = ""
    definition: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.label == 1


def extract_answer(trace: str) -> str:
    """Letters-only answer from a generated trace, or `""`."""
    matches = _ANY_ANSWER_RE.findall(trace or "")
    if matches:
        return answer_letters(matches[-1])
    m = _ANSWER_RE.search(trace or "")
    return answer_letters(m.group(1)) if m else ""


def extract_field(trace: str, key: str) -> str:
    """Pull `definition:` / `wordplay:` out of a trace; `""` if absent.

    Stops at the next field key and drops the `;` that separated them, so the
    value is the reasoning text alone (it gets re-used as training data).
    """
    m = re.search(rf"{key}\s*(.*?)(?=\s*(?:{DEF_KEY}|{WORDPLAY_KEY}|{ANSWER_KEY})|$)",
                  trace or "", re.I | re.S)
    if not m:
        return ""
    return collapse_ws(m.group(1)).rstrip(f"{FIELD_SEP} ")


def _is_degenerate(wordplay: str) -> bool:
    """Catch the classic seq2seq failure: a few tokens repeated forever."""
    words = wordplay.lower().split()
    if len(words) >= 8 and len(set(words)) <= 3:
        return True
    return bool(re.search(r"\b(\w+)( \1){4,}\b", wordplay.lower()))


def verify_trace(
    clue: Clue,
    trace: str,
    require_reasoning: bool = True,
    min_wordplay_chars: int = 8,
    accept_alt_answers: bool = False,
) -> Verdict:
    trace = collapse_ws(trace or "")
    wordplay = extract_field(trace, WORDPLAY_KEY)
    definition = extract_field(trace, DEF_KEY)
    predicted = extract_answer(trace)

    def fail(reason: str) -> Verdict:
        return Verdict(0, reason, predicted, trace, wordplay, definition)

    if ANSWER_KEY.rstrip(":").lower() not in trace.lower():
        return fail(Reason.NO_ANSWER_FIELD)
    if not predicted:
        return fail(Reason.EMPTY_ANSWER)
    if predicted != clue.letters:
        if predicted in clue.alt_letters:
            if not accept_alt_answers:
                return fail(Reason.ALT_ANSWER)
        else:
            # Length is checked first so the reason is the more informative one.
            gold_sig = enum_signature(clue.enumeration)
            if gold_sig and len(predicted) != sum(gold_sig):
                return fail(Reason.WRONG_LENGTH)
            return fail(Reason.WRONG_ANSWER)
    if require_reasoning:
        if len(wordplay.replace("unknown", "")) < min_wordplay_chars:
            return fail(Reason.NO_REASONING)
        if _is_degenerate(wordplay):
            return fail(Reason.DEGENERATE)

    return Verdict(1, Reason.OK, predicted, trace, wordplay, definition)


def exact_match(clue: Clue, trace: str) -> bool:
    """Answer-only EM, ignoring reasoning quality. Used by evaluate.py."""
    return extract_answer(trace) == clue.letters
