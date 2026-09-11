"""The 1/0 labelling rules. These are the tests that protect the training set."""

from cryptic_star.data.cryptonite import Clue
from cryptic_star.verify import Reason, extract_answer, verify_trace

CLUE = Clue(clue="Bird of prey caught out", answer="EAGLE", enumeration="5")


def trace(wordplay="EAGLE is hidden inside the phrase", answer="E A G L E"):
    return f"definition: Bird of prey ; wordplay: {wordplay} ; answer: {answer}"


def test_accepts_a_correct_trace():
    v = verify_trace(CLUE, trace())
    assert v.label == 1 and v.reason == Reason.OK
    assert v.predicted == "EAGLE"
    assert v.wordplay == "EAGLE is hidden inside the phrase"


def test_spaced_and_unspaced_answers_both_parse():
    assert extract_answer(trace(answer="E A G L E")) == "EAGLE"
    assert extract_answer(trace(answer="EAGLE")) == "EAGLE"
    assert extract_answer(trace(answer="eagle")) == "EAGLE"


def test_last_answer_field_wins():
    # Traces sometimes discuss "answer:" mid-reasoning; the final one is the
    # model's actual commitment.
    t = "wordplay: maybe answer: RAVEN then no ; answer: E A G L E"
    assert extract_answer(t) == "EAGLE"


def test_rejects_wrong_answer():
    v = verify_trace(CLUE, trace(answer="RAVEN"))
    assert v.label == 0 and v.reason == Reason.WRONG_ANSWER


def test_rejects_wrong_length_with_its_own_reason():
    v = verify_trace(CLUE, trace(answer="OWL"))
    assert v.label == 0 and v.reason == Reason.WRONG_LENGTH


def test_rejects_missing_answer_field():
    v = verify_trace(CLUE, "definition: Bird of prey ; wordplay: something")
    assert v.label == 0 and v.reason == Reason.NO_ANSWER_FIELD


def test_rejects_right_answer_with_no_reasoning():
    # The whole point is the trace, so a bare correct answer is not a positive.
    v = verify_trace(CLUE, "definition: bird ; wordplay: ; answer: E A G L E")
    assert v.label == 0 and v.reason == Reason.NO_REASONING


def test_rejects_degenerate_repetition():
    v = verify_trace(CLUE, trace(wordplay="hidden hidden hidden hidden hidden hidden"))
    assert v.label == 0 and v.reason == Reason.DEGENERATE


def test_alt_answer_gets_its_own_reason_and_is_not_a_positive():
    # Scoring alt answers as correct would break comparability with the
    # published 7.64% baseline, so by default they are rejected - but with a
    # distinct reason, so the report can say how much error is arguable.
    clue = Clue(clue="Bird of prey", answer="EAGLE", enumeration="5", alt_answers=["RAVEN"])
    v = verify_trace(clue, trace(answer="RAVEN"))
    assert v.label == 0 and v.reason == Reason.ALT_ANSWER


def test_alt_answer_is_accepted_when_asked_for():
    clue = Clue(clue="Bird of prey", answer="EAGLE", enumeration="5", alt_answers=["RAVEN"])
    v = verify_trace(clue, trace(answer="RAVEN"), accept_alt_answers=True)
    assert v.label == 1 and v.reason == Reason.OK


def test_a_plain_wrong_answer_is_still_wrong_when_alt_answers_exist():
    clue = Clue(clue="Bird of prey", answer="EAGLE", enumeration="5", alt_answers=["RAVEN"])
    v = verify_trace(clue, trace(answer="ROBIN"), accept_alt_answers=True)
    assert v.label == 0 and v.reason == Reason.WRONG_ANSWER


def test_multiword_answer_ignores_spacing():
    clue = Clue(clue="Something", answer="PIPE DREAM", enumeration="4,5")
    v = verify_trace(clue, trace(answer="P I P E D R E A M"))
    assert v.label == 1
