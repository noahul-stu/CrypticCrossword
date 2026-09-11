"""Analysis metrics, on hand-built prediction rows."""

from cryptic_star.analyze import (
    analyse,
    bucket,
    candidate_answers,
    definition_is_grounded,
    learning_curve,
    majority_vote,
    to_markdown,
)


def pred(clue, gold, predicted, cands, top1=None, reason="ok", enum=None, publisher="Times"):
    trace = f"definition: {clue.split()[0]} ; wordplay: some reasoning ; answer: {predicted}"
    return {
        "clue": clue,
        "gold": gold,
        "predicted": predicted,
        "enumeration": enum or str(len(gold)),
        "publisher": publisher,
        "top1_correct": (predicted == gold) if top1 is None else top1,
        "any_correct": predicted == gold or gold in cands,
        "reason": reason,
        "trace": trace,
        "candidates": [f"wordplay: r ; answer: {c}" for c in cands],
    }


def test_candidate_answers_are_parsed_and_normalised():
    row = pred("Bird clue", "EAGLE", "EAGLE", ["E A G L E", "raven"])
    assert candidate_answers(row) == ["EAGLE", "RAVEN"]


def test_majority_vote_picks_the_most_agreed_answer():
    row = pred("Bird clue", "EAGLE", "RAVEN", ["EAGLE", "EAGLE", "EAGLE", "RAVEN"])
    answer, votes, total = majority_vote(row)
    assert (answer, votes, total) == ("EAGLE", 3, 4)


def test_majority_vote_breaks_ties_toward_beam_search():
    row = pred("Bird clue", "EAGLE", "RAVEN", ["EAGLE", "RAVEN"])
    answer, _, _ = majority_vote(row)
    assert answer == "RAVEN"  # the top-1/beam answer


def test_majority_vote_with_no_parseable_candidates():
    row = pred("Bird clue", "EAGLE", "EAGLE", [])
    assert majority_vote(row) == ("EAGLE", 0, 0)


def test_majority_vote_can_beat_top1():
    # The self-consistency baseline the discriminator has to beat.
    rows = [pred("A clue here", "EAGLE", "RAVEN", ["EAGLE", "EAGLE", "EAGLE"])]
    report = analyse(rows)
    assert report["headline"]["top1_em"] == 0.0
    assert report["headline"]["majority_vote_em"] == 1.0
    assert report["headline"]["majority_vote_gain_over_top1"] == 1.0


def test_definition_grounding():
    row = pred("Bird of prey caught out", "EAGLE", "EAGLE", [])
    row["trace"] = "definition: Bird of prey ; wordplay: x ; answer: EAGLE"
    assert definition_is_grounded(row) is True

    row["trace"] = "definition: something invented ; wordplay: x ; answer: EAGLE"
    assert definition_is_grounded(row) is False

    # No definition emitted -> excluded from the rate, not counted as failure.
    row["trace"] = "definition: unknown ; wordplay: x ; answer: EAGLE"
    assert definition_is_grounded(row) is None


def test_headline_metrics():
    rows = [
        pred("Clue one here", "EAGLE", "EAGLE", ["EAGLE", "RAVEN"]),
        pred("Clue two here", "OTTER", "RAVEN", ["OTTER", "RAVEN"], reason="wrong_answer"),
        pred("Clue three here", "COT", "OWL", ["OWL", "BAT"], reason="wrong_length"),
    ]
    h = analyse(rows)["headline"]
    assert h["n_clues"] == 3
    assert h["top1_em"] == 1 / 3
    assert h["pass_at_n"] == 2 / 3          # clue two recovered by a sample
    assert abs(h["oracle_gap"] - 1 / 3) < 1e-9
    assert h["n_traces_scored"] == 6
    assert h["trace_precision"] == 2 / 6


def test_failure_reasons_are_counted():
    rows = [
        pred("Clue one here", "OTTER", "RAVEN", [], reason="wrong_answer"),
        pred("Clue two here", "COT", "OWL", [], reason="wrong_length"),
        pred("Clue three here", "COT", "OWL", [], reason="wrong_length"),
    ]
    fr = analyse(rows)["failure_reasons"]
    assert fr["n_failures"] == 3
    assert fr["distribution"]["wrong_length"] == 2
    assert abs(fr["share"]["wrong_length"] - 2 / 3) < 1e-9


def test_unparseable_predictions_are_tracked():
    rows = [pred("Clue one here", "EAGLE", "", [], reason="no_answer_field")]
    assert analyse(rows)["headline"]["unparseable_rate"] == 1.0


def test_small_buckets_are_merged_not_reported_as_findings():
    rows = [pred(f"Clue number {i} here", "EAGLE", "EAGLE", []) for i in range(5)]
    out = bucket(rows, lambda r: r["gold"], min_n=20)
    assert list(out) == ["(small groups)"]
    assert out["(small groups)"]["n"] == 5


def test_buckets_split_when_large_enough():
    rows = [pred(f"Clue {i} here", "EAGLE", "EAGLE", []) for i in range(20)]
    rows += [pred(f"Other {i} here", "COT", "OWL", [], reason="wrong_length") for i in range(20)]
    out = bucket(rows, lambda r: r["gold"], min_n=20)
    assert out["EAGLE"]["top1_em"] == 1.0
    assert out["COT"]["top1_em"] == 0.0


def test_learning_curve_flattens_history():
    history = [
        {
            "iteration": 1,
            "train_examples": 100,
            "new_examples": 40,
            "generation": {"clues_with_at_least_one_correct": 30, "pass_at_n": 0.3},
            "rationalization": {"clues_with_at_least_one_correct": 10},
            "val": {"top1_em": 0.1, "pass_at_n": 0.25},
        }
    ]
    curve = learning_curve(history)
    assert curve[0]["val_top1_em"] == 0.1
    assert curve[0]["rationalized_recovered"] == 10


def test_markdown_renders_without_error():
    rows = [pred("Clue one here", "EAGLE", "EAGLE", ["EAGLE"])]
    md = to_markdown(analyse(rows), learning_curve([]))
    assert "# Generator analysis" in md
    assert "top-1 EM" in md
    assert "unseen clues" in md
