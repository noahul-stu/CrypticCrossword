"""The run tracker: jsonl events, scalar flattening, trace dumps.

TensorBoard is not exercised here (it needs the optional `tensorboard`
package); what is tested is that the tracker never becomes the reason a training
run fails, and that everything is flushed as it is written - a preemptible job
killed at hour 23 must still leave a readable log behind.
"""

import json

from cryptic_star.config import Config
from cryptic_star.tracking import RunTracker, _flatten


def tracker_for(tmp_path, **tracking_kwargs) -> RunTracker:
    cfg = Config()
    cfg.tracking.tensorboard = False  # exercised separately; keeps tests offline
    for key, value in tracking_kwargs.items():
        setattr(cfg.tracking, key, value)
    return RunTracker(tmp_path, cfg)


def events(tmp_path):
    path = tmp_path / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


CAND_ROWS = [
    {
        "clue": "Bird of prey caught out",
        "enumeration": "5",
        "letters": "EAGLE",
        "wordplay": "hidden word",
        "n_correct": 1,
        "candidates": [
            {"label": 1, "reason": "ok", "predicted": "EAGLE", "trace": "wordplay: hidden"},
        ],
    },
    {
        "clue": "Some other clue",
        "enumeration": "4",
        "letters": "OWLS",
        "wordplay": None,
        "n_correct": 0,
        "candidates": [
            {"label": 0, "reason": "wrong_answer", "predicted": "OWL", "trace": "wordplay: guess"},
        ],
    },
]


def test_events_are_written_one_json_object_per_line(tmp_path):
    with tracker_for(tmp_path) as t:
        t.event("generation", iteration=2, scalars={"pass_at_n": 0.25}, model="flan-t5-base")

    rows = events(tmp_path)
    assert len(rows) == 1
    assert rows[0]["event"] == "generation"
    assert rows[0]["iteration"] == 2
    assert rows[0]["pass_at_n"] == 0.25
    assert rows[0]["model"] == "flan-t5-base"
    assert "elapsed_s" in rows[0]


def test_lines_are_flushed_before_close(tmp_path):
    # The whole point on a preemptible partition: a buffered log is an empty log.
    t = tracker_for(tmp_path)
    t.event("iteration_start", iteration=1)
    assert len(events(tmp_path)) == 1
    t.close()


def test_a_resubmitted_run_appends_instead_of_truncating(tmp_path):
    with tracker_for(tmp_path) as t:
        t.event("first_attempt")
    with tracker_for(tmp_path) as t:
        t.event("second_attempt")
    assert [r["event"] for r in events(tmp_path)] == ["first_attempt", "second_attempt"]


def test_disabled_tracker_writes_nothing(tmp_path):
    with tracker_for(tmp_path, enabled=False) as t:
        t.event("generation", scalars={"pass_at_n": 1.0})
        t.config(Config())
        assert t.trace_samples(CAND_ROWS, tmp_path / "x.md", iteration=1) is None
    assert not (tmp_path / "metrics.jsonl").exists()


def test_config_is_recorded_as_the_first_event(tmp_path):
    cfg = Config()
    with tracker_for(tmp_path) as t:
        t.config(cfg)
    rows = events(tmp_path)
    assert rows[0]["event"] == "config"
    assert rows[0]["config"]["star"]["seed"] == cfg.star.seed


def test_flatten_reaches_nested_numbers_and_skips_strings():
    flat = _flatten(
        "generation",
        {
            "pass_at_n": 0.25,
            "reject_reasons": {"ok": 3, "wrong_answer": 7},
            "note": "not a number",
            "answer_disjoint": True,
        },
    )
    assert flat == {
        "generation/pass_at_n": 0.25,
        "generation/reject_reasons/ok": 3.0,
        "generation/reject_reasons/wrong_answer": 7.0,
        "generation/answer_disjoint": 1.0,
    }


def test_trace_samples_show_both_solved_and_failed_clues(tmp_path):
    with tracker_for(tmp_path, trace_samples=4) as t:
        out = t.trace_samples(CAND_ROWS, tmp_path / "traces.md", iteration=3)

    text = out.read_text(encoding="utf-8")
    assert "# Iteration 3 trace samples" in text
    assert "[solved] Bird of prey caught out" in text
    assert "[failed] Some other clue" in text
    assert "`EAGLE`" in text and "wrong_answer" in text
    assert "1/2 clues solved" in text


def test_trace_samples_can_be_switched_off(tmp_path):
    with tracker_for(tmp_path, trace_samples=0) as t:
        assert t.trace_samples(CAND_ROWS, tmp_path / "traces.md", iteration=1) is None
    assert not (tmp_path / "traces.md").exists()


def test_trace_samples_survive_missing_optional_fields(tmp_path):
    rows = [{"n_correct": 0, "candidates": [{"label": 0}]}]
    with tracker_for(tmp_path, trace_samples=2) as t:
        assert t.trace_samples(rows, tmp_path / "traces.md", iteration=1) is not None


def test_scalars_land_in_the_run_s_own_tb_directory(tmp_path):
    """The layout `logging_dir` used to provide, now owned by the tracker.

    transformers 5 removed `logging_dir`, so the Trainer's built-in TensorBoard
    writer can no longer be aimed at `runs/<name>/tb`; train.py routes scalars
    through RunTracker instead. If that regresses, the curves scatter under the
    Trainer's output_dir and `tensorboard --logdir runs/<name>/tb` silently shows
    an empty dashboard - which reads as "training logged nothing".
    """
    import pytest

    pytest.importorskip("torch.utils.tensorboard")

    cfg = Config()
    cfg.tracking.tensorboard = True
    with RunTracker(tmp_path, cfg) as t:
        t.scalars({"loss": 1.5}, step=3, prefix="train/iter1")

    written = list((tmp_path / "tb").glob("events.out.tfevents.*"))
    assert written, f"no TensorBoard event file under {tmp_path / 'tb'}"
    assert written[0].stat().st_size > 0, "event file written but empty"


def test_scalars_do_nothing_when_tensorboard_is_off(tmp_path):
    with tracker_for(tmp_path) as t:
        t.scalars({"loss": 1.0}, step=1, prefix="train/iter1")
    assert not (tmp_path / "tb").exists(), "tb/ created despite tensorboard: false"
