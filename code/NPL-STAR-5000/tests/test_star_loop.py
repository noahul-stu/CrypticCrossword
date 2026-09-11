"""Resume logic and the processed-data cache guard.

None of this needs a GPU or a model: the point is that a preempted job resumes
instead of restarting, and that a cached `data/processed` built with different
row limits is rebuilt rather than silently reused.
"""

import json

from cryptic_star.config import Config
from cryptic_star.star_loop import (
    _PROCESSED_FILES,
    accepted_examples,
    dedup_examples,
    ensure_processed,
    model_is_complete,
    prep_fingerprint,
    reusable_candidates,
)


def cfg_with(tmp_path, **data_kwargs) -> Config:
    cfg = Config()
    cfg.data.processed_dir = str(tmp_path / "processed")
    for key, value in data_kwargs.items():
        setattr(cfg.data, key, value)
    return cfg


# -- model_is_complete -----------------------------------------------------

def test_incomplete_checkpoint_is_not_resumable(tmp_path):
    # save_pretrained writes config.json before the weights, so a job killed
    # mid-save leaves exactly this behind. Resuming from it fails on load.
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert model_is_complete(tmp_path) is False


def test_complete_checkpoint_is_resumable(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    assert model_is_complete(tmp_path) is True


def test_missing_directory_is_not_resumable(tmp_path):
    assert model_is_complete(tmp_path / "nope") is False


def test_weights_without_config_are_not_resumable(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    assert model_is_complete(tmp_path) is False


# -- reusable_candidates ---------------------------------------------------

def write_rows(path, n):
    path.write_text(
        "".join(json.dumps({"clue_id": str(i), "n_correct": 0}) + "\n" for i in range(n)),
        encoding="utf-8",
    )
    return path


def test_candidates_are_reused_when_the_count_matches(tmp_path):
    path = write_rows(tmp_path / "candidates.jsonl", 5)
    rows = reusable_candidates(path, expected=5)
    assert rows is not None and len(rows) == 5


def test_truncated_candidates_are_regenerated(tmp_path):
    # A file cut short mid-generation looks valid; reusing it would quietly
    # shrink the training set for that iteration.
    path = write_rows(tmp_path / "candidates.jsonl", 3)
    assert reusable_candidates(path, expected=5) is None


def test_absent_candidates_are_regenerated(tmp_path):
    assert reusable_candidates(tmp_path / "nothing.jsonl", expected=5) is None


# -- the processed-data cache ----------------------------------------------

def populate(processed, fingerprint=None):
    processed.mkdir(parents=True, exist_ok=True)
    for name in _PROCESSED_FILES:
        (processed / name).write_text("", encoding="utf-8")
    if fingerprint is not None:
        (processed / "prep_fingerprint.json").write_text(
            json.dumps(fingerprint), encoding="utf-8"
        )


def test_matching_cache_is_reused(tmp_path, monkeypatch):
    cfg = cfg_with(tmp_path, limit_train=40000)
    populate(tmp_path / "processed", prep_fingerprint(cfg))

    called = []
    monkeypatch.setattr(
        "cryptic_star.star_loop.united_prepare", lambda c: called.append(c)
    )
    ensure_processed(cfg)
    assert called == []


def test_cache_from_a_smoke_run_is_rebuilt_for_a_full_run(tmp_path, monkeypatch):
    """The bug this guards: a 40k-row smoke cache silently feeding a full run."""
    smoke = cfg_with(tmp_path, limit_train=40000)
    populate(tmp_path / "processed", prep_fingerprint(smoke))

    full = cfg_with(tmp_path, limit_train=None)
    called = []
    monkeypatch.setattr(
        "cryptic_star.star_loop.united_prepare", lambda c: called.append(c)
    )
    ensure_processed(full)
    assert len(called) == 1


def test_cache_without_a_fingerprint_is_rebuilt(tmp_path, monkeypatch):
    cfg = cfg_with(tmp_path)
    populate(tmp_path / "processed")  # files from an older version of the code

    called = []
    monkeypatch.setattr(
        "cryptic_star.star_loop.united_prepare", lambda c: called.append(c)
    )
    ensure_processed(cfg)
    assert len(called) == 1


def test_fingerprint_is_written_after_a_build(tmp_path, monkeypatch):
    cfg = cfg_with(tmp_path, limit_train=100)
    monkeypatch.setattr("cryptic_star.star_loop.united_prepare", lambda c: None)
    ensure_processed(cfg)
    saved = json.loads((tmp_path / "processed" / "prep_fingerprint.json").read_text())
    assert saved == prep_fingerprint(cfg)
    assert saved["limit_train"] == 100


def test_fingerprint_tracks_every_setting_that_changes_the_output(tmp_path):
    base = prep_fingerprint(cfg_with(tmp_path))
    for key, value in [
        ("limit_train", 10),
        ("limit_val", 10),
        ("limit_test", 10),
        ("enforce_answer_split", False),
        ("drop_quick", False),
        ("min_wordplay_chars", 99),
        ("drop_label_only_rationales", True),
        ("dataset", "separate"),
        ("united_dir", "elsewhere"),
        ("use_unmatched_seed", False),
        ("fuzzy_threshold", 0.5),
    ]:
        assert prep_fingerprint(cfg_with(tmp_path, **{key: value})) != base, key


# -- example building (unchanged behaviour, previously untested) -----------

def test_accepted_examples_keeps_every_correct_reasoning_path():
    cfg = Config()
    cfg.star.max_traces_per_clue = 0
    row = {
        "clue_id": "1",
        "clue": "Bird of prey caught out",
        "answer": "EAGLE",
        "letters": "EAGLE",
        "enumeration": "5",
        "candidates": [
            {"label": 1, "trace": "definition: bird ; wordplay: one way ; answer: E A G L E"},
            {"label": 1, "trace": "definition: bird ; wordplay: another way ; answer: E A G L E"},
            {"label": 0, "trace": "definition: bird ; wordplay: nope ; answer: R A V E N"},
        ],
    }
    assert len(accepted_examples([row], cfg)) == 2

    cfg.star.max_traces_per_clue = 1
    assert len(accepted_examples([row], cfg)) == 1


def test_dedup_collapses_cosmetic_variants_only():
    rows = [
        {"source": "s", "target": "wordplay: One Way!"},
        {"source": "s", "target": "wordplay: one way"},
        {"source": "s", "target": "wordplay: another way"},
    ]
    assert len(dedup_examples(rows, normalized=True)) == 2
    assert len(dedup_examples(rows, normalized=False)) == 3
