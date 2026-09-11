"""The STaR loop end to end, with the GPU stages stubbed out.

The loop claimed to be resumable but was not: the warm start checked a
`iter0/config.json` that nothing writes, and the iteration loop had no resume
check at all, so every preemption on `studentkillable` restarted from zero.
These tests run `run()` twice over the same directory and assert the second pass
does no work - which is the only way to catch that without burning a day of GPU.
"""

import json

import pytest

from cryptic_star import star_loop
from cryptic_star.config import Config
from cryptic_star.io_utils import write_jsonl

CLUES = [
    {
        "clue_id": f"train-{i}",
        "clue": f"Bird of prey caught out {i}",
        "answer": "eagle",
        "letters": "EAGLE",
        "enumeration": "5",
        "split": "train",
        "source": "united",
        "definition": "bird of prey",
        "wordplay": "EAGLE is hidden in the phrase",
        "alt_answers": [],
        "publisher": "Times",
        "meta": {},
    }
    for i in range(6)
]


def processed_dir(tmp_path):
    """A believable `data/processed`, so ensure_processed short-circuits."""
    processed = tmp_path / "processed"
    write_jsonl(processed / "seed_train.jsonl", CLUES[:3])
    for split in ("train", "val", "test"):
        rows = [{**c, "split": split, "clue_id": f"{split}-{i}"} for i, c in enumerate(CLUES)]
        write_jsonl(processed / f"cryptonite_{split}.jsonl", rows)
    return processed


def make_cfg(tmp_path, iterations: int = 2) -> Config:
    cfg = Config()
    cfg.data.processed_dir = str(processed_dir(tmp_path))
    (tmp_path / "processed" / "prep_fingerprint.json").write_text(
        json.dumps(star_loop.prep_fingerprint(cfg)), encoding="utf-8"
    )
    cfg.star.out_dir = str(tmp_path / "run")
    cfg.star.iterations = iterations
    cfg.star.rationalize = False
    cfg.data.train_subsample = 0  # use every clue, keeps the counts predictable
    cfg.eval.limit = None
    cfg.tracking.tensorboard = False
    return cfg


class Calls:
    """Counts the expensive stages so a resumed run can be shown to skip them."""

    def __init__(self):
        self.trained: list[str] = []
        self.generated: list[int] = []
        self.evaluated = 0
        self.models_loaded = 0


@pytest.fixture
def stubs(monkeypatch):
    calls = Calls()

    def fake_train(train_file, out_dir, cfg, base_model=None, tracker=None, tag=None, **kw):
        from pathlib import Path

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        calls.trained.append(tag or out_dir.name)
        # Mimic save_pretrained closely enough for model_is_complete().
        (out_dir / "config.json").write_text("{}", encoding="utf-8")
        (out_dir / "model.safetensors").write_bytes(b"weights")
        return out_dir

    def fake_generate(gen, clues, cfg, hints=False):
        from cryptic_star.data.align import clue_to_row

        calls.generated.append(len(clues))
        return [
            {
                **clue_to_row(c),
                "source": "clue: x",
                "hinted": hints,
                "n_correct": 1 if i % 2 == 0 else 0,
                "candidates": [
                    {
                        "trace": f"definition: bird ; wordplay: path {i} ; answer: E A G L E",
                        "label": 1 if i % 2 == 0 else 0,
                        "reason": "ok" if i % 2 == 0 else "wrong_answer",
                        "predicted": "EAGLE" if i % 2 == 0 else "RAVEN",
                        "wordplay": f"path {i}",
                        "definition": "bird of prey",
                    }
                ],
            }
            for i, c in enumerate(clues)
        ]

    def fake_evaluate(gen, clues, cfg):
        calls.evaluated += 1
        rows = [
            {
                "clue_id": c.clue_id,
                "clue": c.clue,
                "enumeration": c.enumeration,
                "publisher": c.publisher,
                "gold": c.letters,
                "predicted": "EAGLE" if i % 2 == 0 else "RAVEN",
                "top1_correct": i % 2 == 0,
                "any_correct": True,
                "alt_answer_match": False,
                "reason": "ok" if i % 2 == 0 else "wrong_answer",
                "trace": "definition: bird of prey ; wordplay: hidden ; answer: E A G L E",
                "candidates": ["definition: bird of prey ; wordplay: hidden ; answer: E A G L E"],
            }
            for i, c in enumerate(clues)
        ]
        n = len(rows) or 1
        metrics = {
            "n": len(rows),
            "top1_em": sum(r["top1_correct"] for r in rows) / n,
            "alt_answer_rate": 0.0,
            "num_beams": cfg.eval.num_beams,
            "pass_at_n": 1.0,
            "n_samples": cfg.generate.n_samples,
            "oracle_gap": 1.0 - sum(r["top1_correct"] for r in rows) / n,
        }
        return metrics, rows

    class FakeGenerator:
        def __init__(self, model_path, cfg, device=None):
            calls.models_loaded += 1
            self.model_path = model_path

    monkeypatch.setattr(star_loop, "train", fake_train)
    monkeypatch.setattr(star_loop, "generate_for_clues", fake_generate)
    monkeypatch.setattr(star_loop, "evaluate", fake_evaluate)
    monkeypatch.setattr(star_loop, "TraceGenerator", FakeGenerator)
    monkeypatch.setattr(star_loop, "export_discriminator", lambda files, out, cfg: 7)
    return calls


def test_a_full_run_trains_every_iteration(tmp_path, stubs):
    cfg = make_cfg(tmp_path)
    summary = star_loop.run(cfg)

    assert stubs.trained == ["iter0", "iter1", "iter2"]
    assert len(summary["history"]) == 2
    assert summary["final_model"].endswith("iter2/model")
    assert (tmp_path / "run" / "summary.json").exists()
    assert (tmp_path / "run" / "analysis" / "analysis.md").exists()


def test_resubmitting_a_finished_run_retrains_nothing(tmp_path, stubs):
    """The preemption case: the same command, twice, over the same out_dir."""
    cfg = make_cfg(tmp_path)
    first = star_loop.run(cfg)

    stubs.trained.clear()
    stubs.generated.clear()
    loaded_before = stubs.models_loaded

    second = star_loop.run(make_cfg(tmp_path))

    assert stubs.trained == [], "a completed run retrained on resubmission"
    assert stubs.generated == [], "a completed run re-generated candidates"
    assert stubs.models_loaded == loaded_before, "a completed run loaded a model"
    assert second["history"] == first["history"]
    assert second["test"] == first["test"]


def test_an_interrupted_iteration_resumes_from_its_candidates(tmp_path, stubs):
    """Killed after generation, before training: reuse the expensive half."""
    cfg = make_cfg(tmp_path, iterations=1)
    star_loop.run(cfg)

    # Drop the trained model but keep candidates.jsonl, exactly what a job killed
    # during training leaves behind.
    model_dir = tmp_path / "run" / "iter1" / "model"
    (model_dir / "model.safetensors").unlink()
    assert (tmp_path / "run" / "iter1" / "candidates.jsonl").exists()

    stubs.trained.clear()
    stubs.generated.clear()
    star_loop.run(make_cfg(tmp_path, iterations=1))

    assert stubs.trained == ["iter1"], "iteration 1 should have been retrained"
    assert stubs.generated == [], "candidates.jsonl should have been reused"


def test_the_warm_start_is_not_repeated(tmp_path, stubs):
    """Guards the old `iter0/config.json` check, which never matched."""
    cfg = make_cfg(tmp_path, iterations=1)
    star_loop.run(cfg)
    assert "iter0" in stubs.trained

    # Remove iteration 1 so the loop has real work, and check iter0 is left alone.
    import shutil

    shutil.rmtree(tmp_path / "run" / "iter1")
    (tmp_path / "run" / "history.json").unlink()
    stubs.trained.clear()
    star_loop.run(make_cfg(tmp_path, iterations=1))

    assert stubs.trained == ["iter1"], "the warm start was retrained"


def test_a_truncated_candidates_file_is_regenerated(tmp_path, stubs):
    cfg = make_cfg(tmp_path, iterations=1)
    star_loop.run(cfg)

    cand = tmp_path / "run" / "iter1" / "candidates.jsonl"
    lines = cand.read_text(encoding="utf-8").splitlines()
    cand.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    (tmp_path / "run" / "iter1" / "model" / "config.json").unlink()

    stubs.generated.clear()
    star_loop.run(make_cfg(tmp_path, iterations=1))
    assert stubs.generated == [len(CLUES)], "a truncated candidates file was reused"


def test_the_run_writes_a_readable_event_stream(tmp_path, stubs):
    cfg = make_cfg(tmp_path, iterations=1)
    star_loop.run(cfg)

    rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    names = [r["event"] for r in rows]
    assert names[0] == "config"
    for expected in ("data_ready", "generation", "training_set", "iteration_done", "test", "run_done"):
        assert expected in names, expected

    done = next(r for r in rows if r["event"] == "iteration_done")
    assert done["iteration"] == 1
    assert "top1_em" in done["val"]
    assert (tmp_path / "run" / "iter1" / "trace_samples.md").exists()
