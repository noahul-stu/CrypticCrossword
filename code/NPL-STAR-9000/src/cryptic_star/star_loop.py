"""The STaR loop: the whole generator pipeline in one command.

    python -m cryptic_star.star_loop --config configs/star_t5_base.yaml

Iteration 0 (warm start)
    Fine-tune the base T5 on the human rationales from cryptic-wordplay
    (`data/processed/seed_train.jsonl`). Without this the model is too weak for
    self-generation to get off the ground.

Iterations 1..K
    a. sample N traces per clue for a subsample of Cryptonite train
    b. keep traces that reach the gold answer with real reasoning        (verify)
    c. re-prompt the failures with the answer as a hint, keep what works,
       strip the hint                                                (rationalize)
    d. rebuild the training set = seed + all accepted traces so far
    e. retrain **from the original base checkpoint**                     (train)
    f. evaluate EM on val, export labelled candidates for the discriminator

Every artefact lands under `star.out_dir/iterK/`, so a run is inspectable and
resumable. Resubmitting the same command after a preemption skips iterations
that already have a complete `iterK/model/` *and* a `history.json` record, and
within an unfinished iteration it reuses `candidates.jsonl` rather than
re-decoding it. This matters on `studentkillable`, where the 1-day cap means a
multi-iteration flan-t5-large run will be interrupted.

Progress is written as it happens, not at the end: `star.out_dir/metrics.jsonl`
(one JSON event per line, `tail -f`-able), `star.out_dir/tb/` for TensorBoard,
and `iterK/trace_samples.md` for the reasoning the model actually produced. See
`tracking.py`.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

from .analyze import analyse, learning_curve, to_markdown
from .config import Config, add_config_args, config_from_args
from .data import align as align_mod
from .data.align import clue_to_row, row_to_clue
from .data.cryptonite import Clue, load_all
from .data.formatting import example_from_trace, make_example
from .data.united import prepare as united_prepare
from .data.wordplay import load_wordplay
from .evaluate import evaluate
from .export_discriminator_data import export as export_discriminator
from .generate import TraceGenerator, generate_for_clues, summarise
from .io_utils import (
    free_gpu,
    read_json,
    read_jsonl,
    seed_everything,
    setup_logging,
    write_json,
    write_jsonl,
)
from .rationalize import failed_clues, rationalized_examples
from .tracking import RunTracker
from .train import train

log = logging.getLogger(__name__)

_PROCESSED_FILES = (
    "seed_train.jsonl",
    "cryptonite_train.jsonl",
    "cryptonite_val.jsonl",
    "cryptonite_test.jsonl",
)


# --------------------------------------------------------------------------
# data prep
# --------------------------------------------------------------------------

def prep_fingerprint(cfg: Config) -> dict:
    """Every config value that changes what `data/processed` contains.

    Written alongside the processed files so a cached directory can be checked
    for *relevance*, not just existence. Without this a smoke run
    (`limit_train: 40000`) leaves four valid-looking files behind and the next
    full run silently trains on 40k rows instead of 474,950 - no error, just a
    quietly wrong result, which is the worst kind.
    """
    d = cfg.data
    return {
        "dataset": d.dataset,
        "united_dir": d.united_dir,
        "cryptonite_dir": d.cryptonite_dir,
        "wordplay_file": d.wordplay_file,
        "enforce_answer_split": d.enforce_answer_split,
        "drop_quick": d.drop_quick,
        "min_wordplay_chars": d.min_wordplay_chars,
        "drop_label_only_rationales": d.drop_label_only_rationales,
        "fuzzy_threshold": d.fuzzy_threshold,
        "use_unmatched_seed": d.use_unmatched_seed,
        "limit_train": d.limit_train,
        "limit_val": d.limit_val,
        "limit_test": d.limit_test,
    }


def _fingerprint_diff(have: dict | None, want: dict) -> str:
    if not have:
        return "it carries no prep_fingerprint.json"
    changed = [k for k in want if have.get(k) != want[k]]
    return ", ".join(f"{k}: {have.get(k)!r} -> {want[k]!r}" for k in changed) or "unknown"


def ensure_processed(cfg: Config) -> None:
    """Build `data/processed` unless a matching build is already there.

    Two sources, same four output files, so everything downstream is unaware of
    which one was used:
      `dataset: united`   - one pre-joined dataset, wordplay already attached
      `dataset: separate` - raw Cryptonite + raw wordplay, joined here

    Calls the loaders directly instead of shelling out, so this run's config
    overrides (subsample sizes, row limits) actually apply.
    """
    processed = Path(cfg.data.processed_dir)
    want = prep_fingerprint(cfg)
    fp_file = processed / "prep_fingerprint.json"

    if all((processed / f).exists() for f in _PROCESSED_FILES):
        have = read_json(fp_file)
        if have == want:
            log.info("using existing processed data in %s", processed)
            return
        log.warning(
            "processed data in %s was built with different settings, rebuilding (%s)",
            processed, _fingerprint_diff(have, want),
        )

    if cfg.data.dataset == "united":
        log.info("building processed data from the united dataset")
        united_prepare(cfg)
        write_json(fp_file, want)
        return
    if cfg.data.dataset != "separate":
        raise ValueError(
            f"data.dataset must be 'united' or 'separate', got {cfg.data.dataset!r}"
        )

    log.info("building processed data by alignment")
    cryptonite = load_all(
        cfg.data.cryptonite_dir,
        limits={
            "train": cfg.data.limit_train,
            "val": cfg.data.limit_val,
            "test": cfg.data.limit_test,
        },
    )
    wordplay = load_wordplay(cfg.data.wordplay_file)
    seed, stats = align_mod.build_seed(
        cryptonite,
        wordplay,
        fuzzy_threshold=cfg.data.fuzzy_threshold,
        use_unmatched=cfg.data.use_unmatched_seed,
    )
    write_jsonl(processed / "seed_train.jsonl", (clue_to_row(c) for c in seed))
    for split, clues in cryptonite.items():
        write_jsonl(processed / f"cryptonite_{split}.jsonl", (clue_to_row(c) for c in clues))
    write_json(processed / "align_report.json", stats)
    write_json(fp_file, want)
    log.info(
        "alignment: seed=%d (exact %d, fuzzy %d, unmatched %d) | dropped for leakage=%d",
        stats["seed_rows"], stats["matched_exact"], stats["matched_fuzzy"],
        stats["unmatched_kept"], stats["dropped_leakage"],
    )


def seed_examples(cfg: Config) -> list[dict]:
    """Human rationales -> seq2seq examples."""
    path = Path(cfg.data.processed_dir) / "seed_train.jsonl"
    rows = [row_to_clue(r) for r in read_jsonl(path)]
    return [{**make_example(c, cfg.format), "provenance": "human_wordplay"} for c in rows]


def sample_train_clues(cfg: Config, iteration: int) -> list[Clue]:
    """A fresh subsample of Cryptonite train for this iteration.

    Cryptonite train is ~470k clues; N=8 samples over all of it is millions of
    generations. Each iteration draws its own subsample (seeded by iteration
    number) so coverage grows across the run instead of re-solving the same
    9k clues every time.
    """
    path = Path(cfg.data.processed_dir) / "cryptonite_train.jsonl"
    clues = [row_to_clue(r) for r in read_jsonl(path)]
    k = cfg.data.train_subsample
    if k and k < len(clues):
        rng = random.Random(cfg.star.seed + 1000 * iteration)
        clues = rng.sample(clues, k)
        log.info(
            "iteration %d: sampled %d of %d train clues (subsampled for compute; "
            "the rest are NOT seen this iteration)",
            iteration, k, len(clues),
        )
    return clues


def accepted_examples(candidate_rows: Sequence[dict], cfg: Config) -> list[dict]:
    """Accepted traces -> fine-tuning examples.

    This is the STaR training signal: for a clue the model got right, every
    distinct correct reasoning path becomes its own `clue -> reasoning + answer`
    example. `star.max_traces_per_clue = 0` keeps all of them; a positive value
    caps them so clues the model finds easy do not dominate the training set.
    """
    cap = cfg.star.max_traces_per_clue
    out: list[dict] = []
    for row in candidate_rows:
        clue = row_to_clue(row)
        kept = 0
        for cand in row["candidates"]:
            if cand["label"] != 1:
                continue
            out.append(
                {
                    **example_from_trace(clue, cand["trace"], cfg.format),
                    "provenance": "self_generated",
                }
            )
            kept += 1
            if cap and kept >= cap:
                break
    return out


def dedup_examples(examples: Sequence[dict], normalized: bool = True) -> list[dict]:
    """Drop repeated (source, target) pairs.

    With `normalized`, traces that differ only in casing or punctuation count as
    the same reasoning - so `max_traces_per_clue: 0` keeps every *distinct* way
    of solving a clue rather than ten cosmetic variants of one way.
    """
    seen: set[tuple[str, str]] = set()
    out = []
    for e in examples:
        target = e["target"]
        key = (e["source"], _norm_trace(target) if normalized else target)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def _norm_trace(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------
#
# `studentkillable` is preemptible and capped at one day, so a 4-iteration
# flan-t5-large run is *expected* to be interrupted and resubmitted. Everything
# below exists so that resubmitting the identical command continues instead of
# starting over: without it each preemption threw away up to a day of GPU time.

_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def model_is_complete(model_dir: str | Path) -> bool:
    """True if `model_dir` holds a fully written HF checkpoint.

    Config plus weights, both. Checking only for the directory - or only for
    `config.json`, which `save_pretrained` writes first - would happily "resume"
    from a checkpoint whose weights were still being written when the job was
    killed, and fail later with a confusing load error.
    """
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        return False
    return any((model_dir / name).exists() for name in _WEIGHT_FILES)


def reusable_candidates(path: str | Path, expected: int) -> list[dict] | None:
    """Candidates from a previous attempt, or None if they cannot be trusted.

    Generation is the expensive half of an iteration (`n_samples` decodes over
    `train_subsample` clues), so it is worth reusing. The row count has to match
    exactly: a file cut short mid-generation is otherwise indistinguishable from
    a complete one, and reusing it would silently shrink the training set.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        rows = list(read_jsonl(path))
    except (OSError, ValueError):
        log.warning("%s is unreadable, regenerating", path)
        return None
    if len(rows) != expected:
        log.warning(
            "%s holds %d rows but this iteration covers %d clues (interrupted "
            "mid-generation?), regenerating",
            path, len(rows), expected,
        )
        return None
    return rows


# --------------------------------------------------------------------------
# loop
# --------------------------------------------------------------------------

def run(cfg: Config, tracker: RunTracker | None = None) -> dict:
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", cfg.to_dict())

    # The tracker flushes every line it writes, so there is nothing buffered to
    # lose if this run is preempted or raises; close() at the end is tidiness,
    # not a correctness requirement, and needs no try/finally.
    owns_tracker = tracker is None
    tracker = tracker or RunTracker(run_dir, cfg)
    tracker.config(cfg)

    ensure_processed(cfg)

    base_model = cfg.train.model_name
    seed_rows = seed_examples(cfg)
    log.info("seed set: %d human-written rationales", len(seed_rows))

    val_clues = [
        row_to_clue(r)
        for r in read_jsonl(
            Path(cfg.data.processed_dir) / "cryptonite_val.jsonl", limit=cfg.eval.limit
        )
    ]
    tracker.event(
        "data_ready",
        scalars={"seed_rows": len(seed_rows), "val_clues": len(val_clues)},
        processed_dir=cfg.data.processed_dir,
    )

    # Records from earlier attempts at this run, keyed by iteration. An iteration
    # is only skipped when both its model and its metrics are on disk.
    prior = {
        r["iteration"]: r
        for r in (read_json(run_dir / "history.json") or [])
        if isinstance(r, dict) and "iteration" in r
    }
    if prior:
        log.info("found completed iterations from a previous attempt: %s",
                 sorted(prior))

    # ---- iteration 0: warm start on human rationales --------------------
    iter0 = cfg.iter_dir(0)
    model_path = base_model
    if cfg.star.warm_start_on_seed and not seed_rows:
        # Nothing to warm-start on. Usually a data-prep problem (wrong
        # united_dir, or a row limit small enough to exclude every annotated
        # clue), so say so instead of failing later on an empty training file.
        log.error(
            "seed set is EMPTY - skipping the warm start. Iteration 1 will sample "
            "from the untuned base model, which on this task solves ~7% of clues, "
            "so expect the loop to barely move. Check data/processed/seed_train.jsonl "
            "and the annotated-row counts in the data report."
        )
    elif cfg.star.warm_start_on_seed:
        # Check the saved checkpoint itself. This used to test `iter0/config.json`,
        # which nothing ever writes - the model's config lands in
        # `iter0/model/config.json` and the run's in `<run_dir>/config.json` - so
        # the warm start was silently redone on every resubmission.
        if model_is_complete(iter0 / "model"):
            log.info("iteration 0 already trained, reusing %s", iter0 / "model")
        else:
            write_jsonl(iter0 / "train.jsonl", seed_rows)
            train(
                iter0 / "train.jsonl",
                iter0 / "model",
                cfg,
                base_model=base_model,
                tracker=tracker,
                tag="iter0",
            )
        model_path = str(iter0 / "model")

    history: list[dict] = []
    accumulated: list[dict] = list(seed_rows)

    for k in range(1, cfg.star.iterations + 1):
        it_dir = cfg.iter_dir(k)
        it_dir.mkdir(parents=True, exist_ok=True)

        # Resume: replay a finished iteration from disk instead of recomputing it.
        # Both halves have to be present - a complete model *and* the metrics
        # record - because the history entry is what the analysis is built from.
        if model_is_complete(it_dir / "model") and k in prior:
            model_path = str(it_dir / "model")
            if cfg.star.accumulate_traces and (it_dir / "train.jsonl").exists():
                accumulated = list(read_jsonl(it_dir / "train.jsonl"))
            history.append(prior[k])
            log.info("iteration %d already complete, reusing %s", k, model_path)
            tracker.event("iteration_reused", iteration=k, model=model_path)
            continue

        log.info("=" * 70)
        log.info("STaR iteration %d/%d  (generator: %s)", k, cfg.star.iterations, model_path)
        tracker.event("iteration_start", iteration=k, generator=model_path)

        # (a) + (b) sample and filter. The generator is loaded lazily: a partially
        # finished iteration may already have every candidate file it needs, and
        # then there is no reason to put a model on the GPU at all.
        clues = sample_train_clues(cfg, k)
        gen: TraceGenerator | None = None
        cand_rows = reusable_candidates(it_dir / "candidates.jsonl", len(clues))
        if cand_rows is None:
            gen = TraceGenerator(model_path, cfg)
            cand_rows = generate_for_clues(gen, clues, cfg, hints=False)
            write_jsonl(it_dir / "candidates.jsonl", cand_rows)
        else:
            log.info("iteration %d: reusing %d candidate rows from disk", k, len(cand_rows))

        gen_stats = summarise(cand_rows)
        new_examples = accepted_examples(cand_rows, cfg)
        log.info(
            "iteration %d: %d/%d clues solved (pass@%d=%.3f) -> %d accepted traces",
            k, gen_stats["clues_with_at_least_one_correct"], len(clues),
            cfg.generate.n_samples, gen_stats["pass_at_n"], len(new_examples),
        )
        tracker.event(
            "generation",
            iteration=k,
            scalars={**gen_stats, "accepted_examples": len(new_examples)},
        )
        tracker.trace_samples(
            cand_rows, it_dir / "trace_samples.md", iteration=k
        )

        # (c) rationalisation on the failures
        rat_stats = {}
        if cfg.star.rationalize:
            failures = failed_clues(cand_rows)
            if failures:
                hinted = reusable_candidates(
                    it_dir / "candidates_hinted.jsonl", len(failures)
                )
                if hinted is None:
                    gen = gen or TraceGenerator(model_path, cfg)
                    hinted = generate_for_clues(gen, failures, cfg, hints=True)
                    write_jsonl(it_dir / "candidates_hinted.jsonl", hinted)
                rat_stats = summarise(hinted)
                recovered = rationalized_examples(hinted, cfg)
                new_examples += recovered
                log.info(
                    "iteration %d: rationalisation recovered %d/%d failed clues "
                    "-> %d extra traces",
                    k, rat_stats["clues_with_at_least_one_correct"], len(failures),
                    len(recovered),
                )
                tracker.event(
                    "rationalization",
                    iteration=k,
                    scalars={**rat_stats, "recovered_examples": len(recovered)},
                )
                tracker.trace_samples(
                    hinted,
                    it_dir / "trace_samples_hinted.md",
                    iteration=k,
                    hinted=True,
                )

        del gen  # free the GPU before training
        free_gpu()

        # (d) build the training set: every correct reasoning path, plus the
        #     human seed rationales.
        nd = cfg.star.normalized_dedup
        if cfg.star.accumulate_traces:
            accumulated = dedup_examples(accumulated + new_examples, normalized=nd)
            train_rows = accumulated
        else:
            train_rows = dedup_examples(seed_rows + new_examples, normalized=nd)
        write_jsonl(it_dir / "train.jsonl", train_rows)

        prov = Counter(r.get("provenance", "?") for r in train_rows)
        per_clue = Counter(r["clue_id"] for r in train_rows)
        log.info(
            "iteration %d training set: %d examples over %d clues "
            "(mean %.2f reasonings/clue, max %d) | provenance %s",
            k, len(train_rows), len(per_clue),
            len(train_rows) / max(len(per_clue), 1),
            max(per_clue.values()) if per_clue else 0,
            dict(prov),
        )

        tracker.event(
            "training_set",
            iteration=k,
            scalars={
                "examples": len(train_rows),
                "clues": len(per_clue),
                "traces_per_clue_mean": len(train_rows) / max(len(per_clue), 1),
                "traces_per_clue_max": max(per_clue.values()) if per_clue else 0,
                "provenance": dict(prov),
            },
        )

        # (e) retrain
        train_from = base_model if cfg.star.retrain_from_base else model_path
        train(
            it_dir / "train.jsonl",
            it_dir / "model",
            cfg,
            base_model=train_from,
            tracker=tracker,
            tag=f"iter{k}",
        )
        model_path = str(it_dir / "model")

        # (f) evaluate + export for the discriminator
        eval_gen = TraceGenerator(model_path, cfg)
        metrics, per_clue = evaluate(eval_gen, val_clues, cfg)
        del eval_gen
        free_gpu()
        write_json(it_dir / "val_metrics.json", metrics)
        write_jsonl(it_dir / "val_predictions.jsonl", per_clue)

        disc_files = [it_dir / "candidates.jsonl"]
        if (it_dir / "candidates_hinted.jsonl").exists():
            disc_files.append(it_dir / "candidates_hinted.jsonl")
        n_disc = export_discriminator(disc_files, it_dir / cfg.discriminator.out_file, cfg)

        record = {
            "iteration": k,
            "train_examples": len(train_rows),
            "new_examples": len(new_examples),
            "generation": gen_stats,
            "rationalization": rat_stats,
            "val": metrics,
            "discriminator_rows": n_disc,
            "model": model_path,
        }
        history.append(record)
        write_json(run_dir / "history.json", history)
        log.info(
            "iteration %d done: val EM %.4f (pass@%s %s) | train set now %d examples",
            k, metrics["top1_em"], metrics.get("n_samples"),
            metrics.get("pass_at_n"), len(train_rows),
        )
        tracker.event(
            "iteration_done",
            iteration=k,
            scalars={"val": metrics, "discriminator_rows": n_disc},
            model=model_path,
        )

    # ---- final test-split evaluation -----------------------------------
    test_clues = [
        row_to_clue(r)
        for r in read_jsonl(
            Path(cfg.data.processed_dir) / "cryptonite_test.jsonl", limit=cfg.eval.limit
        )
    ]
    # Also resumable: the test decode over `eval.limit` clues with beams plus N
    # samples is an hour of GPU on its own, and a job preempted during the
    # analysis step below should not have to repeat it.
    test_metrics = read_json(run_dir / "test_metrics.json")
    pred_file = run_dir / "test_predictions.jsonl"
    if test_metrics and test_metrics.get("n") == len(test_clues) and pred_file.exists():
        test_preds = list(read_jsonl(pred_file))
        log.info("reusing the test evaluation from a previous attempt (%d clues)",
                 len(test_preds))
    else:
        gen = TraceGenerator(model_path, cfg)
        test_metrics, test_preds = evaluate(gen, test_clues, cfg)
        del gen
        free_gpu()
        write_json(run_dir / "test_metrics.json", test_metrics)
        write_jsonl(pred_file, test_preds)
    log.info("FINAL test EM = %.4f | pass@%s = %s", test_metrics["top1_em"],
             test_metrics.get("n_samples"), test_metrics.get("pass_at_n"))
    tracker.event("test", scalars=test_metrics, model=model_path)

    # ---- analysis report (no GPU; re-runnable standalone via -m cryptic_star.analyze)
    report = analyse(test_preds)
    curve = learning_curve(history)
    report["learning_curve"] = curve
    write_json(run_dir / "analysis" / "analysis.json", report)
    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis" / "analysis.md").write_text(
        to_markdown(report, curve), encoding="utf-8"
    )
    h = report["headline"]
    log.info(
        "analysis: top1 %.4f | pass@N %.4f | oracle gap %.4f | majority-vote %.4f",
        h["top1_em"], h["pass_at_n"], h["oracle_gap"], h["majority_vote_em"],
    )
    log.info("full report -> %s", run_dir / "analysis" / "analysis.md")

    summary = {
        "history": history,
        "test": test_metrics,
        "analysis": report["headline"],
        "final_model": model_path,
    }
    write_json(run_dir / "summary.json", summary)
    tracker.event("run_done", scalars=report["headline"], final_model=model_path)
    if owns_tracker:
        tracker.close()
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg: Config = config_from_args(args)
    setup_logging(cfg.log_level)
    seed_everything(cfg.star.seed)
    run(cfg)


if __name__ == "__main__":
    main()
