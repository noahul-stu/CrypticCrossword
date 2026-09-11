"""Exact-Match evaluation on the Cryptonite answer split.

Reports three numbers:

  top1      beam-search EM. The headline generator number; compare against the
            7.64% T5-Large baseline from the Cryptonite paper.
  pass@N    fraction of clues where *any* of N sampled traces is correct. This
            is the ceiling a perfect discriminator could reach, so it bounds
            the teammates' Best-of-N result and is the honest way to report it.
  oracle_gap  pass@N - top1, i.e. how much head-room re-ranking has.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .config import Config, add_config_args, config_from_args
from .data.align import row_to_clue
from .data.cryptonite import Clue
from .data.formatting import render_source
from .generate import TraceGenerator, _chunks
from .io_utils import read_jsonl, seed_everything, setup_logging, write_json, write_jsonl
from .verify import Reason, extract_answer, verify_trace

log = logging.getLogger(__name__)


def evaluate(
    gen: TraceGenerator,
    clues: Sequence[Clue],
    cfg: Config,
) -> tuple[dict, list[dict]]:
    per_clue: list[dict] = []
    n_top1 = n_pass = n_alt = 0

    for batch in _chunks(list(clues), cfg.eval.batch_size):
        sources = [render_source(c, cfg.format) for c in batch]
        beams = gen.greedy(sources, num_beams=cfg.eval.num_beams)
        sampled: list[list[str]] = (
            gen.sample(sources, cfg.generate.n_samples)
            if cfg.eval.report_pass_at_n
            else [[] for _ in batch]
        )
        for clue, top1, cands in zip(batch, beams, sampled):
            correct = extract_answer(top1) == clue.letters
            any_correct = correct or any(extract_answer(t) == clue.letters for t in cands)
            n_top1 += int(correct)
            n_pass += int(any_correct)
            # Scored strictly against the single gold answer, whatever
            # verify.accept_alt_answers says, so top-1 EM stays comparable with
            # the published Cryptonite baseline. Alternative-answer hits are
            # counted separately instead.
            v = verify_trace(
                clue,
                top1,
                require_reasoning=cfg.verify.require_reasoning,
                min_wordplay_chars=cfg.verify.min_wordplay_chars,
                accept_alt_answers=False,
            )
            n_alt += int(v.reason == Reason.ALT_ANSWER)
            per_clue.append(
                {
                    "clue_id": clue.clue_id,
                    "clue": clue.clue,
                    "enumeration": clue.enumeration,
                    "publisher": clue.publisher,  # used by analyze.py breakdowns
                    "gold": clue.letters,
                    "predicted": v.predicted,
                    "top1_correct": correct,
                    "any_correct": any_correct,
                    "alt_answer_match": v.reason == Reason.ALT_ANSWER,
                    "reason": v.reason,
                    "trace": top1,
                    "candidates": cands,
                }
            )
        if len(per_clue) % (cfg.eval.batch_size * 20) == 0:
            log.info("evaluated %d/%d | EM %.3f", len(per_clue), len(clues), n_top1 / len(per_clue))

    n = len(per_clue) or 1
    metrics = {
        "n": len(per_clue),
        "top1_em": n_top1 / n,
        # Predictions that hit a documented alternative answer. Not added to EM;
        # reported so the write-up can say how much of the error is arguable.
        "alt_answer_rate": n_alt / n,
        "num_beams": cfg.eval.num_beams,
    }
    if cfg.eval.report_pass_at_n:
        metrics["pass_at_n"] = n_pass / n
        metrics["n_samples"] = cfg.generate.n_samples
        metrics["oracle_gap"] = metrics["pass_at_n"] - metrics["top1_em"]
    return metrics, per_clue


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--model", required=True)
    parser.add_argument("--clues", default=None, help="defaults to the configured eval split")
    parser.add_argument("--out", required=True, help="output directory for metrics + predictions")
    args = parser.parse_args(argv)

    cfg: Config = config_from_args(args)
    setup_logging(cfg.log_level)
    seed_everything(cfg.star.seed)

    clue_file = args.clues or str(
        Path(cfg.data.processed_dir) / f"cryptonite_{cfg.eval.split}.jsonl"
    )
    clues = [row_to_clue(r) for r in read_jsonl(clue_file, limit=cfg.eval.limit)]
    log.info("evaluating on %d clues from %s", len(clues), clue_file)

    gen = TraceGenerator(args.model, cfg)
    metrics, per_clue = evaluate(gen, clues, cfg)

    out = Path(args.out)
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "predictions.jsonl", per_clue)
    log.info("EM(top-1) = %.4f  pass@%s = %s", metrics["top1_em"],
             metrics.get("n_samples"), metrics.get("pass_at_n"))


if __name__ == "__main__":
    main()
