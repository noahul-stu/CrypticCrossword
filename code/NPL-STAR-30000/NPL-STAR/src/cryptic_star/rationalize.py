"""STaR's rationalisation pass.

Plain STaR throws away every clue the model failed. On cryptic crosswords that
is most of them, so the training set stays tiny and the loop barely moves. The
rationalisation trick recovers them: re-prompt the failed clues with the gold
answer appended as a hint, keep the traces that arrive at that answer with real
reasoning, then **strip the hint** before adding the example to the training set.

The model therefore learns `clue -> reasoning -> answer` from examples it could
only produce with help. Working backwards from the answer is also exactly how
human solvers write these explanations up, which is why it suits this task.

The one thing that must not go wrong: a hint leaking into a training source.
`data.formatting.example_from_trace` rebuilds the source with `hint=None`, and
`assert_no_hint_leak` re-checks the whole file before it is written.
"""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from .config import Config, add_config_args, config_from_args
from .data.align import row_to_clue
from .data.cryptonite import Clue
from .data.formatting import HINT_KEY, example_from_trace
from .generate import TraceGenerator, generate_for_clues, summarise
from .io_utils import read_jsonl, seed_everything, setup_logging, write_json, write_jsonl

log = logging.getLogger(__name__)


def failed_clues(candidate_rows: Sequence[dict]) -> list[Clue]:
    """Clues where not a single sampled trace was accepted."""
    return [row_to_clue(r) for r in candidate_rows if not r.get("n_correct")]


def assert_no_hint_leak(examples: Sequence[dict]) -> None:
    """Guard: no training source may contain the hint field."""
    bad = [e for e in examples if HINT_KEY in e.get("source", "").lower()]
    if bad:
        raise AssertionError(
            f"{len(bad)} rationalised examples still carry '{HINT_KEY}' in their source; "
            "the model would learn to expect the answer as input. First offender:\n"
            f"{bad[0]['source']}"
        )


def rationalized_examples(candidate_rows: Sequence[dict], cfg: Config) -> list[dict]:
    """Accepted hinted traces -> hint-free training examples.

    Honours `star.max_traces_per_clue` the same way as the unhinted path
    (0 = keep every correct reasoning path).
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
                    "provenance": "rationalized",
                }
            )
            kept += 1
            if cap and kept >= cap:
                break
    assert_no_hint_leak(out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidates", required=True, help="candidates jsonl from generate.py")
    parser.add_argument("--out", required=True, help="hinted candidates jsonl")
    args = parser.parse_args(argv)

    cfg: Config = config_from_args(args)
    setup_logging(cfg.log_level)
    seed_everything(cfg.star.seed + 1)

    rows = list(read_jsonl(args.candidates))
    failures = failed_clues(rows)
    log.info("rationalising %d/%d failed clues", len(failures), len(rows))
    if not failures:
        write_jsonl(args.out, [])
        return

    gen = TraceGenerator(args.model, cfg)
    hinted = generate_for_clues(gen, failures, cfg, hints=True)
    write_jsonl(args.out, hinted)
    stats = summarise(hinted)
    write_json(args.out + ".stats.json", stats)
    log.info(
        "rationalisation recovered %d/%d clues (%.1f%%)",
        stats["clues_with_at_least_one_correct"],
        len(failures),
        100 * stats["pass_at_n"],
    )


if __name__ == "__main__":
    main()
