"""Hand-off to the DeBERTa discriminator (proposal step 2).

The generator already produces exactly what a cross-encoder needs: for each
clue, a set of reasoning traces with automatic 1/0 labels. This module reshapes
`candidates.jsonl` into flat pairs and balances them.

Output row:

    {"clue": ..., "enumeration": ..., "trace": ..., "predicted": ...,
     "label": 1, "reason": "ok", "text_a": ..., "text_b": ..., "split": "train"}

`text_a` / `text_b` are pre-joined so the cross-encoder can be trained with a
plain `tokenizer(text_a, text_b)` call and no knowledge of this project.

Two choices worth defending in the write-up:

* `strict_negatives` keeps only traces that committed to a *wrong answer*.
  Unparseable or degenerate outputs are dropped: they teach the discriminator to
  detect malformed text rather than bad reasoning, which is not the job.
* Hinted (rationalised) traces are marked `hinted: true`. They are valid
  positives but they were produced with the answer in view, so they are easier
  than anything seen at inference. Being able to exclude them is why the flag
  is carried through.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Iterable, Sequence

from .config import Config, add_config_args, config_from_args
from .io_utils import read_jsonl, setup_logging, write_json, write_jsonl
from .verify import Reason

log = logging.getLogger(__name__)

# Rejections that reflect bad reasoning (useful negatives) vs. broken decoding.
# `Reason.ALT_ANSWER` is deliberately absent: a trace that reached a documented
# alternative answer is probably sound reasoning for a defensible answer, so
# using it as a hard negative would teach the discriminator to reject good
# reasoning. With `strict_negatives` on it is dropped instead.
MEANINGFUL_REJECTIONS = {Reason.WRONG_ANSWER, Reason.WRONG_LENGTH}


def _pair_text(row: dict, cand: dict) -> tuple[str, str]:
    clue = row["clue"]
    enum = row.get("enumeration", "")
    text_a = f"{clue} ({enum})" if enum else clue
    return text_a, cand["trace"]


def to_pairs(
    rows: Iterable[dict],
    negatives_per_positive: int = 3,
    strict_negatives: bool = True,
    seed: int = 13,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    positives: list[dict] = []
    negatives: list[dict] = []
    stats = {"clues": 0, "positives": 0, "negatives_available": 0, "dropped_noise": 0}

    for row in rows:
        stats["clues"] += 1
        hinted = bool(row.get("hinted", False))
        for cand in row.get("candidates", []):
            text_a, text_b = _pair_text(row, cand)
            item = {
                "clue_id": row.get("clue_id", ""),
                "clue": row["clue"],
                "enumeration": row.get("enumeration", ""),
                "gold": row.get("letters", ""),
                "predicted": cand.get("predicted", ""),
                "trace": cand["trace"],
                "wordplay": cand.get("wordplay", ""),
                "reason": cand.get("reason", ""),
                "hinted": hinted,
                "label": int(cand["label"]),
                "text_a": text_a,
                "text_b": text_b,
            }
            if cand["label"] == 1:
                positives.append(item)
            elif not strict_negatives or cand.get("reason") in MEANINGFUL_REJECTIONS:
                negatives.append(item)
            else:
                stats["dropped_noise"] += 1

    stats["positives"] = len(positives)
    stats["negatives_available"] = len(negatives)

    budget = len(positives) * max(negatives_per_positive, 0)
    if budget and len(negatives) > budget:
        negatives = rng.sample(negatives, budget)
    stats["negatives_kept"] = len(negatives)

    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs, stats


def export(
    candidate_files: Sequence[str | Path],
    out_file: str | Path,
    cfg: Config,
) -> int:
    rows: list[dict] = []
    for path in candidate_files:
        if Path(path).exists():
            rows.extend(read_jsonl(path))
        else:
            log.warning("candidate file missing, skipping: %s", path)

    pairs, stats = to_pairs(
        rows,
        negatives_per_positive=cfg.discriminator.negatives_per_positive,
        strict_negatives=cfg.discriminator.strict_negatives,
        seed=cfg.star.seed,
    )
    n = write_jsonl(out_file, pairs)
    write_json(Path(str(out_file) + ".stats.json"), stats)
    log.info(
        "discriminator data: %d rows (%d pos / %d neg, %d noisy rejections dropped) -> %s",
        n, stats["positives"], stats.get("negatives_kept", 0), stats["dropped_noise"], out_file,
    )
    return n


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    cfg: Config = config_from_args(args)
    setup_logging(cfg.log_level)
    export(args.candidates, args.out, cfg)


if __name__ == "__main__":
    main()
