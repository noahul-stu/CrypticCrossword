"""Join cryptic-wordplay rationales onto Cryptonite, and guard against leakage.

This is the step your idea rests on: Cryptonite gives clue->answer at scale but
no reasoning; cryptic-wordplay gives human wordplay analyses for a much smaller
set of clues. Their intersection is the STaR seed set `D_seed` - real reasoning
traces to warm-start the generator with, so that self-generation has something
better than a 7.6%-accurate model to bootstrap from.

**The leakage rule.** Cryptonite's official split is answer-disjoint: no answer
in test appears in train. cryptic-wordplay is scraped independently and covers
the same newspapers, so some of its clues have answers that live in Cryptonite's
val/test. Training on those destroys the evaluation. So: any wordplay row whose
answer letters appear in the val or test answer set is dropped, whether or not
its clue matched a Cryptonite row. Nothing else in the pipeline re-checks this,
so it happens here, loudly, and the counts go into the alignment report.

Matching is two-pass:
  1. exact on (normalised clue, answer letters)
  2. fallback on answer letters + clue-token Jaccard >= data.fuzzy_threshold
     (catches punctuation and transcription differences between the sources)
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from ..config import Config, add_config_args, config_from_args
from ..io_utils import setup_logging, write_json, write_jsonl
from ..text import clue_tokens, jaccard, norm_clue
from .cryptonite import Clue, answer_sets, load_all
from .wordplay import load_wordplay

log = logging.getLogger(__name__)


def clue_to_row(c: Clue) -> dict:
    return {
        "clue_id": c.clue_id,
        "clue": c.clue,
        "answer": c.answer,
        "letters": c.letters,
        "enumeration": c.enumeration,
        "split": c.split,
        "source": c.source,
        "definition": c.definition,
        "wordplay": c.wordplay,
        "alt_answers": c.alt_answers,
        "publisher": c.publisher,
        "meta": c.meta,
    }


def row_to_clue(row: dict) -> Clue:
    return Clue(
        clue=row["clue"],
        answer=row["answer"],
        enumeration=row.get("enumeration", ""),
        letters=row.get("letters", ""),
        clue_id=row.get("clue_id", ""),
        split=row.get("split", ""),
        publisher=row.get("publisher", ""),
        source=row.get("source", "cryptonite"),
        definition=row.get("definition", ""),
        wordplay=row.get("wordplay", ""),
        alt_answers=row.get("alt_answers") or [],
        meta=row.get("meta", {}) or {},
    )


def build_seed(
    cryptonite: dict[str, list[Clue]],
    wordplay: list[Clue],
    fuzzy_threshold: float = 0.8,
    use_unmatched: bool = True,
) -> tuple[list[Clue], dict]:
    """Return (`seed_rows`, `report`)."""
    answers = answer_sets(cryptonite)
    held_out = answers["val"] | answers["test"]

    # Index Cryptonite train for matching.
    by_exact: dict[tuple[str, str], Clue] = {}
    by_letters: dict[str, list[Clue]] = defaultdict(list)
    for c in cryptonite["train"]:
        by_exact.setdefault((norm_clue(c.clue), c.letters), c)
        by_letters[c.letters].append(c)

    seed: list[Clue] = []
    stats = {
        "wordplay_rows": len(wordplay),
        "dropped_leakage": 0,
        "matched_exact": 0,
        "matched_fuzzy": 0,
        "unmatched_kept": 0,
        "unmatched_dropped": 0,
        "enum_mismatch_fixed": 0,
    }
    leakage_examples: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for wp in wordplay:
        # 1. Leakage guard, before anything else.
        if wp.letters in held_out:
            stats["dropped_leakage"] += 1
            if len(leakage_examples) < 20:
                leakage_examples.append({"clue": wp.clue, "answer": wp.answer})
            continue

        key = (norm_clue(wp.clue), wp.letters)
        if key in seen:  # duplicate analyses of the same clue by different authors
            continue

        match = by_exact.get(key)
        how = "exact" if match else None
        if match is None:
            best, best_score = None, 0.0
            wp_tokens = clue_tokens(wp.clue)
            for cand in by_letters.get(wp.letters, ()):
                score = jaccard(wp_tokens, clue_tokens(cand.clue))
                if score > best_score:
                    best, best_score = cand, score
            if best is not None and best_score >= fuzzy_threshold:
                match, how = best, "fuzzy"

        if match is None and not use_unmatched:
            stats["unmatched_dropped"] += 1
            continue

        seen.add(key)
        row = Clue(
            # Prefer Cryptonite's clue text when we have a match, so the seed
            # examples look exactly like what the model sees at eval time.
            clue=match.clue if match else wp.clue,
            answer=match.answer if match else wp.answer,
            enumeration=(match.enumeration if match else wp.enumeration) or wp.enumeration,
            clue_id=match.clue_id if match else wp.clue_id,
            split="train",
            publisher=wp.publisher or (match.publisher if match else ""),
            source="seed",
            definition=wp.definition,
            wordplay=wp.wordplay,
            meta={
                **wp.meta,
                "matched_cryptonite": match is not None,
                "match_kind": how or "none",
                "wordplay_id": wp.clue_id,
            },
        )
        if match is not None and wp.enum_sig and match.enum_sig and wp.enum_sig != match.enum_sig:
            stats["enum_mismatch_fixed"] += 1
        if how == "exact":
            stats["matched_exact"] += 1
        elif how == "fuzzy":
            stats["matched_fuzzy"] += 1
        else:
            stats["unmatched_kept"] += 1
        seed.append(row)

    stats["seed_rows"] = len(seed)
    stats["seed_with_definition"] = sum(1 for c in seed if c.definition)
    stats["cryptonite_train"] = len(cryptonite["train"])
    stats["cryptonite_val"] = len(cryptonite["val"])
    stats["cryptonite_test"] = len(cryptonite["test"])
    stats["leakage_examples"] = leakage_examples
    return seed, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg: Config = config_from_args(args)
    setup_logging(cfg.log_level)

    cryptonite = load_all(
        cfg.data.cryptonite_dir,
        limits={
            "train": cfg.data.limit_train,
            "val": cfg.data.limit_val,
            "test": cfg.data.limit_test,
        },
    )
    wordplay = load_wordplay(cfg.data.wordplay_file)
    seed, stats = build_seed(
        cryptonite,
        wordplay,
        fuzzy_threshold=cfg.data.fuzzy_threshold,
        use_unmatched=cfg.data.use_unmatched_seed,
    )

    out = Path(cfg.data.processed_dir)
    n = write_jsonl(out / "seed_train.jsonl", (clue_to_row(c) for c in seed))
    for split in ("train", "val", "test"):
        write_jsonl(
            out / f"cryptonite_{split}.jsonl",
            (clue_to_row(c) for c in cryptonite[split]),
        )
    write_json(out / "align_report.json", stats)

    log.info("seed set: %d rows -> %s", n, out / "seed_train.jsonl")
    log.info(
        "matched exact=%d fuzzy=%d unmatched-kept=%d | DROPPED for val/test leakage=%d",
        stats["matched_exact"],
        stats["matched_fuzzy"],
        stats["unmatched_kept"],
        stats["dropped_leakage"],
    )
    if stats["dropped_leakage"]:
        log.warning(
            "%d cryptic-wordplay rationales had answers in Cryptonite val/test and were "
            "discarded. Report this number in the write-up.",
            stats["dropped_leakage"],
        )
    if n == 0:
        log.error("Empty seed set - check that both datasets downloaded correctly.")


if __name__ == "__main__":
    main()
