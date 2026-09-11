"""Loader for `united-cryptonite-wordplay-dataset`.

This is the pre-joined dataset: Cryptonite's clue->answer pairs and the
mdda/cryptic-wordplay rationales already merged into one file per split, so the
fuzzy clue matching in `align.py` is not needed. The join is simply "is the
`wordplay` field non-null on this row".

    train.jsonl.gz   474,950 clues,  5,403 with a wordplay annotation
    val.jsonl.gz      26,387 clues,    300 with a wordplay annotation
    test.jsonl.gz     26,082 clues,      0 with a wordplay annotation

Row schema (all 18 keys present on every row, nulls where absent)::

    id                   "train-000001"
    clue                 Cryptonite style: lowercased, enumeration appended
                         inline, no {} definition markers
    answer               lowercase, may contain spaces
    enumeration          "(4,2)" - with parentheses, commas only
    orientation          "across" / "down"
    number               grid number
    publisher            "Times" / "FT" / ...
    sub_publisher        "The Times" / "Financial Times" / ...
    date                 epoch ms; null on every wordplay-only row
    setter               puzzle setter, often null
    quick                bool - Quick (non-cryptic) puzzle
    wordplay             free-form human analysis, or null
    comment              extra annotator note, or null (107 rows)
    clue_with_definition original case, {} around the definition span(s), or null
    enumeration_raw      hyphenated form where it differs (126 rows), or null
    sources              ["cryptonite"] / ["wordplay"] / both
    alt_answers          other answers seen for this clue text

Two things this loader has to be careful about, because the dataset's own
`verify_dataset.py` does not check them:

1. **Answer-disjointness.** Cryptonite's official split is answer-disjoint, but
   this dataset added 5,253 wordplay-only rows to train from an independently
   scraped source. If any of their answers appear in val/test, the strict split
   is silently broken. `answer_overlap_report` measures it and
   `enforce_answer_split` removes the offenders - the same guard `align.py`
   applies, moved to where the data now comes from.

2. **Label-only rationales.** Some annotations are a bare wordplay *type*
   ("Double Definition") rather than a decomposition. They are legitimate but
   carry no reasoning to imitate, so they are counted and can be filtered with
   `data.drop_label_only_rationales`.

Test has no annotations at all (the wordplay repo asks that no test split be
built from it), so explanation quality can only be measured on val.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from ..config import Config, add_config_args, config_from_args
from ..io_utils import read_jsonl, setup_logging, write_json, write_jsonl
from ..text import (
    collapse_ws,
    enum_signature,
    enumeration_of,
    extract_definitions,
    sanitize_field,
    split_trailing_enumeration,
    surface_answer,
)
from .align import clue_to_row
from .cryptonite import SPLITS, Clue

log = logging.getLogger(__name__)

# Annotations that name the wordplay device without decomposing the clue. Kept
# by default (there are only ~5.7k annotated rows in total and the device name
# is still supervision), but counted so the report says how many there are.
_LABEL_ONLY = {
    "anagram",
    "double definition",
    "triple definition",
    "hidden",
    "hidden word",
    "reversal",
    "charade",
    "container",
    "containment",
    "homophone",
    "deletion",
    "cryptic definition",
    "initial letters",
    "final letters",
    "alternate letters",
    "spoonerism",
    "acrostic",
    "insertion",
    "substitution",
    "and lit",
    "lit",
    "all in one",
    "definition",
}

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def _norm_label(text: str) -> str:
    return collapse_ws(_NON_ALNUM.sub(" ", (text or "").lower()))


def rationale_is_label_only(rationale: str) -> bool:
    """True when the annotation names a device instead of explaining the clue."""
    norm = _norm_label(rationale)
    return not norm or norm in _LABEL_ONLY or len(norm.split()) <= 2


def _bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def split_file(directory: str | Path, split: str) -> Path:
    """`<dir>/<split>.jsonl.gz`, falling back to an already-gunzipped file."""
    directory = Path(directory)
    for name in (f"{split}.jsonl.gz", f"{split}.jsonl"):
        path = directory / name
        if path.exists():
            return path
    matches = sorted(directory.glob(f"*{split}*.jsonl*")) if directory.exists() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No '{split}.jsonl.gz' under {directory}. Point data.united_dir at the "
        "united-cryptonite-wordplay-dataset directory (the one holding "
        "train.jsonl.gz, val.jsonl.gz, test.jsonl.gz and stats.json)."
    )


def _rationale(row: dict) -> str:
    """Merge `wordplay` with `comment`; `comment` sometimes holds the detail."""
    parts = [collapse_ws(str(row.get(key) or "")) for key in ("wordplay", "comment")]
    return sanitize_field(" - ".join(p for p in parts if p))


def _enumeration(row: dict, answer: str, trailing: str) -> str:
    """Prefer `enumeration_raw` (keeps hyphens), then `enumeration`, then derive."""
    for candidate in (row.get("enumeration_raw"), row.get("enumeration"), trailing):
        enum = collapse_ws(str(candidate or "")).strip("()")
        if enum_signature(enum):
            return enum
    return enumeration_of(answer)


def _definition(row: dict) -> str:
    """Definition span(s) from `clue_with_definition`'s {} markers.

    Lowercased to match the `clue` field, which is Cryptonite-style lowercase -
    otherwise the definition the model is trained to emit would not be a literal
    substring of the clue it was given, and the grounding metric in analyze.py
    would under-report.
    """
    marked = collapse_ws(str(row.get("clue_with_definition") or ""))
    if not marked:
        return ""
    spans = [s.lower() for s in extract_definitions(marked)]
    return sanitize_field(" / ".join(spans))


def row_to_clue(row: dict, split: str, index: int) -> Clue | None:
    """One dataset row -> `Clue`, or None if unusable."""
    raw_clue = collapse_ws(str(row.get("clue") or ""))
    answer = surface_answer(row.get("answer") or "")
    if not raw_clue or not answer:
        return None
    clue_text, trailing = split_trailing_enumeration(raw_clue)
    rationale = _rationale(row)
    return Clue(
        clue=clue_text,
        answer=answer,
        enumeration=_enumeration(row, answer, trailing),
        clue_id=str(row.get("id") or f"{split}-{index}"),
        split=split,
        quick=_bool(row.get("quick")),
        publisher=str(row.get("publisher") or ""),
        source="united",
        definition=_definition(row),
        wordplay=rationale,
        alt_answers=[surface_answer(a) for a in (row.get("alt_answers") or [])],
        meta={
            "sources": row.get("sources") or [],
            "sub_publisher": row.get("sub_publisher"),
            "setter": row.get("setter"),
            "orientation": row.get("orientation"),
            "number": row.get("number"),
            "date": row.get("date"),
            "label_only_rationale": bool(rationale)
            and rationale_is_label_only(rationale),
        },
    )


def load_united_split(
    united_dir: str | Path,
    split: str,
    limit: int | None = None,
    drop_quick: bool = True,
    min_wordplay_chars: int = 8,
    drop_label_only_rationales: bool = False,
) -> list[Clue]:
    """Load one split of the united dataset.

    `drop_quick` removes Quick (non-cryptic) puzzles: they are plain definition
    clues with no wordplay, and training on them teaches the generator to skip
    the reasoning step. Rationales shorter than `min_wordplay_chars` are dropped
    (the clue itself is kept, it just moves to the unannotated pool).
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    path = split_file(united_dir, split)

    clues: list[Clue] = []
    stats = {
        "quick": 0,
        "malformed": 0,
        "rationale_too_short": 0,
        "rationale_label_only": 0,
    }
    for i, row in enumerate(read_jsonl(path)):
        if drop_quick and _bool(row.get("quick")):
            stats["quick"] += 1
            continue
        clue = row_to_clue(row, split, i)
        if clue is None:
            stats["malformed"] += 1
            continue
        if clue.wordplay:
            if len(clue.wordplay) < min_wordplay_chars:
                stats["rationale_too_short"] += 1
                clue.wordplay = ""
            elif clue.meta.get("label_only_rationale"):
                stats["rationale_label_only"] += 1
                if drop_label_only_rationales:
                    clue.wordplay = ""
        clues.append(clue)
        if limit is not None and len(clues) >= limit:
            break

    annotated = sum(1 for c in clues if c.has_rationale)
    log.info(
        "united/%s: %d clues from %s, %d with a rationale (skipped %d quick, "
        "%d malformed; %d rationales too short, %d device-label only)",
        split, len(clues), path.name, annotated,
        stats["quick"], stats["malformed"],
        stats["rationale_too_short"], stats["rationale_label_only"],
    )
    return clues


def load_united(
    united_dir: str | Path,
    limits: dict[str, int | None] | None = None,
    drop_quick: bool = True,
    min_wordplay_chars: int = 8,
    drop_label_only_rationales: bool = False,
) -> dict[str, list[Clue]]:
    limits = limits or {}
    return {
        split: load_united_split(
            united_dir,
            split,
            limits.get(split),
            drop_quick=drop_quick,
            min_wordplay_chars=min_wordplay_chars,
            drop_label_only_rationales=drop_label_only_rationales,
        )
        for split in SPLITS
    }


# --------------------------------------------------------------------------
# the answer split
# --------------------------------------------------------------------------

def answer_overlap_report(splits: dict[str, list[Clue]]) -> dict:
    """How badly the answer split is violated, per pair of splits.

    Cryptonite's official split is answer-disjoint by construction. Anything
    non-zero here comes from the wordplay rows this dataset added to train, and
    means the val/test numbers are optimistic until it is removed.
    """
    letters = {name: {c.letters for c in clues} for name, clues in splits.items()}
    report: dict = {
        "answers_per_split": {k: len(v) for k, v in letters.items()},
        "clues_per_split": {k: len(v) for k, v in splits.items()},
        "overlaps": {},
        "answer_disjoint": True,
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = letters.get(a, set()) & letters.get(b, set())
        rows_a = sum(1 for c in splits.get(a, []) if c.letters in shared)
        annotated_a = sum(
            1 for c in splits.get(a, []) if c.letters in shared and c.has_rationale
        )
        report["overlaps"][f"{a}|{b}"] = {
            "shared_answers": len(shared),
            f"{a}_rows_affected": rows_a,
            f"{a}_annotated_rows_affected": annotated_a,
            "examples": sorted(shared)[:20],
        }
        if shared:
            report["answer_disjoint"] = False
    return report


def enforce_answer_split(splits: dict[str, list[Clue]]) -> tuple[dict[str, list[Clue]], dict]:
    """Drop rows that break answer-disjointness, keeping the eval splits intact.

    Rows are removed from the *earlier* split of each pair (train before val,
    val before test) so that test - the split the headline number is reported on
    - is never shrunk, and the comparison against Cryptonite's published
    baseline stays on the same clue set.
    """
    test_answers = {c.letters for c in splits.get("test", [])}
    val_answers = {c.letters for c in splits.get("val", [])}

    kept = dict(splits)
    stats: dict = {}
    for name, forbidden in (("train", val_answers | test_answers), ("val", test_answers)):
        before = splits.get(name, [])
        after = [c for c in before if c.letters not in forbidden]
        removed = len(before) - len(after)
        stats[f"{name}_rows_dropped"] = removed
        stats[f"{name}_annotated_rows_dropped"] = sum(
            1 for c in before if c.letters in forbidden and c.has_rationale
        )
        kept[name] = after
        if removed:
            log.warning(
                "answer split: dropped %d %s rows whose answer appears in a later "
                "split (%d of them carried a human rationale). Report this number.",
                removed, name, stats[f"{name}_annotated_rows_dropped"],
            )
    return kept, stats


# --------------------------------------------------------------------------
# seed / pool
# --------------------------------------------------------------------------

def split_seed_and_pool(clues: list[Clue]) -> tuple[list[Clue], list[Clue]]:
    """Annotated rows (STaR seed set) vs the rest (self-generation pool).

    This replaces `align.py`'s matching step: the join is already in the data.
    Seed rows stay in the pool as well - a clue with one human rationale can
    still gain more self-generated ones, which is the point of feeding back
    every correct reasoning path.
    """
    seed = [c for c in clues if c.has_rationale]
    for c in seed:
        c.meta = {**c.meta, "matched_cryptonite": "cryptonite" in (c.meta.get("sources") or [])}
    return seed, clues


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg: Config = config_from_args(args)
    setup_logging(cfg.log_level)
    prepare(cfg)


def prepare(cfg: Config) -> dict:
    """Read the united dataset, enforce the answer split, write `data/processed`.

    Output filenames match what `align.py` produces so the rest of the pipeline
    does not care which source the clues came from.
    """
    splits = load_united(
        cfg.data.united_dir,
        limits={
            "train": cfg.data.limit_train,
            "val": cfg.data.limit_val,
            "test": cfg.data.limit_test,
        },
        drop_quick=cfg.data.drop_quick,
        min_wordplay_chars=cfg.data.min_wordplay_chars,
        drop_label_only_rationales=cfg.data.drop_label_only_rationales,
    )

    report = {"before": answer_overlap_report(splits)}
    if cfg.data.enforce_answer_split:
        splits, drop_stats = enforce_answer_split(splits)
        report["enforced"] = drop_stats
        report["after"] = answer_overlap_report(splits)
    else:
        log.warning(
            "data.enforce_answer_split is off: val/test answers may also appear in "
            "train, which inflates every number this project reports."
        )

    seed, pool = split_seed_and_pool(splits["train"])
    out = Path(cfg.data.processed_dir)
    write_jsonl(out / "seed_train.jsonl", (clue_to_row(c) for c in seed))
    for split, clues in splits.items():
        write_jsonl(out / f"cryptonite_{split}.jsonl", (clue_to_row(c) for c in clues))

    report["seed_rows"] = len(seed)
    report["seed_with_definition"] = sum(1 for c in seed if c.definition)
    report["seed_label_only"] = sum(
        1 for c in seed if c.meta.get("label_only_rationale")
    )
    report["pool_rows"] = len(pool)
    report["val_annotated"] = sum(1 for c in splits["val"] if c.has_rationale)
    report["test_annotated"] = sum(1 for c in splits["test"] if c.has_rationale)
    write_json(out / "united_report.json", report)

    log.info(
        "united: seed=%d (%d with a definition span, %d device-label only) | "
        "train pool=%d | val=%d | test=%d",
        report["seed_rows"], report["seed_with_definition"], report["seed_label_only"],
        report["pool_rows"], len(splits["val"]), len(splits["test"]),
    )
    if not report["before"]["answer_disjoint"]:
        log.warning(
            "the shipped splits were NOT answer-disjoint: %s",
            {k: v["shared_answers"] for k, v in report["before"]["overlaps"].items()},
        )
    if report["test_annotated"] == 0:
        log.info(
            "test has no wordplay annotations, as expected - measure explanation "
            "quality on val (%d annotated clues).", report["val_annotated"]
        )
    if report["seed_rows"] == 0:
        log.error("Empty seed set - check data.united_dir points at the right directory.")
    return report


if __name__ == "__main__":
    main()
