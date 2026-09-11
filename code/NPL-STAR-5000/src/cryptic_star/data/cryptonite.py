"""Cryptonite loader (Efrat et al., 2021).

One jsonl row per clue, fields: `clue`, `answer`, `enumeration`, `publisher`,
`date`, `quick`, `id`. Answers are lowercase and multi-word answers contain
spaces; the clue text usually repeats the enumeration in trailing parentheses.

Only the *official* split is loaded by default. That split is answer-disjoint
("answer split" in the paper) which is what the proposal evaluates on, and it
is what makes the leakage check in `align.py` possible at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..io_utils import read_jsonl
from ..text import (
    answer_letters,
    collapse_ws,
    enum_signature,
    enumeration_of,
    split_trailing_enumeration,
    surface_answer,
)

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


@dataclass
class Clue:
    """A clue from either dataset, in one normalised shape."""

    clue: str  # definition markers stripped, enumeration stripped
    answer: str  # surface form, uppercase, spaces/hyphens kept
    enumeration: str  # "7" / "3,4" / "5-4"
    letters: str = ""  # answer with everything but A-Z removed
    clue_id: str = ""
    split: str = ""
    quick: bool = False
    publisher: str = ""
    source: str = "cryptonite"
    # Populated only for rows that carry a human rationale (cryptic-wordplay).
    definition: str = ""
    wordplay: str = ""
    # Other answers documented for this same clue text. The united dataset
    # records these where two sources disagreed; scoring against them would
    # inflate EM relative to the published baseline, so they are reported
    # separately and only accepted when `verify.accept_alt_answers` is on.
    alt_answers: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.letters:
            self.letters = answer_letters(self.answer)

    @property
    def enum_sig(self) -> tuple[int, ...]:
        return enum_signature(self.enumeration)

    @property
    def has_rationale(self) -> bool:
        return bool(self.wordplay)

    @property
    def alt_letters(self) -> set[str]:
        return {answer_letters(a) for a in self.alt_answers if answer_letters(a)}


def _split_file(directory: Path, split: str) -> Path:
    """Find `cryptonite-<split>.jsonl` without hardcoding the exact filename."""
    candidates = sorted(directory.glob(f"*{split}*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No '*{split}*.jsonl' under {directory}. "
            "Run scripts/download_data.sh first, or point data.cryptonite_dir "
            "at the directory holding the official-split jsonl files."
        )
    if len(candidates) > 1:
        log.warning("Several %s files in %s, using %s", split, directory, candidates[0].name)
    return candidates[0]


def load_split(
    cryptonite_dir: str | Path,
    split: str,
    limit: int | None = None,
    drop_quick: bool = True,
) -> list[Clue]:
    """Load one Cryptonite split.

    `drop_quick` removes non-cryptic "Quick" puzzles, which are plain
    definition clues with no wordplay and would teach the generator to skip
    the reasoning step entirely.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    path = _split_file(Path(cryptonite_dir), split)

    clues: list[Clue] = []
    skipped_quick = skipped_bad = 0
    for row in read_jsonl(path):
        quick = bool(row.get("quick", False))
        if drop_quick and quick:
            skipped_quick += 1
            continue
        raw_clue = collapse_ws(str(row.get("clue", "")))
        answer = surface_answer(row.get("answer", ""))
        if not raw_clue or not answer:
            skipped_bad += 1
            continue
        clue_text, trailing = split_trailing_enumeration(raw_clue)
        enumeration = collapse_ws(str(row.get("enumeration", ""))) or trailing or ""
        enumeration = enumeration.strip("()")
        if not enum_signature(enumeration):
            # Fall back to deriving it from the answer's word lengths.
            enumeration = enumeration_of(answer)
        clues.append(
            Clue(
                clue=clue_text,
                answer=answer,
                enumeration=enumeration,
                clue_id=str(row.get("id", "")) or f"{split}-{len(clues)}",
                split=split,
                quick=quick,
                publisher=str(row.get("publisher", "")),
                source="cryptonite",
                meta={"date": row.get("date")},
            )
        )
        if limit is not None and len(clues) >= limit:
            break

    log.info(
        "cryptonite/%s: %d clues from %s (skipped %d quick, %d malformed)",
        split,
        len(clues),
        path.name,
        skipped_quick,
        skipped_bad,
    )
    return clues


def load_all(
    cryptonite_dir: str | Path,
    limits: dict[str, int | None] | None = None,
    drop_quick: bool = True,
) -> dict[str, list[Clue]]:
    limits = limits or {}
    return {
        split: load_split(cryptonite_dir, split, limits.get(split), drop_quick)
        for split in SPLITS
    }


def answer_sets(splits: dict[str, list[Clue]]) -> dict[str, set[str]]:
    """Letter-only answer set per split; the basis of the leakage check."""
    return {name: {c.letters for c in clues} for name, clues in splits.items()}
