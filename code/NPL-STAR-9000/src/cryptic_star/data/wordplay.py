"""cryptic-wordplay loader (mdda/cryptic-wordplay).

Per-line fields, per the dataset README:

    clue         clue text, definition part(s) wrapped in '{}'
    pattern      enumeration as printed in the puzzle
    ad           'A' across / 'D' down
    answer       uppercase grid answer, may contain spaces and '-'
    wordplay     free-form human analysis of the wordplay
    author       who wrote the analysis
    setter       puzzle setter
    publication  where it appeared
    is_quick     'Quick' variant flag

The `{}` markers and the `wordplay` strings are the only human-written
reasoning in this project, so this loader is deliberately permissive about
formatting and strict about the two things that matter: an answer exists, and
the wordplay string is long enough to be an actual explanation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..io_utils import read_jsonl
from ..text import (
    collapse_ws,
    enum_signature,
    enumeration_of,
    extract_definitions,
    sanitize_field,
    strip_definition_markers,
    surface_answer,
)
from .cryptonite import Clue

log = logging.getLogger(__name__)

# Analyses shorter than this are things like "anagram" or "dd" - too terse to
# train a reasoning model on.
MIN_WORDPLAY_CHARS = 8


def _is_quick(row: dict) -> bool:
    v = row.get("is_quick", False)
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes", "y"}
    return bool(v)


def load_wordplay(
    path: str | Path,
    limit: int | None = None,
    drop_quick: bool = True,
    min_wordplay_chars: int = MIN_WORDPLAY_CHARS,
) -> list[Clue]:
    path = Path(path)
    if not path.exists():
        # Accept a directory of shards as well as a single file.
        parent = path if path.is_dir() else path.parent
        shards = sorted(parent.glob("*.jsonl")) if parent.exists() else []
        if not shards:
            raise FileNotFoundError(
                f"No cryptic-wordplay jsonl at {path}. Run scripts/download_data.sh "
                "or set data.wordplay_file."
            )
        rows = (r for shard in shards for r in read_jsonl(shard))
        log.info("wordplay: reading %d shard(s) from %s", len(shards), parent)
    else:
        rows = read_jsonl(path)

    out: list[Clue] = []
    dropped = {"quick": 0, "no_answer": 0, "no_wordplay": 0, "no_clue": 0}
    for i, row in enumerate(rows):
        if drop_quick and _is_quick(row):
            dropped["quick"] += 1
            continue
        raw_clue = collapse_ws(str(row.get("clue", "")))
        if not raw_clue:
            dropped["no_clue"] += 1
            continue
        answer = surface_answer(row.get("answer", ""))
        if not answer:
            dropped["no_answer"] += 1
            continue
        wordplay = sanitize_field(str(row.get("wordplay", "")))
        if len(wordplay) < min_wordplay_chars:
            dropped["no_wordplay"] += 1
            continue

        definitions = extract_definitions(raw_clue)
        enumeration = collapse_ws(str(row.get("pattern", ""))).strip("()")
        if not enum_signature(enumeration):
            enumeration = enumeration_of(answer)

        out.append(
            Clue(
                clue=strip_definition_markers(raw_clue),
                answer=answer,
                enumeration=enumeration,
                clue_id=f"wp-{i}",
                split="",  # assigned by align.py from the Cryptonite answer split
                quick=_is_quick(row),
                publisher=str(row.get("publication", "")),
                source="wordplay",
                definition=sanitize_field(" / ".join(definitions)),
                wordplay=wordplay,
                meta={
                    "ad": row.get("ad"),
                    "author": row.get("author"),
                    "setter": row.get("setter"),
                    "raw_clue": raw_clue,  # keeps the {} markers for inspection
                },
            )
        )
        if limit is not None and len(out) >= limit:
            break

    log.info("wordplay: %d usable rationales (dropped %s)", len(out), dropped)
    n_def = sum(1 for c in out if c.definition)
    log.info("wordplay: %d/%d rows have a {} definition span", n_def, len(out))
    return out
