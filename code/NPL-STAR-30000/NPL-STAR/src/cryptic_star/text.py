"""Text normalisation shared by every stage.

Two jobs live here, and they must not be confused with each other:

1. `norm_*` functions produce *join keys and comparison keys*. They are
   aggressive and lossy on purpose (Cryptonite and cryptic-wordplay quote the
   same clue with different punctuation, casing and enumeration placement).
2. `sanitize_for_t5` produces *model-visible text*. T5's SentencePiece
   vocabulary has no tokens for `{ } [ ] < > \\ ^ ~ | ` `, so those characters
   are silently dropped by the tokenizer. cryptic-wordplay marks definitions
   with `{}` and its wordplay strings are full of brackets, so leaving them in
   would mean the model trains on text it can never reproduce.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------
# T5-safe text
# --------------------------------------------------------------------------

# Characters absent from the T5 SentencePiece vocab -> closest safe stand-in.
_T5_REPLACEMENTS = {
    "{": "(",
    "}": ")",
    "[": "(",
    "]": ")",
    "<": "(",
    ">": ")",
    "\\": "/",
    "|": ",",
    "^": "",
    "~": "-",
    "`": "'",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
    " ": " ",
    "\t": " ",
    "\n": " ",
    "\r": " ",
}

_T5_TABLE = str.maketrans(_T5_REPLACEMENTS)

# Field separator used in the seq2seq target. Must survive the tokenizer and
# must not appear inside a field value, so field values get it stripped.
FIELD_SEP = ";"


def sanitize_for_t5(text: str) -> str:
    """Make `text` round-trippable through the T5 tokenizer."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_T5_TABLE)
    return collapse_ws(text)


def sanitize_field(text: str) -> str:
    """T5-safe *and* free of the field separator, for use inside a target."""
    return collapse_ws(sanitize_for_t5(text).replace(FIELD_SEP, ","))


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Clue handling
# --------------------------------------------------------------------------

# A trailing enumeration such as "(7)", "(3,4)", "(5-4)", "(2,3,4)", "(7,hyph)".
_TRAILING_ENUM = re.compile(r"\s*\(([0-9][0-9,\-\s\.]*[a-z,\.\s]*)\)\s*$", re.I)
_BRACED = re.compile(r"\{([^}]*)\}")


def strip_definition_markers(clue: str) -> str:
    """`"Bird {of prey}"` -> `"Bird of prey"`.

    The markers are supervision, not input: they must never reach the encoder,
    or the model is told half the answer.
    """
    return collapse_ws(_BRACED.sub(r"\1", clue))


def extract_definitions(clue: str) -> list[str]:
    """Pull out the `{...}` definition span(s) from a cryptic-wordplay clue."""
    return [collapse_ws(d) for d in _BRACED.findall(clue) if collapse_ws(d)]


def split_trailing_enumeration(clue: str) -> tuple[str, str | None]:
    """Separate a clue from a trailing `(7)`-style enumeration, if present.

    Cryptonite stores the enumeration both in its own field and (usually) at
    the end of the clue text; cryptic-wordplay stores it only in `pattern`.
    """
    m = _TRAILING_ENUM.search(clue)
    if not m:
        return collapse_ws(clue), None
    return collapse_ws(clue[: m.start()]), collapse_ws(m.group(1))


def norm_clue(clue: str) -> str:
    """Join key for a clue: letters and digits only, lowercase."""
    clue = strip_definition_markers(clue)
    clue, _ = split_trailing_enumeration(clue)
    clue = unicodedata.normalize("NFKD", clue)
    clue = "".join(c for c in clue if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", clue.lower())


def clue_tokens(clue: str) -> set[str]:
    """Bag of word tokens, for the fuzzy fallback in the alignment step."""
    clue = strip_definition_markers(clue)
    clue, _ = split_trailing_enumeration(clue)
    return {t for t in re.split(r"[^a-z0-9]+", clue.lower()) if t}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------
# Answer handling
# --------------------------------------------------------------------------

_NON_ALPHA = re.compile(r"[^A-Z]")


def answer_letters(answer: str) -> str:
    """`"pipe dream"` -> `"PIPEDREAM"`. The canonical comparison key."""
    answer = unicodedata.normalize("NFKD", answer or "")
    answer = "".join(c for c in answer if not unicodedata.combining(c))
    return _NON_ALPHA.sub("", answer.upper())


def surface_answer(answer: str) -> str:
    """Display form: uppercase, spaces and hyphens kept."""
    return collapse_ws(sanitize_for_t5(answer or "").upper())


def enumeration_of(answer: str) -> str:
    """Derive `(3,5)`-style enumeration content from a surface answer.

    `"pipe dream"` -> `"4,5"`;  `"well-known"` -> `"4-5"`.
    """
    parts = re.split(r"([ \-])", collapse_ws(answer or ""))
    out: list[str] = []
    for part in parts:
        if part == " ":
            out.append(",")
        elif part == "-":
            out.append("-")
        elif part:
            out.append(str(len(_NON_ALPHA.sub("", part.upper()))))
    return "".join(out)


def enum_signature(enumeration: str | None) -> tuple[int, ...]:
    """Comparable form of an enumeration: `"3,4"` and `"3-4"` -> `(3, 4)`.

    Word *lengths* are the signal; whether the split is a space or a hyphen is
    an inconsistency between the two datasets, so it is deliberately ignored.
    Returns `()` when the enumeration is missing or unparseable.
    """
    if not enumeration:
        return ()
    nums = re.findall(r"\d+", enumeration)
    return tuple(int(n) for n in nums)


def spaced_letters(answer: str) -> str:
    """`"PIPEDREAM"` -> `"P I P E D R E A M"`.

    T5 tokenises `PIPEDREAM` as a couple of subwords, which makes letter-level
    wordplay (anagrams, hidden words, first letters) nearly impossible to
    express. Spacing the letters forces one token per letter. Enabled with
    `format.spaced_answer` and worth an ablation in the write-up.
    """
    return " ".join(answer_letters(answer))


def unspace_letters(text: str) -> str:
    """Inverse of `spaced_letters`, tolerant of the model's spacing mistakes."""
    return answer_letters(text)
