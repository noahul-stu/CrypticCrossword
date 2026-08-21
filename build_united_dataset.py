#!/usr/bin/env python3
"""Build the united Cryptonite + Wordplay cryptic-crossword dataset.

Run from the parent directory that holds `cryptonite-official-split/` and
`cryptic-wordplay-main/`:

    python3 united-cryptonite-wordplay-dataset/build_united_dataset.py

Build decisions (see README.md for the rationale):
  * Full union of both sources, one unified schema.
  * `clue` is rendered in Cryptonite style: lowercased, definition `{}` markers
    stripped, enumeration appended inline.
  * Deduplication is by normalized clue text and is GLOBAL, with split
    precedence train > val > test, so a clue text appears exactly once in the
    whole dataset.
  * Nothing is silently discarded: every dropped row is written to reports/.
"""

import collections
import json
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CN = os.path.join(ROOT, "cryptonite-official-split", "cryptonite-%s.jsonl")
WP = os.path.join(ROOT, "cryptic-wordplay-main", "prebuilt", "sample_teacow_%s.jsonl")
REPORTS = os.path.join(HERE, "reports")

SPLITS = ["train", "val", "test"]  # order matters: dedup precedence

ENUM_RE = re.compile(r"\s*\(([\d,\-\s]*)\)\s*$")


# ---------------------------------------------------------------- normalization

def _unify(text):
    """NFKC + fold the unicode punctuation variants that differ between sources."""
    s = unicodedata.normalize("NFKC", text)
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-"), ("−", "-"), (" ", " ")):
        s = s.replace(a, b)
    # Source artifact: one Cryptonite clue encodes a comma as '{cm}'.
    s = s.replace("{cm}", ",")
    return re.sub(r"\s+", " ", s).strip()


def clue_words(clue, strip_braces):
    """Soft-normalized clue: no {} markers, no trailing enumeration, lowercase."""
    s = _unify(clue)
    if strip_braces:
        s = s.replace("{", "").replace("}", "")
    s = ENUM_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def dedup_key(clue, strip_braces):
    """Punctuation-insensitive key: alphanumerics of the clue, enumeration excluded."""
    return re.sub(r"[^a-z0-9]+", "", clue_words(clue, strip_braces))


def norm_answer(answer):
    return re.sub(r"[^a-z0-9]+", "", _unify(answer).lower())


def to_cryptonite_enumeration(pattern):
    """Wordplay `pattern` ('7-2-3', '11,3') -> Cryptonite enumeration ('(7,2,3)').

    Cryptonite never uses hyphens in enumerations (verified: 0 of 1,588 forms),
    so hyphens are folded to commas. The original is preserved separately.
    Braces are stripped first: one Wordplay row has a definition marker that
    leaked into its `pattern` field ('{8}').
    """
    cleaned = pattern.replace("{", "").replace("}", "").strip()
    return "(%s)" % re.sub(r"[-\s]+", ",", cleaned)


# ------------------------------------------------------------------- record I/O

PUBLICATION_MAP = {
    "financial-times": ("FT", "Financial Times"),
    "guardian-cryptic": ("Guardian", "Guardian Cryptic"),
    "guardian-prize": ("Guardian", "Guardian Prize"),
}


def blank_to_none(value):
    return value if value else None


def from_cryptonite(row, split):
    clue = _unify(row["clue"])
    enumeration = _unify(row["enumeration"])
    return {
        "clue": clue.lower(),
        "answer": _unify(row["answer"]).lower(),
        "enumeration": enumeration,
        "enumeration_raw": enumeration,
        "clue_with_definition": None,
        "wordplay": None,
        "wordplay_author": None,
        "orientation": blank_to_none(row.get("orientation")),
        "number": row.get("number"),
        "publisher": blank_to_none(row.get("publisher")),
        "sub_publisher": blank_to_none(row.get("sub_publisher")),
        "date": row.get("date"),
        "setter": blank_to_none(row.get("author")),
        "quick": bool(row.get("quick")),
        "sources": ["cryptonite"],
        "split": split,
        "alt_answers": [],
    }


def from_wordplay(row, split):
    marked = _unify(row["clue"])                      # keeps {} and original case
    bare = marked.replace("{", "").replace("}", "")
    enumeration = to_cryptonite_enumeration(row["pattern"])
    raw_enum = "(%s)" % row["pattern"].replace("{", "").replace("}", "").strip()
    publisher, sub_publisher = PUBLICATION_MAP.get(
        row.get("publication"), (blank_to_none(row.get("publication")), None))
    return {
        "clue": ("%s %s" % (bare.lower(), enumeration)).strip(),
        "answer": _unify(row["answer"]).lower(),
        "enumeration": enumeration,
        "enumeration_raw": raw_enum,
        "clue_with_definition": marked,
        "wordplay": _unify(row["wordplay"]) or None,
        "wordplay_author": blank_to_none(row.get("author")),
        "orientation": {"A": "across", "D": "down"}.get(row.get("ad")),
        "number": row.get("num"),
        "publisher": publisher,
        "sub_publisher": sub_publisher,
        "date": None,
        "setter": blank_to_none(row.get("setter")),
        "quick": bool(row.get("is_quick")),
        "sources": ["wordplay"],
        "split": split,
        "alt_answers": [],
    }


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------- merging

FIELDS_FROM_DONOR = ("clue_with_definition", "wordplay", "wordplay_author",
                     "orientation", "number", "publisher", "sub_publisher",
                     "date", "setter")


def merge_metadata(primary, donor):
    """Fill gaps in `primary` from `donor` (same clue AND same answer only)."""
    for field in FIELDS_FROM_DONOR:
        if primary.get(field) in (None, "") and donor.get(field) not in (None, ""):
            primary[field] = donor[field]
    for source in donor["sources"]:
        if source not in primary["sources"]:
            primary["sources"].append(source)


def pick_primary(rows, split):
    """Choose the surviving row for a clue group.

    Only rows belonging to `split` are eligible: the split precedence is
    train > val > test, so a duplicate found in a later split must never
    replace the earlier split's record. Among eligible rows, prefer one with a
    wordplay breakdown (it justifies the answer), then the earliest date, then
    first seen.
    """
    eligible = [(i, r) for i, r in enumerate(rows) if r["split"] == split]

    def rank(item):
        index, row = item
        return (0 if row["wordplay"] else 1,
                row["date"] if row["date"] is not None else float("inf"),
                index)
    return min(eligible, key=rank)[1]


def main():
    os.makedirs(REPORTS, exist_ok=True)

    # ---- load ------------------------------------------------------------
    loaded = {}
    for split in SPLITS:
        rows = [from_cryptonite(r, split) for r in read_jsonl(CN % split)]
        wp_path = WP % split
        if os.path.exists(wp_path):
            rows += [from_wordplay(r, split) for r in read_jsonl(wp_path)]
        loaded[split] = rows
        print("loaded %-5s : %d rows" % (split, len(rows)))

    # ---- group by clue key, globally, honouring split precedence ----------
    groups = collections.OrderedDict()   # key -> {"split": s, "rows": [...]}
    cross_split = []
    for split in SPLITS:
        for row in loaded[split]:
            key = dedup_key(row["clue"], strip_braces=False)
            if not key:
                # Punctuation-only clues are real (Cryptonite has '? (8)' ->
                # 'clueless'); key them on the punctuation so they survive.
                key = "punct:" + clue_words(row["clue"], False)
            if key in groups:
                group = groups[key]
                if group["split"] != split:
                    cross_split.append({"clue_key": key,
                                        "kept_in_split": group["split"],
                                        "dropped_from_split": split,
                                        "kept_clue": group["rows"][0]["clue"],
                                        "dropped_clue": row["clue"],
                                        "dropped_answer": row["answer"]})
                group["rows"].append(row)
            else:
                groups[key] = {"split": split, "rows": [row]}

    # ---- collapse each group to one record --------------------------------
    united = {s: [] for s in SPLITS}
    dropped, answer_conflicts, enum_conflicts = [], [], []

    for key, group in groups.items():
        rows = group["rows"]
        primary = pick_primary(rows, group["split"])
        primary_answer = norm_answer(primary["answer"])

        alt_answers, alt_enums = [], []
        for row in rows:
            if row is primary:
                continue
            same_answer = norm_answer(row["answer"]) == primary_answer
            same_split = row["split"] == group["split"]
            if same_answer and same_split:
                # Only merge metadata within a split; a cross-split duplicate is
                # discarded outright so the official split boundaries are exact.
                merge_metadata(primary, row)
            elif not same_answer and same_split:
                if row["answer"] not in alt_answers:
                    alt_answers.append(row["answer"])
            if (same_split and row["enumeration"] != primary["enumeration"]
                    and row["enumeration"] not in alt_enums):
                alt_enums.append(row["enumeration"])
            dropped.append({
                "clue_key": key,
                "reason": ("cross_split_duplicate" if not same_split else
                           "duplicate_clue_same_answer" if same_answer else
                           "duplicate_clue_different_answer"),
                "kept_split": group["split"], "row_split": row["split"],
                "kept_clue": primary["clue"], "kept_answer": primary["answer"],
                "dropped_clue": row["clue"], "dropped_answer": row["answer"],
                "dropped_sources": row["sources"],
            })

        if alt_answers:
            answer_conflicts.append({
                "clue_key": key, "clue": primary["clue"],
                "kept_answer": primary["answer"], "alt_answers": alt_answers,
                "kept_sources": primary["sources"],
            })
        if alt_enums:
            enum_conflicts.append({
                "clue_key": key, "clue": primary["clue"],
                "kept_enumeration": primary["enumeration"], "alt_enumerations": alt_enums,
            })

        primary["alt_answers"] = alt_answers
        primary["n_source_rows"] = len(rows)
        united[group["split"]].append(primary)

    # ---- stamp ids and write ---------------------------------------------
    field_order = ["id", "split", "clue", "answer", "enumeration", "orientation",
                   "number", "publisher", "sub_publisher", "date", "setter",
                   "quick", "wordplay", "clue_with_definition", "wordplay_author",
                   "enumeration_raw", "sources", "alt_answers", "n_source_rows"]
    for split in SPLITS:
        for index, row in enumerate(united[split], start=1):
            row["id"] = "%s-%06d" % (split, index)
        write_jsonl(os.path.join(HERE, "%s.jsonl" % split),
                    [{k: row[k] for k in field_order} for row in united[split]])
        print("wrote %-5s.jsonl : %d unique clues" % (split, len(united[split])))

    write_jsonl(os.path.join(REPORTS, "dropped_duplicate_rows.jsonl"), dropped)
    write_jsonl(os.path.join(REPORTS, "answer_conflicts.jsonl"), answer_conflicts)
    write_jsonl(os.path.join(REPORTS, "enumeration_conflicts.jsonl"), enum_conflicts)
    write_jsonl(os.path.join(REPORTS, "cross_split_duplicates.jsonl"), cross_split)

    # ---- overlap report: the clues that existed in BOTH sources -----------
    overlap = []
    for split in SPLITS:
        for row in united[split]:
            if len(row["sources"]) < 2:
                continue
            wp_text = row["clue_with_definition"].replace("{", "").replace("}", "")
            cn_text = ENUM_RE.sub("", row["clue"])
            if wp_text == cn_text:
                verdict = "identical"
            elif wp_text.lower() == cn_text.lower():
                verdict = "case_only_difference"
            elif clue_words(wp_text, False).split() == clue_words(cn_text, False).split():
                verdict = "punctuation_or_spacing_difference"
            else:
                verdict = "word_level_difference"
            overlap.append({"id": row["id"], "split": split, "clue": row["clue"],
                            "answer": row["answer"],
                            "wordplay_clue": row["clue_with_definition"],
                            "exactness": verdict})
    write_jsonl(os.path.join(REPORTS, "source_overlap.jsonl"), overlap)

    # ---- stats -----------------------------------------------------------
    stats = {"splits": {}, "totals": {}}
    for split in SPLITS:
        rows = united[split]
        stats["splits"][split] = {
            "unique_clues": len(rows),
            "source_rows_consumed": len(loaded[split]),
            "from_cryptonite_only": sum(1 for r in rows if r["sources"] == ["cryptonite"]),
            "from_wordplay_only": sum(1 for r in rows if r["sources"] == ["wordplay"]),
            "from_both_sources": sum(1 for r in rows if len(r["sources"]) > 1),
            "with_wordplay_annotation": sum(1 for r in rows if r["wordplay"]),
            "quick": sum(1 for r in rows if r["quick"]),
            "with_alt_answers": sum(1 for r in rows if r["alt_answers"]),
            "publishers": dict(collections.Counter(r["publisher"] for r in rows).most_common()),
        }
    stats["totals"] = {
        "unique_clues": sum(len(united[s]) for s in SPLITS),
        "source_rows_consumed": sum(len(loaded[s]) for s in SPLITS),
        "rows_dropped_as_duplicates": len(dropped),
        "answer_conflicts": len(answer_conflicts),
        "enumeration_conflicts": len(enum_conflicts),
        "cross_split_duplicates_dropped": len(cross_split),
        "with_wordplay_annotation": sum(
            stats["splits"][s]["with_wordplay_annotation"] for s in SPLITS),
    }
    with open(os.path.join(HERE, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("\n" + json.dumps(stats["totals"], indent=2))


if __name__ == "__main__":
    main()
