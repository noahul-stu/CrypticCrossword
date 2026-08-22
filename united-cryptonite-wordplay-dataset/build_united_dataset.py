#!/usr/bin/env python3
"""Build the united Cryptonite + Wordplay dataset.

    python3 build_united_dataset.py

Union of both sources in one schema. Clues are rendered Cryptonite-style
(lowercase, enumeration inline, no `{}`). Dedup is by normalized clue text and
is global, with split precedence train > val > test. Every dropped row lands in
reports/.
"""

import collections
import gzip
import json
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_source_root():
    """Find the directory holding the source folders; they may be beside or above."""
    for candidate in (os.path.dirname(HERE), HERE,
                      os.path.dirname(os.path.dirname(HERE))):
        if os.path.isdir(os.path.join(candidate, "cryptonite-official-split")):
            return candidate
    return os.path.dirname(HERE)


ROOT = _find_source_root()
CN = os.path.join(ROOT, "cryptonite-official-split", "cryptonite-%s.jsonl")
WP = os.path.join(ROOT, "cryptic-wordplay-main", "prebuilt", "sample_teacow_%s.jsonl")
REPORTS = os.path.join(HERE, "reports")

SPLITS = ["train", "val", "test"]   # order is the dedup precedence

# Plain train.jsonl is 205 MB, over GitHub's 100 MB limit; gzipped it is 19.7 MB.
COMPRESS_SPLITS = True

ENUM_RE = re.compile(r"\s*\(([\d,\-\s]*)\)\s*$")


def sources_available():
    return all(os.path.exists(CN % split) for split in ("train", "val", "test"))


def split_filename(split):
    return "%s.jsonl.gz" % split if COMPRESS_SPLITS else "%s.jsonl" % split


# --- normalization ---

def _unify(text):
    """Fold the unicode punctuation variants that differ between the sources."""
    s = unicodedata.normalize("NFKC", text)
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-"), ("−", "-"), (" ", " ")):
        s = s.replace(a, b)
    s = s.replace("{cm}", ",")      # one Cryptonite clue encodes a comma this way
    return re.sub(r"\s+", " ", s).strip()


def clue_words(clue, strip_braces):
    """Clue without {} markers, trailing enumeration, or case."""
    s = _unify(clue)
    if strip_braces:
        s = s.replace("{", "").replace("}", "")
    s = ENUM_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def dedup_key(clue, strip_braces):
    """Alphanumerics of the clue. Punctuation-insensitive, enumeration excluded."""
    return re.sub(r"[^a-z0-9]+", "", clue_words(clue, strip_braces))


def norm_answer(answer):
    return re.sub(r"[^a-z0-9]+", "", _unify(answer).lower())


def to_cryptonite_enumeration(pattern):
    """'7-2-3' -> '(7,2,3)'. Cryptonite never hyphenates, so fold to commas.

    Braces are stripped first: one Wordplay row has '{8}' as its pattern.
    """
    cleaned = pattern.replace("{", "").replace("}", "").strip()
    return "(%s)" % re.sub(r"[-\s]+", ",", cleaned)


# --- source rows -> united rows ---

PUBLICATION_MAP = {
    "financial-times": ("FT", "Financial Times"),
    "guardian-cryptic": ("Guardian", "Guardian Cryptic"),
    "guardian-prize": ("Guardian", "Guardian Prize"),
}


def blank_to_none(value):
    return value if value else None


def from_cryptonite(row, split):
    enumeration = _unify(row["enumeration"])
    return {
        "clue": _unify(row["clue"]).lower(),
        "answer": _unify(row["answer"]).lower(),
        "enumeration": enumeration,
        "enumeration_raw": None,       # Cryptonite is already in canonical form
        "clue_with_definition": None,
        "wordplay": None,
        "comment": None,
        "orientation": blank_to_none(row.get("orientation")),
        "number": row.get("number"),
        "publisher": blank_to_none(row.get("publisher")),
        "sub_publisher": blank_to_none(row.get("sub_publisher")),
        "date": row.get("date"),
        "setter": blank_to_none(row.get("author")),   # their `author` is the setter
        "quick": bool(row.get("quick")),
        "sources": ["cryptonite"],
        "split": split,
        "alt_answers": [],
    }


def from_wordplay(row, split):
    marked = _unify(row["clue"])                      # keeps {} and original case
    bare = marked.replace("{", "").replace("}", "")
    enumeration = to_cryptonite_enumeration(row["pattern"])
    raw = "(%s)" % row["pattern"].replace("{", "").replace("}", "").strip()
    publisher, sub_publisher = PUBLICATION_MAP.get(
        row.get("publication"), (blank_to_none(row.get("publication")), None))
    return {
        "clue": ("%s %s" % (bare.lower(), enumeration)).strip(),
        "answer": _unify(row["answer"]).lower(),
        "enumeration": enumeration,
        # Only kept when it differs, i.e. when the answer is hyphenated.
        "enumeration_raw": raw if raw != enumeration else None,
        "clue_with_definition": marked,
        "wordplay": _unify(row["wordplay"]) or None,
        # On 107 rows only. Sometimes holds the decomposition when `wordplay` is
        # just a label like "Double Definition".
        "comment": blank_to_none(_unify(row.get("comment") or "")),
        "orientation": {"A": "across", "D": "down"}.get(row.get("ad")),
        "number": row.get("num"),
        "publisher": publisher,
        "sub_publisher": sub_publisher,
        "date": None,                                 # this source has no dates
        "setter": blank_to_none(row.get("setter")),
        "quick": bool(row.get("is_quick")),
        "sources": ["wordplay"],
        "split": split,
        "alt_answers": [],
    }


# --- io ---

def read_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    """Write .jsonl, or .jsonl.gz when the path says so.

    gzip mtime is pinned to 0 so an unchanged rebuild produces identical bytes
    and no new git blob.
    """
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.endswith(".gz"):
        with open(path, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9,
                               mtime=0) as handle:
                handle.write(body.encode("utf-8"))
    else:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(body)


# --- merging ---

FIELDS_FROM_DONOR = ("clue_with_definition", "wordplay", "comment", "orientation",
                     "number", "publisher", "sub_publisher", "date", "setter")


def merge_metadata(primary, donor):
    """Fill gaps in primary from donor. Callers must check the answers match."""
    for field in FIELDS_FROM_DONOR:
        if primary.get(field) in (None, "") and donor.get(field) not in (None, ""):
            primary[field] = donor[field]
    for source in donor["sources"]:
        if source not in primary["sources"]:
            primary["sources"].append(source)


def pick_primary(rows, split):
    """Pick the survivor of a clue group.

    Only rows from `split` are eligible, so a duplicate in a later split can
    never displace the earlier one. Prefer an annotated row, then the earliest.
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

    loaded = {}
    for split in SPLITS:
        rows = [from_cryptonite(r, split) for r in read_jsonl(CN % split)]
        wp_path = WP % split
        if os.path.exists(wp_path):
            rows += [from_wordplay(r, split) for r in read_jsonl(wp_path)]
        loaded[split] = rows
        print("loaded %-5s : %d rows" % (split, len(rows)))

    # Group by clue key across all splits, earliest split wins.
    groups = collections.OrderedDict()
    cross_split = []
    for split in SPLITS:
        for row in loaded[split]:
            key = dedup_key(row["clue"], strip_braces=False)
            if not key:
                # Punctuation-only clues are real: Cryptonite has '? (8)' -> clueless.
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
                merge_metadata(primary, row)
            elif not same_answer and same_split:
                if row["answer"] not in alt_answers:
                    alt_answers.append(row["answer"])
            if (same_split and row["enumeration"] != primary["enumeration"]
                    and row["enumeration"] not in alt_enums):
                alt_enums.append(row["enumeration"])
            # Cross-split rows contribute nothing: merging or listing their
            # answers would leak val/test content into train.
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
                "kept_enumeration": primary["enumeration"],
                "alt_enumerations": alt_enums,
            })

        primary["alt_answers"] = alt_answers
        united[group["split"]].append(primary)

    # `split` is internal only: the filename and the id prefix both carry it.
    field_order = ["id", "clue", "answer", "enumeration", "orientation", "number",
                   "publisher", "sub_publisher", "date", "setter", "quick",
                   "wordplay", "comment", "clue_with_definition",
                   "enumeration_raw", "sources", "alt_answers"]
    for split in SPLITS:
        for index, row in enumerate(united[split], start=1):
            row["id"] = "%s-%06d" % (split, index)
        path = os.path.join(HERE, split_filename(split))
        write_jsonl(path, [{k: row[k] for k in field_order} for row in united[split]])
        print("wrote %-16s : %6d unique clues, %5.1f MB"
              % (split_filename(split), len(united[split]),
                 os.path.getsize(path) / 1048576.0))

    write_jsonl(os.path.join(REPORTS, "dropped_duplicate_rows.jsonl"), dropped)
    write_jsonl(os.path.join(REPORTS, "answer_conflicts.jsonl"), answer_conflicts)
    write_jsonl(os.path.join(REPORTS, "enumeration_conflicts.jsonl"), enum_conflicts)
    write_jsonl(os.path.join(REPORTS, "cross_split_duplicates.jsonl"), cross_split)

    # How exactly the two sources agreed on the clues they shared.
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
            "with_annotator_comment": sum(1 for r in rows if r["comment"]),
            "with_hyphenated_enumeration": sum(1 for r in rows if r["enumeration_raw"]),
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
