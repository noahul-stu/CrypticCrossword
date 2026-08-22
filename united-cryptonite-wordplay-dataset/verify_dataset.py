#!/usr/bin/env python3
"""Check the united dataset's invariants. Exits non-zero on any failure.

    python3 verify_dataset.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_united_dataset as B  # noqa: E402

SPLITS = ["train", "val", "test"]

FIELDS = ["id", "clue", "answer", "enumeration", "orientation", "number",
          "publisher", "sub_publisher", "date", "setter", "quick", "wordplay",
          "comment", "clue_with_definition", "enumeration_raw", "sources",
          "alt_answers"]

# Every source field and where it went. None means dropped on purpose.
# A source key missing from here is an unmapped field - that is how the
# `comment` field went missing in the first place.
CRYPTONITE_MAP = {
    "clue": "clue", "answer": "answer", "enumeration": "enumeration",
    "orientation": "orientation", "number": "number", "publisher": "publisher",
    "sub_publisher": "sub_publisher", "date": "date", "quick": "quick",
    "author": "setter",
}
WORDPLAY_MAP = {
    "clue": "clue + clue_with_definition", "answer": "answer",
    "pattern": "enumeration + enumeration_raw", "ad": "orientation",
    "num": "number", "publication": "publisher + sub_publisher",
    "is_quick": "quick", "setter": "setter", "wordplay": "wordplay",
    "comment": "comment",
    "author": None,   # always "teacow"; identical information to wordplay != null
}

failures = []


def check(condition, message):
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)


def main():
    data = {s: list(B.read_jsonl(os.path.join(B.HERE, B.split_filename(s))))
            for s in SPLITS}

    print("schema")
    for split in SPLITS:
        bad = [r["id"] for r in data[split] if list(r.keys()) != FIELDS]
        check(not bad, "%s: every row has the exact field set, in order" % split)

    print("\nsource field coverage")
    if not B.sources_available():
        print("  skip   source files not present")
    else:
        for label, paths, mapping in (
                ("cryptonite", [B.CN % s for s in SPLITS], CRYPTONITE_MAP),
                ("wordplay", [B.WP % s for s in SPLITS], WORDPLAY_MAP)):
            seen = set()
            for path in paths:
                if os.path.exists(path):
                    for row in B.read_jsonl(path):
                        seen.update(row.keys())
            unmapped = seen - set(mapping)
            check(not unmapped, "%s: all %d source fields accounted for%s"
                  % (label, len(seen), " (unmapped: %s)" % sorted(unmapped)
                     if unmapped else ""))

    print("\nuniqueness")
    keys = {}
    for split in SPLITS:
        for row in data[split]:
            keys.setdefault(clue_key(row), []).append(row["id"])
    dupes = {k: v for k, v in keys.items() if len(v) > 1}
    check(not dupes, "clue text is globally unique across train+val+test (%d dupes)"
          % len(dupes))
    for split in SPLITS:
        ids = [r["id"] for r in data[split]]
        check(len(ids) == len(set(ids)), "%s: ids are unique" % split)
        # Stands in for the dropped `split` field. This caught a val row that
        # had been written into train.jsonl.
        check(all(r["id"].startswith(split + "-") for r in data[split]),
              "%s: every id is prefixed with the file's split" % split)

    print("\nno cross-split leakage")
    per_split = {s: {clue_key(r) for r in data[s]} for s in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = per_split[a] & per_split[b]
        check(not shared, "%s n %s = 0 shared clues (%d)" % (a, b, len(shared)))

    print("\nfield integrity")
    for split in SPLITS:
        rows = data[split]
        check(all(r["clue"].strip() for r in rows), "%s: no empty clue" % split)
        check(all(r["answer"].strip() for r in rows), "%s: no empty answer" % split)
        check(all(r["clue"] == r["clue"].lower() for r in rows),
              "%s: clue is lowercased (Cryptonite style)" % split)
        check(all(r["answer"] == r["answer"].lower() for r in rows),
              "%s: answer is lowercased" % split)
        check(all(not re.search(r"\{[^}]*\}", r["clue"]) for r in rows),
              "%s: no {} definition span leaked into clue" % split)
        check(all(r["clue"].rstrip().endswith(r["enumeration"]) for r in rows),
              "%s: clue ends with its enumeration" % split)
        check(all("-" not in r["enumeration"] for r in rows),
              "%s: enumeration uses Cryptonite comma style" % split)
        check(all(r["enumeration_raw"] != r["enumeration"] for r in rows),
              "%s: enumeration_raw is null unless it differs" % split)
        check(all(r["wordplay"] is None or r["clue_with_definition"] for r in rows),
              "%s: every wordplay row keeps its {} clue" % split)
        check(all(("wordplay" in r["sources"]) == (r["wordplay"] is not None)
                  for r in rows),
              "%s: sources agree with presence of a wordplay breakdown" % split)

    print("\nenumeration vs answer length")
    for split in SPLITS:
        mismatched = 0
        for row in data[split]:
            expected = [int(n) for n in re.findall(r"\d+", row["enumeration"])]
            actual = [len(w) for w in re.findall(r"[a-z0-9]+", row["answer"])]
            if expected and sum(expected) != sum(actual):
                mismatched += 1
        pct = 100.0 * mismatched / len(data[split])
        check(pct < 1.0, "%s: answer length matches enumeration (%d off, %.2f%%)"
              % (split, mismatched, pct))

    print("\nrow accounting vs the source files")
    if not B.sources_available():
        print("  skip   source files not present")
    else:
        consumed = 0
        for split in SPLITS:
            consumed += sum(1 for _ in B.read_jsonl(B.CN % split))
            if os.path.exists(B.WP % split):
                consumed += sum(1 for _ in B.read_jsonl(B.WP % split))
        kept = sum(len(data[s]) for s in SPLITS)
        dropped = sum(1 for _ in B.read_jsonl(
            os.path.join(B.REPORTS, "dropped_duplicate_rows.jsonl")))
        check(kept + dropped == consumed,
              "kept %d + dropped %d == %d source rows" % (kept, dropped, consumed))

        wp_unique = set()
        for split in SPLITS:
            if os.path.exists(B.WP % split):
                for row in B.read_jsonl(B.WP % split):
                    wp_unique.add(B.dedup_key(row["clue"], strip_braces=True))
        annotated = count(data, lambda r: r["wordplay"])
        check(annotated == len(wp_unique),
              "all %d unique wordplay clues survived (found %d)"
              % (len(wp_unique), annotated))

    print("\nexpected counts (independent of the source files)")
    for split, n in (("train", 474950), ("val", 26387), ("test", 26082)):
        check(len(data[split]) == n,
              "%s has %d rows (expected %d)" % (split, len(data[split]), n))
    for label, n, pred in (
            ("wordplay breakdowns", 5703, lambda r: r["wordplay"]),
            ("annotator comments", 107, lambda r: r["comment"]),
            ("hyphenated enumerations", 126, lambda r: r["enumeration_raw"]),
            ("clues with alt answers", 51, lambda r: r["alt_answers"]),
            ("clues from both sources", 160, lambda r: len(r["sources"]) > 1)):
        found = count(data, pred)
        check(found == n, "%s: %d (expected %d)" % (label, found, n))
    check(all(r["comment"] is None for s in SPLITS for r in data[s]
              if not r["wordplay"]),
          "comment only appears on wordplay-annotated rows")

    print("\ngit hosting limits")
    names = sorted(os.listdir(B.HERE)) + [
        os.path.join("reports", f) for f in sorted(os.listdir(B.REPORTS))]
    for name in names:
        path = os.path.join(B.HERE, name)
        if not os.path.isfile(path):
            continue
        mb = os.path.getsize(path) / 1048576.0
        check(mb < 100, "%s is %.1f MB (< 100 MB GitHub hard limit)" % (name, mb))
        if mb >= 50:
            print("  note   %s is %.1f MB - over GitHub's 50 MB warning" % (name, mb))

    print("\n" + ("PASS - all invariants hold" if not failures else
                  "FAILED %d check(s):\n  - %s"
                  % (len(failures), "\n  - ".join(failures))))
    return 1 if failures else 0


def clue_key(row):
    return (B.dedup_key(row["clue"], strip_braces=False)
            or "punct:" + B.clue_words(row["clue"], False))


def count(data, pred):
    return sum(1 for s in SPLITS for r in data[s] if pred(r))


if __name__ == "__main__":
    sys.exit(main())
