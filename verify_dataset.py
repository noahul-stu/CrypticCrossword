#!/usr/bin/env python3
"""Verify the invariants of the united dataset. Exits non-zero on any failure.

    python3 united-cryptonite-wordplay-dataset/verify_dataset.py
"""

import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_united_dataset as B  # noqa: E402

SPLITS = ["train", "val", "test"]
FIELDS = ["id", "split", "clue", "answer", "enumeration", "orientation", "number",
          "publisher", "sub_publisher", "date", "setter", "quick", "wordplay",
          "clue_with_definition", "wordplay_author", "enumeration_raw",
          "sources", "alt_answers", "n_source_rows"]

failures = []


def check(condition, message):
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)


def main():
    data = {}
    for split in SPLITS:
        path = os.path.join(B.HERE, "%s.jsonl" % split)
        data[split] = [json.loads(line) for line in open(path, encoding="utf-8")]

    print("schema")
    for split in SPLITS:
        bad = [r["id"] for r in data[split] if list(r.keys()) != FIELDS]
        check(not bad, "%s: every row has the exact field set, in order" % split)

    print("\nuniqueness")
    keys = {}
    for split in SPLITS:
        for row in data[split]:
            key = B.dedup_key(row["clue"], strip_braces=False) or (
                "punct:" + B.clue_words(row["clue"], False))
            keys.setdefault(key, []).append(row["id"])
    dupes = {k: v for k, v in keys.items() if len(v) > 1}
    check(not dupes, "clue text is globally unique across train+val+test (%d dupes)"
          % len(dupes))
    for split in SPLITS:
        ids = [r["id"] for r in data[split]]
        check(len(ids) == len(set(ids)), "%s: ids are unique" % split)
        check(all(r["split"] == split for r in data[split]),
              "%s: split field matches the file" % split)

    print("\nno cross-split leakage")
    per_split = {s: {B.dedup_key(r["clue"], False) or ("punct:" + B.clue_words(r["clue"], False))
                     for r in data[s]} for s in SPLITS}
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
        check(all(r["wordplay"] is None or r["clue_with_definition"] for r in rows),
              "%s: every wordplay row keeps its {} clue" % split)
        check(all(("wordplay" in r["sources"]) == (r["wordplay"] is not None)
                  for r in rows),
              "%s: sources agree with presence of a wordplay breakdown" % split)

    print("\nenumeration vs answer length")
    for split in SPLITS:
        mismatched = 0
        for row in rows_of(data, split):
            expected = [int(n) for n in re.findall(r"\d+", row["enumeration"])]
            actual = [len(w) for w in re.findall(r"[a-z0-9]+", row["answer"])]
            if expected and sum(expected) != sum(actual):
                mismatched += 1
        pct = 100.0 * mismatched / len(data[split])
        check(pct < 1.0, "%s: answer length matches enumeration (%d off, %.2f%%)"
              % (split, mismatched, pct))

    print("\nsource accounting vs the original files")
    consumed = 0
    for split in SPLITS:
        consumed += sum(1 for _ in B.read_jsonl(B.CN % split))
        path = B.WP % split
        if os.path.exists(path):
            consumed += sum(1 for _ in B.read_jsonl(path))
    kept = sum(len(data[s]) for s in SPLITS)
    dropped = sum(1 for _ in B.read_jsonl(
        os.path.join(B.REPORTS, "dropped_duplicate_rows.jsonl")))
    check(kept + dropped == consumed,
          "kept %d + dropped %d == %d source rows" % (kept, dropped, consumed))

    annotated = sum(1 for s in SPLITS for r in data[s] if r["wordplay"])
    wp_unique = set()
    for split in SPLITS:
        path = B.WP % split
        if os.path.exists(path):
            for row in B.read_jsonl(path):
                wp_unique.add(B.dedup_key(row["clue"], strip_braces=True))
    check(annotated == len(wp_unique),
          "all %d unique wordplay clues survived (found %d)" % (len(wp_unique), annotated))

    print("\n" + ("PASS - all invariants hold" if not failures
                  else "FAILED %d check(s):\n  - %s" % (len(failures), "\n  - ".join(failures))))
    return 1 if failures else 0


def rows_of(data, split):
    return data[split]


if __name__ == "__main__":
    sys.exit(main())
