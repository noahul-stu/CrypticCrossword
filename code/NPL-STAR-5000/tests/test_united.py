"""united-cryptonite-wordplay-dataset loader, on a hand-built miniature copy.

The rows here use the real 18-field schema, including the awkward parts: the
enumeration arrives with parentheses, the clue already has it appended inline,
the answer is lowercase, and the definition markers live in a separate
`clue_with_definition` field rather than in `clue`.
"""

from __future__ import annotations

import gzip
import json

from cryptic_star.data.united import (
    answer_overlap_report,
    enforce_answer_split,
    load_united_split,
    rationale_is_label_only,
    split_seed_and_pool,
)


def row(**over) -> dict:
    base = {
        "id": "train-000001",
        "clue": "make progress socially in stated region (5)",
        "answer": "climb",
        "enumeration": "(5)",
        "orientation": "across",
        "number": 6,
        "publisher": "Times",
        "sub_publisher": "The Times",
        "date": 971654400000,
        "setter": None,
        "quick": False,
        "wordplay": None,
        "comment": None,
        "clue_with_definition": None,
        "enumeration_raw": None,
        "sources": ["cryptonite"],
        "alt_answers": [],
    }
    return {**base, **over}


def write_split(tmp_path, split: str, rows: list[dict], gzipped: bool = True):
    path = tmp_path / (f"{split}.jsonl.gz" if gzipped else f"{split}.jsonl")
    text = "".join(json.dumps(r) + "\n" for r in rows)
    if gzipped:
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text, encoding="utf-8")
    return path


def test_reads_gzipped_split_and_normalises_the_row(tmp_path):
    write_split(tmp_path, "train", [row()])
    (clue,) = load_united_split(tmp_path, "train")

    assert clue.clue == "make progress socially in stated region"  # enum stripped off
    assert clue.answer == "CLIMB"
    assert clue.enumeration == "5"  # parentheses removed
    assert clue.letters == "CLIMB"
    assert clue.clue_id == "train-000001"
    assert clue.split == "train"
    assert not clue.has_rationale


def test_plain_jsonl_also_works(tmp_path):
    write_split(tmp_path, "val", [row(id="val-1")], gzipped=False)
    assert len(load_united_split(tmp_path, "val")) == 1


def test_wordplay_row_becomes_a_rationale_with_a_definition(tmp_path):
    write_split(
        tmp_path,
        "train",
        [
            row(
                id="train-006570",
                clue="call round (4)",
                answer="ring",
                enumeration="(4)",
                wordplay="Double definition: to telephone, and a circle",
                clue_with_definition="{Call} {round}",
                sources=["wordplay", "cryptonite"],
            )
        ],
    )
    (clue,) = load_united_split(tmp_path, "train")
    assert clue.has_rationale
    # Lowercased so it stays a literal substring of the (lowercase) clue.
    assert clue.definition == "call / round"
    assert "telephone" in clue.wordplay


def test_comment_is_merged_into_the_rationale(tmp_path):
    write_split(
        tmp_path,
        "train",
        [row(wordplay="Anagram", comment="letters of 'stated region' rearranged")],
    )
    (clue,) = load_united_split(tmp_path, "train")
    assert clue.wordplay == "Anagram - letters of 'stated region' rearranged"
    # With the comment attached it is no longer a bare device label.
    assert not clue.meta["label_only_rationale"]


def test_bare_device_labels_are_flagged_and_droppable(tmp_path):
    write_split(tmp_path, "train", [row(wordplay="Double Definition")])

    (kept,) = load_united_split(tmp_path, "train")
    assert kept.meta["label_only_rationale"] is True
    assert kept.has_rationale  # kept by default

    (dropped,) = load_united_split(tmp_path, "train", drop_label_only_rationales=True)
    assert not dropped.has_rationale  # clue survives, rationale does not


def test_rationale_is_label_only():
    assert rationale_is_label_only("Double Definition")
    assert rationale_is_label_only("anagram")
    assert rationale_is_label_only("")
    assert not rationale_is_label_only("RING = telephone, hidden in the phrase")


def test_short_rationale_is_discarded_but_the_clue_is_kept(tmp_path):
    write_split(tmp_path, "train", [row(wordplay="dd")])
    (clue,) = load_united_split(tmp_path, "train")
    assert not clue.has_rationale
    assert clue.answer == "CLIMB"


def test_quick_puzzles_are_dropped_by_default(tmp_path):
    write_split(tmp_path, "train", [row(quick=True), row(id="train-2")])
    assert len(load_united_split(tmp_path, "train")) == 1
    assert len(load_united_split(tmp_path, "train", drop_quick=False)) == 2


def test_hyphenated_enumeration_is_preserved(tmp_path):
    write_split(
        tmp_path,
        "train",
        [row(answer="pipe dream", enumeration="(4,5)", enumeration_raw="(4-5)")],
    )
    (clue,) = load_united_split(tmp_path, "train")
    assert clue.enumeration == "4-5"
    assert clue.enum_sig == (4, 5)


def test_alt_answers_are_carried_onto_the_clue(tmp_path):
    write_split(tmp_path, "train", [row(alt_answers=["climbs"])])
    (clue,) = load_united_split(tmp_path, "train")
    assert clue.alt_letters == {"CLIMBS"}


# -- the answer split ------------------------------------------------------

def test_answer_overlap_is_detected(tmp_path):
    write_split(tmp_path, "train", [row(answer="climb"), row(id="t2", answer="eagle")])
    write_split(tmp_path, "val", [row(id="v1", answer="eagle")])
    write_split(tmp_path, "test", [row(id="s1", answer="otter")])
    splits = {s: load_united_split(tmp_path, s) for s in ("train", "val", "test")}

    report = answer_overlap_report(splits)
    assert report["answer_disjoint"] is False
    assert report["overlaps"]["train|val"]["shared_answers"] == 1
    assert report["overlaps"]["train|val"]["train_rows_affected"] == 1
    assert report["overlaps"]["train|test"]["shared_answers"] == 0


def test_enforce_answer_split_shrinks_train_not_test(tmp_path):
    write_split(
        tmp_path,
        "train",
        [
            row(answer="climb"),
            row(id="t2", answer="eagle", wordplay="hidden inside the phrase somewhere"),
            row(id="t3", answer="otter"),
        ],
    )
    write_split(tmp_path, "val", [row(id="v1", answer="eagle")])
    write_split(tmp_path, "test", [row(id="s1", answer="otter")])
    splits = {s: load_united_split(tmp_path, s) for s in ("train", "val", "test")}

    kept, stats = enforce_answer_split(splits)
    assert [c.letters for c in kept["train"]] == ["CLIMB"]
    assert stats["train_rows_dropped"] == 2
    assert stats["train_annotated_rows_dropped"] == 1  # cost of the guard, reported
    assert len(kept["val"]) == 1 and len(kept["test"]) == 1
    assert answer_overlap_report(kept)["answer_disjoint"] is True


def test_val_rows_sharing_a_test_answer_are_dropped(tmp_path):
    write_split(tmp_path, "train", [row()])
    write_split(tmp_path, "val", [row(id="v1", answer="otter")])
    write_split(tmp_path, "test", [row(id="s1", answer="otter")])
    splits = {s: load_united_split(tmp_path, s) for s in ("train", "val", "test")}

    kept, stats = enforce_answer_split(splits)
    assert stats["val_rows_dropped"] == 1
    assert len(kept["test"]) == 1


# -- seed / pool -----------------------------------------------------------

def test_seed_is_the_annotated_rows_and_the_pool_is_everything(tmp_path):
    write_split(
        tmp_path,
        "train",
        [
            row(),
            row(id="t2", answer="ring", wordplay="double definition of call and round"),
        ],
    )
    clues = load_united_split(tmp_path, "train")
    seed, pool = split_seed_and_pool(clues)
    assert [c.letters for c in seed] == ["RING"]
    # Annotated clues stay in the pool too: they can gain self-generated traces.
    assert len(pool) == 2
