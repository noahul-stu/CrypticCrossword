"""Alignment + leakage tests, on hand-written rows (no dataset required).

The leakage test is the important one in this file: if it ever fails, every EM
number the project reports is invalid.
"""

from cryptic_star.data.align import build_seed
from cryptic_star.data.cryptonite import Clue


def cn(clue, answer, enumeration, split, cid):
    return Clue(clue=clue, answer=answer, enumeration=enumeration, split=split, clue_id=cid)


def wp(clue, answer, enumeration, wordplay, definition=""):
    return Clue(
        clue=clue,
        answer=answer,
        enumeration=enumeration,
        source="wordplay",
        wordplay=wordplay,
        definition=definition,
        clue_id="wp-x",
    )


CRYPTONITE = {
    "train": [
        cn("Bird of prey caught out", "EAGLE", "5", "train", "t1"),
        cn("Sleepy hollow contains a bed", "COT", "3", "train", "t2"),
    ],
    "val": [cn("Some val clue", "OTTER", "5", "val", "v1")],
    "test": [cn("Some test clue", "RAVEN", "5", "test", "s1")],
}


def test_exact_match_on_clue_and_answer():
    seed, stats = build_seed(CRYPTONITE, [wp("Bird of prey caught out", "EAGLE", "5", "hidden word")])
    assert stats["matched_exact"] == 1
    assert seed[0].split == "train"
    assert seed[0].wordplay == "hidden word"
    assert seed[0].meta["matched_cryptonite"] is True


def test_fuzzy_match_survives_punctuation_and_markers():
    rows = [wp("{Bird of prey}, caught out! (5)", "eagle", "5", "EAGLE is hidden")]
    seed, stats = build_seed(CRYPTONITE, rows)
    # norm_clue removes punctuation so this is still an exact key match.
    assert stats["matched_exact"] + stats["matched_fuzzy"] == 1
    assert len(seed) == 1


def test_wordplay_row_with_test_answer_is_dropped():
    rows = [wp("Some other clue for a bird", "RAVEN", "5", "anagram of NAVER")]
    seed, stats = build_seed(CRYPTONITE, rows)
    assert stats["dropped_leakage"] == 1
    assert seed == []


def test_wordplay_row_with_val_answer_is_dropped():
    rows = [wp("A river creature", "OTTER", "5", "hidden in the clue")]
    seed, stats = build_seed(CRYPTONITE, rows)
    assert stats["dropped_leakage"] == 1
    assert seed == []


def test_unmatched_but_safe_row_is_kept_as_extra_supervision():
    rows = [wp("Totally unseen clue", "ZEBRA", "5", "anagram of BRAZE")]
    seed, stats = build_seed(CRYPTONITE, rows, use_unmatched=True)
    assert stats["unmatched_kept"] == 1
    assert len(seed) == 1 and seed[0].meta["matched_cryptonite"] is False


def test_unmatched_row_can_be_excluded():
    rows = [wp("Totally unseen clue", "ZEBRA", "5", "anagram of BRAZE")]
    seed, stats = build_seed(CRYPTONITE, rows, use_unmatched=False)
    assert seed == [] and stats["unmatched_dropped"] == 1


def test_duplicate_analyses_of_the_same_clue_collapse():
    rows = [
        wp("Bird of prey caught out", "EAGLE", "5", "first author's analysis"),
        wp("Bird of prey caught out", "EAGLE", "5", "second author's analysis"),
    ]
    seed, _ = build_seed(CRYPTONITE, rows)
    assert len(seed) == 1
