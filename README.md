# United Cryptonite + Wordplay Dataset

A single cryptic-crossword dataset merging **Cryptonite** (`cryptonite-official-split/`)
and the **Wordplay** sample (`cryptic-wordplay-main/prebuilt/`) into one schema,
deduplicated to unique clue texts, split into train / val / test.

Built by `build_united_dataset.py`; every invariant below is checked by
`verify_dataset.py` (currently **PASS**).

## Contents

| File | Rows | Size | What it is |
|---|---|---|---|
| `train.jsonl` | 474,950 | 205 MB | Training clues |
| `val.jsonl` | 26,387 | 11 MB | Validation clues |
| `test.jsonl` | 26,082 | 11 MB | Test clues (Cryptonite-only, no wordplay annotations) |
| `stats.json` | — | — | Machine-readable counts per split |
| `build_united_dataset.py` | — | — | Reproducible build; rerun to regenerate everything |
| `verify_dataset.py` | — | — | Asserts all invariants; exits non-zero on failure |
| `reports/*.jsonl` | 1,691 | — | Full audit trail of everything dropped or in conflict |

Regenerate from the two source folders (run from their shared parent directory):

```bash
python3 united-cryptonite-wordplay-dataset/build_united_dataset.py
python3 united-cryptonite-wordplay-dataset/verify_dataset.py
```

## Composition

| | train | val | test |
|---|---|---|---|
| Unique clues | **474,950** | **26,387** | **26,082** |
| From Cryptonite only | 469,547 | 26,087 | 26,082 |
| From Wordplay only | 5,253 | 290 | 0 |
| Present in **both** sources | 150 | 10 | 0 |
| With a wordplay breakdown | 5,403 | 300 | 0 |
| Quick-crossword clues | 37,256 | 2,111 | 2,069 |

Publishers — train: Times 277,939 · Telegraph 191,608 · FT 5,224 · Guardian 179.
Cryptonite supplies Times/Telegraph; the Wordplay sample adds FT and Guardian.

Row accounting is exact: **528,829** source rows in → **527,419** kept +
**1,410** dropped as duplicates.

## Schema

One JSON object per line, fields always present and in this order:

| Field | Type | Notes |
|---|---|---|
| `id` | str | `"train-000001"`, unique within the file |
| `split` | str | `train` / `val` / `test` |
| `clue` | str | **Cryptonite style**: lowercased, enumeration appended inline, no `{}` markers |
| `answer` | str | Lowercased; may contain spaces (`"running buffet"`) |
| `enumeration` | str | `"(4,2)"` — Cryptonite style, commas only |
| `orientation` | str/null | `across` / `down` |
| `number` | int/null | Clue number in the grid |
| `publisher` | str/null | `Times`, `Telegraph`, `FT`, `Guardian` |
| `sub_publisher` | str/null | e.g. `The Sunday Times`, `Guardian Prize` |
| `date` | int/null | Epoch ms; **always null for Wordplay-only rows** (that source has no dates) |
| `setter` | str/null | Puzzle compiler |
| `quick` | bool | Quick-crossword variant |
| `wordplay` | str/null | The human breakdown, e.g. `"DA (lawyer) + T[ribunal] in SEE (diocese)"` |
| `clue_with_definition` | str/null | Original-case clue with definition spans in `{}` |
| `wordplay_author` | str/null | Annotator (`teacow` throughout this sample) |
| `enumeration_raw` | str | As-published; keeps hyphens (`"(4-2)"`) that `enumeration` folds to commas |
| `sources` | list | `["cryptonite"]`, `["wordplay"]`, or both |
| `alt_answers` | list | Other answers published for this same clue text (see conflicts below) |
| `n_source_rows` | int | How many source rows collapsed into this record |

Get the reasoning-annotated subset with a filter — no separate file needed:

```python
annotated = [r for r in rows if r["wordplay"]]     # 5,703 clues across train+val
```

## Build decisions

1. **Full union**, one schema. Wordplay-specific fields are `null` on rows that
   have no annotation.
2. **`clue` is Cryptonite style** — lowercased, enumeration inline, `{}` stripped.
   Wordplay clues are converted *to* this style, and their enumeration is
   rebuilt from the `pattern` field.
   *Deviation from a strict reading of that choice:* Wordplay's original-case
   clue with its `{}` definition spans is preserved in `clue_with_definition`
   rather than discarded. Nothing forces you to use it, and dropping it would
   have destroyed the annotation irreversibly.
3. **Dedup by normalized clue text, globally.** The key is the clue's
   alphanumeric characters, lowercased, with the enumeration excluded — so
   `"carpet burn"` and `"carpet burn?"` collapse, as do `"hardwood"` and
   `"hard wood"`.
4. **Split precedence train > val > test.** A clue text appearing in two splits
   is kept in the earlier one and deleted from the later. This makes the dataset
   globally leak-free but means val/test differ slightly from official
   Cryptonite — see the caveat below.
5. **Which duplicate survives:** a row with a wordplay breakdown wins (its
   annotation justifies the answer), then the earliest-dated row, then first
   seen. Same-split duplicates with the same answer donate their missing
   metadata to the survivor — this is how 160 clues carry both sources' fields.
6. **Nothing is silently dropped.** All 1,410 removed rows are in
   `reports/dropped_duplicate_rows.jsonl` with a reason.

## Caveats

**Val and test are 3 and 7 clues smaller than official Cryptonite.** Cryptonite's
own splits share 11 clue texts across split boundaries; per the global-dedup
choice these were kept in train and removed from val/test
(`reports/cross_split_duplicates.jsonl`). If you need numbers directly
comparable to the published Cryptonite paper, use the original files.

**51 clues had a genuine answer conflict.** Different setters reused the same
surface text for different answers — `"flower girl"` is DAISY in one puzzle and
VIOLET in another; `"fast food?"` is DIET and also RUNNING BUFFET. Clue-text
dedup keeps one and records the rest in `alt_answers` plus
`reports/answer_conflicts.jsonl`. These are not annotation errors; they are
distinct riddles that happen to share a surface reading. **If your task needs
every valid clue→answer pair, this dataset is the wrong shape** — rebuild with a
`(clue, answer)` dedup key instead.

**59 clues had conflicting enumerations** (`"(4)"` vs `"(7,6)"` for the same
words) — same cause, recorded in `reports/enumeration_conflicts.jsonl`.

**`date` is null for all 5,703 Wordplay-annotated rows**, so don't filter or
sort the whole dataset by date without handling nulls.

**The annotated subset is one annotator, one style.** All 5,703 breakdowns are by
`teacow`, mostly Financial Times puzzles. The `wordplay` field is free-form text
with no fixed grammar — treat its format as a per-annotator convention, not a
schema.

**No wordplay annotations exist in test at all.** The Wordplay repo deliberately
ships no test split and asks that one not be created, so a wordplay-annotated
test set is not available. Evaluate explanation quality on `val`.

**Source-data artifacts fixed during the build**, all isolated single cases:
one Cryptonite clue encoded a comma as `{cm}`; one Wordplay `pattern` field had
a leaked `{8}` definition marker; one Cryptonite clue is literally `"? (8)"` →
`clueless`, which normalizes to an empty key and needed special handling to
survive. One source typo is left as-is: `"supply drug (crack} (5)"`.

## How the two sources overlapped

Before merging, 153 train and 10 val clue texts existed in both sources. None
were byte-identical, because Cryptonite lowercases every clue and appends the
enumeration while Wordplay preserves case and marks definitions. After
normalizing those two systematic differences, of the 153 train matches: 132 were
identical apart from letter case, 21 differed only in punctuation (a trailing
`?`, curly vs straight apostrophes), and exactly 1 differed in words
(`hardwood` / `hard wood`). Answers agreed on 150 of 153; the 3 exceptions are
the answer conflicts described above. Val: 7 case-only, 3 punctuation, 10/10
answers agreeing. Per-clue classifications are in
`reports/source_overlap.jsonl`.

## Reports

| File | Rows | Contents |
|---|---|---|
| `dropped_duplicate_rows.jsonl` | 1,410 | Every removed row, with the kept clue/answer and a reason |
| `source_overlap.jsonl` | 160 | The clues found in both sources, with an exactness verdict |
| `answer_conflicts.jsonl` | 51 | Same clue text, different published answers |
| `enumeration_conflicts.jsonl` | 59 | Same clue text, different enumerations |
| `cross_split_duplicates.jsonl` | 11 | Clues removed from val/test because train had them |

## Provenance and citation

Derived from two independently licensed sources; cite both.

```latex
@inproceedings{efrat2021cryptonite,
  title={Cryptonite: A Cryptic Crossword Benchmark for Extreme Ambiguity in Language},
  author={Efrat, Avia and Shaham, Uri and Kilman, Dan and Levy, Omer},
  booktitle={EMNLP}, year={2021}
}
@software{Wordplay_dataset_repo,
  author={Andrews, Martin}, title={{Wordplay Dataset}},
  url={https://github.com/mdda/cryptic-wordplay}, version={0.0.1}, year={2024}
}
```

Note the Wordplay repo's request that no test split be created from its scrapers
and that test data not be published, to avoid contaminating future LLM training
sets.
