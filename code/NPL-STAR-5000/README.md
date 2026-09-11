# cryptic-star — STaR generator for cryptic crosswords

The **generator** half of the Best-of-N project. It fine-tunes T5 to output a
*reasoning trace* plus an answer for a cryptic clue, using STaR
(Self-Taught Reasoner): the model generates its own reasoning, keeps only the
traces that land on the gold answer, retrains on them, and repeats.

Two data sources, playing different roles:

| Dataset | Role |
| --- | --- |
| **Cryptonite** | scale — ~470k clue→answer pairs, no reasoning. The STaR training pool and the evaluation set (official answer-disjoint split). |
| **cryptic-wordplay** | reasoning — human wordplay analyses plus `{}`-marked definition spans, for the clues it covers. Used as the **seed set** that warm-starts the model. |

They arrive **already joined** as `united-cryptonite-wordplay-dataset`
(`data.dataset: united`, the default): one file per split, with a `wordplay`
field that is non-null on the rows a human annotated. `data.dataset: separate`
still reads the two raw datasets and joins them itself via `data/align.py`.

Without the seed set, STaR has nothing to bootstrap from: plain T5-Large solves
7.64% of Cryptonite (Efrat et al., 2021), so nearly every sampled trace is
rejected and the training set barely grows. The human rationales give the model
the *format and style* of cryptic reasoning first; self-generation then scales it.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # or just export PYTHONPATH=src
```

The unit tests need no dataset and no GPU:

```bash
pytest -q                 # 115 tests
```

## Adding the data

Nothing downloads automatically.

### `data.dataset: united` (default)

Put the dataset directory here — gzip stays as-is, the loader reads `.gz`:

```
data/raw/united-cryptonite-wordplay-dataset/
    train.jsonl.gz        474,950 clues,  5,403 with a wordplay annotation
    val.jsonl.gz           26,387 clues,    300 with a wordplay annotation
    test.jsonl.gz          26,082 clues,      0 with a wordplay annotation
    stats.json
```

From the `CrypticCrossword` repo/zip:

```bash
scripts/install_united_dataset.sh ~/Downloads/CrypticCrossword-main.zip
```

Then build the processed splits and **read the report before training anything**:

```bash
python -m cryptic_star.data.united -c configs/base.yaml
cat data/processed/united_report.json
```

The report answers three questions that decide whether the run is valid:

- `before.answer_disjoint` — were the shipped splits answer-disjoint? The
  dataset's own `verify_dataset.py` does not check this, and 5,253 wordplay-only
  rows were added to train from an independently scraped source.
- `enforced.train_rows_dropped` / `train_annotated_rows_dropped` — how much
  train (and how much precious annotated train) the guard had to remove.
- `seed_rows`, `seed_label_only` — how large the warm-start set really is, and
  how much of it is a bare device name (`"Double Definition"`) rather than a
  decomposition.

`test_annotated` is expected to be **0**: the wordplay repo asks that no test
split be built from it, so explanation quality can only be measured on **val**.
Answer accuracy is still measured on test.

### `data.dataset: separate`

```
data/raw/cryptonite/cryptonite-official-split/
    cryptonite-train.jsonl
    cryptonite-val.jsonl
    cryptonite-test.jsonl
data/raw/wordplay/wordplay.jsonl
```

Filenames are globbed (`*train*.jsonl`), so minor naming differences are fine; a
directory of wordplay shards works too. Use the **official** Cryptonite split —
it is answer-disjoint, which is what the proposal evaluates on and what makes
the leakage check possible.

Expected fields:

- Cryptonite: `clue`, `answer`, `enumeration`, `publisher`, `date`, `quick`, `id`
- cryptic-wordplay: `clue` (definitions in `{}`), `pattern`, `ad`, `answer`,
  `wordplay`, `author`, `setter`, `publication`, `is_quick`

Then build the aligned seed set and inspect the report before training anything:

```bash
python -m cryptic_star.data.align -c configs/base.yaml
cat data/processed/align_report.json
```

Either way the outputs are the same four files, so nothing downstream changes:

```
data/processed/seed_train.jsonl        annotated rows -> STaR warm start
data/processed/cryptonite_train.jsonl  the self-generation pool
data/processed/cryptonite_val.jsonl
data/processed/cryptonite_test.jsonl
```

## Running

```bash
# 1. smoke test — a few minutes on a laptop, proves the plumbing works
python -m cryptic_star.star_loop -c configs/base.yaml -c configs/smoke.yaml

# 2. real run
python -m cryptic_star.star_loop -c configs/base.yaml -c configs/star_t5_base.yaml
```

Any config value can be overridden inline:

```bash
python -m cryptic_star.star_loop -c configs/base.yaml -o generate.n_samples=16 -o star.iterations=5
```

Individual stages are runnable on their own — useful when a long run dies
halfway:

```bash
python -m cryptic_star.train      --train-file runs/x/iter1/train.jsonl --out runs/x/iter1/model
python -m cryptic_star.generate   --model runs/x/iter1/model --clues data/processed/cryptonite_train.jsonl --out cands.jsonl
python -m cryptic_star.evaluate   --model runs/x/iter1/model --out runs/x/eval_test
```

## On the cluster (Slurm)

Submit through the wrapper. Do **not** call `sbatch scripts/slurm/star.sbatch`
directly:

```bash
scripts/slurm/submit.sh configs/star_t5_large.yaml
```

`star.sbatch` declares `--output=logs/star-%j.out`, and Slurm opens that file
*before* the job script runs. If `logs/` does not exist the job is killed
instantly with `ExitCode 1:0` and writes no `.out` or `.err` at all — which
looks exactly like a silent crash and is impossible to debug from the outside.
`submit.sh` creates the directory on the submit host, which is the only place it
can be done in time.

**First-time setup, on the login node.** Two things must happen there and cannot
happen inside a job:

```bash
# 1. the environment. Conda is picked up automatically from ~/anaconda3,
#    ~/miniconda3, or $CONDA_BASE; set CONDA_ENV if it is not `nlp_env`.
#    Without conda, star.sbatch falls back to a virtualenv at $STORAGE_ROOT/venv
#    — build it here, because the compute nodes' python3.12 has no ensurepip and
#    `python3 -m venv` fails on them:
#      python3 -m venv --without-pip "$VENV" && source "$VENV/bin/activate"
#      curl -sS https://bootstrap.pypa.io/get-pip.py | python
pip install -r requirements.txt

# 2. the model weights (~3GB for flan-t5-large). Compute nodes may have no route
#    to huggingface.co, and a preempted job would re-download every time.
python -m cryptic_star.prefetch -c configs/base.yaml -c configs/star_t5_large.yaml
```

`star.sbatch` keeps every cache off `$HOME` — `HF_HOME`, `XDG_CACHE_HOME`,
`PIP_CACHE_DIR`, `MPLCONFIGDIR`, `TRITON_CACHE_DIR` all land under
`$STORAGE_ROOT` (default `/home/morg/NLP_2526b/$USER`, export it to move them).
The CS home directories have a quota too small for a 3GB model cache, and
overflowing it surfaces as `OSError: [Errno 122] Disk quota exceeded` from
whatever happened to be writing, tens of minutes into the run.

**Resuming after a preemption.** `studentkillable` is preemptible with a one-day
cap, so a multi-iteration flan-t5-large run *will* be interrupted. The job asks
Slurm to `--requeue` it, so it goes back in the queue by itself; `--open-mode=append`
keeps the killed attempt's log instead of truncating it, since a requeue reuses
the job id. Re-submitting the identical command by hand works the same way:
iterations with a complete `iterK/model/` and a `history.json` record are skipped,
an interrupted iteration reuses its `candidates.jsonl` instead of re-decoding it,
and a finished test evaluation is not repeated.

**If training OOMs.** Change the optimizer before the batch size — see the
commented `adafactor` block in `configs/star_t5_large.yaml`. AdamW holds two fp32
moments per parameter, about 6GB of state for flan-t5-large's 780M.

**Watching a run.** All three are written as the run progresses, so a job killed
at hour 23 still leaves everything behind:

```bash
tail -f logs/star-<jobid>.out         # the job's stdout
tail -f runs/<name>/metrics.jsonl     # one JSON event per line
less runs/<name>/iter1/trace_samples.md   # what the model actually reasoned
tensorboard --logdir runs/<name>/tb   # loss curves, EM per iteration
```

See `tracking.py`; `tracking.enabled: false` turns all of it off. There is
deliberately no Weights & Biases — compute nodes have no outbound network, so a
tracker that phones home would stall the job or have to be disabled exactly when
it is most useful.

## What the loop does

```
iteration 0   warm start: fine-tune base T5 on the human wordplay rationales
              (seed_train.jsonl)

iteration k   a. sample N traces per clue for a subsample of Cryptonite train
              b. keep traces reaching the gold answer with real reasoning
              c. RATIONALISE the failures: re-prompt with the answer as a hint,
                 keep what works, then STRIP the hint
              d. training set = seed + every accepted trace so far
              e. retrain from the ORIGINAL base checkpoint
              f. evaluate EM on val; export labelled candidates for DeBERTa
```

Two details that are easy to get wrong and are deliberate here:

**Retraining from base each iteration** (`star.retrain_from_base`) rather than
continuing from the last checkpoint. This is what the STaR paper does; continued
training compounds the model's own errors across iterations.

**Rationalisation** (`star.rationalize`) is what makes the loop move on a task
this hard. Plain STaR discards every failed clue; here the failures are
re-attempted with the answer visible, and the successful explanations are added
with the hint removed — so the model learns reasoning it could not yet produce
unaided. `rationalize.assert_no_hint_leak` re-checks every training file, because
a leaked hint would silently invalidate the entire experiment.

## Prompt format

```
source:  solve the cryptic clue. clue: Bird of prey caught out ; enumeration: 5 ; letters: 5
target:  definition: Bird of prey ; wordplay: EAGLE hidden in the surface ; answer: E A G L E
```

- The `{}` markers from cryptic-wordplay become the `definition:` field of the
  **target**, never the input — otherwise the model is handed half the answer.
- `{ } [ ] < > |` are absent from T5's SentencePiece vocabulary and are silently
  dropped by the tokenizer, so `text.sanitize_for_t5` maps them to safe
  characters everywhere. `;` is the field separator and is stripped from values.
- `format.spaced_answer` emits `E A G L E` instead of `EAGLE`, forcing one token
  per letter. T5 tokenises `EAGLE` as a couple of subwords, which makes
  letter-level wordplay (anagrams, hidden words, initials) hard to express. Worth
  an ablation in the report.

## Outputs

```
runs/<name>/
├── config.json                     exact config used
├── history.json                    per-iteration stats — the main results table
├── metrics.jsonl                   live event stream, one JSON object per line
├── tb/                             TensorBoard event files
├── test_metrics.json               final EM / pass@N
├── test_predictions.jsonl
├── analysis/analysis.md            error breakdown + learning curve
└── iterK/
    ├── candidates.jsonl            every sampled trace + label + reject reason
    ├── candidates_hinted.jsonl     rationalisation pass
    ├── trace_samples.md            solved and failed reasoning, for reading
    ├── train.jsonl                 what this iteration trained on
    ├── model/                      the fine-tuned generator
    ├── val_metrics.json
    └── discriminator_data.jsonl    → the DeBERTa cross-encoder
```

`data/processed/prep_fingerprint.json` records the `data.*` settings that built
the processed files. `ensure_processed` compares against it and rebuilds on a
mismatch, so a smoke run's `limit_train: 40000` cache cannot silently be reused
by a full run — which would produce a plausible, quietly wrong result.

Reported metrics:

- `top1_em` — beam-search Exact Match. The headline number; compare to 7.64%.
- `pass_at_n` — any of N samples correct. The ceiling a perfect re-ranker could
  reach, so it bounds the Best-of-N result.
- `oracle_gap` — `pass_at_n − top1_em`, i.e. the head-room re-ranking has. If
  this is near zero, Best-of-N cannot help and that is the finding.

## Hand-off to the discriminator

`discriminator_data.jsonl` rows are ready for a cross-encoder:

```json
{"text_a": "Bird of prey caught out (5)", "text_b": "definition: ... ; answer: E A G L E",
 "label": 1, "reason": "ok", "hinted": false, "predicted": "EAGLE", "gold": "EAGLE"}
```

`strict_negatives` (default on) keeps only traces that committed to a *wrong
answer* as negatives, and drops unparseable or degenerate output — otherwise the
discriminator learns to spot malformed text instead of bad reasoning. Hinted
positives are flagged, since they were produced with the answer in view and are
easier than anything seen at inference time.

## Leakage guard

Cryptonite's official split is answer-disjoint. cryptic-wordplay is scraped
independently and covers the same newspapers, so **some of its clues have
answers that live in Cryptonite's val/test**. Training on those would inflate
every number in the report.

The guard runs in whichever loader produced the data, and nothing downstream
re-checks it — so read the number and quote it in the write-up:

| dataset | where | what to read |
| --- | --- | --- |
| `united` | `data/united.py` → `enforce_answer_split` | `united_report.json`: `before.answer_disjoint`, `enforced.train_rows_dropped` |
| `separate` | `data/align.py` → `build_seed` | `align_report.json`: `dropped_leakage` |

For `united` the rows are removed from the *earlier* split of each pair (train
before val, val before test) so that test — the split the headline EM is
reported on — is never shrunk and stays comparable with the published baseline.
Set `data.enforce_answer_split: false` only to measure how much difference the
guard makes; never for a reported number.

**Alternative answers.** The united dataset records `alt_answers` where two
sources gave different answers for the same clue text. Scoring those as correct
would make our EM non-comparable with Cryptonite's 7.64%, so
`verify.accept_alt_answers` is off: they are rejected with their own reason
(`alt_answer`) and surface as `alt_answer_rate` in the metrics, which tells you
how much of the error is arguable rather than wrong.

## Known constraints

- **Compute.** N=8 samples over 470k train clues is millions of generations.
  `data.train_subsample` caps each iteration (default 5k clues) and draws a
  *different* subsample per iteration, so coverage grows across the run. The
  logs state how many clues were skipped — do not report results as full-train.
- **T5 and letters.** Subword tokenisation fights letter-level wordplay; the
  spaced-answer option mitigates it but does not remove the problem.
- **Seed set size.** Only ~5.4k of 475k train clues carry a human rationale
  (~1.1%), and the leakage guard shrinks that further. That is the whole warm
  start, which is why `star.rationalize` matters so much: it is the only way the
  training set grows on clues the model cannot yet solve unaided.
- **Rationale quality is uneven.** All the wordplay comes from one annotator
  (`teacow`), mostly Financial Times puzzles, in free form with no fixed grammar
  — and some entries only name the device rather than decomposing the clue
  (`data.drop_label_only_rationales` filters those; `seed_label_only` counts
  them). The seed set is therefore stylistically narrow, and the generator will
  imitate that style.
- **No wordplay in test.** Explanation quality can only be evaluated against
  human annotations on **val** (~300 clues). Test gives answer accuracy only.

## Layout

```
src/cryptic_star/
├── config.py                       YAML → typed config, CLI overrides
├── text.py                         normalisation, T5-safe characters
├── io_utils.py                     jsonl, logging, seeding, device pick
├── tracking.py                     metrics.jsonl, TensorBoard, trace dumps
├── data/
│   ├── cryptonite.py               Cryptonite loader, Clue dataclass
│   ├── wordplay.py                 cryptic-wordplay loader
│   ├── align.py                    the join + LEAKAGE GUARD → seed set
│   ├── united.py                   pre-joined dataset + ANSWER-SPLIT GUARD
│   └── formatting.py               prompt/target templates
├── train.py                        T5 fine-tuning (+ optional LoRA/QLoRA)
├── generate.py                     sample N traces per clue
├── verify.py                       trace → label 1/0 + reject reason
├── rationalize.py                  hint pass on failures, hint-leak guard
├── evaluate.py                     EM, pass@N, oracle gap
├── prefetch.py                     warm the HF cache on the login node
├── export_discriminator_data.py    → DeBERTa
└── star_loop.py                    orchestrator (+ resume, data-prep cache)
```

```
scripts/slurm/submit.sh             submit this, not sbatch directly
scripts/slurm/star.sbatch           the job: conda, env pre-check, star_loop
```
