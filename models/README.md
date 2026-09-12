# Model checkpoints

Nothing in this directory is committed — checkpoints are hundreds of MB to
several GB, well past GitHub's 100 MB per-file limit, and the repo has already
been made unpushable once by a large committed file. Only this README is tracked.

## Current state

| Model | Where it is | Status |
|---|---|---|
| Reasoning scorer (stage 2) | `code/deberta/with-finetuning/trained_deberta/` | ✅ **delivered 2026-09-12**, verified below |
| Generator (stage 1) | `hugTAU/cryptonite-flant5-large` (Hub) | ✅ available; cache it locally to save a ~3GB download per job |

### What the delivered scorer actually is

Read from the safetensors header, not just `config.json`:

| | |
|---|---|
| Architecture | **deberta-v3-`large`** — 435.1M params, 24 layers, hidden 1024 |
| Head | `classifier.weight [2, 1024]`, `id2label = {0: INCORRECT, 1: CORRECT}` |
| Precision | fp32 |
| Tokenizer | fast `tokenizer.json` present — no sentencepiece→fast conversion needed |
| Provenance | `training_args.bin` present; transformers 5.15.1 |

It is `large`, not the `small` the repo's fine-tuning notebook trains. That is why
`config/t5large_deberta_large.json` sets `batch_size: 16` (down from 32) and pins
`dtype: float32` — see the comments in that file.

**One thing is unverified:** nothing in this repo records the input template this
checkpoint was fine-tuned with. The repo's with-finetuning notebook is the *small*
model's. The config assumes the same layout,
`CLUE: {clue}\nANSWER: {answer}\nREASONING: {reason}`, and a mismatched template
does **not** error — it produces confident nonsense. Confirm it before trusting
any result:

```bash
python code/pipeline/run_pipeline.py --verify-scorer 200
```

That scores human `wordplay` annotations against two kinds of negative and prints
a verdict. See "Verifying a scorer" below.

## Cache the generator locally (optional, saves download time)

```bash
source code/deberta/without-finetuning/activate_env.sh
python - <<'PY'
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
src, dest = "hugTAU/cryptonite-flant5-large", os.environ["NLPDIR"] + "/models/cryptonite-flant5-large"
AutoModelForSeq2SeqLM.from_pretrained(src).save_pretrained(dest)
AutoTokenizer.from_pretrained(src).save_pretrained(dest)
PY
```

The config tries `{storage}/models/cryptonite-flant5-large` first and falls back
to the Hub id, so this is purely an optimization — nothing to change either way.

## Swapping either model is a config change, by design

The pipeline talks to each stage through one interface (`Scorer.score(records)`,
`Generator.generate(records)` in `adapters.py`). Everything that could differ
between checkpoints is a config field:

| Field | Why it may need changing |
|---|---|
| `scorer.model` | the checkpoint itself. A `;`-separated preference list; first existing local directory wins, Hub id last |
| `scorer.template` | **must match what the scorer was fine-tuned on** — a scorer fed a layout it never saw returns confident nonsense, not an error |
| `scorer.positive_label` | `null` reads the positive class from the checkpoint's own `label2id`, so a model trained with reversed labels does not silently invert the ranking. Set to `"CORRECT"` or an int to force it |
| `scorer.max_length` | must stay ≤ 512 (position embeddings). The tokenizer ships the "unlimited" `model_max_length` sentinel, so nothing else would stop you |
| `scorer.batch_size`, `dtype` | a larger scorer needs a smaller batch |

Per-run override, no file edit:

```bash
python code/pipeline/run_pipeline.py --split test --limit 1000 \
    --set scorer.model=<path-or-hub-id> --set scorer.batch_size=8
```

### Note: no base-model fallback

`config/t5large_deberta_large.json` deliberately has **no** fallback to a bare
`microsoft/deberta-v3-*`. A base model loads fine but its classification head is
randomly initialized, so its scores are noise and the resulting accuracy table is
void. Now that a real checkpoint exists, a missing one should fail loudly rather
than quietly downgrade to nonsense. The detector is still in place — the scorer
warns on placeholder `LABEL_n` labels and stamps `"scorer_head_untrained": true`
into `metrics.json` — so an accidental `--set scorer.model=microsoft/...` is still
caught.

## Verifying a scorer

```bash
python code/pipeline/run_pipeline.py --verify-scorer 200
```

Needs no generator, so it is fast and cannot be confounded by generation quality.
It scores three inputs per annotated val clue and reports two columns:

- **training-style negative** — `(clue_i, answer_j, reason_j)`, a different clue's
  answer *and* reasoning. This is exactly how the fine-tuning notebook built its
  negatives, so this AUC is comparable to the pairwise ranking accuracy the
  training run reported. **If this is near 0.5, the setup is broken** — suspect
  the template, then the positive-class index, then a truncated checkpoint. Below
  0.5 means the ranking is inverted.
- **rerank-style negative** — `(clue_i, answer_i, reason_j)`, the *correct* answer
  with someone else's reasoning. Harder, and much closer to what reranking
  actually asks. The scorer never saw this during training, so a large gap between
  the two columns is the out-of-distribution problem quantified, and predicts a
  weak reranking delta before you spend a GPU-hour finding out.

## A/B two scorers without regenerating

Generation is by far the expensive half of the pipeline, so generate once and
score twice:

```bash
# generate once
python code/pipeline/run_pipeline.py --split test --limit 10000 --stage generate

# score that same generation with each candidate scorer
python code/pipeline/run_pipeline.py --split test --limit 10000 \
    --stage score --candidates-from runs/<that-run> \
    --set scorer.model=<scorer-A> --tag scorerA
python code/pipeline/run_pipeline.py --split test --limit 10000 \
    --stage score --candidates-from runs/<that-run> \
    --set scorer.model=<scorer-B> --tag scorerB
```

A genuinely different *kind* of scorer (a regression head, an ensemble, an LLM
judge) needs one new class in `adapters.py` plus one entry in the `SCORERS`
registry at the bottom of that file. The rest of the pipeline does not change.
