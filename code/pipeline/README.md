# Candidate-generate → score → select pipeline

```
clue ──▶ [1 GENERATE]  T5 proposes k (answer, reason) candidates
                       dedup, then drop answers that cannot fit the enumeration
     ──▶ [2 SCORE]     DeBERTa scores every (clue, answer, reason) → [0,1]
     ──▶ [3 SELECT]    argmax → the returned answer + reason
```

One entry point serves one clue and ten thousand. Both models are swappable from a
config file, and every run reports the pipeline **against its own baseline** on
exactly the same clues.

```bash
# one clue, full stage-by-stage trace
python code/pipeline/run_pipeline.py --clue "attack general at end of month (6)"

# is the scorer wired up correctly? do this first, it needs no generator
python code/pipeline/run_pipeline.py --verify-scorer 200

# 8 clues on a laptop CPU, tiny models, checks the wiring only
python code/pipeline/run_pipeline.py --config smoke --split val --limit 8

# the real evaluation, on a GPU node
sbatch code/pipeline/run_pipeline.sbatch --split test --limit 1000
```

`pipeline_demo.ipynb` is a thin wrapper over the same modules — no logic of its
own, so a fix never has to be applied twice.

---

## The models

| Stage | Model | Notes |
|---|---|---|
| 1 generate | `hugTAU/cryptonite-flant5-large` | flan-t5-large fine-tuned on Cryptonite |
| 2 score | `code/deberta/with-finetuning/trained_deberta` | fine-tuned **deberta-v3-large** (435M params, 24 layers), `id2label {0: INCORRECT, 1: CORRECT}`, fp32 |

Run this once, before anything else — it checks the scorer in isolation, needs no
generator, and catches the failure mode that does not raise:

```bash
python code/pipeline/run_pipeline.py --verify-scorer 200
```

Nothing in this repo records the input template the delivered scorer was
fine-tuned with (the with-finetuning notebook here is the *small* model's). The
config assumes `CLUE:/ANSWER:/REASONING:`, and a mismatched template produces
confident nonsense rather than an error. `--verify-scorer` is what tells you.
It reports two AUCs — see [`models/README.md`](../../models/README.md) for how to
read them and for the swap/A-B procedure.

There is deliberately no fallback to a bare `microsoft/deberta-v3-*`: a base model
loads fine but has a random classification head, so its scores are noise. The
detector remains in place anyway — the scorer warns on placeholder `LABEL_n`
labels, reprints a banner after the results table, and stamps
`"scorer_head_untrained": true` into `metrics.json`.

---

## Reading the results

`format_metrics` prints five accuracies, and they only mean something together:

| Number | What it is | What it tells you |
|---|---|---|
| `random_from_candidates` | pick uniformly from the k candidates | the **floor**. A "reranker" that lands here has learned nothing and is being credited for the generator's list |
| `baseline_generator_top1` | the generator's own rank-0 decode | what **"a simple LLM generator"** scores. The thing to beat |
| `baseline_after_filters` | generator top-1 among candidates that survived the enumeration filter | isolates how much of any gain is the **free length filter** rather than the scorer |
| `pipeline` | scorer argmax | the claim |
| `oracle_at_k` | was the gold answer *anywhere* in the k candidates? | the **hard ceiling**. A scorer cannot pick an answer that was never proposed |

Plus:

- **`mcnemar_p_value`** — exact paired test over the clues where the two systems
  disagree. Both systems answer the same clues, so only disagreements carry
  information; an unpaired test on two accuracy numbers throws the pairing away.
  At n=1000 a two-point gap is well inside noise, so a delta without this is not
  a result.
- **`headroom_captured`** — of the baseline's mistakes that *were* fixable (gold
  was in the list), what fraction did the scorer fix? A cleaner read on the
  scorer than raw delta, which is capped by how often the generator was already
  right.
- **`scorer.candidate_auc`** — given a correct and an incorrect candidate, how
  often does the scorer rank the correct one higher? The scorer's quality in
  isolation. 0.5 is chance; below 0.5 means the ranking is inverted.

The summary block ends with a **diagnosis line** that names the bottleneck:
oracle@k flat against baseline → work on the generator; headroom present but
delta ≤ 0 → work on the scorer; delta positive but p ≥ 0.05 → run more clues.

### Where the headroom comes from

The scorer was fine-tuned to tell a human `wordplay` annotation apart from *an
unrelated clue's* annotation. Reranking asks a harder and different question:
separate near-miss candidates for the **same** clue. That is out of distribution,
and it is the most likely reason for a disappointing delta — see the "recommended
next improvement" note at the end of
`DeBERTa_small_Wordplay_Reasoning_Scorer_with_finetuning.ipynb`, which calls for
training on generated hard negatives. `selection.combine="weighted"` is the cheap
mitigation: blend the scorer with the generator's own confidence.

---

## Files

| File | Role |
|---|---|
| `run_pipeline.py` | launcher; works by path or as `python -m pipeline.run_pipeline` |
| `cli.py` | argument parsing, run directory, artifacts, `--probe`, `--verify-scorer`, `--score-gold` |
| `pipeline.py` | the three stages, filters, selection, chunked resume |
| `adapters.py` | `Seq2SeqGenerator`, `SequenceClassificationScorer`, and the registry they plug into |
| `metrics.py` | accuracies, oracle@k, McNemar, Wilson CI, AUC, three figures |
| `data.py` | split loading, reproducible sampling, sharding, ad-hoc clues |
| `config_io.py` | JSON-with-comments configs, `extends`, `--set` overrides |
| `paths.py` | cluster-vs-local paths, cache redirection off `$HOME` |
| `selftest.py` | 103 checks over the full logic, with torch stubbed and both models faked; needs no GPU and no download |
| `run_pipeline.sbatch` | Slurm wrapper following the existing repo conventions |
| `config/*.json` | `default` (base) · `t5large_deberta_large` (**the real one, default**) · `smoke` (laptop) |

Run `python code/pipeline/selftest.py` after any change to `pipeline.py`,
`metrics.py` or `data.py`. A few seconds of CPU, though wall-clock varies with
filesystem state.

---

## Swapping models

Nothing model-specific is a code path. Either edit the config or override per run:

```bash
--set scorer.model=<path-or-hub-id>            # different scorer
--set scorer.batch_size=8                      # if a card OOMs on the large scorer
--set generator.model=<path-or-hub-id>         # different generator
--set generator.num_candidates=20 --set generator.beam_candidates=8
--set generator.strategy=beam        # ablation: no sampling pass, poor oracle@k
--set selection.combine=weighted               # blend scorer + generator confidence
--set selection.combine=generator              # ablation: reproduces the baseline
--set filters.enumeration=false                # ablation: how much is the length filter worth?
```

A `model` field is a `;`-separated preference list — first existing local
directory wins, hub id last — so one config serves both the cluster and a laptop.

A genuinely different *kind* of model (a causal LM generator, an ensemble or
LLM-judge scorer) needs one new class in `adapters.py` plus one entry in the
`GENERATORS` / `SCORERS` registry at the bottom of that file. Nothing else moves.

### Decode strategy is the biggest lever on the ceiling

The default is `beam_sample_union`: a beam pass then a sampling pass, merged. The
two things this pipeline needs from decoding are in tension — the **baseline** must
be what a plain T5 returns (beam search), while the **candidate pool** must be
diverse or the reranker has nothing to rerank (and beam search is bad at that,
since its beams share prefixes). Separate passes get both, and beam runs first so
the pool's rank 0 is the true beam top-1.

`beam_candidates` splits `num_candidates` between the two passes (default: half).

> **`diverse_beam` is not usable on transformers 5.** Group beam search was removed
> from core; it now lives in the remote `transformers-community/group-beam-search`
> repo and needs `trust_remote_code=True` plus a live route to hf.co *at generation
> time* — which a compute node may not have. Selecting it raises immediately with
> instructions rather than failing mid-job. Opt in with
> `--set generator.allow_remote_code=true` if you really want it.

Two environment quirks the code works around, both verified by running it:

- **Tokenizer/model mismatch.** A BERT-lineage tokenizer emits `token_type_ids`,
  which T5's `generate()` rejects outright (it does not ignore unknown kwargs).
  Inputs are filtered against the model's real `forward` signature, with a notice.
- **MPS + sampling.** On Apple MPS, transformers 5.17 raises
  `'EncoderDecoderCache' object has no attribute 'layers'` whenever
  `return_dict_in_generate=True` meets a non-beam path. `use_cache=False` avoids
  it, applied automatically on MPS only. CPU and CUDA are unaffected, so the
  cluster never sees this.

### First run against an unfamiliar checkpoint: `--probe`

```bash
python code/pipeline/run_pipeline.py --probe 5 --split test
```

Dumps raw decodes with no parsing. `generator.parse.patterns` cannot be written
without seeing this, and **a wrong pattern does not raise** — it silently yields
empty reasons and leaves the scorer judging `(clue, answer)` only. The run also
reports how many decodes matched a pattern, and warns loudly if none did.

Note the generator prompt defaults to the bare `{clue}` because
`train_cryptonite.py` fine-tuned on bare clue text (the clue already carries its
enumeration inline). Adding an instruction prefix pushes the input off the
distribution the checkpoint was trained on, which costs accuracy silently.

---

## Scaling to 10k

- **Streaming + resume.** Each finished clue is appended to `records.jsonl`
  immediately, and a rerun with the same `--run-dir` skips what is already there.
  `studentkillable` is preemptible and the sbatch uses `--requeue`, so a killed
  job continues rather than restarting. A truncated final line from a mid-write
  kill is detected and that one clue redone.
- **Sharding.** `--shard i/n` splits the same clue selection across parallel jobs.
  Sampling happens before sharding, so 4 shards of `--limit 10000` cover exactly
  the clues one unsharded job would.
- **Two-phase.** `--stage generate` then `--stage score --candidates-from <run>`.
  Generation is by far the expensive half; this is how you evaluate a new scorer
  on an existing 10k generation for the price of the scoring alone.
- **Cost.** Generation dominates: ~1–3 clues/s for flan-t5-large with 12 diverse
  beams on one card, so ~1000 clues is well under an hour and 10k is a few hours
  — prefer 4 shards over one long job. Scoring 12 × 10k pairs with
  DeBERTa-small is minutes.
- **Artifacts** per run: `config.json` (resolved, after `extends` and `--set`),
  `environment.json` (versions, GPU, git commit), `records.jsonl` (every
  candidate with scores and drop reasons — a full audit trail), `metrics.json`,
  `predictions.csv`, `figures/*.png`.

## Setup on the cluster

The pipeline needs `torch`, `transformers`, `sentencepiece`, `protobuf`, `numpy`
and `matplotlib` — exactly what `code/deberta/without-finetuning/setup_env.sh`
already installs. Run that on the **login node** (compute nodes' python3.12 has
no `ensurepip`, so `python3 -m venv` fails there), `mkdir -p slurm_logs`, then
submit. The sbatch preflights the dependency list and fails in seconds with the
exact `pip install` command if anything is missing.
