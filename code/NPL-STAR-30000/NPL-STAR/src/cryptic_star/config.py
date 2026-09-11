"""Typed config, loaded from `configs/base.yaml` plus one override file.

Every path and hyper-parameter the pipeline touches comes from here, so the
same code runs on a laptop (`star_t5_base.yaml`) and on the GPU cluster
(`star_t5_large.yaml`) with no edits.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataCfg:
    # "united"   - one pre-joined dataset (united-cryptonite-wordplay-dataset),
    #              wordplay already merged onto the Cryptonite rows.
    # "separate" - raw Cryptonite + raw mdda/cryptic-wordplay, joined by align.py.
    dataset: str = "united"
    # Directory holding train.jsonl.gz / val.jsonl.gz / test.jsonl.gz.
    united_dir: str = "data/raw/united-cryptonite-wordplay-dataset"
    # Drop rows whose answer also appears in a later split. The united dataset
    # ships without this check, so leaving it off silently breaks Cryptonite's
    # answer-disjoint evaluation.
    enforce_answer_split: bool = True
    # Drop Quick (non-cryptic) puzzles: definition-only clues with no wordplay.
    drop_quick: bool = True
    # Rationales shorter than this are not explanations ("dd", "anag").
    min_wordplay_chars: int = 8
    # Also discard annotations that only name the device ("Double Definition")
    # instead of decomposing the clue. Off: with ~5.7k annotated rows in total,
    # the device name is still worth something as a warm start.
    drop_label_only_rationales: bool = False
    # -- dataset: "separate" only ---------------------------------------
    # Directory holding cryptonite-{train,val,test}.jsonl (official answer split).
    cryptonite_dir: str = "data/raw/cryptonite/cryptonite-official-split"
    # Single jsonl from mdda/cryptic-wordplay.
    wordplay_file: str = "data/raw/wordplay/wordplay.jsonl"
    # -------------------------------------------------------------------
    processed_dir: str = "data/processed"
    # Fuzzy-match threshold for the clue-token Jaccard fallback in align.py.
    fuzzy_threshold: float = 0.8
    # Keep wordplay rows that never matched a Cryptonite clue? Their answers are
    # still checked against the val/test answer sets, so they are leakage-safe
    # extra supervision.
    use_unmatched_seed: bool = True
    # Clues sampled from Cryptonite train per STaR iteration. Full train is
    # ~470k clues; N samples over all of it is not affordable.
    train_subsample: int = 30000
    # Cap on rows loaded from each file. None = all. Set small for smoke tests.
    limit_train: int | None = None
    limit_val: int | None = None
    limit_test: int | None = None


@dataclass
class FormatCfg:
    # Emit the answer as "P I P E" instead of "PIPE" (one token per letter).
    spaced_answer: bool = True
    # Ask the model for the definition span before the wordplay.
    include_definition: bool = True
    # Show "letters: 7" alongside the enumeration in the prompt.
    include_letter_count: bool = True
    max_source_len: int = 96
    max_target_len: int = 128


@dataclass
class GenCfg:
    n_samples: int = 8
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 0
    max_new_tokens: int = 128
    batch_size: int = 16
    # Deduplicate identical traces produced for the same clue.
    dedup: bool = True


@dataclass
class TrainCfg:
    model_name: str = "google/flan-t5-base"
    learning_rate: float = 1e-4
    epochs: float = 3.0
    batch_size: int = 8
    grad_accum: int = 4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    label_smoothing: float = 0.0
    # Optimizer and schedule, exposed because they decide whether flan-t5-large
    # fits in memory at all. `adamw_torch` keeps two fp32 moments per parameter
    # (~6GB of optimizer state for 780M params); `adafactor` factors the second
    # moment and drops the first, which is what the Cryptonite paper used for its
    # T5 baseline (with learning_rate 1e-3 and lr_scheduler_type "constant").
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    # LoRA / QLoRA. Off by default: full fine-tuning of flan-t5-base|large fits
    # on one cluster GPU and avoids the peft/bitsandbytes debugging tax.
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = False
    num_workers: int = 2


@dataclass
class VerifyCfg:
    """How a generated trace earns its `1` label."""

    # Require real reasoning, not just the right answer.
    require_reasoning: bool = True
    min_wordplay_chars: int = 8
    # Count a documented alternative answer as correct. Off by default: the
    # 7.64% Cryptonite baseline scores against the single gold answer, so
    # turning this on makes the headline EM non-comparable. Either way the
    # matches are recorded as their own rejection reason, so the analysis
    # reports how many there were.
    accept_alt_answers: bool = False


@dataclass
class StarCfg:
    iterations: int = 3
    # STaR retrains from the *original* pretrained checkpoint each iteration
    # rather than continuing from the last one, to avoid compounding drift.
    # Set False for the cheaper continued-training variant.
    retrain_from_base: bool = True
    # Warm-start iteration 0 on the human wordplay rationales before any
    # self-generation. Without this, flan-t5 is too weak to bootstrap.
    warm_start_on_seed: bool = True
    # Second STaR pass: re-prompt failed clues with the gold answer as a hint.
    rationalize: bool = True
    # Keep at most this many accepted traces per clue. 0 = keep ALL of them
    # (every distinct correct reasoning path for that answer).
    # Trade-off: more traces per clue = more reasoning diversity, but easy clues
    # (which the model solves many ways) start to dominate the training set.
    max_traces_per_clue: int = 0
    # Treat traces differing only in casing/punctuation as the same reasoning.
    normalized_dedup: bool = True
    # Carry accepted traces from earlier iterations into later training sets.
    accumulate_traces: bool = True
    out_dir: str = "runs/default"
    seed: int = 13


@dataclass
class EvalCfg:
    split: str = "test"
    limit: int | None = 2000
    num_beams: int = 4
    max_new_tokens: int = 128
    batch_size: int = 16
    # Also report pass@N / oracle accuracy using sampled candidates.
    report_pass_at_n: bool = True


@dataclass
class DiscriminatorCfg:
    """Export settings for the DeBERTa cross-encoder teammates train."""

    out_file: str = "discriminator_data.jsonl"
    # Negatives per positive, sampled from wrong-answer traces.
    negatives_per_positive: int = 3
    # Drop negatives whose answer is right but whose reasoning was rejected for
    # another reason; they are label noise for a right/wrong classifier.
    strict_negatives: bool = True


@dataclass
class TrackingCfg:
    """Live metrics, reasoning traces and TensorBoard scalars. See tracking.py.

    File-based on purpose. Cluster compute nodes have no outbound network, so a
    tracker that phones home (Weights & Biases) would either stall the job or
    have to be switched off exactly when it is most useful. TensorBoard reads
    the event files afterwards, over an ssh tunnel or on a laptop.
    """

    enabled: bool = True
    # Append-only, one JSON object per line, under star.out_dir. `tail -f`-able
    # while the job runs, and one `pd.read_json(lines=True)` afterwards.
    metrics_file: str = "metrics.jsonl"
    # TensorBoard event files under star.out_dir/<tb_dir>. Needs `tensorboard`
    # installed; missing, it degrades to the jsonl stream with a warning.
    tensorboard: bool = True
    tb_dir: str = "tb"
    # Trainer logging cadence, in optimizer steps.
    log_steps: int = 25
    # Solved/failed traces written to iterK/trace_samples.md each iteration, to
    # eyeball how the model actually reasons. 0 disables.
    trace_samples: int = 15


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    format: FormatCfg = field(default_factory=FormatCfg)
    generate: GenCfg = field(default_factory=GenCfg)
    verify: VerifyCfg = field(default_factory=VerifyCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    star: StarCfg = field(default_factory=StarCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    discriminator: DiscriminatorCfg = field(default_factory=DiscriminatorCfg)
    tracking: TrackingCfg = field(default_factory=TrackingCfg)
    device: str = "auto"
    log_level: str = "INFO"

    # -- loading ---------------------------------------------------------
    @classmethod
    def load(cls, *paths: str | Path) -> "Config":
        merged: dict[str, Any] = {}
        for path in paths:
            if path is None:
                continue
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            merged = _deep_merge(merged, raw)
        return _from_dict(cls, merged)

    def to_dict(self) -> dict:
        return _to_dict(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.star.out_dir)

    def iter_dir(self, i: int) -> Path:
        return self.run_dir / f"iter{i}"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _from_dict(cls, data: dict):
    kwargs = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in (data or {}).items():
        if key not in known:
            raise ValueError(f"Unknown config key '{key}' for {cls.__name__}")
        f = known[key]
        if is_dataclass(f.type) or (isinstance(value, dict) and _is_cfg(f)):
            kwargs[key] = _from_dict(_cfg_type(f), value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


_CFG_TYPES = {
    "data": DataCfg,
    "format": FormatCfg,
    "generate": GenCfg,
    "verify": VerifyCfg,
    "train": TrainCfg,
    "star": StarCfg,
    "eval": EvalCfg,
    "discriminator": DiscriminatorCfg,
    "tracking": TrackingCfg,
}


def _is_cfg(f) -> bool:
    return f.name in _CFG_TYPES


def _cfg_type(f):
    return _CFG_TYPES[f.name]


def _to_dict(obj):
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj


DEFAULT_CONFIG = "configs/base.yaml"


def add_config_args(parser) -> None:
    """Shared CLI surface: `-c a.yaml -c b.yaml -o k.k2=v`.

    `-c` is spelled out as an explicit short option: argparse only abbreviates
    *long* options, so `-c` is not a stand-in for `--config` and every command
    in the README used to die with "unrecognized arguments" before any logging
    was configured - which on Slurm looks like a silent crash.

    `default` stays None rather than [DEFAULT_CONFIG], because `action="append"`
    appends to its default instead of replacing it: the documented
    `-c configs/base.yaml -c configs/smoke.yaml` would otherwise load base.yaml
    twice. `config_from_args` supplies the default when nothing was passed.
    """
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        default=None,
        help=f"YAML config; repeatable, later files override earlier ones. "
        f"Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="dotted.key=value",
        help="Ad-hoc override, e.g. -o generate.n_samples=4",
    )


def config_from_args(args) -> Config:
    cfg = Config.load(*(args.config or [DEFAULT_CONFIG]))
    for item in args.override or []:
        key, _, value = item.partition("=")
        _apply_override(cfg, key.strip(), value.strip())
    return cfg


def _apply_override(cfg: Config, dotted: str, raw: str) -> None:
    target: Any = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise ValueError(f"Unknown config key '{dotted}'")
    current = getattr(target, leaf)
    setattr(target, leaf, _coerce(raw, current))


def _coerce(raw: str, current: Any) -> Any:
    if raw.lower() in {"none", "null"}:
        return None
    if isinstance(current, bool) or raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return yaml.safe_load(raw) if raw.startswith(("[", "{")) else raw
