"""Fine-tune T5 on a jsonl of {source, target} rows.

Used for both the warm start on human rationales and every STaR iteration; the
only difference is which file is passed in. Optional LoRA/QLoRA path for the
larger generator mentioned in the proposal.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import Config, add_config_args, config_from_args
from .io_utils import free_gpu, read_jsonl, seed_everything, setup_logging, write_json
from .tracking import RunTracker

log = logging.getLogger(__name__)


def _build_dataset(rows: Sequence[dict], tokenizer, cfg: Config):
    from datasets import Dataset

    ds = Dataset.from_list([{"source": r["source"], "target": r["target"]} for r in rows])

    def tok(batch):
        model_inputs = tokenizer(
            batch["source"],
            max_length=cfg.format.max_source_len,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=cfg.format.max_target_len,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return ds.map(tok, batched=True, remove_columns=["source", "target"])


def _maybe_wrap_lora(model, cfg: Config):
    if not cfg.train.use_lora:
        return model
    from peft import LoraConfig, TaskType, get_peft_model

    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg.train.lora_r,
        lora_alpha=cfg.train.lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        target_modules=["q", "v"],  # standard choice for T5
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


# Keyword arguments transformers has renamed, newest name first. The Trainer API
# is not stable across minor versions and a cluster environment is rarely the
# version you developed against, so every one of these is a `TypeError` raised
# the moment iteration 0 starts training - after data prep has already run.
_RENAMES: tuple[tuple[str, ...], ...] = (
    ("eval_strategy", "evaluation_strategy"),  # renamed in transformers 4.41
    ("processing_class", "tokenizer"),  # Trainer arg renamed in 4.46, gone in 5.0
)


def _compat_kwargs(target: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Rename or drop keyword arguments this transformers version rejects.

    Checks the real signature rather than the version number, because the same
    argument moved at different times in different classes and a version string
    does not tell you which patch release you are on. Anything genuinely unknown
    is dropped with a warning: losing `bf16` should be visible in the log, not a
    silent slowdown, and it should not abort a run that would otherwise work.
    """
    params = inspect.signature(target).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)  # **kwargs swallows everything; nothing to check

    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        name = key
        if name not in params:
            for group in _RENAMES:
                if key in group:
                    name = next((alt for alt in group if alt in params), key)
                    break
        if name in params:
            out[name] = value
        else:
            log.warning(
                "%s in this transformers version (%s) takes no `%s`; ignoring it",
                getattr(target, "__qualname__", target),
                _transformers_version(),
                key,
            )
    return out


def _warmup_kwargs(cfg: Config, n_examples: int, params) -> dict[str, Any]:
    """Warmup as a ratio where that exists, converted to a step count where it does not.

    transformers 5 dropped `warmup_ratio` and kept only `warmup_steps`. These are
    not interchangeable names for one argument, so `_compat_kwargs` cannot help:
    letting the ratio fall on the floor would silently remove warmup from the
    schedule for the whole run - a real change to the training recipe, invisible
    except as a slightly different curve.

    The step count is derived the same way the Trainer derives its own total:
    one optimizer step per `batch_size * grad_accum` examples, times the epoch
    count.
    """
    ratio = float(cfg.train.warmup_ratio or 0.0)
    if ratio <= 0:
        return {}
    if "warmup_ratio" in params:
        return {"warmup_ratio": ratio}

    per_step = max(int(cfg.train.batch_size), 1) * max(int(cfg.train.grad_accum), 1)
    steps_per_epoch = max(math.ceil(n_examples / per_step), 1)
    total_steps = max(math.ceil(steps_per_epoch * float(cfg.train.epochs)), 1)
    steps = max(round(ratio * total_steps), 1)
    log.info(
        "warmup_ratio %.3g is not supported here; using warmup_steps=%d of ~%d total",
        ratio,
        steps,
        total_steps,
    )
    return {"warmup_steps": steps}


def _transformers_version() -> str:
    """For log messages only, so it must not be able to raise.

    `_compat_kwargs` exists to keep a version mismatch from aborting a run;
    letting its own diagnostic throw ImportError would defeat the point.
    """
    try:
        import transformers
    except ImportError:
        return "not installed"
    return getattr(transformers, "__version__", "unknown")


def _tracker_callback(tracker: RunTracker, tag: str):
    """Send the Trainer's log dict to both of the tracker's artefacts.

    Deliberately not `report_to=["tensorboard"]`: transformers 5 removed
    `logging_dir`, so the built-in integration can no longer be aimed at the
    run's `tb/` directory and would scatter event files under the Trainer's
    output_dir instead. Writing the scalars here works on every version and puts
    the warm start and each STaR iteration under its own `train/<tag>/` prefix,
    so they are comparable in one TensorBoard view.

    The jsonl half is the one that matters on a cluster: it can be read with
    `tail -f` over ssh, with no port forwarding and nothing to install.
    """
    from transformers import TrainerCallback

    class _Mirror(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            numeric = {
                k: v for k, v in logs.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if not numeric:
                return
            step = int(state.global_step or 0)
            # Plain fields, not `scalars=`, so `event` does not also plot them:
            # the explicit `scalars` call below owns the tag layout.
            tracker.event(f"train/{tag}", step=step, **numeric)
            tracker.scalars(numeric, step=step, prefix=f"train/{tag}")

    return _Mirror()


def train(
    train_file: str | Path,
    out_dir: str | Path,
    cfg: Config,
    base_model: str | None = None,
    eval_file: str | Path | None = None,
    tracker: RunTracker | None = None,
    tag: str | None = None,
) -> Path:
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    base_model = base_model or cfg.train.model_name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or out_dir.name

    rows = list(read_jsonl(train_file))
    if not rows:
        raise ValueError(f"No training rows in {train_file}")
    log.info("training %s on %d examples -> %s", base_model, len(rows), out_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    load_kwargs: dict = {}
    if cfg.train.load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model, **load_kwargs)
    model = _maybe_wrap_lora(model, cfg)

    train_ds = _build_dataset(rows, tokenizer, cfg)
    eval_ds = None
    if eval_file:
        eval_rows = list(read_jsonl(eval_file))
        if eval_rows:
            eval_ds = _build_dataset(eval_rows, tokenizer, cfg)

    cuda = torch.cuda.is_available()
    # fp16 is deliberately separate from bf16 and off by default: T5 was trained
    # in bfloat16 and its activations overflow float16's range, which shows up as
    # a loss that goes to nan a few hundred steps in rather than as an error.
    if cfg.train.fp16 and cuda:
        log.warning(
            "train.fp16 is on. T5 family models frequently produce nan losses in "
            "float16; prefer train.bf16 on any Ampere-or-newer GPU."
        )
    arg_params = inspect.signature(Seq2SeqTrainingArguments).parameters
    targs = Seq2SeqTrainingArguments(
        **_compat_kwargs(
            Seq2SeqTrainingArguments,
            dict(
                output_dir=str(out_dir / "_hf"),
                learning_rate=cfg.train.learning_rate,
                optim=cfg.train.optim,
                lr_scheduler_type=cfg.train.lr_scheduler_type,
                num_train_epochs=cfg.train.epochs,
                per_device_train_batch_size=cfg.train.batch_size,
                per_device_eval_batch_size=cfg.train.batch_size,
                gradient_accumulation_steps=cfg.train.grad_accum,
                weight_decay=cfg.train.weight_decay,
                label_smoothing_factor=cfg.train.label_smoothing,
                bf16=cfg.train.bf16 and cuda,
                fp16=cfg.train.fp16 and cuda,
                gradient_checkpointing=cfg.train.gradient_checkpointing,
                dataloader_num_workers=cfg.train.num_workers,
                logging_steps=max(int(cfg.tracking.log_steps), 1),
                save_strategy="no",  # the final model is saved explicitly below
                eval_strategy="epoch" if eval_ds is not None else "no",
                # Scalars go through _tracker_callback, not the Trainer's own
                # TensorBoard writer - see that function for why.
                report_to=[],
                seed=cfg.star.seed,
                predict_with_generate=False,
                **_warmup_kwargs(cfg, len(train_ds), arg_params),
            ),
        )
    )
    trainer = Seq2SeqTrainer(
        **_compat_kwargs(
            Seq2SeqTrainer,
            dict(
                model=model,
                args=targs,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                # `processing_class` under transformers >= 4.46, `tokenizer`
                # before it; the old name was removed outright in 5.0.
                processing_class=tokenizer,
                data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            ),
        )
    )
    if tracker is not None:
        trainer.add_callback(_tracker_callback(tracker, tag))
    result = trainer.train()

    # Merge LoRA so downstream stages can load the directory as a plain model.
    if cfg.train.use_lora:
        model = model.merge_and_unload()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    metrics = {"n_examples": len(rows), "base_model": base_model, **result.metrics}
    write_json(out_dir / "train_metrics.json", metrics)
    if tracker is not None:
        tracker.event(
            "train_done",
            scalars={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            tag=tag,
            base_model=base_model,
            out_dir=str(out_dir),
        )
    log.info("saved fine-tuned generator -> %s", out_dir)

    # Drop the model and the Trainer's optimizer/gradient state before returning:
    # the caller immediately loads a generator onto the same GPU, and torch's
    # caching allocator does not release this on its own.
    del trainer, model
    free_gpu()
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--eval-file", default=None)
    args = parser.parse_args(argv)

    cfg: Config = config_from_args(args)
    setup_logging(cfg.log_level)
    seed_everything(cfg.star.seed)
    out = Path(args.out)
    # Standalone runs get their own metrics.jsonl / tb next to the model, rather
    # than the run-level ones star_loop.py keeps.
    with RunTracker(out.parent, cfg) as tracker:
        tracker.config(cfg)
        train(
            args.train_file,
            out,
            cfg,
            base_model=args.base_model,
            eval_file=args.eval_file,
            tracker=tracker,
            tag=out.name,
        )


if __name__ == "__main__":
    main()
