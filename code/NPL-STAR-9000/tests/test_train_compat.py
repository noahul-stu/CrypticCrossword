"""The transformers version-compatibility shim in train.py.

`Seq2SeqTrainingArguments` renamed `evaluation_strategy` to `eval_strategy` in
4.41, and `Trainer` renamed `tokenizer` to `processing_class` in 4.46 and removed
the old spelling in 5.0. A cluster environment is rarely the version you
developed against, and the mismatch is a TypeError raised the moment iteration 0
starts training - after data prep has already spent its hour. These tests stand
in for the transformers versions we cannot install here.
"""

import logging

from cryptic_star.config import Config
from cryptic_star.train import _compat_kwargs, _warmup_kwargs


def new_style(eval_strategy="no", bf16=False, learning_rate=1e-4):
    """A signature like transformers >= 4.41."""


def old_style(evaluation_strategy="no", bf16=False, learning_rate=1e-4):
    """A signature like transformers < 4.41."""


def trainer_new(model=None, processing_class=None, data_collator=None):
    """A signature like Trainer in transformers >= 4.46."""


def trainer_old(model=None, tokenizer=None, data_collator=None):
    """A signature like Trainer in transformers < 4.46."""


def takes_anything(model=None, **kwargs):
    """A signature that swallows everything."""


def test_the_new_name_is_kept_when_the_signature_has_it():
    got = _compat_kwargs(new_style, {"eval_strategy": "epoch", "bf16": True})
    assert got == {"eval_strategy": "epoch", "bf16": True}


def test_the_new_name_is_translated_back_for_an_older_version():
    got = _compat_kwargs(old_style, {"eval_strategy": "epoch", "bf16": True})
    assert got == {"evaluation_strategy": "epoch", "bf16": True}


def test_processing_class_becomes_tokenizer_on_an_older_trainer():
    got = _compat_kwargs(trainer_old, {"model": 1, "processing_class": "tok"})
    assert got == {"model": 1, "tokenizer": "tok"}


def test_processing_class_survives_on_a_newer_trainer():
    got = _compat_kwargs(trainer_new, {"model": 1, "processing_class": "tok"})
    assert got == {"model": 1, "processing_class": "tok"}


def test_a_renamed_key_is_never_passed_under_both_names():
    """Passing both spellings is a TypeError, so only one may come out."""
    for target in (new_style, old_style):
        got = _compat_kwargs(target, {"eval_strategy": "epoch"})
        assert len(got) == 1, got


def test_an_unknown_key_is_dropped_and_logged(caplog):
    """Dropped rather than fatal - but never silently.

    Losing `bf16` halves training throughput, so it has to be visible in the
    job log; aborting a run that would otherwise complete is worse.
    """
    with caplog.at_level(logging.WARNING):
        got = _compat_kwargs(new_style, {"bf16": True, "invented_arg": 3})
    assert got == {"bf16": True}
    assert "invented_arg" in caplog.text


def test_a_signature_with_var_keyword_is_left_alone():
    kwargs = {"model": 1, "anything_at_all": 2}
    assert _compat_kwargs(takes_anything, kwargs) == kwargs


def warmup_cfg(ratio=0.1, examples_per_step=8, epochs=1.0):
    cfg = Config()
    cfg.train.warmup_ratio = ratio
    cfg.train.batch_size = examples_per_step
    cfg.train.grad_accum = 1
    cfg.train.epochs = epochs
    return cfg


def test_warmup_stays_a_ratio_where_the_argument_exists():
    got = _warmup_kwargs(warmup_cfg(), 800, {"warmup_ratio": None, "warmup_steps": None})
    assert got == {"warmup_ratio": 0.1}


def test_warmup_becomes_a_step_count_where_the_ratio_was_removed():
    """transformers 5 kept only `warmup_steps`, and the units differ.

    800 examples at 8 per optimizer step is 100 steps for one epoch, so 10% is
    10 steps. Dropping the ratio instead would remove warmup from the schedule
    for the whole run - a silent change to the training recipe.
    """
    got = _warmup_kwargs(warmup_cfg(), 800, {"warmup_steps": None})
    assert got == {"warmup_steps": 10}


def test_warmup_step_count_accounts_for_accumulation_and_epochs():
    cfg = warmup_cfg(ratio=0.5, examples_per_step=4, epochs=2.0)
    cfg.train.grad_accum = 2  # 4 * 2 = 8 examples per optimizer step
    # 80 examples -> 10 steps/epoch -> 20 total -> half of that is 10.
    assert _warmup_kwargs(cfg, 80, {"warmup_steps": None}) == {"warmup_steps": 10}


def test_no_warmup_passes_nothing_at_all():
    """A zero ratio must not become `warmup_steps=1`."""
    for params in ({"warmup_ratio": None}, {"warmup_steps": None}):
        assert _warmup_kwargs(warmup_cfg(ratio=0.0), 800, params) == {}


def test_a_tiny_but_nonzero_warmup_survives_rounding():
    """Rounding to zero would silently disable a warmup that was asked for."""
    got = _warmup_kwargs(warmup_cfg(ratio=0.001), 8, {"warmup_steps": None})
    assert got == {"warmup_steps": 1}


def test_the_real_training_arguments_accept_what_train_passes():
    """The actual guarantee, against whatever transformers is installed here.

    Skipped on a machine without transformers - which is the laptop this suite
    usually runs on, and precisely why the fake signatures above exist.
    """
    import inspect

    import pytest

    pytest.importorskip("transformers")
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    cfg = Config()
    params = inspect.signature(Seq2SeqTrainingArguments).parameters
    passed = {
        "output_dir": "x",
        "learning_rate": cfg.train.learning_rate,
        "optim": cfg.train.optim,
        "lr_scheduler_type": cfg.train.lr_scheduler_type,
        "eval_strategy": "no",
        "bf16": False,
        "predict_with_generate": False,
        **_warmup_kwargs(cfg, 800, params),
    }
    accepted = _compat_kwargs(Seq2SeqTrainingArguments, passed)
    # Nothing was silently dropped: every argument train() passes is understood
    # by the transformers actually installed here.
    assert set(accepted) == set(passed), f"dropped: {set(passed) - set(accepted)}"
    assert Seq2SeqTrainingArguments(**accepted) is not None

    trainer_args = _compat_kwargs(Seq2SeqTrainer, {"model": None, "processing_class": None})
    assert set(trainer_args) <= set(inspect.signature(Seq2SeqTrainer).parameters)
    assert len(trainer_args) == 2, "the tokenizer argument was dropped entirely"
