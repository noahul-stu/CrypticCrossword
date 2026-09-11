"""The CLI surface every entry point shares.

There were no tests here, which is how `-c` - the flag used by the sbatch script
and by every command in the README - shipped without being registered at all.
argparse exited 2 before logging was configured, so on Slurm it looked like a
silent crash with no log files.
"""

import argparse

import pytest

from cryptic_star.config import (
    DEFAULT_CONFIG,
    Config,
    add_config_args,
    config_from_args,
)


def parse(*argv):
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    return parser.parse_args(list(argv))


def write_cfg(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_short_and_long_config_flags_both_work():
    assert parse("-c", "a.yaml").config == ["a.yaml"]
    assert parse("--config", "a.yaml").config == ["a.yaml"]


def test_repeated_config_does_not_also_load_the_default():
    # `action="append"` appends to its default, so a non-empty default would
    # make the documented `-c base.yaml -c smoke.yaml` load base.yaml twice.
    assert parse("-c", "base.yaml", "-c", "smoke.yaml").config == [
        "base.yaml",
        "smoke.yaml",
    ]


def test_no_config_flag_falls_back_to_the_default_file():
    args = parse()
    assert args.config is None
    assert DEFAULT_CONFIG == "configs/base.yaml"
    assert config_from_args(args).to_dict() == Config.load(DEFAULT_CONFIG).to_dict()


def test_later_config_overrides_earlier_one(tmp_path):
    base = write_cfg(tmp_path, "base.yaml", "star:\n  iterations: 3\n  seed: 13\n")
    over = write_cfg(tmp_path, "over.yaml", "star:\n  iterations: 1\n")
    cfg = Config.load(base, over)
    assert cfg.star.iterations == 1
    assert cfg.star.seed == 13  # untouched keys survive the merge


def test_overrides_are_applied_and_coerced(tmp_path):
    base = write_cfg(tmp_path, "base.yaml", "star:\n  iterations: 3\n")
    args = parse(
        "-c", base,
        "-o", "generate.n_samples=4",
        "-o", "star.retrain_from_base=false",
        "-o", "train.learning_rate=5e-5",
        "-o", "eval.limit=none",
    )
    cfg = config_from_args(args)
    assert cfg.generate.n_samples == 4 and isinstance(cfg.generate.n_samples, int)
    assert cfg.star.retrain_from_base is False
    assert cfg.train.learning_rate == pytest.approx(5e-5)
    assert cfg.eval.limit is None


def test_unknown_override_key_is_rejected(tmp_path):
    base = write_cfg(tmp_path, "base.yaml", "star:\n  iterations: 3\n")
    args = parse("-c", base, "-o", "star.no_such_knob=1")
    with pytest.raises(ValueError, match="no_such_knob"):
        config_from_args(args)


def test_unknown_yaml_key_is_rejected(tmp_path):
    path = write_cfg(tmp_path, "bad.yaml", "star:\n  itterations: 3\n")
    with pytest.raises(ValueError, match="itterations"):
        Config.load(path)


def test_shipped_configs_load_and_compose():
    """The real files, in the order the README and the sbatch use them."""
    base = Config.load("configs/base.yaml")
    assert base.star.iterations >= 1
    assert base.tracking.enabled is True

    smoke = Config.load("configs/base.yaml", "configs/smoke.yaml")
    assert smoke.train.model_name == "google/flan-t5-small"
    assert smoke.star.iterations == 1

    large = Config.load("configs/base.yaml", "configs/star_t5_large.yaml")
    assert large.train.model_name == "google/flan-t5-large"


def test_to_dict_round_trips_through_load(tmp_path):
    import yaml

    cfg = Config.load("configs/base.yaml")
    path = tmp_path / "dumped.yaml"
    path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    assert Config.load(path).to_dict() == cfg.to_dict()
