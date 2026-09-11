"""Download the model weights into the HF cache before a job needs them.

Run this on the LOGIN node, which has outbound network, so the job itself never
downloads anything:

    python -m cryptic_star.prefetch -c configs/base.yaml -c configs/star_t5_large.yaml

Two reasons this is a separate step rather than something the run does for
itself:

* A compute node behind the university firewall may have no route to
  huggingface.co. `from_pretrained` then hangs on a connection timeout deep
  inside iteration 0, long after data prep has spent its hour.
* flan-t5-large is ~3GB. Paying for that inside a `studentkillable` job spends
  wall-clock on a partition that can preempt you at any moment, and pays it
  again on every resubmission.

`star.sbatch` calls this too, as a fail-fast preflight. Once the cache is warm
it returns in under a second, so running it in both places costs nothing.
"""

from __future__ import annotations

import argparse
import logging
import os

from .config import Config, add_config_args, config_from_args
from .io_utils import setup_logging

log = logging.getLogger(__name__)


def prefetch(cfg: Config) -> str:
    """Pull the generator's tokenizer and weights into the local HF cache."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    name = cfg.train.model_name
    log.info("caching %s into HF_HOME=%s", name, os.environ.get("HF_HOME", "<default>"))
    AutoTokenizer.from_pretrained(name)
    # Loaded and thrown away: the point is the download, and from_pretrained is
    # the only thing guaranteed to fetch exactly the files the run will ask for.
    AutoModelForSeq2SeqLM.from_pretrained(name)
    log.info("cached %s", name)
    return name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args(argv)

    cfg: Config = config_from_args(args)
    setup_logging(cfg.log_level)
    name = prefetch(cfg)
    print(f"model cached: {name}")
    print("the run can now be submitted with no network on the compute node:")
    print("  HF_HUB_OFFLINE=1 scripts/slurm/submit.sh <config>")


if __name__ == "__main__":
    main()
