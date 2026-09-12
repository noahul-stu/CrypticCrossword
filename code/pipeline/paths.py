"""Where things live, and how the config's `{placeholders}` resolve.

Cluster-first with a local fallback, matching the convention already used by
`code/deberta/without-finetuning/run_deberta_slurm_bySize.sbatch`: real runs
happen under the course storage at /home/morg/NLP_2526b/$USER, but the same
command has to work on a laptop for a one-clue smoke test.

Every cache is redirected off $HOME. That is not decoration -- $HOME on the TAU
CS cluster is quota-limited and nbconvert has already died there with
"OSError: [Errno 122] Disk quota exceeded: '~/.ipython'".

IMPORTANT: import this module (or call `setup_environment()`) BEFORE importing
transformers / datasets / huggingface_hub. They read HF_HOME at import time to
resolve cache paths, so setting it afterwards silently has no effect.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- repo layout
# paths.py lives at <repo>/code/pipeline/paths.py, so the repo root is 2 up.
REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PIPELINE_DIR / "config"
DATASET_DIR = REPO_ROOT / "united-cryptonite-wordplay-dataset"

CLUSTER_STORAGE = Path(f"/home/morg/NLP_2526b/{os.environ.get('USER', 'nobody')}")


def storage_root() -> Path:
    """Course storage when it exists, else the repo (laptop / Colab).

    Override with STORAGE_ROOT=/some/path to point at a different scratch area.
    """
    override = os.environ.get("STORAGE_ROOT")
    if override:
        return Path(override)
    if CLUSTER_STORAGE.is_dir():
        return CLUSTER_STORAGE
    return REPO_ROOT


def on_cluster() -> bool:
    return storage_root() == CLUSTER_STORAGE


def runs_root() -> Path:
    """Parent of the per-run output directories.

    Kept in the repo tree (runs/ is gitignored) even on the cluster, because run
    artifacts are small -- JSONL, CSV, PNG -- and you want them next to the code
    that produced them. Only model weights and HF caches go to storage.
    """
    return Path(os.environ.get("RUNS_ROOT", REPO_ROOT / "runs"))


def setup_environment() -> None:
    """Redirect every cache off $HOME. Idempotent; safe to call more than once.

    Uses setdefault throughout so an sbatch script that already exported these
    (run_pipeline.sbatch does) always wins over these defaults.
    """
    root = storage_root()
    cache = root / ".cache"

    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("PIP_CACHE_DIR", str(cache / "pip"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("IPYTHONDIR", str(root / ".ipython"))
    # A dataloader worker per tokenizer thread deadlocks on fork; the warning
    # transformers prints about it is noise in a Slurm log.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    for key in ("HF_HOME", "PIP_CACHE_DIR", "MPLCONFIGDIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def substitutions() -> dict[str, str]:
    """The `{placeholders}` a config file may use in any path-valued field."""
    return {
        "repo": str(REPO_ROOT),
        "storage": str(storage_root()),
        "dataset": str(DATASET_DIR),
        "user": os.environ.get("USER", "nobody"),
    }


def expand(value: str) -> str:
    """Expand `{repo}` / `{storage}` / `{dataset}` / `{user}` plus $VARS and ~."""
    return os.path.expanduser(os.path.expandvars(value.format(**substitutions())))


def resolve_model_path(spec: str) -> str:
    """Turn a config `model` field into something from_pretrained accepts.

    A spec may be a Hugging Face hub id ("microsoft/deberta-v3-small"), an
    absolute path, or a `;`-separated preference list of candidate local paths
    with a hub id last:

        "{storage}/output_model;google/flan-t5-large"

    The first entry that exists on disk wins; if none do, the last entry is
    returned as-is and left for the hub to resolve. This is what lets one config
    file serve both the cluster (where the fine-tuned checkpoint exists) and a
    laptop (where it does not, and the hub baseline is the sane fallback).
    """
    candidates = [expand(part.strip()) for part in spec.split(";") if part.strip()]
    if not candidates:
        raise ValueError(f"empty model spec: {spec!r}")

    for candidate in candidates:
        if Path(candidate).is_dir():
            return candidate

    return candidates[-1]
