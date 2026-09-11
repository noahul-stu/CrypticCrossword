#!/bin/bash
#SBATCH --job-name=cryptic-star
#SBATCH --account=gpu-students
#SBATCH --partition=studentkillable
#SBATCH --output=logs/star-%j.out
#SBATCH --error=logs/star-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#
# Submit through the wrapper, NOT with sbatch directly - Slurm opens the
# --output file above before this script runs, so `logs/` has to exist already
# or the job dies instantly with ExitCode 1:0 and no log files at all:
#
#   scripts/slurm/submit.sh configs/star_t5_large.yaml
#
# studentkillable is preemptible and capped at 1 day, so --requeue lets Slurm
# put the job straight back in the queue instead of leaving it to be noticed by
# hand. That is only safe because the loop resumes: iterations with a complete
# iterK/model/ and a history.json record are skipped, and an unfinished
# iteration reuses its candidates.jsonl instead of decoding it again.
# --open-mode=append goes with it: a requeued job keeps the same job id, and the
# default open mode would truncate the log of the attempt that was killed -
# throwing away the only record of why.
#
# Live progress, while the job runs:
#   tail -f logs/star-<jobid>.out
#   tail -f runs/*/metrics.jsonl          # one JSON event per line
#   less runs/*/iter1/trace_samples.md    # what the model actually reasoned
# and afterwards, from a machine that can reach the node:
#   tensorboard --logdir runs/<name>/tb
#
# NOTE: no `set -e`. A failing line here must still reach the echo/diagnostics
# below, otherwise the job exits silently and the logs say nothing.
set -uo pipefail

CONFIG="${1:-configs/star_t5_base.yaml}"

# sbatch copies this script to a node-local spool directory, so its own path is
# /var/spool/slurmd/... and not the repo. SLURM_SUBMIT_DIR is where sbatch ran.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ---------------------------------------------------------------------------
# Caches, all of them, off $HOME.
#
# The CS cluster home directories have a small quota, and every one of these
# tools writes there by default. Overflowing it does not produce a tidy error at
# the top of the run: it surfaces as `OSError: [Errno 122] Disk quota exceeded`
# from whatever happened to be writing at the time, tens of minutes in.
# STORAGE_ROOT is the course allocation; export it to point somewhere else.
# ---------------------------------------------------------------------------
STORAGE_ROOT="${STORAGE_ROOT:-/home/morg/NLP_2526b/$USER}"
if [[ ! -d "$STORAGE_ROOT" ]]; then
  echo "note: $STORAGE_ROOT not found, keeping caches under $PWD"
  STORAGE_ROOT="$PWD"
fi
export HF_HOME="${HF_HOME:-$STORAGE_ROOT/.cache/huggingface}"   # ~3GB of weights
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STORAGE_ROOT/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$STORAGE_ROOT/.cache/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$STORAGE_ROOT/.cache/matplotlib}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$STORAGE_ROOT/.cache/triton}"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
# Unbuffered, so progress shows up in the .out file live instead of at exit.
export PYTHONUNBUFFERED=1

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cwd=$PWD config=$CONFIG"
echo "restart=${SLURM_RESTART_COUNT:-0} storage=$STORAGE_ROOT hf_home=$HF_HOME"
echo "started $(date)"

module load cuda 2>/dev/null || true

# ---------------------------------------------------------------------------
# Python environment: conda if there is one, otherwise a virtualenv.
#
# The venv path is not a nicety - the compute nodes' system python3.12 ships
# without ensurepip, so `python3 -m venv` fails *there*. A venv has to be built
# on the login node (see README) and is then only activated here.
# ---------------------------------------------------------------------------
CONDA_SH=""
for candidate in \
    "${CONDA_BASE:-}" \
    "$HOME/anaconda3" \
    "$HOME/miniconda3" \
    "/specific/a/home/cc/students/csguests/shanimakori/anaconda3" ; do
  if [[ -n "$candidate" && -f "$candidate/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="$candidate/etc/profile.d/conda.sh"
    break
  fi
done

VENV="${VENV:-$STORAGE_ROOT/venv}"

if [[ -n "$CONDA_SH" ]]; then
  echo "conda: $CONDA_SH"
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "${CONDA_ENV:-nlp_env}" || {
    echo "FATAL: conda activate ${CONDA_ENV:-nlp_env} failed."
    echo "       Fix the env, or point at a virtualenv instead:"
    echo "         VENV=/path/to/venv sbatch scripts/slurm/star.sbatch $CONFIG"
    exit 1
  }
elif [[ -x "$VENV/bin/python" ]]; then
  echo "venv: $VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "FATAL: no python environment. Looked for conda.sh under CONDA_BASE,"
  echo "       ~/anaconda3, ~/miniconda3, and a virtualenv at $VENV."
  echo "       Export CONDA_BASE=/path/to/anaconda3 or VENV=/path/to/venv."
  exit 1
fi
echo "python: $(command -v python)"

# ---------------------------------------------------------------------------
# Preflight. Every check here is one that otherwise fails much later, with a
# worse error, after the run has already spent real time.
# ---------------------------------------------------------------------------
python - <<'PY' || { echo "FATAL: environment check failed"; exit 1; }
import importlib.util
import sys

# module name -> pip name. protobuf is not decorative: flan-t5 ships a
# SentencePiece spiece.model, and when a repo has no prebuilt tokenizer.json
# transformers needs protobuf to convert it. Without protobuf it falls through
# to a TikToken extractor and dies on a missing tiktoken instead.
needed = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "accelerate": "accelerate",          # Trainer refuses to start without it
    "sentencepiece": "sentencepiece",
    "google.protobuf": "protobuf",
    "yaml": "PyYAML",
}
optional = {"tensorboard": "tensorboard"}


def present(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


missing = sorted({pip for mod, pip in needed.items() if not present(mod)})
if missing:
    sys.exit(
        "ERROR: this environment is missing: " + " ".join(missing) + "\n"
        "       Install them on the LOGIN node, not in a job:\n"
        "         " + sys.prefix + "/bin/pip install -U " + " ".join(missing)
    )

for mod, pip in optional.items():
    if not present(mod):
        print(f"note: {pip} absent - metrics.jsonl still written, no TensorBoard curves")

import torch
import transformers

import cryptic_star  # noqa: F401  - proves PYTHONPATH reaches the package

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB",
          "bf16" if torch.cuda.is_bf16_supported() else "no-bf16")
else:
    print("WARNING: no GPU visible - the run would fall back to CPU and never finish")
PY

nvidia-smi || true

# Fetch the weights now, so a node with no route to huggingface.co fails here in
# seconds with a clear message instead of hanging inside iteration 0. Instant
# once the cache is warm; warm it on the login node with the same command.
python -m cryptic_star.prefetch --config configs/base.yaml --config "$CONFIG" \
  || { echo "FATAL: could not fetch the model. On the login node, run:"
       echo "         python -m cryptic_star.prefetch -c configs/base.yaml -c $CONFIG"
       exit 1; }

python -m cryptic_star.star_loop --config configs/base.yaml --config "$CONFIG"
status=$?
echo "star_loop exit=$status at $(date)"
if [[ $status -ne 0 ]]; then
  echo "the run's own event log, if it got that far:"
  tail -n 20 runs/*/metrics.jsonl 2>/dev/null || echo "  (none written)"
fi
exit $status
