#!/usr/bin/env bash
# ============================================================================
# One-time setup, on the cluster LOGIN node. Run this once, then submit jobs.
#
#     cd <this repo>
#     scripts/slurm/setup.sh
#
# Safe to re-run: it reuses whatever already exists and only fills in the gaps.
#
# Everything here has to happen on the login node and cannot happen inside a
# job, which is why it is a separate script:
#
#   * pip needs outbound network, and compute nodes may have none.
#   * `python3 -m venv` fails on the compute nodes - their system python3.12
#     ships without ensurepip - so a virtualenv can only be built here.
#   * the model weights (~3GB for flan-t5-large) must be in the cache before a
#     preemptible job starts, or the download is paid again on every restart.
#   * logs/ must exist before sbatch runs, because Slurm opens the job's
#     --output file before the job script does anything. A missing logs/ kills
#     the job with ExitCode 1:0 and writes no log at all.
#
# Override any of these:
#   STORAGE_ROOT=/path   where caches and the venv live (default: course storage)
#   CONDA_ENV=name       conda environment to use if conda is present
#   CONFIG=configs/x     which config's model to pre-download
# ============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/star_t5_base.yaml}"

fail() { echo; echo "SETUP FAILED: $*" >&2; exit 1; }

echo "repo:   $ROOT"

# --- 1. where the big files go ----------------------------------------------
# $HOME on the CS cluster has a quota too small for a 3GB model cache. Anything
# that defaults to writing there eventually dies with
# "OSError: [Errno 122] Disk quota exceeded", tens of minutes into a run.
STORAGE_ROOT="${STORAGE_ROOT:-/home/morg/NLP_2526b/$USER}"
if [[ ! -d "$STORAGE_ROOT" ]]; then
  echo "note:   $STORAGE_ROOT does not exist, using $ROOT instead."
  echo "        If your course allocation is elsewhere, re-run as:"
  echo "          STORAGE_ROOT=/path/to/allocation scripts/slurm/setup.sh"
  STORAGE_ROOT="$ROOT"
fi
export HF_HOME="${HF_HOME:-$STORAGE_ROOT/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STORAGE_ROOT/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$STORAGE_ROOT/.cache/pip}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" logs || fail "could not create cache directories"
echo "storage: $STORAGE_ROOT"

# --- 2. python environment ---------------------------------------------------
# Prefer an existing conda environment, since that is what the sbatch looks for
# first; otherwise build a virtualenv here, where venv creation still works.
CONDA_SH=""
for candidate in "${CONDA_BASE:-}" "$HOME/anaconda3" "$HOME/miniconda3"; do
  if [[ -n "$candidate" && -f "$candidate/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="$candidate/etc/profile.d/conda.sh"
    break
  fi
done

VENV="${VENV:-$STORAGE_ROOT/venv}"

if [[ -n "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  CONDA_ENV="${CONDA_ENV:-nlp_env}"
  if conda activate "$CONDA_ENV" 2>/dev/null; then
    echo "python: conda env $CONDA_ENV"
  else
    echo "conda found but env '$CONDA_ENV' does not exist - creating it"
    conda create -y -n "$CONDA_ENV" python=3.11 || fail "conda create failed"
    conda activate "$CONDA_ENV" || fail "conda activate $CONDA_ENV failed"
    echo "python: conda env $CONDA_ENV (new)"
  fi
elif [[ -x "$VENV/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate" || fail "could not activate $VENV"
  echo "python: venv $VENV (existing)"
else
  echo "no conda and no venv - creating one at $VENV"
  # --without-pip plus get-pip.py, because the cluster python has no ensurepip.
  python3 -m venv --without-pip "$VENV" || fail "python3 -m venv failed at $VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate" || fail "could not activate the new $VENV"
  curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip-$$.py \
    || fail "could not download get-pip.py (no network on this node?)"
  python /tmp/get-pip-$$.py || fail "pip bootstrap failed"
  rm -f /tmp/get-pip-$$.py
  echo "python: venv $VENV (new)"
  echo
  echo "IMPORTANT: this venv is outside the repo, so tell the job where it is:"
  echo "  VENV=$VENV scripts/slurm/submit.sh $CONFIG"
  echo "(only needed if you have no conda; export VENV in your shell to persist)"
fi
echo "        $(command -v python)  ($(python --version 2>&1))"

# --- 3. dependencies ---------------------------------------------------------
echo
echo "installing requirements (this is the slow part - torch is ~2.5GB)"
python -m pip install --upgrade pip || fail "pip self-upgrade failed"
python -m pip install -r requirements.txt || fail "pip install -r requirements.txt failed"

# --- 4. verify ---------------------------------------------------------------
# The same check star.sbatch runs, so a green result here means a submitted job
# will not die on a missing dependency. cuda_available is expected to be False
# on a login node - it has no GPU.
echo
echo "verifying"
PYTHONPATH="$ROOT/src" python - <<'PY' || fail "dependency check failed - see above"
import importlib.util
import sys

needed = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "accelerate": "accelerate",
    "sentencepiece": "sentencepiece",
    "google.protobuf": "protobuf",
    "yaml": "PyYAML",
    "tensorboard": "tensorboard",
}


def present(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


missing = sorted({pip for mod, pip in needed.items() if not present(mod)})
if missing:
    sys.exit("  MISSING: " + " ".join(missing))

import torch
import transformers

import cryptic_star  # noqa: F401  - proves PYTHONPATH reaches the package

print(f"  torch        {torch.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  cryptic_star imports OK")
print(f"  cuda_available {torch.cuda.is_available()}  (False on a login node is normal)")
PY

# --- 5. the dataset ----------------------------------------------------------
# Not downloaded automatically - it is not ours to fetch. Say exactly what to do
# rather than letting the first job fail on a missing file.
echo
DATA_DIR="data/raw/united-cryptonite-wordplay-dataset"
if [[ -d "$DATA_DIR" ]]; then
  echo "dataset: $DATA_DIR present"
else
  echo "dataset: MISSING at $DATA_DIR"
  echo "         Copy it from your laptop, then re-run this script:"
  echo "           scp -r $DATA_DIR <user>@<login-host>:$ROOT/data/raw/"
  echo "         or install it from the CrypticCrossword zip:"
  echo "           scripts/install_united_dataset.sh ~/CrypticCrossword-main.zip"
  fail "no dataset - nothing can run without it"
fi

# --- 6. warm the model cache -------------------------------------------------
echo
echo "pre-downloading models into $HF_HOME"
# Both the smoke model and the real one, since step 1 below runs the smoke test
# first and a firewalled compute node cannot fetch either of them itself.
for c in configs/smoke.yaml "$CONFIG"; do
  echo "  $c"
  PYTHONPATH="$ROOT/src" python -m cryptic_star.prefetch \
    --config configs/base.yaml --config "$c" \
    || fail "could not download the model for $c (no network on this node?)"
done

# --- done --------------------------------------------------------------------
cat <<EOF

============================================================================
Setup finished. Nothing above needs doing again.

Next, in order:

  1. A smoke test - a few minutes, proves the whole pipeline runs on a GPU.
     Its accuracy numbers are meaningless by design.

       scripts/slurm/submit.sh configs/smoke.yaml

     Wait for it, then check it got to the end:

       squeue --me
       tail -n 30 logs/star-<jobid>.out
       cat runs/smoke/summary.json

  2. The real run, once the smoke test is green:

       scripts/slurm/submit.sh $CONFIG

     Watch it:
       tail -f runs/*/metrics.jsonl              # live metrics, one JSON per line
       less runs/*/iter1/trace_samples.md        # what the model actually reasoned

  If it gets preempted, Slurm requeues it and the loop continues where it
  stopped. Re-submitting the same command by hand does the same thing.
============================================================================
EOF
