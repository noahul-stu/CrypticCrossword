#!/bin/bash
#SBATCH --job-name=deberta-wordplay
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=studentkillable
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Runs one of the "without finetuning" DeBERTa notebooks end-to-end on a GPU
# node of the TAU CS Slurm cluster.  Submit from the repository root:
#
#     sbatch run_deberta_slurm_bySize.sbatch small   # microsoft/deberta-v3-small
#     sbatch run_deberta_slurm_bySize.sbatch large   # microsoft/deberta-v3-large
#     sbatch run_deberta_slurm_bySize.sbatch         # defaults to large
#
# The two sizes are independent (separate venvs, separate output notebooks), so
# both can be queued at the same time and may run concurrently.
#
# --time is sized for a COLD run: creating the venv and downloading the CUDA
# torch wheel (~2.5GB) dominates, and costs far more than the model itself. Once
# a size's venv and HF cache are warm, a rerun finishes in minutes:
#   `sbatch --time=00:30:00 run_deberta_slurm_bySize.sbatch small` is then plenty.
#
# Notes from the course guidelines (NLP_course_2025b___project_guidelines.pdf, 2.2):
#   * partition is `studentkillable` -> jobs are preemptible, hence --requeue.
#     The notebooks are pure inference, so a restart from scratch is cheap and safe.
#   * fill in https://www.cs.tau.ac.il/system/SlurmRequestForm before the first run
#     (PI/lab: "NLP class 2025/2026"); cluster docs: https://www.cs.tau.ac.il/system/slurm
#   * venv / HF cache / datasets belong on the course storage, not in $HOME.

set -euo pipefail

# sbatch copies this script to a node-local spool directory before running it,
# so "${BASH_SOURCE[0]}" is /var/spool/slurmd/job*/slurm_script and NOT the repo
# - cd-ing there and mkdir-ing gives "Permission denied". $SLURM_SUBMIT_DIR is
# the directory sbatch was invoked from; fall back to the script's own location
# for local `bash run_deberta_slurm_bySize.sbatch` runs outside Slurm.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_DIR"

# figures/ and outputs/ are only written later, so creating them here is fine.
# slurm_logs/ is different: Slurm opens the --output/--error paths above when it
# launches the job step, i.e. BEFORE this script runs, so this mkdir cannot save
# a missing slurm_logs/ - the job would already have died with "Unable to open
# file". Create it by hand once, or keep a committed slurm_logs/.gitkeep.
mkdir -p slurm_logs figures outputs

# --- model size selection -------------------------------------------------
# First positional argument wins, then $MODEL_SIZE from the environment, then
# the `large` default.
MODEL_SIZE="${1:-${MODEL_SIZE:-large}}"

case "$MODEL_SIZE" in
    small)
        MODEL_NAME="microsoft/deberta-v3-small"
        NOTEBOOK_NAME="DeBERTa_Slurm_small_Wordplay_without_finetuning.ipynb"
        ;;
    large)
        MODEL_NAME="microsoft/deberta-v3-large"
        NOTEBOOK_NAME="DeBERTa_Slurm_large_Wordplay_without_finetuning.ipynb"
        ;;
    *)
        echo "ERROR: unknown model size '$MODEL_SIZE' (expected 'small' or 'large')" >&2
        exit 2
        ;;
esac

# Look for the notebook in the repo layout (code/) and in a flat layout, so the
# script works whether it sits in a clone of the repo or in a plain working
# directory on the cluster. Override with NOTEBOOK=/path/to/nb.ipynb.
if [ -n "${NOTEBOOK:-}" ]; then
    :
else
    for candidate in "code/$NOTEBOOK_NAME" "$NOTEBOOK_NAME" "notebooks/$NOTEBOOK_NAME"; do
        if [ -f "$candidate" ]; then
            NOTEBOOK="$candidate"
            break
        fi
    done
fi

if [ -z "${NOTEBOOK:-}" ] || [ ! -f "$NOTEBOOK" ]; then
    echo "ERROR: could not find $NOTEBOOK_NAME under $PROJECT_DIR" >&2
    echo "       Looked in: code/  ./  notebooks/" >&2
    echo "       Notebooks actually present here:" >&2
    find . -maxdepth 2 -name '*.ipynb' -not -path './outputs/*' 2>/dev/null \
        | sed 's/^/         /' >&2 || true
    echo "       Either submit from the directory holding the notebook, or pass" >&2
    echo "       an explicit path:  NOTEBOOK=/full/path/nb.ipynb sbatch ... $MODEL_SIZE" >&2
    exit 2
fi

OUT_NOTEBOOK="outputs/DeBERTa_${MODEL_SIZE}_Wordplay_${SLURM_JOB_ID:-local}.ipynb"

echo "Model size:   $MODEL_SIZE"
echo "Model name:   $MODEL_NAME"
echo "Notebook:     $NOTEBOOK"
echo "Output:       $OUT_NOTEBOOK"

# --- storage --------------------------------------------------------------
# Persistent (not backed-up) course storage. Falls back to the project dir when
# running somewhere else, e.g. a laptop.
STORAGE_ROOT="${STORAGE_ROOT:-/home/morg/NLP_2526b/$USER}"
if [ ! -d "$STORAGE_ROOT" ]; then
    echo "WARNING: $STORAGE_ROOT not found, keeping venv and caches under $PROJECT_DIR"
    STORAGE_ROOT="$PROJECT_DIR"
fi

export HF_HOME="$STORAGE_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$STORAGE_ROOT/.cache"
export PIP_CACHE_DIR="$STORAGE_ROOT/.cache/pip"
export PREBUILT_DIR="$PROJECT_DIR/cryptic-wordplay-main/prebuilt"
export FIG_DIR="$PROJECT_DIR/figures"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

# cryptic-wordplay-main/ is gitignored (re-downloadable). Without it the
# notebook falls back to raw.githubusercontent.com, which a firewalled compute
# node cannot reach - so say so loudly here rather than failing mid-notebook.
if [ ! -d "$PREBUILT_DIR" ]; then
    echo "WARNING: $PREBUILT_DIR not found."
    echo "         The notebook will try to read the prebuilt jsonl files from GitHub."
    echo "         If the compute node has no outbound HTTP, clone the data first:"
    echo "           git clone --depth 1 https://github.com/mdda/cryptic-wordplay \\"
    echo "               $PROJECT_DIR/cryptic-wordplay-main"
fi

# Both deberta-v3 checkpoints are public and ungated, so no HF token is
# required. A token only raises the download rate limit; drop one in
# ~/.hf_token to use it.
if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi

# --- environment ---------------------------------------------------------
module load cuda 2>/dev/null || true

# The compute nodes' system python3.12 ships without ensurepip (python3.12-venv
# is not installed, and `apt install` needs sudo we do not have), so
# `python3 -m venv` fails there with "ensurepip is not available". Reuse a
# pre-built environment instead of trying to create one.
# Override with VENV=/path/to/venv when submitting.
VENV="${VENV:-$STORAGE_ROOT/venv}"

if [ ! -x "$VENV/bin/python" ]; then
    echo "ERROR: no usable virtualenv at $VENV" >&2
    echo "       This script cannot create one: the compute nodes' python3 has no" >&2
    echo "       ensurepip, so 'python3 -m venv' fails there." >&2
    echo "       Build it once on a node that can, then resubmit; or point at an" >&2
    echo "       existing one:  VENV=/path/to/venv sbatch ... $MODEL_SIZE" >&2
    exit 2
fi

source "$VENV/bin/activate"
echo "Virtualenv:   $VENV"

# SKIP_PIP=1 is the default because $VENV is shared between the small and large
# jobs. Neither job writes to it, so both can run concurrently and safely. The
# notebook's own install cell reads this variable too, for the same reason.
# Set SKIP_PIP=0 to let this job install/upgrade - only do that with one job
# running at a time.
SKIP_PIP="${SKIP_PIP:-1}"
export SKIP_PIP

if [ "$SKIP_PIP" = "1" ]; then
    echo "SKIP_PIP=1  -> using $VENV as-is, installing nothing"
else
    pip install -q -U pip
    pip install -q -U torch transformers datasets sentencepiece \
        pandas numpy matplotlib nbconvert nbclient ipykernel
fi

# Fail fast with the exact pip command to run, rather than dying deep inside
# nbconvert or halfway through the notebook.
python - <<'PY'
import importlib.util as u, sys

needed = ["torch", "transformers", "datasets", "pandas", "numpy",
          "matplotlib", "sentencepiece", "nbconvert", "nbclient", "ipykernel"]
missing = [m for m in needed if u.find_spec(m) is None]

if missing:
    sys.exit(
        "ERROR: this venv is missing: " + " ".join(missing) + "\n"
        "       Install them once, on the login node:\n"
        "         " + sys.prefix + "/bin/pip install -U " + " ".join(missing)
    )
print("Dependencies OK.")
PY

# --- run -----------------------------------------------------------------
nvidia-smi || true

# Each notebook hardcodes its own MODEL_NAME. Check it against the case branch
# above before spending a download on it, so the two can never drift apart and
# silently report the wrong model's numbers.
MODEL_NAME="$MODEL_NAME" NOTEBOOK="$NOTEBOOK" python - <<'PY'
import json, os, re, sys

expected = os.environ["MODEL_NAME"]
nb = json.load(open(os.environ["NOTEBOOK"]))

found = [
    m.group(1)
    for cell in nb["cells"]
    if cell["cell_type"] == "code"
    for m in [re.search(r'^MODEL_NAME\s*=\s*["\'](.+?)["\']', "".join(cell["source"]), re.M)]
    if m
]

if found != [expected]:
    sys.exit(
        f"ERROR: {os.environ['NOTEBOOK']} sets MODEL_NAME={found}, "
        f"but this job selected {expected!r}."
    )
print("MODEL_NAME matches notebook:", expected)
PY

# Download the weights before the notebook starts, so a node without a route to
# huggingface.co fails here with a clear error instead of hanging mid-notebook.
# Once the cache is warm you can add:  export HF_HUB_OFFLINE=1
MODEL_NAME="$MODEL_NAME" python - <<'PY'
import os

from transformers import AutoModel, AutoTokenizer

MODEL_NAME = os.environ["MODEL_NAME"]
AutoTokenizer.from_pretrained(MODEL_NAME)
AutoModel.from_pretrained(MODEL_NAME)
print("Model cached:", MODEL_NAME)
PY

# `python -m nbconvert` rather than `jupyter nbconvert`, so the run does not
# depend on the jupyter_core launcher being installed in the reused venv.
python -m nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output "$PROJECT_DIR/$OUT_NOTEBOOK" \
    "$NOTEBOOK"

echo "Executed notebook written to $OUT_NOTEBOOK"
echo "Figures written to $FIG_DIR"
