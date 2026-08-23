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
# node of the TAU CS Slurm cluster.  Submit from the directory holding the
# notebooks:
#
#     sbatch run_deberta_slurm_bySize.sbatch small   # microsoft/deberta-v3-small
#     sbatch run_deberta_slurm_bySize.sbatch large   # microsoft/deberta-v3-large
#     sbatch run_deberta_slurm_bySize.sbatch         # defaults to large
#
# The two sizes share one venv but never write to it (SKIP_PIP=1) and write
# different output notebooks, so both can be queued at once and run concurrently.
#
# ==========================================================================
# ONE-TIME SETUP - run these on the LOGIN node, not inside a job
# ==========================================================================
# The compute nodes' system python3.12 has no ensurepip, so `python3 -m venv`
# fails there. The venv must already exist; this script only activates it.
#
#   1. Python packages. Everything the notebooks and this script need:
#
#        torch          transformers    datasets      pandas    numpy
#        sentencepiece  protobuf        matplotlib
#        nbconvert      nbclient        ipykernel
#
#      sentencepiece AND protobuf are both required, not just the first:
#      deberta-v3 ships its tokenizer as a SentencePiece spm.model, and
#      transformers needs protobuf to parse it into the fast-tokenizer format.
#      Without protobuf it falls back to a TikToken extractor and dies on a
#      missing tiktoken. nbconvert/nbclient/ipykernel are what execute the
#      notebook; matplotlib draws the section-15 figures.
#
#        cd /home/morg/NLP_2526b/$USER
#        venv/bin/pip install -U torch transformers datasets pandas numpy \
#            sentencepiece protobuf matplotlib nbconvert nbclient ipykernel
#
#      Verify (should print nothing missing):
#        venv/bin/python -c "import torch,transformers,datasets,pandas,numpy,\
#        sentencepiece,google.protobuf,nbconvert,nbclient,ipykernel; print('ok')"
#
#   2. The Wordplay data. Without it the notebook falls back to GitHub at run
#      time; a local copy is faster and works on a firewalled node:
#
#        git clone --depth 1 https://github.com/mdda/cryptic-wordplay \
#            /home/morg/NLP_2526b/$USER/cryptic-wordplay-main
#
#   3. The log directory. Slurm opens --output/--error BEFORE this script runs,
#      so the mkdir below cannot create it in time:
#
#        mkdir -p /home/morg/NLP_2526b/$USER/slurm_logs
#
#   4. Optional - warm the model cache, so the job does not download anything:
#
#        HF_HOME=$PWD/.cache/huggingface venv/bin/python -c "
#        from transformers import AutoModel, AutoTokenizer
#        for m in ['microsoft/deberta-v3-small', 'microsoft/deberta-v3-large']:
#            AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"
#
# This script's dependency preflight checks item 1 and prints the exact pip
# command for anything missing, so a bad environment fails in seconds.
# ==========================================================================
#
# --time is sized for a COLD run, where the HF download dominates (~0.3GB for
# small, ~1.7GB for large). Once the cache is warm a rerun takes minutes:
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

# $HOME on this cluster is small and goes over quota, and several tools write
# there by default - nbconvert died with
#   OSError: [Errno 122] Disk quota exceeded: '~/.ipython'
# because IPython creates its directory before a kernel can start. Redirect
# every one of them onto the course storage.
export IPYTHONDIR="$STORAGE_ROOT/.ipython"                  # IPython  ~/.ipython
export JUPYTER_CONFIG_DIR="$STORAGE_ROOT/.jupyter"          # jupyter  ~/.jupyter
export JUPYTER_DATA_DIR="$STORAGE_ROOT/.jupyter/data"       # kernelspecs
export JUPYTER_RUNTIME_DIR="$STORAGE_ROOT/.jupyter/runtime" # kernel connection files
export MPLCONFIGDIR="$STORAGE_ROOT/.cache/matplotlib"       # font cache
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$IPYTHONDIR" "$JUPYTER_CONFIG_DIR" \
    "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$MPLCONFIGDIR"

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
    pip install -q -U torch transformers datasets sentencepiece protobuf \
        pandas numpy matplotlib nbconvert nbclient ipykernel
fi

# Fail fast with the exact pip command to run, rather than dying deep inside
# nbconvert or halfway through the notebook.
python - <<'PY'
import importlib.util, sys

# module name -> pip name. protobuf is not optional: deberta-v3 ships a
# SentencePiece spm.model, and transformers needs protobuf to convert it to the
# fast tokenizer format. Without it, it falls back to a TikToken extractor and
# dies on a missing tiktoken instead.
needed = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "sentencepiece": "sentencepiece",
    "google.protobuf": "protobuf",
    "nbconvert": "nbconvert",
    "nbclient": "nbclient",
    "ipykernel": "ipykernel",
}


def present(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


missing = sorted({pip for mod, pip in needed.items() if not present(mod)})

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
