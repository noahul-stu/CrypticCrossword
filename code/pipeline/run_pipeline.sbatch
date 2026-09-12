#!/bin/bash
#SBATCH --job-name=cc-pipeline
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=studentkillable
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#
# Runs the candidate-generate -> score -> select pipeline on a GPU node.
# Submit from the repository root:
#
#   sbatch code/pipeline/run_pipeline.sbatch --split test --limit 1000
#   sbatch code/pipeline/run_pipeline.sbatch --split test --limit 10000 --shard 0/4
#
# Every argument is forwarded verbatim to run_pipeline.py, so anything the CLI
# accepts works here. See `python code/pipeline/run_pipeline.py --help`.
#
# ---------------------------------------------------------------------------
# ONE-TIME SETUP, on the LOGIN node (the compute nodes' python3.12 has no
# ensurepip, so `python3 -m venv` fails there):
#
#   cd /home/morg/NLP_2526b/$USER
#   ./setup_env.sh                      # from code/deberta/without-finetuning/
#   mkdir -p slurm_logs                 # Slurm opens the log paths BEFORE this
#                                       # script runs, so a mkdir here is too late
#
# Then warm both checkpoints so a job never spends its wall clock downloading:
#
#   source code/deberta/without-finetuning/activate_env.sh
#   python - <<'PY'
#   import os
#   from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#   dest = os.environ["NLPDIR"] + "/models/cryptonite-flant5-large"
#   src = "hugTAU/cryptonite-flant5-large"
#   AutoModelForSeq2SeqLM.from_pretrained(src).save_pretrained(dest)
#   AutoTokenizer.from_pretrained(src).save_pretrained(dest)
#   PY
#
# The fine-tuned scorer (deberta-v3-large) is expected at either
#   $NLPDIR/models/trained_deberta
#   <repo>/code/deberta/with-finetuning/trained_deberta
# There is deliberately NO fallback to a base deberta-v3: a base model loads fine
# but its classification head is random, so its scores would be noise. A missing
# checkpoint fails loudly instead.
#
# Sanity-check the scorer before spending GPU-hours on a full run -- this needs no
# generator and takes under a minute:
#
#   python code/pipeline/run_pipeline.py --verify-scorer 200
#
# ---------------------------------------------------------------------------
# Sizing --time: generation dominates. flan-t5-large with 12 candidates (beam +
# sampling union) runs at roughly 1-3 clues/s on one card, so ~1000 clues is well
# under an hour and 10,000 is a few hours. `studentkillable` is preemptible, hence
# --requeue: the pipeline appends each finished clue to records.jsonl and resumes
# from there, so a requeued job continues rather than restarting. For 10k, prefer
# 4 shards in parallel over one long job.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p slurm_logs runs

STORAGE_ROOT="${STORAGE_ROOT:-/home/morg/NLP_2526b/$USER}"
if [ ! -d "$STORAGE_ROOT" ]; then
    echo "WARNING: $STORAGE_ROOT not found; keeping venv and caches under $PROJECT_DIR"
    STORAGE_ROOT="$PROJECT_DIR"
fi
export STORAGE_ROOT

# $HOME on this cluster is quota-limited and several tools write there by
# default; nbconvert has already died with
#   OSError: [Errno 122] Disk quota exceeded: '~/.ipython'
# paths.setup_environment() sets these too, but exporting here means they are
# also right for anything this script runs before Python starts.
export HF_HOME="$STORAGE_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$STORAGE_ROOT/.cache"
export PIP_CACHE_DIR="$STORAGE_ROOT/.cache/pip"
export MPLCONFIGDIR="$STORAGE_ROOT/.cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

module load cuda 2>/dev/null || true

VENV="${VENV:-$STORAGE_ROOT/venv}"
if [ ! -x "$VENV/bin/python" ]; then
    echo "ERROR: no usable virtualenv at $VENV" >&2
    echo "       Build it on the login node first: code/deberta/without-finetuning/setup_env.sh" >&2
    echo "       or point at an existing one:  VENV=/path/to/venv sbatch ..." >&2
    exit 2
fi
source "$VENV/bin/activate"

echo "Project:    $PROJECT_DIR"
echo "Storage:    $STORAGE_ROOT"
echo "Virtualenv: $VENV"
nvidia-smi || true

# Fail in seconds with the exact fix, rather than deep inside model loading.
python - <<'PY'
import importlib.util, sys

needed = {
    "torch": "torch",
    "transformers": "transformers",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "sentencepiece": "sentencepiece",   # deberta-v3 ships a SentencePiece spm.model
    "google.protobuf": "protobuf",      # ...and transformers needs protobuf to parse it
}
missing = sorted({pip for mod, pip in needed.items() if importlib.util.find_spec(mod) is None})
if missing:
    sys.exit("ERROR: venv missing: " + " ".join(missing) +
             "\n       " + sys.prefix + "/bin/pip install -U " + " ".join(missing))
print("Dependencies OK.")
PY

srun python code/pipeline/run_pipeline.py "$@"
