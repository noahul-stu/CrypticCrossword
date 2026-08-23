#!/bin/bash
# ============================================================================
# setup_env.sh - ONE-TIME first-run setup on the TAU CS cluster.
#
# Run on the LOGIN node (it needs the network, and the compute nodes cannot
# create a venv - see below):
#
#     cd /home/morg/NLP_2526b/$USER
#     ./setup_env.sh
#
# Safe to re-run: it reuses an existing venv and skips work already done.
# To rebuild the venv from scratch (slow - torch is ~2.5GB):
#
#     RECREATE_VENV=1 ./setup_env.sh
#
# After this, day-to-day work only needs `source activate_env.sh`, and jobs are
# submitted with run_deberta_slurm_bySize.sbatch (which sets its own env).
# ============================================================================

set -euo pipefail

export NLPDIR="${NLPDIR:-/home/morg/NLP_2526b/$USER}"
export VENV="${VENV:-$NLPDIR/venv}"
RECREATE_VENV="${RECREATE_VENV:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$NLPDIR"
cd "$NLPDIR"
echo "==> Working in $NLPDIR"

# --- 1. virtualenv ----------------------------------------------------------
# The cluster's system python3.12 ships without ensurepip (python3.12-venv is
# not installed and `apt install` needs sudo we do not have), so a plain
# `python3 -m venv` fails with "ensurepip is not available". Create the venv
# --without-pip and bootstrap pip from get-pip.py instead.
if [ "$RECREATE_VENV" = "1" ]; then
    echo "==> RECREATE_VENV=1: deleting $VENV"
    rm -rf "$VENV"
fi

if [ -x "$VENV/bin/python" ]; then
    echo "==> Reusing existing venv at $VENV"
else
    echo "==> Creating venv at $VENV (without pip, then bootstrapping pip)"
    python3 -m venv --without-pip "$VENV"
    source "$VENV/bin/activate"
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip-$$.py
    python /tmp/get-pip-$$.py
    rm -f /tmp/get-pip-$$.py
fi

# --- 2. environment ---------------------------------------------------------
# Sets NLPDIR/HF_HOME/IPYTHONDIR/JUPYTER_*/MPLCONFIGDIR/PREBUILT_DIR, creates
# the directories, loads cuda, activates the venv. Single source of truth, so
# this script and interactive shells cannot drift apart.
echo "==> Loading environment from activate_env.sh"
source "$SCRIPT_DIR/activate_env.sh"

# --- 3. python packages -----------------------------------------------------
# Why each of the less obvious ones is here:
#   sentencepiece  deberta-v3 ships its tokenizer as a SentencePiece spm.model
#   protobuf       needed to PARSE that spm.model into the fast-tokenizer
#                  format. Without it transformers falls back to a TikToken
#                  extractor and dies on a missing tiktoken - this is a real
#                  failure, not a warning.
#   nbconvert      executes the notebook in batch
#   nbclient       the execution engine nbconvert drives
#   ipykernel      provides the python3 kernelspec nbclient starts
#   matplotlib     the section-15 figures
#   transformers>=5  the notebooks pass `dtype=` to from_pretrained, which v4
#                  spells `torch_dtype=`
echo "==> Installing python packages"
pip install --upgrade pip
pip install -U \
    torch \
    "transformers>=5" \
    accelerate \
    datasets \
    huggingface_hub \
    pandas \
    numpy \
    sentencepiece \
    protobuf \
    matplotlib \
    nbconvert \
    nbclient \
    ipykernel

# --- 4. Wordplay data -------------------------------------------------------
# Without this the notebooks read the prebuilt jsonl files over HTTP at run
# time, which is slower and fails on a node with no outbound route.
if [ -d "$NLPDIR/cryptic-wordplay-main" ]; then
    echo "==> Wordplay data already present"
else
    echo "==> Cloning Wordplay data"
    git clone --depth 1 https://github.com/mdda/cryptic-wordplay \
        "$NLPDIR/cryptic-wordplay-main"
fi

# --- 5. warm the model cache ------------------------------------------------
# Download both checkpoints now, on a node that definitely has network, so the
# jobs spend their wall clock computing rather than downloading.
echo "==> Caching model weights into $HF_HOME"
python - <<'PY'
from transformers import AutoModel, AutoTokenizer

for name in ("microsoft/deberta-v3-small", "microsoft/deberta-v3-large"):
    AutoTokenizer.from_pretrained(name)
    AutoModel.from_pretrained(name)
    print("  cached:", name)
PY

# --- 6. verify --------------------------------------------------------------
# Same check run_deberta_slurm_bySize.sbatch performs, so a green run here means
# a submitted job will not fail on a missing dependency.
echo "==> Verifying dependencies"
python - <<'PY'
import importlib.util
import sys

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
    sys.exit("  MISSING: " + " ".join(missing))

import torch
import transformers

print(f"  torch        {torch.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  cuda visible {torch.cuda.is_available()}  (False on a login node is expected)")
PY

cat <<EOF

==> Setup completed successfully.

Next:
  cd $NLPDIR
  sbatch run_deberta_slurm_bySize.sbatch small
  sbatch run_deberta_slurm_bySize.sbatch large

For interactive work later, you only need:
  source activate_env.sh
EOF
