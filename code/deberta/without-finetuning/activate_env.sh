#!/bin/bash
# ============================================================================
# activate_env.sh - environment only. No installs, no downloads.
#
# SOURCE this (do not execute it) whenever you want to work interactively on the
# cluster: run python by hand, poke at the notebook, warm a cache, or submit a
# job from a shell with the right variables already set.
#
#     cd /home/morg/NLP_2526b/$USER
#     source activate_env.sh
#
# For a first-time machine, run ./setup_env.sh instead - it does everything this
# does, plus the installs and downloads.
#
# run_deberta_slurm_bySize.sbatch sets all of these itself, so you do NOT need to
# source this before sbatch. It is purely for interactive use.
# ============================================================================

# Deliberately no `set -e`: this file is sourced into your interactive shell, and
# a failing command must not kill that shell.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "ERROR: activate_env.sh must be sourced, not executed:" >&2
    echo "         source activate_env.sh" >&2
    exit 1
fi

# --- where everything lives -------------------------------------------------
# Course storage: persistent, not backed up, and far larger than $HOME.
export NLPDIR="${NLPDIR:-/home/morg/NLP_2526b/$USER}"
export VENV="${VENV:-$NLPDIR/venv}"

# --- keep every cache off $HOME ---------------------------------------------
# $HOME is small and goes over quota. Anything that defaults to writing there
# will eventually fail with "OSError: [Errno 122] Disk quota exceeded" - that is
# what killed nbconvert before IPYTHONDIR was set.
export HF_HOME="$NLPDIR/.cache/huggingface"                # model weights
export XDG_CACHE_HOME="$NLPDIR/.cache"
export PIP_CACHE_DIR="$NLPDIR/.cache/pip"
export IPYTHONDIR="$NLPDIR/.ipython"                       # ~/.ipython
export JUPYTER_CONFIG_DIR="$NLPDIR/.jupyter"               # ~/.jupyter
export JUPYTER_DATA_DIR="$NLPDIR/.jupyter/data"            # kernelspecs
export JUPYTER_RUNTIME_DIR="$NLPDIR/.jupyter/runtime"      # kernel conn files
export MPLCONFIGDIR="$NLPDIR/.cache/matplotlib"            # font cache

# --- data + misc ------------------------------------------------------------
# The notebooks read the prebuilt Wordplay jsonl files from here, and fall back
# to raw.githubusercontent.com when the directory is missing.
export PREBUILT_DIR="$NLPDIR/cryptic-wordplay-main/prebuilt"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1

# A token is optional; it only raises the HF download rate limit.
if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$IPYTHONDIR" "$JUPYTER_CONFIG_DIR" \
    "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$MPLCONFIGDIR" \
    "$NLPDIR/slurm_logs" "$NLPDIR/outputs"

# --- GPU toolchain ----------------------------------------------------------
module load cuda 2>/dev/null || true

# --- venv -------------------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
    echo "WARNING: no virtualenv at $VENV - run ./setup_env.sh first." >&2
else
    source "$VENV/bin/activate"
fi

echo "NLPDIR:  $NLPDIR"
echo "venv:    ${VIRTUAL_ENV:-<none>}"
echo "HF_HOME: $HF_HOME"
if [ ! -d "$PREBUILT_DIR" ]; then
    echo "WARNING: $PREBUILT_DIR missing - run ./setup_env.sh to clone it." >&2
fi
