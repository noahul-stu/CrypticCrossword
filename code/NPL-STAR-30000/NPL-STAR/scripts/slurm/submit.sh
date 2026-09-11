#!/usr/bin/env bash
# Submit the STaR loop to Slurm.
#
#   scripts/slurm/submit.sh                          # configs/star_t5_base.yaml
#   scripts/slurm/submit.sh configs/star_t5_large.yaml
#
# Use this instead of calling `sbatch scripts/slurm/star.sbatch` yourself.
# star.sbatch declares `--output=logs/star-%j.out`, and Slurm opens that file
# *before* the job script runs: if `logs/` does not exist the job is killed
# immediately with ExitCode 1:0 and writes no .out/.err at all, which looks
# exactly like a silent crash. Creating the directory here, on the submit host,
# is the only place it can be done in time.
set -euo pipefail

CONFIG="${1:-configs/star_t5_base.yaml}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

cd "$ROOT"

if [[ ! -f "$CONFIG" ]]; then
  echo "no such config: $CONFIG" >&2
  echo "available:" >&2
  ls configs/*.yaml >&2
  exit 2
fi

mkdir -p logs

# star.sbatch asks for 24h/48G, which is right for a multi-iteration
# flan-t5-large run and badly wrong for a smoke test - on a shared partition a
# request that large can queue for hours before a 5-minute job gets to run.
# Anything else goes through SBATCH_ARGS, e.g.
#   SBATCH_ARGS="--time=04:00:00 --gres=gpu:a100:1" scripts/slurm/submit.sh ...
EXTRA=(${SBATCH_ARGS:-})
if [[ "$(basename "$CONFIG")" == "smoke.yaml" && -z "${SBATCH_ARGS:-}" ]]; then
  EXTRA=(--time=01:00:00 --mem=16G --job-name=cryptic-smoke)
  echo "smoke config: requesting 1h/16G instead of 24h/48G so it queues quickly"
fi

# The compute node may have no route to huggingface.co, and flan-t5-large is
# ~3GB. Warming the cache here - on the login node, which does have network -
# means the job starts computing immediately, and does not re-download after
# every preemption. Only a warning: a node with network handles it fine.
STORAGE_ROOT="${STORAGE_ROOT:-/home/morg/NLP_2526b/$USER}"
[[ -d "$STORAGE_ROOT" ]] || STORAGE_ROOT="$ROOT"
HF_CACHE="${HF_HOME:-$STORAGE_ROOT/.cache/huggingface}"
if [[ ! -d "$HF_CACHE/hub" ]]; then
  echo "note: no warm model cache at $HF_CACHE"
  echo "      if the compute node has no outbound network, run this first:"
  echo "        python -m cryptic_star.prefetch -c configs/base.yaml -c $CONFIG"
  echo
fi

# `${EXTRA[@]+...}` rather than a plain "${EXTRA[@]}": expanding an *empty*
# array under `set -u` is an unbound-variable error on bash 3.2, and the login
# node's bash version is not something to bet the submission on.
JOB="$(sbatch --parsable ${EXTRA[@]+"${EXTRA[@]}"} scripts/slurm/star.sbatch "$CONFIG")"
echo "submitted job $JOB  (config=$CONFIG)"
echo
echo "watch it:"
echo "  squeue --me"
echo "  tail -f logs/star-$JOB.out"
echo "  tail -f logs/star-$JOB.err"
echo "  sacct -l -j $JOB          # why it died, if it did"
