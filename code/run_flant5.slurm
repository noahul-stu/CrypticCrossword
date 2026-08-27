#!/bin/bash
#SBATCH --job-name=cryptonite_t5
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output="/home/morg/NLP_2526b/hullernoa/logs/%x_%j.out"
#SBATCH --error="/home/morg/NLP_2526b/hullernoa/logs/%x_%j.err"
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hullernoa@mail.tau.ac.il

STORAGE_DIR="/home/morg/NLP_2526b/hullernoa"

# Route caches so you don't hit quotas
export HF_HOME="${STORAGE_DIR}/cache/huggingface"
export HF_DATASETS_CACHE="${STORAGE_DIR}/cache/huggingface/datasets"
export XDG_CACHE_HOME="${STORAGE_DIR}/cache/xdg"
export TORCH_HOME="${STORAGE_DIR}/cache/torch"
export PIP_CACHE_DIR="${STORAGE_DIR}/cache/pip"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate venv
source "${STORAGE_DIR}/venv/bin/activate"

# Run the script
cd "${STORAGE_DIR}/code"
srun python train_cryptonite.py