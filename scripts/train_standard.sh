#!/bin/bash
#SBATCH --job-name=skill_jepa_standard
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=96:00:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

cd /home/tcouso/Skill-JEPA || exit 1

echo "[$(date)] JOB_ID: $SLURM_JOB_ID | NODE: $SLURM_NODELIST"
echo "[$(date)] Allocated GPUs: $CUDA_VISIBLE_DEVICES"

export PYTHONUNBUFFERED=1

export XDG_CACHE_HOME="/workspace1/tcouso/.cache"
export HF_HOME="/workspace1/tcouso/.cache/huggingface"
export TORCH_HOME="/workspace1/tcouso/.cache/torch"
export PIP_CACHE_DIR="/workspace1/tcouso/.cache/pip"

export WANDB_DIR="/workspace1/tcouso/wandb_logs"
export WANDB_CACHE_DIR="/workspace1/tcouso/.cache/wandb"
export WANDB_CONFIG_DIR="/workspace1/tcouso/.config/wandb"
export WANDB_DATA_DIR="/workspace1/tcouso/.local/share/wandb"

export WANDB_PROJECT="skill-jepa"
export WANDB_MODE="online"

mkdir -p $XDG_CACHE_HOME $HF_HOME $TORCH_HOME $PIP_CACHE_DIR
mkdir -p $WANDB_DIR $WANDB_CACHE_DIR $WANDB_CONFIG_DIR $WANDB_DATA_DIR
mkdir -p logs

set -a
source .env
set +a

uv run python -c "import torch; print(f'Torch CUDA available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

srun uv run python -u src/train.py \
    predictor.mode=standard \
    dataset.env_name=lewm-tworooms \
    dataset.cache_dir=/workspace1/tcouso/skill-jepa/datasets
