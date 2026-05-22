#!/bin/bash
#SBATCH --job-name=lewm_latent_train
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=log/lewm_latent_train_%j.log
#SBATCH --error=log/lewm_latent_train_%j.err

set -euo pipefail

echo "=== SLURM latent planner training started ==="
echo "Node: $(hostname)"
echo "GPUs allocated: ${SLURM_GPUS:-${CUDA_VISIBLE_DEVICES:-unknown}}"

# ---------------------------
# Load CUDA, GCC, and Conda
# ---------------------------
module purge
module load gcc/13.4.0
module load cuda/13.0.0
module load conda

# ---------------------------
# Initialize Conda
# ---------------------------
conda activate "${CONDA_ENV:-lewm}"

# ---------------------------
# Cache directories
# ---------------------------
export HF_HOME=/gpfs/projects/embodied3d/hf-cache
export HF_HUB_CACHE=/gpfs/projects/embodied3d/hf-cache
export TRANSFORMERS_CACHE=/gpfs/projects/embodied3d/hf-cache
export HF_DATASETS_CACHE=/gpfs/projects/embodied3d/hf-cache
export XDG_CACHE_HOME=/gpfs/projects/embodied3d/hf-cache
export WANDB_DIR=/gpfs/projects/embodied3d/hf-cache/wandb
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# stable-worldmodel stores datasets/checkpoints here.
export STABLEWM_HOME="${STABLEWM_HOME:-/gpfs/scrubbed/hwhuang/stable-wm}"

echo "Using python: $(which python)"
echo "Using pip: $(which pip)"
echo "CUDA: $(which nvcc)"
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available in Torch: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "STABLEWM_HOME: ${STABLEWM_HOME}"

# ---------------------------
# Run latent planner training
# ---------------------------
cd /gpfs/projects/embodied3d/hsiangwei/le-wm
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Override these at submit time if needed, e.g.:
#   sbatch --export=ALL,TASK=pusht,RUN_NAME=pusht_h10_ep10 train_latent_planner.slurm
#   sbatch --export=ALL,TASK=reacher,RUN_NAME=reacher_h10_ep10 train_latent_planner.slurm
#   sbatch --export=ALL,TASK=cube,RUN_NAME=cube_h10_ep10 train_latent_planner.slurm
TASK="${TASK:-tworoom}"
case "${TASK}" in
  tworoom)
    DEFAULT_LEWM_CHECKPOINT="tworoom/lewm"
    DEFAULT_DATASET_NAME="tworoom"
    DEFAULT_KEYS_TO_LOAD="[pixels,action,proprio]"
    DEFAULT_KEYS_TO_CACHE="[action,proprio]"
    ;;
  pusht)
    DEFAULT_LEWM_CHECKPOINT="pusht/lewm"
    DEFAULT_DATASET_NAME="pusht_expert_train"
    DEFAULT_KEYS_TO_LOAD="[pixels,action,proprio,state]"
    DEFAULT_KEYS_TO_CACHE="[action,proprio,state]"
    ;;
  reacher)
    DEFAULT_LEWM_CHECKPOINT="reacher/lewm"
    DEFAULT_DATASET_NAME="reacher"
    DEFAULT_KEYS_TO_LOAD="[pixels,action,observation]"
    DEFAULT_KEYS_TO_CACHE="[action,observation]"
    ;;
  cube)
    DEFAULT_LEWM_CHECKPOINT="cube/lewm"
    DEFAULT_DATASET_NAME="ogbench/cube_single_expert"
    DEFAULT_KEYS_TO_LOAD="[pixels,action,observation]"
    DEFAULT_KEYS_TO_CACHE="[action,observation]"
    ;;
  *)
    echo "Unknown TASK=${TASK}. Expected one of: tworoom, pusht, reacher, cube." >&2
    exit 2
    ;;
esac

LEWM_CHECKPOINT="${LEWM_CHECKPOINT:-${DEFAULT_LEWM_CHECKPOINT}}"
DATASET_NAME="${DATASET_NAME:-${DEFAULT_DATASET_NAME}}"
KEYS_TO_LOAD="${KEYS_TO_LOAD:-${DEFAULT_KEYS_TO_LOAD}}"
KEYS_TO_CACHE="${KEYS_TO_CACHE:-${DEFAULT_KEYS_TO_CACHE}}"
HORIZON="${HORIZON:-5}"
MAX_HORIZON="${MAX_HORIZON:-20}"
ACTION_BLOCK="${ACTION_BLOCK:-5}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
RUN_NAME="${RUN_NAME:-latent_planner_${DATASET_NAME}_h${HORIZON}_ep${EPOCHS}}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lewm-latent-planner}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

python train_latent_planner.py \
  lewm_checkpoint="${LEWM_CHECKPOINT}" \
  data.dataset.name="${DATASET_NAME}" \
  data.dataset.keys_to_load="${KEYS_TO_LOAD}" \
  data.dataset.keys_to_cache="${KEYS_TO_CACHE}" \
  planner.horizon="${HORIZON}" \
  planner.max_horizon="${MAX_HORIZON}" \
  planner.action_block="${ACTION_BLOCK}" \
  epochs="${EPOCHS}" \
  log_interval="${LOG_INTERVAL}" \
  loader.batch_size="${BATCH_SIZE}" \
  subdir="latent_planner/${RUN_NAME}" \
  wandb.enabled="${WANDB_ENABLED}" \
  wandb.project="${WANDB_PROJECT}" \
  wandb.entity="${WANDB_ENTITY}" \
  wandb.name="${RUN_NAME}" \
  wandb.mode="${WANDB_MODE}" \
  ${EXTRA_OVERRIDES}

echo "=== Latent planner training finished ==="
