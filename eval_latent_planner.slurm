#!/bin/bash
#SBATCH --job-name=lewm_latent_eval
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=log/lewm_latent_eval_%j.log
#SBATCH --error=log/lewm_latent_eval_%j.err

set -euo pipefail

echo "=== SLURM latent planner eval started ==="
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
export TOKENIZERS_PARALLELISM=false

# stable-worldmodel stores datasets/checkpoints here.
export STABLEWM_HOME="${STABLEWM_HOME:-/gpfs/scrubbed/hwhuang/stable-wm}"

echo "Using python: $(which python)"
echo "Using pip: $(which pip)"
echo "CUDA: $(which nvcc)"
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available in Torch: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "STABLEWM_HOME: ${STABLEWM_HOME}"

# ---------------------------
# Run latent planner evaluation
# ---------------------------
cd /gpfs/projects/embodied3d/hsiangwei/le-wm
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Override these at submit time if needed, e.g.:
#   sbatch --export=ALL,CONFIG_NAME=pusht.yaml,PLANNER_CHECKPOINT=latent_planner/pusht_h10_ep10_24h/latent_planner.pt,HORIZON=10 eval_latent_planner.slurm
CONFIG_NAME="${CONFIG_NAME:-tworoom.yaml}"
SOLVER="${SOLVER:-latent_flow}"
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-latent_planner/latent_planner_h10/latent_planner.pt}"
HORIZON="${HORIZON:-5}"
ACTION_BLOCK="${ACTION_BLOCK:-5}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
FLOW_STEPS="${FLOW_STEPS:-16}"
SCORE_MODE="${SCORE_MODE:-rollout_goal}"
NUM_EVAL="${NUM_EVAL:-50}"
SEED="${SEED:-}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-}"
EVAL_BUDGET="${EVAL_BUDGET:-}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-}"
EPISODE_SPLIT_ENABLED="${EPISODE_SPLIT_ENABLED:-false}"
EPISODE_SPLIT_FILE="${EPISODE_SPLIT_FILE:-}"
EPISODE_SPLIT_MODE="${EPISODE_SPLIT_MODE:-eval}"

EXTRA_ARGS=()
if [[ -n "${GOAL_OFFSET_STEPS}" ]]; then
  EXTRA_ARGS+=("eval.goal_offset_steps=${GOAL_OFFSET_STEPS}")
fi
if [[ -n "${EVAL_BUDGET}" ]]; then
  EXTRA_ARGS+=("eval.eval_budget=${EVAL_BUDGET}")
fi
if [[ -n "${OUTPUT_FILENAME}" ]]; then
  EXTRA_ARGS+=("output.filename=${OUTPUT_FILENAME}")
fi
if [[ -n "${SEED}" ]]; then
  EXTRA_ARGS+=("seed=${SEED}")
fi
if [[ "${EPISODE_SPLIT_ENABLED}" == "true" ]]; then
  EXTRA_ARGS+=("episode_split.enabled=true")
  EXTRA_ARGS+=("episode_split.mode=${EPISODE_SPLIT_MODE}")
  EXTRA_ARGS+=("episode_split.split_file=${EPISODE_SPLIT_FILE}")
fi

python eval.py --config-name="${CONFIG_NAME}" \
  solver="${SOLVER}" \
  policy="${PLANNER_CHECKPOINT}" \
  plan_config.horizon="${HORIZON}" \
  plan_config.receding_horizon="${HORIZON}" \
  plan_config.action_block="${ACTION_BLOCK}" \
  solver.num_samples="${NUM_SAMPLES}" \
  solver.flow_steps="${FLOW_STEPS}" \
  solver.score_mode="${SCORE_MODE}" \
  eval.num_eval="${NUM_EVAL}" \
  "${EXTRA_ARGS[@]}"

echo "=== Latent planner eval finished ==="
