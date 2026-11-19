#!/usr/bin/env bash
#
# Multi-GPU CTC training for combined-dataset Amharic ASR experiments.
#
# Usage:
#   sbatch --export=ALL,CONFIG=configs/A_waxal_am_baseline.yaml \
#          --job-name=am-A-baseline \
#          combined_dataset_experiment/slurm_scripts/train.sbatch
#
#SBATCH --job-name=am-train
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --gpus=4
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/path/to/Ethio-ASR/combined_dataset_experiment/logs/%x_%j.out
#SBATCH --error=/path/to/Ethio-ASR/combined_dataset_experiment/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="/path/to/Ethio-ASR"
REPO_DIR="$PROJECT_ROOT/Ethio-ASR"
EXP_DIR="$PROJECT_ROOT/combined_dataset_experiment"
CONFIG="${CONFIG:?Set CONFIG=path/to/config.yaml when calling sbatch}"

# Resolve config to absolute path
if [[ "$CONFIG" != /* ]]; then
    CONFIG="$EXP_DIR/$CONFIG"
fi

module purge
module load pytorch/2.8.0
source "$PROJECT_ROOT/slurm_scripts/setup_env.sh"

# DDP env for torchrun / HF Trainer
export OMP_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false
NGPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-4}}"

echo "=== combined_dataset_experiment — multi-GPU CTC training ==="
echo "Host:      $(hostname)"
echo "Job:       ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?})"
echo "Started:   $(date -Iseconds)"
echo "Config:    $CONFIG"
echo "GPUs:      $NGPUS"
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.total --format=csv || true

trap 'echo "[SLURM] Forwarding SIGUSR1"; kill -USR1 $PID 2>/dev/null || true' USR1

cd "$REPO_DIR"
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NGPUS \
    scripts/train_model.py --config "$CONFIG" &
PID=$!
wait $PID
STATUS=$?

echo "Train exited with status $STATUS"
echo "End: $(date -Iseconds)"
exit $STATUS
