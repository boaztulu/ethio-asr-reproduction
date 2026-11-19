#!/bin/bash
#
# XLS-R-1B without the adapter (configs/05b_new_xls-r-1b_no_adapter.yaml), second attempt.
#
# The first attempt (job 50146121, 2026-11-07) was cancelled by the cluster's idle-GPU
# check: the length-grouped sampler spent the first hour measuring every training input
# on the CPU while the B200 sat idle. This run starts training through
# train_fast_sampler.py, which hands the sampler the stored length column instead
# (same lengths, same batch order). Config, data, seed and resources are unchanged.
#
#SBATCH --job-name=ethio-xlsr1b-noadapter-v2
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gpus=1
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/path/to/Ethio-ASR/logs/%x_%j.out
#SBATCH --error=/path/to/Ethio-ASR/logs/%x_%j.err

set -uo pipefail

PROJECT_ROOT="/path/to/Ethio-ASR"
REPO_DIR="$PROJECT_ROOT/Ethio-ASR"
A="$PROJECT_ROOT/assessment_2026-11"
CONFIG="$PROJECT_ROOT/configs/05b_new_xls-r-1b_no_adapter.yaml"

module purge
module load pytorch/2.8.0
source "$PROJECT_ROOT/slurm_scripts/setup_env.sh"

# Job-specific dataset cache, removed when the script exits so the 2026-11-07 cleanup stays effective.
export HF_DATASETS_CACHE="$A/tmp_ds_cache_xlsr_${SLURM_JOB_ID}"
mkdir -p "$HF_DATASETS_CACHE"
trap 'rm -rf "$HF_DATASETS_CACHE"' EXIT

echo "=== Ethio-ASR training (CTC), stored-length sampler ==="
echo "Host:      $(hostname)"
echo "Job:       ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?})"
echo "Started:   $(date -Iseconds)"
echo "Config:    $CONFIG"
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "Free disk:"
df -h "$PROJECT_ROOT" | tail -1

# Forward SIGUSR1 (time limit or requeue) and SIGTERM to Python.
trap 'echo "[SLURM] Forwarding SIGUSR1"; kill -USR1 $PID 2>/dev/null || true' USR1
trap 'echo "[SLURM] Forwarding SIGTERM"; kill -TERM $PID 2>/dev/null || true' TERM

cd "$REPO_DIR"
python3 -u "$A/scripts/train_fast_sampler.py" --config "$CONFIG" &
PID=$!
STATUS=0
wait $PID || STATUS=$?
# A trapped signal ends `wait` early; keep waiting until Python has really exited.
while kill -0 $PID 2>/dev/null; do
    STATUS=0
    wait $PID || STATUS=$?
done

echo "Train exited with status $STATUS"
echo "End: $(date -Iseconds)"
exit $STATUS
