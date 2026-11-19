#!/usr/bin/env bash
#SBATCH --job-name=snapwre-inspect
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/path/to/Ethio-ASR/combined_dataset_experiment/logs/inspect_%j.out
#SBATCH --error=/path/to/Ethio-ASR/combined_dataset_experiment/logs/inspect_%j.err

set -euo pipefail
PROJECT_ROOT="/path/to/Ethio-ASR"
source "$PROJECT_ROOT/slurm_scripts/setup_env.sh"
module purge
module load pytorch/2.8.0

echo "[job] Inspection start $(date -Iseconds) on $(hostname)"
python3 -u "$PROJECT_ROOT/combined_dataset_experiment/data_prep/01_download_and_inspect.py"
echo "[job] Done $(date -Iseconds)"
