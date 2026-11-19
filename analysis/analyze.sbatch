#!/bin/bash
# Metrics, bootstrap intervals and checks from the eval_v2 predictions (CPU only).
#SBATCH --job-name=ethio-analyze
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/Ethio-ASR/assessment_2026-11/logs/%x_%j.out
#SBATCH --error=/path/to/Ethio-ASR/assessment_2026-11/logs/%x_%j.err
set -uo pipefail
P=/path/to/Ethio-ASR
module purge
module load pytorch/2.8.0
source "$P/slurm_scripts/setup_env.sh"
python3 -u "$P/assessment_2026-11/scripts/analyze.py"
echo "done $(date -Iseconds)"
