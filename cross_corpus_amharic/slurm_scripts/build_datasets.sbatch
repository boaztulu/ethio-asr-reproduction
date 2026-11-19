#!/usr/bin/env bash
#SBATCH --job-name=build-combined
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/path/to/Ethio-ASR/combined_dataset_experiment/logs/build_%j.out
#SBATCH --error=/path/to/Ethio-ASR/combined_dataset_experiment/logs/build_%j.err

set -euo pipefail
PROJECT_ROOT="/path/to/Ethio-ASR"
source "$PROJECT_ROOT/slurm_scripts/setup_env.sh"
module purge
module load pytorch/2.8.0

echo "[build] Start $(date -Iseconds) on $(hostname)"
python3 -u "$PROJECT_ROOT/combined_dataset_experiment/data_prep/02_build_combined_dataset.py"
echo "[build] Bundle sizes:"
du -sh /path/to/Ethio-ASR/combined_dataset_experiment/data/*
echo "[build] Done $(date -Iseconds)"
