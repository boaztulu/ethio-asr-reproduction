#!/usr/bin/env bash
# Submit A, B, C training jobs in parallel.
# Prereqs: build_datasets.sbatch has completed AND 03_generate_configs.py has run.

set -euo pipefail
EXP_DIR="/path/to/Ethio-ASR/combined_dataset_experiment"

for RUN in A_waxal_am_baseline B_snapwre_only C_combined_am; do
    CONFIG="$EXP_DIR/configs/${RUN}.yaml"
    if [[ ! -f "$CONFIG" ]]; then
        echo "[skip] $CONFIG not found — run 03_generate_configs.py first."
        continue
    fi
    JOB_ID=$(sbatch --parsable \
        --export=ALL,CONFIG="$CONFIG" \
        --job-name="am-${RUN}" \
        "$EXP_DIR/slurm_scripts/train.sbatch")
    echo "[launch] ${RUN} -> job ${JOB_ID}  (config: ${CONFIG})"
done

echo
echo "Current queue:"
squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.4D %R"
