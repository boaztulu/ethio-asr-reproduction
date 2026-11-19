#!/usr/bin/env bash
#
# Cross-evaluation: 3 trained models × 3 test sets = 9 WER/CER runs.
#
# Usage:
#   sbatch combined_dataset_experiment/slurm_scripts/evaluate_matrix.sbatch
#
#SBATCH --job-name=am-eval-matrix
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/Ethio-ASR/combined_dataset_experiment/logs/eval_matrix_%j.out
#SBATCH --error=/path/to/Ethio-ASR/combined_dataset_experiment/logs/eval_matrix_%j.err
# (removed --mem-per-cpu; conflicts with --mem)

set -euo pipefail

PROJECT_ROOT="/path/to/Ethio-ASR"
EXP_DIR="$PROJECT_ROOT/combined_dataset_experiment"
EVAL_PY="$PROJECT_ROOT/slurm_scripts/evaluate_ctc.py"

module purge
module load pytorch/2.8.0
source "$PROJECT_ROOT/slurm_scripts/setup_env.sh"

# === Discovery: map A/B/C stable names -> timestamped model dirs ===
# train_model.py names dirs via datetime.now(); the linker script parses each
# training log for "Vocabulary saved to <path>/ctc_tokenizer" and symlinks
# models/{A,B,C}_* -> the real dir.  Aborts the job if any link fails.
python3 -u "$EXP_DIR/data_prep/04_link_models.py"

MODELS=(
    "$EXP_DIR/models/A_waxal_am_baseline"
    "$EXP_DIR/models/B_snapwre_only"
    "$EXP_DIR/models/C_combined_am"
)
TEST_SETS=(
    "$EXP_DIR/data/waxal_am"
    "$EXP_DIR/data/snapwre_am"
    "$EXP_DIR/data/combined_am"
)
TEST_NAMES=(waxal_am snapwre_am combined_am)

mkdir -p "$EXP_DIR/results"

for M in "${MODELS[@]}"; do
    if [[ ! -e "$M" ]]; then
        echo "[skip] Model $M not found (run not finished?)"
        continue
    fi
    MNAME=$(basename "$M")
    for i in "${!TEST_SETS[@]}"; do
        T="${TEST_SETS[$i]}"
        TNAME="${TEST_NAMES[$i]}"
        OUT="$EXP_DIR/results/${MNAME}__on__${TNAME}.json"
        echo "=== eval $MNAME on $TNAME ==="
        python3 -u "$EVAL_PY" \
            --model_dir "$M" \
            --dataset "$T" \
            --split test \
            --batch_size 8 \
            --apply_postproc \
            --out_json "$OUT" || echo "[warn] failed $MNAME on $TNAME"
    done
done

echo "=== eval matrix done $(date -Iseconds) ==="
ls -la "$EXP_DIR/results/"

echo "=== analysis / comparison report ==="
python3 -u "$EXP_DIR/analysis/compare.py"
echo "=== report saved: $EXP_DIR/analysis/report.md ==="
cat "$EXP_DIR/analysis/report.md" || true
