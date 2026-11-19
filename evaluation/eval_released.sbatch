#!/bin/bash
# Corrected evaluation, released checkpoints only (loaded from their local snapshot folders; the
# by-name offline load failed in job 50146120). Same steps as eval_v2.sbatch: per-utterance predictions for every multilingual
# model on the full WAXAL test split and for the three combined-experiment models on the combined
# Amharic test set, then real-time factor at batch size 1 (GPU for 8 models, CPU for 4).
# Finished outputs are skipped on resubmission. Metrics come later from analyze.py.
#SBATCH --job-name=ethio-eval-v2-released
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/Ethio-ASR/assessment_2026-11/logs/%x_%j.out
#SBATCH --error=/path/to/Ethio-ASR/assessment_2026-11/logs/%x_%j.err
set -uo pipefail
P=/path/to/Ethio-ASR
A=$P/assessment_2026-11
module purge
module load pytorch/2.8.0
source "$P/slurm_scripts/setup_env.sh"
export HF_DATASETS_CACHE=$A/tmp_ds_cache_${SLURM_JOB_ID}
export HF_HUB_OFFLINE=1
mkdir -p "$A/predictions" "$A/rtf"
nvidia-smi -L

snap() { ls -d "$P"/hf_cache/hub/models--badrex--Ethio-ASR-multilingual-"$1"/snapshots/*/ | head -1; }
MODELS=(
  "released_94M|$(snap 94M)"
  "released_300M|$(snap 300M)"
  "released_1B|$(snap 1B)"
  "released_600M|$(snap 600M)"
)
run_predict() {  # name path testset
  local out="$A/predictions/${1}__${3}.jsonl"
  if [ -s "$out" ]; then echo "skip $1 $3 (done)"; return; fi
  python3 -u "$A/scripts/predict.py" --model "$2" --name "$1" --testset "$3" --out "$out.tmp"     && mv "$out.tmp" "$out" && mv "$out.tmp.meta.json" "$out.meta.json" || echo "FAILED predict $1 $3"
}
for m in "${MODELS[@]}"; do run_predict "${m%%|*}" "${m#*|}" waxal; done
for m in "${MODELS[@]}"; do
  name=${m%%|*}; out="$A/rtf/${name}__cuda.json"
  [ -s "$out" ] || python3 -u "$A/scripts/rtf.py" --model "${m#*|}" --name "$name" --device cuda --n 300 --out "$out" || echo "FAILED rtf $name"
done
rm -rf "$HF_DATASETS_CACHE"
echo "done $(date -Iseconds)"
