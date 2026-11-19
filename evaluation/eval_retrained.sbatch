#!/bin/bash
# Corrected evaluation (approved 2026-11-07): per-utterance predictions for every multilingual
# model on the full WAXAL test split and for the three combined-experiment models on the combined
# Amharic test set, then real-time factor at batch size 1 (GPU for 8 models, CPU for 4).
# Finished outputs are skipped on resubmission. Metrics come later from analyze.py.
#SBATCH --job-name=ethio-eval-v2
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
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

MODELS=(
  "released_94M|badrex/Ethio-ASR-multilingual-94M"
  "released_300M|badrex/Ethio-ASR-multilingual-300M"
  "released_1B|badrex/Ethio-ASR-multilingual-1B"
  "released_600M|badrex/Ethio-ASR-multilingual-600M"
  "ours_afrihubert|$P/models/ajesujoba/AfriHuBERT-11062026-234348"
  "ours_mms300m|$P/models/facebook/mms-300m-09062026-205411"
  "ours_mms1b|$P/models/facebook/mms-1b-09062026-203848"
  "ours_w2vbert|$P/models/facebook/w2v-bert-2.0-09062026-212456"
  "ours_w2vbert_43k|$P/models/facebook/w2v-bert-2.0-10062026-003217/checkpoint-43200"
  "ours_xlsr_adapter|$P/models/facebook/wav2vec2-xls-r-1b-09062026-230639"
)
COMBINED=(
  "A_waxal_am|$P/combined_dataset_experiment/models/A_waxal_am_baseline"
  "B_snapwre|$P/combined_dataset_experiment/models/B_snapwre_only"
  "C_combined|$P/combined_dataset_experiment/models/C_combined_am"
)

run_predict() {  # name path testset
  local out="$A/predictions/${1}__${3}.jsonl"
  if [ -s "$out" ]; then echo "skip $1 $3 (done)"; return; fi
  python3 -u "$A/scripts/predict.py" --model "$2" --name "$1" --testset "$3" --out "$out.tmp" \
    && mv "$out.tmp" "$out" && mv "$out.tmp.meta.json" "$out.meta.json" || echo "FAILED predict $1 $3"
}
for m in "${MODELS[@]}"; do run_predict "${m%%|*}" "${m#*|}" waxal; done
for m in "${COMBINED[@]}"; do run_predict "${m%%|*}" "${m#*|}" combined_am; done

for m in "${MODELS[@]:0:8}"; do
  name=${m%%|*}; out="$A/rtf/${name}__cuda.json"
  [ -s "$out" ] || python3 -u "$A/scripts/rtf.py" --model "${m#*|}" --name "$name" --device cuda --n 300 --out "$out" || echo "FAILED rtf $name"
done
for m in "${MODELS[@]:4:4}"; do
  name=${m%%|*}; out="$A/rtf/${name}__cpu.json"
  [ -s "$out" ] || python3 -u "$A/scripts/rtf.py" --model "${m#*|}" --name "$name" --device cpu --n 40 --threads 8 --out "$out" || echo "FAILED rtf-cpu $name"
done
rm -rf "$HF_DATASETS_CACHE"
echo "done $(date -Iseconds)"
