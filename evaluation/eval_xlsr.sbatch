#!/bin/bash
# Evaluation of the XLS-R-1B run without the output adapter (training job 50160141).
#
# Submitted on 2026-11-07 with --dependency=afterok:50160141 --kill-on-invalid-dep=yes, so it
# starts only if training ends successfully and is cancelled otherwise. The steps are those of
# eval_v2.sbatch for the other models: per-utterance predictions on the full WAXAL test split
# (predict.py, same batching and precision), then the real-time factor at batch size 1 on the
# GPU (300 utterances) and on the CPU (40 utterances, 8 threads). Metrics come from analyze.py.
#SBATCH --job-name=ethio-eval-xlsr
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/Ethio-ASR/assessment_2026-11/logs/%x_%j.out
#SBATCH --error=/path/to/Ethio-ASR/assessment_2026-11/logs/%x_%j.err
set -uo pipefail
P=/path/to/Ethio-ASR
A=$P/assessment_2026-11
RUN=$P/models/facebook/wav2vec2-xls-r-1b-07112026-162109
NAME=ours_xlsr_noadapter
module purge
module load pytorch/2.8.0
source "$P/slurm_scripts/setup_env.sh"
export HF_DATASETS_CACHE=$A/tmp_ds_cache_${SLURM_JOB_ID}
export HF_HUB_OFFLINE=1
mkdir -p "$HF_DATASETS_CACHE" "$A/predictions" "$A/rtf"
trap 'rm -rf "$HF_DATASETS_CACHE"' EXIT
nvidia-smi -L
echo "Started: $(date -Iseconds)"

# The trainer writes the best checkpoint (load_best_model_at_end) to the run folder only after training ends.
if [ ! -s "$RUN/model.safetensors" ] || [ ! -s "$RUN/config.json" ]; then
    echo "no final model in $RUN"; exit 1
fi
ls -la "$RUN"

STATUS=0
out="$A/predictions/${NAME}__waxal.jsonl"
if [ -s "$out" ]; then
    echo "skip predictions (done)"
else
    python3 -u "$A/scripts/predict.py" --model "$RUN" --name "$NAME" --testset waxal --out "$out.tmp" \
        && mv "$out.tmp" "$out" && mv "$out.tmp.meta.json" "$out.meta.json" || { echo "FAILED predict"; STATUS=1; }
fi

out="$A/rtf/${NAME}__cuda.json"
[ -s "$out" ] || python3 -u "$A/scripts/rtf.py" --model "$RUN" --name "$NAME" --device cuda --n 300 --out "$out" || { echo "FAILED rtf cuda"; STATUS=1; }
out="$A/rtf/${NAME}__cpu.json"
[ -s "$out" ] || python3 -u "$A/scripts/rtf.py" --model "$RUN" --name "$NAME" --device cpu --n 40 --threads 8 --out "$out" || { echo "FAILED rtf cpu"; STATUS=1; }

echo "End: $(date -Iseconds) status $STATUS"
exit $STATUS
