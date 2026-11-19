#!/bin/bash
# Sensitivity of the WAXAL test WER to language tags repeated inside a hypothesis (added 2026-11-08).
#
# XLS-R-1B without the adapter repeats the language tag at the end of most transcripts ("baala.[SID]"),
# MMS-1B in about one in seven. The primary scoring (analyze.py default) removes only the leading tag, so a
# repeated tag counts as a word error, as it does in the original evaluation code. This job
#   1. runs the edited analyze.py with default settings into a scratch folder and requires the output to be
#      byte-identical to results/ (the edit must not change the primary numbers);
#   2. reruns it with TAG_SCORING=all (every tag removed from the hypotheses) into results_aux/tag_sensitivity/,
#      with the same speaker resamples;
#   3. writes the per-model counts of repeated tags to results_aux/repeated_tags.json.
#SBATCH --job-name=ethio-tag-sens
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/path/to/Ethio-ASR/assessment_2026-11/logs/%x_%j.out
#SBATCH --error=/path/to/Ethio-ASR/assessment_2026-11/logs/%x_%j.err
set -uo pipefail
P=/path/to/Ethio-ASR
A=$P/assessment_2026-11
module purge
module load pytorch/2.8.0
source "$P/slurm_scripts/setup_env.sh"
echo "Started: $(date -Iseconds)"

CHK="$A/results_aux/check_default_${SLURM_JOB_ID}"
ANALYSIS_OUT="$CHK" python3 -u "$A/scripts/analyze.py" > "$CHK.log" 2>&1 || { echo "FAILED default run"; tail -20 "$CHK.log"; exit 1; }
if cmp -s "$CHK/analysis.json" "$A/results/analysis.json" && cmp -s "$CHK/summary.txt" "$A/results/summary.txt"; then
    echo "default settings reproduce results/analysis.json and summary.txt byte for byte"
    rm -rf "$CHK" "$CHK.log"
else
    echo "MISMATCH: the default run differs from results/ (kept in $CHK)"; exit 1
fi

TAG_SCORING=all ANALYSIS_OUT="$A/results_aux/tag_sensitivity" python3 -u "$A/scripts/analyze.py" || { echo "FAILED sensitivity run"; exit 1; }

python3 -u "$A/scripts/count_repeated_tags.py" > "$A/results_aux/repeated_tags.json.new" \
    && mv "$A/results_aux/repeated_tags.json.new" "$A/results_aux/repeated_tags.json" || { echo "FAILED tag counts"; exit 1; }
echo "done $(date -Iseconds)"
