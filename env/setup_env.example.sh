#!/bin/bash
# Example of the setup_env.sh that the Slurm jobs source from the project root.
# Copy it to /path/to/Ethio-ASR/slurm_scripts/setup_env.sh and adapt it to your cluster.

export PROJECT_ROOT=/path/to/Ethio-ASR

# Hugging Face caches inside the project, so models and datasets are downloaded once.
export HF_HOME="$PROJECT_ROOT/hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/hf_cache/datasets"

# The original Ethio-ASR code (https://github.com/badrex/Ethio-ASR) cloned into the project root.
export PYTHONPATH="$PROJECT_ROOT/Ethio-ASR:$PYTHONPATH"

# No token is needed for the public datasets and checkpoints used here. If you need one,
# read it from a file outside the repository; never write it into a script.
