# Independent reproduction of Ethio-ASR

Code and measured results for the paper "Independent reproduction of joint speech recognition and language identification for five Ethiopian languages, with a cross-corpus study for Amharic" (Boaz B. Tulu).

The study retrains the four encoders of Ethio-ASR (Abdullah et al., 2026, [arXiv:2603.23654](https://doi.org/10.48550/arXiv.2603.23654)) from public artefacts. It scores the released checkpoints and the retrained models with one pipeline on the full WAXAL test set, with speaker-level bootstrap intervals. It also adds a corrected XLS-R-1B run and a cross-corpus study for Amharic.

Model training uses the original Ethio-ASR code, [github.com/badrex/Ethio-ASR](https://github.com/badrex/Ethio-ASR), unchanged apart from software pinning. This repository holds what the study added.

## Contents

| Folder | What it holds |
|---|---|
| `configs/` | Training configurations for the multilingual runs (AfriHuBERT, MMS-300M, MMS-1B, w2v-BERT-2.0, XLS-R-1B with and without the output adapter, and the long w2v-BERT-2.0 schedule) and the script that writes them |
| `training/` | The Slurm job for the original training script and the wrapper used for the corrected XLS-R-1B run. The wrapper passes the stored utterance lengths to the length-grouped sampler; `test_fast_sampler.py` checks that it gives the same lengths and batch order as the library. `xlsr_feasibility.py` counts utterances whose CTC targets cannot be aligned. |
| `evaluation/` | Prediction over the full test set with every utterance's output saved (`predict.py`), real-time factor on GPU and CPU (`rtf.py`), and the Slurm jobs |
| `analysis/` | Scoring, normalisation, speaker-level bootstrap intervals, language identification, gender, phonology and inference cost (`analyze.py`). Also the repeated-language-token analysis (`count_repeated_tags.py`, `tag_sensitivity_detail.py`), the dataset audit and the export of validation curves. |
| `cross_corpus_amharic/` | Data preparation, configurations, Slurm jobs and per-model results for the three Amharic-only models (WAXAL, snapwre and both) |
| `paper/` | `make_paper_assets.py`, which generates every number, table and figure in the paper from the files in `results/` and checks the data-dependent statements in the text |
| `results/` | Measured metrics: scores, intervals, paired differences, validation curves, repeated-token counts and the data audit. No audio and no transcripts. |

## Scoring conventions

Scores are computed after the leading language token is removed and text is lower-cased ("raw"). A language token emitted again later stays in the hypothesis and counts as an error, as in the original evaluation code.

"Post-processed" scores also remove Ethiopic and ASCII punctuation and Ge'ez numerals. They merge four groups of homophone letters in all seven vowel orders:
- the ha series (U+1200, U+1210, U+1280);
- the sa series (U+1230, U+1220);
- the glottal series (U+12A0, U+12D0);
- the ts'a series (U+1338, U+1340).

Setting `TAG_SCORING=all` in `analysis/analyze.py` removes every language token instead. The defaults reproduce the paper's main results.

## Running it

Paths in the scripts point to `/path/to/Ethio-ASR`, the project root, and Slurm settings use `YOUR_ACCOUNT`, `YOUR_QOS` and `YOUR_*_PARTITION`. Replace them for your cluster, together with the `module load` lines. The original Ethio-ASR code is expected as a clone in `Ethio-ASR/` under the project root. The Slurm jobs source `slurm_scripts/setup_env.sh` from the project root; `env/setup_env.example.sh` shows what it needs to set.

Main software: Python 3.10 or later, PyTorch 2.8.0 (CUDA 12.8), Transformers 4.55.4, Datasets 3.6.0 (see `requirements.txt`). Training used one NVIDIA B200 GPU per job.

The data are public:
- **WAXAL:** [huggingface.co/datasets/google/WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP), CC-BY-SA-4.0 for the five Ethiopian languages, used through the Ethio-ASR copy `badrex/waxalNLP-ethiopic-final` (revision c5a7948).
- **Dataset.ET Amharic Speech v0.1.0:** [huggingface.co/datasets/snapwre/amharic-speech](https://huggingface.co/datasets/snapwre/amharic-speech), revision 9ca0183, CC-BY-4.0.

## Licence

The code is released under the MIT licence (see `LICENSE`). The datasets and the released Ethio-ASR checkpoints keep their own licences.
