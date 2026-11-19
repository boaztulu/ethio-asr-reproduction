#!/usr/bin/env python
"""
Generate 3 training configs (A, B, C) from the base template, filling
in dataset_path and max_steps scaled by DATASET SIZE (per user instruction).

Scaling formula: fixed number of EPOCHS across all runs → fair matched
training. Each run does N_EPOCHS full passes through its data.

    scaled_steps = round(N_EPOCHS * n_train_rows / effective_batch)

where effective_batch = per_device_batch * grad_accum * NGPUS (= 8 * 1 * 4 = 32).

This is equivalent to scaling by hours (assuming similar avg-utterance length)
but avoids the audio-decode step and matches the paper's actual training
budget (~6 epochs on WAXAL 5-lang).

A minimum floor of MIN_STEPS is applied so tiny datasets (snapwre alone)
still get enough optimizer steps to converge past the LR warmup phase.
"""

import json
from pathlib import Path

EXP = Path("/path/to/Ethio-ASR/combined_dataset_experiment")
BASE = EXP / "configs" / "_base_w2v_bert_am.yaml"
REPORT = EXP / "data_prep" / "inspection_report.json"
DATA_ROOT = EXP / "data"

# Training budget knobs (match paper's implied ~6 epochs on WAXAL 5-lang)
N_EPOCHS = 6
PER_DEVICE_BATCH = 8
GRAD_ACCUM = 1
N_GPUS = 4
EFFECTIVE_BATCH = PER_DEVICE_BATCH * GRAD_ACCUM * N_GPUS  # 32
MIN_STEPS = 3000  # floor for tiny datasets

RUNS = [
    dict(
        tag="A",
        name="A_waxal_am_baseline",
        project="Ethio-ASR-combined-A-waxal-am-baseline",
        dataset_dir=str(DATA_ROOT / "waxal_am"),
        rows_source=("waxal_amharic", "train"),
    ),
    dict(
        tag="B",
        name="B_snapwre_only",
        project="Ethio-ASR-combined-B-snapwre-only",
        dataset_dir=str(DATA_ROOT / "snapwre_am"),
        rows_source=("snapwre", "train"),
    ),
    dict(
        tag="C",
        name="C_combined_am",
        project="Ethio-ASR-combined-C-waxal-am-plus-snapwre",
        dataset_dir=str(DATA_ROOT / "combined_am"),
        rows_source=("__combined__", "train"),
    ),
]


def _rows_of(report, ds_key, split):
    if ds_key == "__combined__":
        w = report["waxal_amharic"]["splits"].get(split, {}).get("rows", 0)
        s = report["snapwre"]["splits"].get(split, {}).get("rows", 0)
        return w + s
    return report[ds_key]["splits"].get(split, {}).get("rows", 0)


def main():
    base_yaml = BASE.read_text()
    report = json.loads(REPORT.read_text())

    print(f"Scaling: N_EPOCHS={N_EPOCHS}, effective_batch={EFFECTIVE_BATCH}, "
          f"min_steps={MIN_STEPS}")

    for run in RUNS:
        ds_key, split = run["rows_source"]
        n_rows = _rows_of(report, ds_key, split)
        raw_steps = int(round(N_EPOCHS * n_rows / EFFECTIVE_BATCH))
        steps = max(raw_steps, MIN_STEPS)
        # Round save/eval cadence to ~10% of total steps, floored at 200
        cadence = max(200, (steps // 10 // 100) * 100 or 200)

        yaml_text = base_yaml
        yaml_text = yaml_text.replace("project: PLACEHOLDER", f"project: {run['project']}")
        yaml_text = yaml_text.replace("max_steps: PLACEHOLDER", f"max_steps: {steps}")
        yaml_text = yaml_text.replace("dataset_path: PLACEHOLDER", f"dataset_path: {run['dataset_dir']}")
        yaml_text = yaml_text.replace("save_steps: 400", f"save_steps: {cadence}")
        yaml_text = yaml_text.replace("eval_steps: 400", f"eval_steps: {cadence}")

        out = EXP / "configs" / f"{run['name']}.yaml"
        out.write_text(yaml_text)
        print(f"  [{run['tag']}] {out.name}: rows={n_rows}  raw_steps={raw_steps}  "
              f"final_steps={steps}  cadence={cadence}")


if __name__ == "__main__":
    main()
