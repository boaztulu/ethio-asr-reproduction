#!/usr/bin/env python
"""
Build three local DatasetDict bundles for training:
  - waxal_am     : WAXAL filtered to language=='amharic' only (baseline A)
  - snapwre_am   : snapwre/amharic-speech, columns renamed to pipeline convention (B)
  - combined_am  : train = concat(waxal_am.train, snapwre_am.train)
                   validation = concat(waxal_am.validation, snapwre_am.validation)
                   test = concat(waxal_am.test, snapwre_am.test)   (C)

All bundles are saved under combined_dataset_experiment/data/ and loaded via
`use_custom_dataset: true` + `dataset_path: <bundle-path>` in training configs.

Column contract for the training pipeline (src/data/dataset.py):
  - audio         : Audio(sampling_rate=16000)
  - transcription : str
  - language      : str (canonical: "amharic")
  - audio_duration: float (seconds)

Snapwre remap:
  sentence   -> transcription
  duration_s -> audio_duration
  language   'am' -> 'amharic'
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/path/to/Ethio-ASR")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "hf_cache" / "datasets"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / "hf_cache" / "hub"))

from datasets import (
    load_dataset,
    concatenate_datasets,
    DatasetDict,
    Audio,
    Value,
    Features,
    disable_caching,
)
disable_caching()

OUT_ROOT = PROJECT_ROOT / "combined_dataset_experiment" / "data"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

WAXAL_AM_DIR   = OUT_ROOT / "waxal_am"
SNAPWRE_DIR    = OUT_ROOT / "snapwre_am"
COMBINED_DIR   = OUT_ROOT / "combined_am"

KEEP_COLS = ["audio", "transcription", "language", "audio_duration"]


def _keep_only(ds, cols=KEEP_COLS):
    drop = [c for c in ds.column_names if c not in cols]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


def _align_features(ds):
    """Cast to a canonical schema so bundles can be concatenated later.
    WAXAL uses float64 for audio_duration; snapwre uses float32. Standardise
    on float64 to avoid ValueError in concatenate_datasets.
    """
    features = ds.features.copy()
    features["audio_duration"] = Value("float64")
    features["transcription"] = Value("string")
    features["language"] = Value("string")
    return ds.cast(features)


def _fast_filter_amharic(s):
    """Filter to amharic without decoding audio.

    dataset.filter(...) iterates row-by-row and materialises the audio bytes
    for each row (huge waste when the mask depends only on the language
    column). Reading only the language column via select_columns is Arrow-
    native and completes in seconds even on 200k rows.

    WAXAL uses ISO codes ('amh'); accept 'amharic' too for safety.
    """
    lang_col = list(s.select_columns(["language"])["language"])
    indices = [i for i, l in enumerate(lang_col) if l in ("amh", "amharic")]
    print(f"  [fast_filter] kept {len(indices)}/{len(s)} amharic rows")
    return s.select(indices)


def build_waxal_amharic():
    print("=" * 60)
    print("[waxal_am] Loading WAXAL and filtering to amharic")
    print("=" * 60)
    ds = load_dataset("badrex/waxalNLP-ethiopic-final", verification_mode="no_checks")
    out = {}
    for split_name in ("train", "validation", "test"):
        if split_name not in ds:
            print(f"  [warn] {split_name} not in WAXAL — skipping")
            continue
        s = _fast_filter_amharic(ds[split_name])
        # transcription column (WAXAL uses 'transcription' already but be safe)
        if "transcription" not in s.column_names:
            for cand in ("transcript", "text"):
                if cand in s.column_names:
                    s = s.rename_column(cand, "transcription")
                    break
        # duration column
        if "audio_duration" not in s.column_names and "duration" in s.column_names:
            s = s.rename_column("duration", "audio_duration")
        # Normalize language: 'amh' (WAXAL ISO code) -> 'amharic' to match
        # the training pipeline / eval script conventions.
        s = s.map(
            lambda x: {"language": "amharic"},
            desc=f"waxal-am {split_name} normalize language",
        )
        s = s.cast_column("audio", Audio(sampling_rate=16000))
        s = _keep_only(s)
        s = _align_features(s)
        print(f"  [waxal_am/{split_name}] rows={len(s)} cols={s.column_names}")
        out[split_name] = s
    dd = DatasetDict(out)
    print(f"[waxal_am] Saving to {WAXAL_AM_DIR}")
    dd.save_to_disk(str(WAXAL_AM_DIR), num_proc=4)
    return dd


def build_snapwre():
    print("=" * 60)
    print("[snapwre_am] Loading snapwre/amharic-speech and remapping columns")
    print("=" * 60)
    ds = load_dataset("snapwre/amharic-speech")
    out = {}
    for split_name in ("train", "validation", "test"):
        if split_name not in ds:
            print(f"  [warn] {split_name} not in snapwre — skipping")
            continue
        s = ds[split_name]

        # rename sentence -> transcription
        if "sentence" in s.column_names and "transcription" not in s.column_names:
            s = s.rename_column("sentence", "transcription")
        # rename duration_s -> audio_duration
        if "duration_s" in s.column_names and "audio_duration" not in s.column_names:
            s = s.rename_column("duration_s", "audio_duration")
        # normalize language: 'am' -> 'amharic'
        s = s.map(
            lambda x: {"language": "amharic"},
            desc=f"snapwre {split_name} normalize language",
        )
        s = s.cast_column("audio", Audio(sampling_rate=16000))
        s = _keep_only(s)
        s = _align_features(s)
        print(f"  [snapwre_am/{split_name}] rows={len(s)} cols={s.column_names}")
        out[split_name] = s
    dd = DatasetDict(out)
    print(f"[snapwre_am] Saving to {SNAPWRE_DIR}")
    dd.save_to_disk(str(SNAPWRE_DIR), num_proc=4)
    return dd


def build_combined(waxal_am: DatasetDict, snapwre_am: DatasetDict):
    print("=" * 60)
    print("[combined_am] Concatenating WAXAL-amharic + snapwre-amharic")
    print("=" * 60)
    out = {}
    for split_name in ("train", "validation", "test"):
        parts = []
        if split_name in waxal_am:
            parts.append(waxal_am[split_name])
        if split_name in snapwre_am:
            parts.append(snapwre_am[split_name])
        if not parts:
            continue
        cat = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        print(f"  [combined_am/{split_name}] rows={len(cat)} "
              f"(waxal={len(waxal_am.get(split_name,[]))} + snapwre={len(snapwre_am.get(split_name,[]))})")
        out[split_name] = cat
    dd = DatasetDict(out)
    print(f"[combined_am] Saving to {COMBINED_DIR}")
    dd.save_to_disk(str(COMBINED_DIR), num_proc=4)
    return dd


def main():
    waxal_am = build_waxal_amharic()
    snapwre_am = build_snapwre()
    _ = build_combined(waxal_am, snapwre_am)
    print("\n[done] Bundles written under", OUT_ROOT)
    for p in (WAXAL_AM_DIR, SNAPWRE_DIR, COMBINED_DIR):
        print(f"  {p}")


if __name__ == "__main__":
    main()
