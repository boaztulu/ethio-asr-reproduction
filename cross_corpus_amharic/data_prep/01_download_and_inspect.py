#!/usr/bin/env python
"""
Download snapwre/amharic-speech + WAXAL, print inspection stats side-by-side.
Emits data_prep/inspection_report.json for downstream planning.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", "/path/to/Ethio-ASR/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/path/to/Ethio-ASR/hf_cache/datasets")

from datasets import load_dataset, disable_caching
disable_caching()

OUT_DIR = Path("/path/to/Ethio-ASR/combined_dataset_experiment/data_prep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def split_hours(split, duration_col):
    total_sec = sum(float(x) for x in split[duration_col])
    return total_sec / 3600.0


def compute_audio_hours(split):
    total_sec = 0.0
    for a in split["audio"]:
        try:
            total_sec += len(a["array"]) / a["sampling_rate"]
        except Exception:
            pass
    return total_sec / 3600.0


def inspect_snapwre():
    print("=" * 60)
    print("SNAPWRE / amharic-speech")
    print("=" * 60)
    ds = load_dataset("snapwre/amharic-speech")
    report = {"name": "snapwre/amharic-speech", "splits": {}}
    for split_name in ("train", "validation", "test"):
        if split_name not in ds:
            continue
        s = ds[split_name]
        hours = split_hours(s, "duration_s")
        n = len(s)
        char_hist = Counter()
        for t in s["sentence"][: min(2000, n)]:
            char_hist.update(t)
        n_speakers = len(set(s["speaker_id"]))
        report["splits"][split_name] = {
            "rows": n,
            "hours": round(hours, 3),
            "avg_duration_s": round(hours * 3600 / n, 2),
            "unique_speakers": n_speakers,
            "unique_chars_in_sample": len(char_hist),
        }
        print(f"[{split_name}] rows={n}  hours={hours:.2f}  speakers={n_speakers}  chars={len(char_hist)}")
        if split_name == "train":
            top_chars = char_hist.most_common(30)
            report["splits"][split_name]["top_chars_sample"] = [
                {"char": c, "count": ct} for c, ct in top_chars
            ]
    print(f"columns: {ds['train'].column_names}")
    report["columns"] = ds["train"].column_names
    return report, ds


def inspect_waxal_amharic():
    print("=" * 60)
    print("WAXAL amharic subset")
    print("=" * 60)
    ds = load_dataset("badrex/waxalNLP-ethiopic-final", verification_mode="no_checks")
    report = {"name": "badrex/waxalNLP-ethiopic-final (amharic filter)", "splits": {}}
    for split_name in ("train", "validation", "test"):
        if split_name not in ds:
            continue
        s = ds[split_name]
        # Fast Amharic filter: read only the language column (no audio decode).
        # WAXAL uses ISO codes ('amh'), but also accept 'amharic' just in case.
        lang_col = list(s.select_columns(["language"])["language"])
        indices = [i for i, l in enumerate(lang_col) if l in ("amh", "amharic")]
        print(f"  [fast_filter] {split_name}: kept {len(indices)}/{len(s)}")
        s_am = s.select(indices)
        n = len(s_am)
        # audio_duration column may already exist
        if "audio_duration" in s_am.column_names:
            hours = split_hours(s_am, "audio_duration")
        elif "duration" in s_am.column_names:
            hours = split_hours(s_am, "duration")
        else:
            hours = compute_audio_hours(s_am)
        # transcription column
        tcol = None
        for c in ("transcription", "transcript", "text"):
            if c in s_am.column_names:
                tcol = c
                break
        char_hist = Counter()
        if tcol:
            for t in s_am[tcol][: min(2000, n)]:
                char_hist.update(t)
        report["splits"][split_name] = {
            "rows": n,
            "hours": round(hours, 3),
            "avg_duration_s": round(hours * 3600 / max(n, 1), 2),
            "unique_chars_in_sample": len(char_hist),
        }
        print(f"[{split_name}] rows={n}  hours={hours:.2f}  chars={len(char_hist)}")
    print(f"columns: {ds['train'].column_names}")
    report["columns"] = ds["train"].column_names
    return report


def main():
    snapwre_report, _ = inspect_snapwre()
    waxal_report = inspect_waxal_amharic()

    # Combined size projection
    snapwre_train_h = snapwre_report["splits"].get("train", {}).get("hours", 0)
    waxal_train_h = waxal_report["splits"].get("train", {}).get("hours", 0)

    # Original paper baseline: 36800 steps on full 5-lang WAXAL (~196h train)
    # Scale by dataset hours (per user request)
    PAPER_STEPS = 36800
    PAPER_HOURS = 196.0  # from project memory
    STEPS_PER_HOUR = PAPER_STEPS / PAPER_HOURS

    scale = {
        "A_waxal_am_only": {
            "train_hours": waxal_train_h,
            "scaled_steps": int(round(STEPS_PER_HOUR * waxal_train_h)),
        },
        "B_snapwre_only": {
            "train_hours": snapwre_train_h,
            "scaled_steps": int(round(STEPS_PER_HOUR * snapwre_train_h)),
        },
        "C_combined": {
            "train_hours": waxal_train_h + snapwre_train_h,
            "scaled_steps": int(round(STEPS_PER_HOUR * (waxal_train_h + snapwre_train_h))),
        },
    }

    report = {
        "snapwre": snapwre_report,
        "waxal_amharic": waxal_report,
        "training_step_projection": {
            "baseline_paper_steps": PAPER_STEPS,
            "baseline_paper_hours": PAPER_HOURS,
            "steps_per_hour": round(STEPS_PER_HOUR, 2),
            "per_run": scale,
        },
    }

    out_path = OUT_DIR / "inspection_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path}")
    print("\nTRAINING-STEP PROJECTION:")
    for k, v in scale.items():
        print(f"  {k}: {v['train_hours']:.2f}h  ->  {v['scaled_steps']} steps")


if __name__ == "__main__":
    main()
