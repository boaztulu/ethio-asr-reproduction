#!/usr/bin/env python3
"""Run one CTC model over a test set and save one JSON line per utterance (no metrics here).

Inference follows slurm_scripts/evaluate_ctc.py exactly (same processor call, bf16 inputs,
batch size 8 with padding, greedy argmax decoding), so the corpus-level WER recomputed from these
files must reproduce the earlier eval_test*.json values. Differences from that script:
  - the WAXAL test split is read straight from its parquet files (only the test split is prepared),
  - every utterance is saved: id, speaker, language, gender, duration, reference, hypothesis,
    the predicted language tag (raw), and the model's output length in frames.
Metrics are computed afterwards by analyze.py.

  python predict.py --model <dir or HF id> --name <label> --testset waxal|combined_am --out <file.jsonl>
"""
import argparse
import glob
import json
import os
import re
import time
from pathlib import Path

P = Path("/path/to/Ethio-ASR")
os.environ.setdefault("HF_HOME", str(P / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(P / "hf_cache" / "hub"))

import torch  # noqa: E402
from datasets import Audio, load_dataset, load_from_disk  # noqa: E402
from transformers import AutoModelForCTC, AutoProcessor  # noqa: E402

TAG = re.compile(r"^\s*\[([A-Za-z]+)\]\|?\s*")


def load_test(testset):
    if testset == "waxal":
        data = sorted(glob.glob(str(P / "hf_cache/hub/datasets--badrex--waxalNLP-ethiopic-final/snapshots/*/data/test-*.parquet")))
        assert len(data) == 10, data
        ds = load_dataset("parquet", data_files={"test": data}, split="test")
    elif testset == "combined_am":
        ds = load_from_disk(str(P / "combined_dataset_experiment/data/combined_am"))["test"]
    else:
        raise SystemExit(f"unknown testset {testset}")
    return ds.cast_column("audio", Audio(sampling_rate=16000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--testset", default="waxal")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    dev = "cuda"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    ds = load_test(args.testset)
    n = len(ds)
    print(f"[predict] {args.name}: {args.model} on {args.testset} ({n} utterances)", flush=True)

    t_start = time.time()
    audio_seconds = 0.0
    with open(args.out, "w", encoding="utf-8") as fo:
        for i in range(0, n, args.batch_size):
            batch = ds[i:i + args.batch_size]
            audio = [a["array"] for a in batch["audio"]]
            sr = batch["audio"][0]["sampling_rate"]
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
            kw = {}
            if "input_values" in inputs:
                kw["input_values"] = inputs["input_values"].to(dev, dtype=torch.bfloat16)
            if "input_features" in inputs:
                kw["input_features"] = inputs["input_features"].to(dev, dtype=torch.bfloat16)
            if "attention_mask" in inputs:
                kw["attention_mask"] = inputs["attention_mask"].to(dev)
            with torch.no_grad():
                logits = model(**kw).logits
            hyps = processor.batch_decode(logits.argmax(dim=-1))
            out_len = [None] * len(audio)
            try:
                if "attention_mask" in inputs:
                    ol = model._get_feat_extract_output_lengths(inputs["attention_mask"].sum(-1))
                    out_len = [int(x) for x in ol]
            except Exception:  # not every model class exposes this
                pass
            for j, hyp in enumerate(hyps):
                k = i + j
                m = TAG.match(hyp)
                dur = len(audio[j]) / sr
                audio_seconds += dur
                rec = {
                    "idx": k,
                    "id": batch.get("id", [None] * len(audio))[j],
                    "speaker_id": batch.get("speaker_id", [None] * len(audio))[j],
                    "language": batch.get("language", [None] * len(audio))[j],
                    "gender": batch.get("gender", [None] * len(audio))[j],
                    "duration": round(dur, 3),
                    "ref": batch["transcription"][j],
                    "hyp": hyp,
                    "pred_tag": m.group(1).lower() if m else None,
                    "out_frames": out_len[j],
                }
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i // args.batch_size) % 250 == 0:
                print(f"  {min(i + args.batch_size, n)}/{n}", flush=True)
    wall = time.time() - t_start
    meta = {"name": args.name, "model": args.model, "testset": args.testset, "n": n,
            "batch_size": args.batch_size, "wall_seconds": round(wall, 1),
            "audio_seconds": round(audio_seconds, 1), "throughput_rtf": round(wall / audio_seconds, 5),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    Path(args.out + ".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[predict] done {args.name}: {meta}", flush=True)


if __name__ == "__main__":
    main()
