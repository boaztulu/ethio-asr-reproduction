#!/usr/bin/env python3
"""Real-time factor at batch size 1 on a fixed random subset of the WAXAL test split.

For each utterance the time for feature extraction (CPU) and for the model forward pass, argmax
and decoding (GPU synchronised) are measured separately. RTF = compute time / audio duration;
values below 1 are faster than real time. Five warm-up utterances are not timed.

  python rtf.py --model <dir or HF id> --name <label> --device cuda|cpu --n 300 --out <file.json>
"""
import argparse
import glob
import json
import os
import statistics
import time
from pathlib import Path

P = Path("/path/to/Ethio-ASR")
os.environ.setdefault("HF_HOME", str(P / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(P / "hf_cache" / "hub"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402
from transformers import AutoModelForCTC, AutoProcessor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dev = args.device
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model, torch_dtype=dtype).to(dev).eval()
    data = sorted(glob.glob(str(P / "hf_cache/hub/datasets--badrex--waxalNLP-ethiopic-final/snapshots/*/data/test-*.parquet")))
    ds = load_dataset("parquet", data_files={"test": data}, split="test").cast_column("audio", Audio(sampling_rate=16000))
    idx = sorted(np.random.RandomState(42).choice(len(ds), size=args.n + 5, replace=False).tolist())
    warm, timed = idx[:5], idx[5:]

    def run(i):
        ex = ds[int(i)]
        wav = ex["audio"]["array"]
        t0 = time.perf_counter()
        inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
        t1 = time.perf_counter()
        kw = {}
        if "input_values" in inputs:
            kw["input_values"] = inputs["input_values"].to(dev, dtype=dtype)
        if "input_features" in inputs:
            kw["input_features"] = inputs["input_features"].to(dev, dtype=dtype)
        if "attention_mask" in inputs:
            kw["attention_mask"] = inputs["attention_mask"].to(dev)
        with torch.no_grad():
            logits = model(**kw).logits
        _ = processor.batch_decode(logits.argmax(dim=-1))
        if dev == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        return len(wav) / 16000, t1 - t0, t2 - t1

    for i in warm:
        run(i)
    rows = [run(i) for i in timed]
    dur = sum(r[0] for r in rows)
    feat = sum(r[1] for r in rows)
    mod = sum(r[2] for r in rows)
    lat = [r[1] + r[2] for r in rows]
    res = {
        "name": args.name, "model": args.model, "device": dev, "n": len(rows),
        "audio_seconds": round(dur, 2), "rtf_total": round((feat + mod) / dur, 5), "rtf_model": round(mod / dur, 5),
        "rtf_features": round(feat / dur, 5), "median_latency_s": round(statistics.median(lat), 4),
        "p90_latency_s": round(sorted(lat)[int(0.9 * len(lat)) - 1], 4),
        "hardware": torch.cuda.get_device_name(0) if dev == "cuda" else f"CPU, {args.threads} threads",
        "dtype": str(dtype), "torch": torch.__version__,
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
    }
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("[rtf]", res, flush=True)


if __name__ == "__main__":
    main()
