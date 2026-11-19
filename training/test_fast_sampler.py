"""Check that train_fast_sampler.py gives the same sampler order as transformers' own path.

Small synthetic dataset, CPU only (safe on a login node):
    python3 test_fast_sampler.py /path/to/train_fast_sampler.py
"""
import importlib.util
import sys
import types

import datasets
import numpy as np
import torch
from transformers import Trainer

spec = importlib.util.spec_from_file_location("tfs", sys.argv[1])
tfs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tfs)  # not __main__, so nothing is patched or run

rng = np.random.default_rng(0)
rows = [{"input_values": rng.standard_normal(int(n)).astype("float32").tolist(), "labels": [1, 2, 3]}
        for n in rng.integers(100, 3000, 400)]
full = datasets.Dataset.from_list(rows)
full = full.map(lambda b: {"length": [len(x) for x in b["input_values"]]}, batched=True)
reduced = full.remove_columns(["length"])  # what Trainer._get_dataloader hands the sampler

args = types.SimpleNamespace(group_by_length=True, length_column_name="length",
                             train_batch_size=4, gradient_accumulation_steps=8,
                             eval_batch_size=4, use_legacy_prediction_loop=False)
fake = types.SimpleNamespace(args=args,
                             processing_class=types.SimpleNamespace(model_input_names=["input_values", "attention_mask"]),
                             train_dataset=full, eval_dataset=full)

ok = True
for label, orig_fn, new_fn in (("train", Trainer._get_train_sampler, tfs._train_sampler),
                               ("eval", Trainer._get_eval_sampler, tfs._eval_sampler)):
    s_orig = orig_fn(fake, reduced)
    s_new = new_fn(fake, reduced)
    same_lengths = list(s_orig.lengths) == list(s_new.lengths)
    for seed in (42, 1234):
        torch.manual_seed(seed)
        a = list(iter(s_orig))
        torch.manual_seed(seed)
        b = list(iter(s_new))
        print(f"{label}: seed {seed}: lengths identical {same_lengths}, order identical {a == b}, n={len(a)}")
        ok = ok and same_lengths and a == b

# Fallback: with the column present, the original code path is used unchanged.
s = tfs._train_sampler(fake, full)
print("column present -> original path, lengths from column:", list(s.lengths) == full["length"])
print("ALL OK" if ok else "MISMATCH")
