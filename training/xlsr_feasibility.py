"""CPU check: how many WAXAL utterances can CTC align with and without the XLS-R adapter?

CTC needs at least U + R output frames for a label of U tokens with R adjacent repeats.
Output frames of a wav2vec2-family encoder (conv kernels 10,3,3,3,3,2,2; strides 5,2,2,2,2,2,2)
and of the optional adapter (3 conv layers, kernel 3, stride 2, padding 1) follow the same
formulas as transformers' _get_feat_extract_output_lengths. Labels follow the training text
pipeline closely (NFC, lower case, configured character set, one language token).
Reads only metadata columns from the parquet files.
"""
import glob
import json
import re
import unicodedata

import pyarrow.parquet as pq

P = "/path/to/Ethio-ASR"
WAX = glob.glob(P + "/hf_cache/hub/datasets--badrex--waxalNLP-ethiopic-final/snapshots/*/data")[0]
CFG = P + "/configs/05_new_xls-r-1b.yaml"
m = re.search(r"character_set:\s*'(.*)'\s*$", open(CFG, encoding="utf-8").read(), re.M)
CHARSET = set(m.group(1).replace("''", "'"))


def conv_len(n_samples):
    t = n_samples
    for k, s in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
        t = (t - k) // s + 1
    return t


def adapter_len(t, layers=3):
    for _ in range(layers):
        t = (t + 2 * 1 - 3) // 2 + 1
    return t


out = {}
for split in ("train", "validation", "test"):
    rows = {"n": 0, "infeasible_adapter": 0, "infeasible_no_adapter": 0, "label_tokens": 0}
    per_lang = {}
    for f in sorted(glob.glob(f"{WAX}/{split}-*.parquet")):
        t = pq.read_table(f, columns=["transcription", "language", "audio_duration"]).to_pydict()
        for text, lang, dur in zip(t["transcription"], t["language"], t["audio_duration"]):
            s = "".join(c for c in unicodedata.normalize("NFC", text or "").lower() if c in CHARSET)
            s = " ".join(s.split())
            u = 1 + len(s)                      # language token + characters (spaces are word delimiters)
            r = sum(1 for a, b in zip(s, s[1:]) if a == b)
            tc = conv_len(int(round((dur or 0) * 16000)))
            ta = adapter_len(tc)
            pl = per_lang.setdefault(lang, {"n": 0, "infeasible_adapter": 0, "infeasible_no_adapter": 0})
            for d in (rows, pl):
                d["n"] += 1
                d["infeasible_adapter"] += int(ta < u + r)
                d["infeasible_no_adapter"] += int(tc < u + r)
            rows["label_tokens"] += u
    rows["pct_infeasible_adapter"] = round(100 * rows["infeasible_adapter"] / rows["n"], 2)
    rows["pct_infeasible_no_adapter"] = round(100 * rows["infeasible_no_adapter"] / rows["n"], 2)
    for pl in per_lang.values():
        pl["pct_infeasible_adapter"] = round(100 * pl["infeasible_adapter"] / pl["n"], 2)
    rows["per_language"] = per_lang
    out[split] = rows
    print(split, {k: v for k, v in rows.items() if k != "per_language"})
    print("   per language, % infeasible with adapter:", {k: v["pct_infeasible_adapter"] for k, v in sorted(per_lang.items())})
print("XLSR_JSON " + json.dumps(out))
