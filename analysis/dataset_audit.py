"""Read-only metadata audit of the WAXAL-Ethiopic and snapwre Amharic datasets used by Ethio-ASR.

Reads only metadata columns from the cached parquet files (no audio is decoded, nothing is
written on the cluster). Prints a text report followed by one JSON line starting with
'AUDIT_JSON ' that holds the numbers.
"""
import collections
import glob
import json
import os
import re
import sys
import unicodedata

import pyarrow.parquet as pq

P = "/path/to/Ethio-ASR"
HUB = P + "/hf_cache/hub"
WAX = glob.glob(HUB + "/datasets--badrex--waxalNLP-ethiopic-final/snapshots/*/data")[0]
SNP = glob.glob(HUB + "/datasets--snapwre--amharic-speech/snapshots/*/data")[0]
CFG = P + "/Ethio-ASR/reproduction_uf/configs/04_paper_w2v-bert-2.0.yaml"

m = re.search(r'character_set:\s*["\'](.*)["\']\s*$', open(CFG, encoding="utf-8").read(), re.M)
CHARSET = set(m.group(1).replace("''", "'")) if m else set()

PUNCT = set(".,;:?!\"'()[]{}\\/|<>-_=+*&^%$#@~`") | set("።፣፤፥፦፧፨፠፡")


def norm(t):
    t = unicodedata.normalize("NFC", t or "").lower()
    t = "".join(" " if c in PUNCT else c for c in t)
    return " ".join(t.split())


def read(dirpath, split, cols):
    files = sorted(glob.glob(f"{dirpath}/{split}-*.parquet"))
    rows = collections.defaultdict(list)
    for f in files:
        schema_names = pq.read_schema(f).names
        use = [c for c in cols if c in schema_names]
        t = pq.read_table(f, columns=use)
        for c in use:
            rows[c].extend(t.column(c).to_pylist())
    return dict(rows), len(files)


out = {"waxal": {}, "snapwre": {}, "cross": {}}
print("character_set size from config:", len(CHARSET))

# ------------------------------------------------------------------ WAXAL
wcols = ["id", "speaker_id", "transcription", "language", "gender", "audio_duration"]
W = {}
for split in ("train", "validation", "test"):
    W[split], nf = read(WAX, split, wcols)
    print(f"WAXAL {split}: {len(W[split]['id'])} rows from {nf} files; columns {sorted(W[split])}")

langs = sorted(set(W["train"]["language"]))
print("languages:", langs)
print("\n== WAXAL per split and language: rows, hours, speakers, gender hours, duration range")
for split in ("train", "validation", "test"):
    d = W[split]
    for lang in langs:
        idx = [i for i, l in enumerate(d["language"]) if l == lang]
        dur = [d["audio_duration"][i] or 0.0 for i in idx]
        spk = {d["speaker_id"][i] for i in idx}
        gh = collections.Counter()
        for i in idx:
            gh[str(d["gender"][i]).lower()] += (d["audio_duration"][i] or 0.0) / 3600
        rec = {"rows": len(idx), "hours": round(sum(dur) / 3600, 2), "speakers": len(spk),
               "gender_hours": {k: round(v, 2) for k, v in sorted(gh.items())},
               "dur_min": round(min(dur), 2) if dur else None, "dur_max": round(max(dur), 2) if dur else None,
               "dur_mean": round(sum(dur) / len(dur), 2) if dur else None}
        out["waxal"].setdefault(split, {})[lang] = rec
        print(f"  {split:10s} {lang:4s} rows {rec['rows']:6d}  hours {rec['hours']:7.2f}  speakers {rec['speakers']:4d}  "
              f"gender_h {rec['gender_hours']}  dur {rec['dur_min']}-{rec['dur_max']} (mean {rec['dur_mean']})")

print("\n== WAXAL speaker overlap between splits (per language)")
for lang in langs:
    s = {sp: {W[sp]["speaker_id"][i] for i, l in enumerate(W[sp]["language"]) if l == lang} for sp in W}
    tr_te = s["train"] & s["test"]
    tr_va = s["train"] & s["validation"]
    va_te = s["validation"] & s["test"]
    te_rows = [i for i, l in enumerate(W["test"]["language"]) if l == lang]
    te_in_train = sum(1 for i in te_rows if W["test"]["speaker_id"][i] in s["train"])
    rec = {"train_test_speakers": len(tr_te), "train_val_speakers": len(tr_va), "val_test_speakers": len(va_te),
           "test_utts_with_train_speaker_pct": round(100 * te_in_train / max(1, len(te_rows)), 2)}
    out["waxal"].setdefault("speaker_overlap", {})[lang] = rec
    print(f"  {lang}: {rec}")
all_spk = {sp: set(W[sp]["speaker_id"]) for sp in W}
print("  any language: train&test", len(all_spk["train"] & all_spk["test"]), "| train&val", len(all_spk["train"] & all_spk["validation"]))

print("\n== WAXAL transcript checks")
train_norm = collections.Counter(norm(t) for t in W["train"]["transcription"])
for split in ("train", "validation", "test"):
    d = W[split]
    for lang in langs:
        idx = [i for i, l in enumerate(d["language"]) if l == lang]
        texts = [d["transcription"][i] or "" for i in idx]
        empty = sum(1 for t in texts if not t.strip())
        short = sum(1 for t in texts if 0 < len(t.strip()) < 3)
        nt = [norm(t) for t in texts]
        dup_within = len(nt) - len(set(nt))
        in_train = sum(1 for t in nt if t and train_norm.get(t, 0) > 0) if split != "train" else None
        cps = []
        for i in idx:
            dur = d["audio_duration"][i] or 0
            if dur > 0:
                cps.append(len((d["transcription"][i] or "").replace(" ", "")) / dur)
        cps_hi = sum(1 for c in cps if c > 25)
        cps_lo = sum(1 for c in cps if c < 1.5)
        chars = collections.Counter()
        for t in texts:
            chars.update(unicodedata.normalize("NFC", t).lower())
        tot = sum(chars.values())
        oov = {c: n for c, n in chars.items() if c not in CHARSET and c != " "}
        oov_n = sum(oov.values())
        top_oov = sorted(oov.items(), key=lambda x: -x[1])[:6]
        rec = {"empty": empty, "very_short": short, "dup_within_split": dup_within,
               "in_train_text": in_train, "in_train_text_pct": None if in_train is None else round(100 * in_train / max(1, len(idx)), 2),
               "cps_gt25": cps_hi, "cps_lt1.5": cps_lo, "oov_char_pct": round(100 * oov_n / max(1, tot), 3),
               "top_oov": [(f"U+{ord(c):04X}", n) for c, n in top_oov]}
        out["waxal"].setdefault("text", {}).setdefault(split, {})[lang] = rec
        print(f"  {split:10s} {lang}: {rec}")
ids = collections.Counter(i for sp in W for i in W[sp]["id"])
print("  duplicate ids across all splits:", sum(1 for v in ids.values() if v > 1))
out["waxal"]["duplicate_ids"] = sum(1 for v in ids.values() if v > 1)

# ------------------------------------------------------------------ snapwre
scols = ["clip_id", "sentence", "speaker_id", "language", "duration_s", "speech_s", "gender", "age_band",
         "region", "sample_rate", "up_votes", "down_votes"]
S = {}
print()
for split in ("train", "validation", "test"):
    S[split], nf = read(SNP, split, scols)
    d = S[split]
    n = len(d["clip_id"])
    hours = sum(x or 0 for x in d["duration_s"]) / 3600
    gh = collections.Counter()
    for g, x in zip(d["gender"], d["duration_s"]):
        gh[str(g).lower()] += (x or 0) / 3600
    rec = {"rows": n, "hours": round(hours, 2), "speakers": len(set(d["speaker_id"])),
           "gender_hours": {k: round(v, 2) for k, v in sorted(gh.items())},
           "sample_rates": dict(collections.Counter(d["sample_rate"]).most_common(4)),
           "age_bands": dict(collections.Counter(d["age_band"]).most_common(6)),
           "regions": len(set(d["region"])),
           "down_gt_up": sum(1 for u, dn in zip(d["up_votes"], d["down_votes"]) if (dn or 0) > (u or 0)),
           "empty": sum(1 for t in d["sentence"] if not (t or "").strip())}
    out["snapwre"][split] = rec
    print(f"snapwre {split}: {rec}")
ss = {sp: set(S[sp]["speaker_id"]) for sp in S}
rec = {"train_test_speakers": len(ss["train"] & ss["test"]), "train_val_speakers": len(ss["train"] & ss["validation"]),
       "val_test_speakers": len(ss["validation"] & ss["test"])}
out["snapwre"]["speaker_overlap"] = rec
print("snapwre speaker overlap:", rec)
snp_train = collections.Counter(norm(t) for t in S["train"]["sentence"])
for split in ("validation", "test"):
    nt = [norm(t) for t in S[split]["sentence"]]
    k = sum(1 for t in nt if t and snp_train.get(t, 0) > 0)
    out["snapwre"].setdefault("text_in_train_pct", {})[split] = round(100 * k / max(1, len(nt)), 2)
    print(f"snapwre {split} sentences also in snapwre train: {k}/{len(nt)}")
nt = [norm(t) for t in S["train"]["sentence"]]
out["snapwre"]["dup_within_train"] = len(nt) - len(set(nt))
print("snapwre duplicate sentences within train:", len(nt) - len(set(nt)))

# ------------------------------------------------------------------ cross-dataset (Amharic)
am = "amh" if "amh" in langs else langs[0]
wax_am_train = collections.Counter(norm(W["train"]["transcription"][i]) for i, l in enumerate(W["train"]["language"]) if l == am)
wax_am_test = [norm(W["test"]["transcription"][i]) for i, l in enumerate(W["test"]["language"]) if l == am]
snp_test = [norm(t) for t in S["test"]["sentence"]]
c1 = sum(1 for t in snp_test if t and wax_am_train.get(t, 0) > 0)
c2 = sum(1 for t in wax_am_test if t and snp_train.get(t, 0) > 0)
wspk = {str(x) for sp in W for x in W[sp]["speaker_id"]}
sspk = {str(x) for sp in S for x in S[sp]["speaker_id"]}
out["cross"] = {"snapwre_test_text_in_waxal_am_train": c1, "waxal_am_test_text_in_snapwre_train": c2,
                "speaker_id_strings_shared": len(wspk & sspk)}
print("\ncross-dataset:", out["cross"])
print("AUDIT_JSON " + json.dumps(out, ensure_ascii=True))
