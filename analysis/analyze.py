#!/usr/bin/env python3
"""Metrics from the per-utterance predictions of eval_v2 (CPU only).

Reads  assessment_2026-11/predictions/*.jsonl  and  assessment_2026-11/rtf/*.json
Writes assessment_2026-11/results/analysis.json  and  results/summary.txt

Text handling is imported from slurm_scripts/evaluate_ctc.py (strip_lang_token, paper_postprocess),
so the corpus-level WER/CER here are computed exactly as in the original evaluation; the script
checks this against the earlier eval_test*.json files. On top of that it adds:
  - 95% confidence intervals from a speaker-level cluster bootstrap within each language
    (speakers resampled with replacement, 2,000 resamples, seed 1234);
  - paired differences (retrained minus released, C minus A) using the same resamples;
  - language identification: accuracy over all utterances (a missing tag counts as an error)
    and the confusion counts;
  - per-gender WER per language (post-processed) and the male-minus-female gap with its interval;
  - vowel-length and gemination normalisation for the Latin-script languages
    (same functions as slurm_scripts/detailed_test_eval.py).

Options (environment variables, added 2026-11-08; the defaults give exactly the earlier output):
  TAG_SCORING=leading  only the leading language tag is removed from a hypothesis, so a tag repeated later
                       counts as an error, as in the original evaluation code (default);
  TAG_SCORING=all      every language tag is removed from the hypothesis before scoring (sensitivity analysis;
                       language identification still uses the leading tag);
  ANALYSIS_OUT=<dir>   write analysis.json and summary.txt to <dir> instead of results/.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

P = Path("/path/to/Ethio-ASR")
A = P / "assessment_2026-11"
sys.path.insert(0, str(P / "slurm_scripts"))
from evaluate_ctc import LANG_TOKEN_RE, paper_postprocess, strip_lang_token  # noqa: E402
import evaluate  # noqa: E402
import jiwer  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

LANGS = ["amh", "orm", "sid", "tir", "wal"]
TAG_SCORING = os.environ.get("TAG_SCORING", "leading")
assert TAG_SCORING in ("leading", "all"), TAG_SCORING
OUTDIR = Path(os.environ.get("ANALYSIS_OUT", str(A / "results")))
B = 2000
RNG = np.random.default_rng(1234)
WER_M, CER_M = evaluate.load("wer"), evaluate.load("cer")


def vowel_length_norm(t):
    for v in "aeiou":
        t = re.sub(v + "+", v, t)
    return t


def geminate_norm(t):
    for c in "bcdfgjklmnpqrstvwxyz":
        t = re.sub(c + "+", c, t)
    return t


def counts(refs, hyps):
    """Per-utterance (errors, reference length) at word and character level."""
    we, wn, ce, cn = [], [], [], []
    for r, h in zip(refs, hyps):
        if r.strip():
            w = jiwer.process_words(r, h)
            c = jiwer.process_characters(r, h)
            we.append(w.substitutions + w.deletions + w.insertions); wn.append(w.substitutions + w.deletions + w.hits)
            ce.append(c.substitutions + c.deletions + c.insertions); cn.append(c.substitutions + c.deletions + c.hits)
        else:
            we.append(len(h.split())); wn.append(0); ce.append(len(h)); cn.append(0)
    return np.array(we), np.array(wn), np.array(ce), np.array(cn)


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    for r in rows:
        hyp, tag = strip_lang_token(r["hyp"])
        if TAG_SCORING == "all":
            hyp = LANG_TOKEN_RE.sub(" ", hyp)  # a repeated tag is often glued to the last word ("baala.[SID]")
        ref, _ = strip_lang_token(r["ref"])
        r["hyp_raw"], r["ref_raw"] = hyp.lower(), ref.lower()
        r["hyp_pp"], r["ref_pp"] = paper_postprocess(r["hyp_raw"]), paper_postprocess(r["ref_raw"])
    return rows


def boot_groups(keys):
    """Index arrays for a cluster bootstrap: B resamples of the unique keys."""
    uniq = sorted(set(keys))
    pos = {k: i for i, k in enumerate(uniq)}
    member = np.array([pos[k] for k in keys])
    draws = RNG.integers(0, len(uniq), size=(B, len(uniq)))
    weight = np.zeros((B, len(uniq)))
    for b in range(B):
        np.add.at(weight[b], draws[b], 1)
    return member, weight        # utterance -> cluster, and B x clusters multiplicity


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# ---------------------------------------------------------------- multilingual models
files = sorted(glob.glob(str(A / "predictions" / "*__waxal.jsonl")))
models = {Path(f).name.split("__")[0]: load(f) for f in files}
names = sorted(models)
ref_model = models[names[0]]
for n in names:  # every file must list the same utterances in the same order
    assert [r["id"] for r in models[n]] == [r["id"] for r in ref_model], n
lang = np.array([r["language"] for r in ref_model])
spk = [f'{r["language"]}:{r["speaker_id"]}' for r in ref_model]
gender = np.array([str(r["gender"]).lower() for r in ref_model])
member, weight = {}, {}
for L in LANGS:
    idx = np.where(lang == L)[0]
    member[L], weight[L] = boot_groups([spk[i] for i in idx])

out = {"n_utterances": len(ref_model), "bootstrap": {"resamples": B, "unit": "speaker within language", "seed": 1234},
       "models": {}, "paired": {}, "rtf": {}, "combined": {}}
per_utt = {}
for n in names:
    rows = models[n]
    m = {"per_language": {}, "lid": {}, "gender": {}, "phonology": {}}
    boot_wer = np.zeros(B); boot_cer = np.zeros(B)
    pu = {}
    for L in LANGS:
        idx = np.where(lang == L)[0]
        R = [rows[i] for i in idx]
        res = {"n": len(idx), "speakers": int(weight[L].shape[1])}
        for kind in ("raw", "pp"):
            refs = [r["ref_" + kind] for r in R]; hyps = [r["hyp_" + kind] for r in R]
            we, wn, ce, cn = counts(refs, hyps)
            wer_eval = WER_M.compute(predictions=hyps, references=refs)
            cer_eval = CER_M.compute(predictions=hyps, references=refs)
            res[kind] = {"wer": wer_eval, "cer": cer_eval, "wer_from_counts": we.sum() / wn.sum(), "cer_from_counts": ce.sum() / cn.sum()}
            if kind == "pp":
                Wc = np.zeros(weight[L].shape[1]); Wn = np.zeros_like(Wc); Cc = np.zeros_like(Wc); Cn = np.zeros_like(Wc)
                np.add.at(Wc, member[L], we); np.add.at(Wn, member[L], wn); np.add.at(Cc, member[L], ce); np.add.at(Cn, member[L], cn)
                bw = (weight[L] @ Wc) / (weight[L] @ Wn); bc = (weight[L] @ Cc) / (weight[L] @ Cn)
                res["pp"]["wer_ci"], res["pp"]["cer_ci"] = ci(bw), ci(bc)
                boot_wer += bw / len(LANGS); boot_cer += bc / len(LANGS)
                pu[L] = (Wc, Wn)
        m["per_language"][L] = res
        # language identification (tags are codes such as AMH; the dataset column holds codes)
        tags = [(r["pred_tag"] or "none").lower() for r in R]
        conf = defaultdict(int)
        for t in tags:
            conf[t] += 1
        m["lid"][L] = {"correct": conf.get(L, 0), "n": len(R), "confusion": dict(conf)}
        # gender (post-processed), Wolaytta has essentially no male test speech
        if L != "wal":
            g = {}
            for G in ("male", "female"):
                sub = [r for r, gg in zip(R, gender[idx]) if gg == G]
                if sub:
                    g[G] = {"n": len(sub), "wer": WER_M.compute(predictions=[r["hyp_pp"] for r in sub], references=[r["ref_pp"] for r in sub])}
            # bootstrap of the gap: resample speakers within the language (each speaker has one gender)
            we, wn, _, _ = counts([r["ref_pp"] for r in R], [r["hyp_pp"] for r in R])
            ismale = np.array([gg == "male" for gg in gender[idx]])
            Wm = np.zeros(weight[L].shape[1]); Nm = np.zeros_like(Wm); Wf = np.zeros_like(Wm); Nf = np.zeros_like(Wm)
            np.add.at(Wm, member[L][ismale], we[ismale]); np.add.at(Nm, member[L][ismale], wn[ismale])
            np.add.at(Wf, member[L][~ismale], we[~ismale]); np.add.at(Nf, member[L][~ismale], wn[~ismale])
            with np.errstate(invalid="ignore", divide="ignore"):
                gap = (weight[L] @ Wm) / (weight[L] @ Nm) - (weight[L] @ Wf) / (weight[L] @ Nf)
            gap = gap[np.isfinite(gap)]
            if "male" in g and "female" in g:
                g["gap_male_minus_female"] = g["male"]["wer"] - g["female"]["wer"]
                g["gap_ci"] = ci(gap) if len(gap) else None
                g["speakers"] = {"male": int(len(set(np.array(spk)[idx][ismale]))), "female": int(len(set(np.array(spk)[idx][~ismale])))}
            m["gender"][L] = g
        if L in ("orm", "sid", "wal"):
            ph = {}
            for nm, fn in (("vowel", vowel_length_norm), ("gemination", geminate_norm), ("both", lambda t: geminate_norm(vowel_length_norm(t)))):
                ph[nm] = WER_M.compute(predictions=[fn(r["hyp_pp"]) for r in R], references=[fn(r["ref_pp"]) for r in R])
            ph["none"] = res["pp"]["wer"]
            m["phonology"][L] = ph
    m["macro"] = {k: float(np.mean([m["per_language"][L][k]["wer"] for L in LANGS])) for k in ("raw", "pp")}
    m["macro_cer"] = {k: float(np.mean([m["per_language"][L][k]["cer"] for L in LANGS])) for k in ("raw", "pp")}
    m["macro"]["pp_ci"], m["macro_cer"]["pp_ci"] = ci(boot_wer), ci(boot_cer)
    tot = sum(v["n"] for v in m["lid"].values()); cor = sum(v["correct"] for v in m["lid"].values())
    m["lid_overall"] = {"accuracy": cor / tot, "correct": cor, "n": tot,
                        "missing_tag": sum(v["confusion"].get("none", 0) for v in m["lid"].values())}
    out["models"][n] = m
    per_utt[n] = pu
    print(f"{n:22s} macro pp WER {100 * m['macro']['pp']:.2f} [{100 * m['macro']['pp_ci'][0]:.2f}, {100 * m['macro']['pp_ci'][1]:.2f}]  LID {100 * m['lid_overall']['accuracy']:.2f}%", flush=True)

# paired differences: retrained minus released, same speaker resamples
PAIRS = [("ours_afrihubert", "released_94M"), ("ours_mms300m", "released_300M"), ("ours_mms1b", "released_1B"),
         ("ours_w2vbert", "released_600M"), ("ours_w2vbert_43k", "ours_w2vbert"),
         # added 2026-11-07: XLS-R-1B without the adapter (job 50160141) against MMS-1B, the same
         # wav2vec 2.0 architecture and size with different pre-training data; skipped until both exist
         ("ours_xlsr_noadapter", "ours_mms1b"),
         # added 2026-11-07: the two best released checkpoints, to test the published ranking (600M best)
         ("released_600M", "released_1B")]
for a, b in PAIRS:
    if a in per_utt and b in per_utt:
        d = np.zeros(B)
        for L in LANGS:
            (Wa, Na), (Wb, Nb) = per_utt[a][L], per_utt[b][L]
            d += ((weight[L] @ Wa) / (weight[L] @ Na) - (weight[L] @ Wb) / (weight[L] @ Nb)) / len(LANGS)
        point = out["models"][a]["macro"]["pp"] - out["models"][b]["macro"]["pp"]
        out["paired"][f"{a}-minus-{b}"] = {"diff": point, "ci": ci(d), "share_below_zero": float((d < 0).mean())}

# consistency with the earlier evaluation files
OLD = {"released_94M": "eval_results/paper_published_pp/94M_test_pp.json", "released_300M": "eval_results/paper_published_pp/300M_test_pp.json",
       "released_1B": "eval_results/paper_published_pp/1B_test_pp.json", "released_600M": "eval_results/paper_published_pp/600M_test_pp.json",
       "ours_afrihubert": "Ethio-ASR/reproduction_uf/results/ours/ajesujoba_AfriHuBERT-11062026-234348__eval_test_pp.json",
       "ours_mms300m": "Ethio-ASR/reproduction_uf/results/ours/facebook_mms-300m-09062026-205411__eval_test_pp.json",
       "ours_mms1b": "Ethio-ASR/reproduction_uf/results/ours/facebook_mms-1b-09062026-203848__eval_test_pp.json",
       "ours_w2vbert": "Ethio-ASR/reproduction_uf/results/ours/facebook_w2v-bert-2.0-09062026-212456__eval_test_pp.json",
       "ours_w2vbert_43k": "Ethio-ASR/reproduction_uf/results/ours/facebook_w2v-bert-2.0-10062026-003217__eval_test_pp.json",
       "ours_xlsr_adapter": "Ethio-ASR/reproduction_uf/results/ours/facebook_wav2vec2-xls-r-1b-09062026-230639__eval_test_pp.json"}
out["consistency"] = {}
for n, rel in OLD.items():
    if n in out["models"] and (P / rel).exists():
        old = json.load(open(P / rel))["averages"]["macro_wer"]
        out["consistency"][n] = {"old_macro_pp": old, "new_macro_pp": out["models"][n]["macro"]["pp"],
                                 "diff_pp_points": 100 * (out["models"][n]["macro"]["pp"] - old)}

# real-time factor
for f in sorted(glob.glob(str(A / "rtf" / "*.json"))):
    r = json.load(open(f)); out["rtf"][Path(f).stem] = r

# ---------------------------------------------------------------- combined Amharic experiment
cfiles = sorted(glob.glob(str(A / "predictions" / "*__combined_am.jsonl")))
if cfiles:
    snap = glob.glob(str(P / "hf_cache/hub/datasets--badrex--waxalNLP-ethiopic-final/snapshots/*/data"))[0]
    wspk = []
    for f in sorted(glob.glob(snap + "/test-*.parquet")):
        t = pq.read_table(f, columns=["language", "speaker_id"]).to_pydict()
        wspk += [f"waxal:{s}" for l, s in zip(t["language"], t["speaker_id"]) if l in ("amh", "amharic")]
    ssnap = glob.glob(str(P / "hf_cache/hub/datasets--snapwre--amharic-speech/snapshots/*/data"))[0]
    sspk = []
    for f in sorted(glob.glob(ssnap + "/test-*.parquet")):
        sspk += [f"snapwre:{s}" for s in pq.read_table(f, columns=["speaker_id"]).to_pydict()["speaker_id"]]
    cm = {Path(f).name.split("__")[0]: load(f) for f in cfiles}
    nW = len(wspk)
    keys = wspk + sspk
    n_all = len(next(iter(cm.values())))
    assert n_all == len(keys), (n_all, len(keys))
    parts = {"waxal_am": np.arange(0, nW), "snapwre": np.arange(nW, n_all), "combined": np.arange(0, n_all)}
    pw = {}
    for part, idx in parts.items():
        mem, wgt = boot_groups([keys[i] for i in idx])
        pw[part] = (mem, wgt)
    for n, rows in cm.items():
        out["combined"][n] = {}
        for part, idx in parts.items():
            R = [rows[i] for i in idx]
            refs = [r["ref_pp"] for r in R]; hyps = [r["hyp_pp"] for r in R]
            we, wn, ce, cn = counts(refs, hyps)
            mem, wgt = pw[part]
            Wc = np.zeros(wgt.shape[1]); Wn = np.zeros_like(Wc); np.add.at(Wc, mem, we); np.add.at(Wn, mem, wn)
            out["combined"][n][part] = {"n": len(idx), "speakers": int(wgt.shape[1]),
                                        "wer": WER_M.compute(predictions=hyps, references=refs),
                                        "cer": CER_M.compute(predictions=hyps, references=refs),
                                        "wer_ci": ci((wgt @ Wc) / (wgt @ Wn)), "_w": (Wc, Wn)}
    if "C_combined" in out["combined"] and "A_waxal_am" in out["combined"]:
        out["combined"]["paired_C_minus_A"] = {}
        for part in parts:
            (Wc, Wn), (Wa, Na) = out["combined"]["C_combined"][part]["_w"], out["combined"]["A_waxal_am"][part]["_w"]
            wgt = pw[part][1]
            d = (wgt @ Wc) / (wgt @ Wn) - (wgt @ Wa) / (wgt @ Na)
            out["combined"]["paired_C_minus_A"][part] = {
                "diff": out["combined"]["C_combined"][part]["wer"] - out["combined"]["A_waxal_am"][part]["wer"],
                "ci": ci(d), "share_below_zero": float((d < 0).mean())}
    for n in cm:
        for part in parts:
            out["combined"][n][part].pop("_w", None)
    # consistency with the earlier 3 x 3 results files
    OLDC = {"A_waxal_am": "A_waxal_am_baseline", "B_snapwre": "B_snapwre_only", "C_combined": "C_combined_am"}
    PARTF = {"waxal_am": "waxal_am", "snapwre": "snapwre_am", "combined": "combined_am"}
    out["consistency_combined"] = {}
    for n, stem in OLDC.items():
        for part, pf in PARTF.items():
            f = P / "combined_dataset_experiment/results" / f"{stem}__on__{pf}.json"
            if n in out["combined"] and f.exists():
                old = json.load(open(f))["averages"]["macro_wer"]
                out["consistency_combined"][f"{n}/{part}"] = {"old": old, "new": out["combined"][n][part]["wer"],
                                                             "diff_pp_points": 100 * (out["combined"][n][part]["wer"] - old)}

if TAG_SCORING != "leading":
    out["tag_scoring"] = TAG_SCORING  # recorded only off the default, so the default output is unchanged
OUTDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / "analysis.json").write_text(json.dumps(out, indent=1, default=float))
lines = []
for n, m in out["models"].items():
    lines.append(f"{n:22s} macro WER pp {100 * m['macro']['pp']:6.2f} CI [{100 * m['macro']['pp_ci'][0]:.2f}, {100 * m['macro']['pp_ci'][1]:.2f}]"
                 f"  raw {100 * m['macro']['raw']:6.2f}  CER pp {100 * m['macro_cer']['pp']:5.2f}  LID {100 * m['lid_overall']['accuracy']:6.2f}% (missing {m['lid_overall']['missing_tag']})")
for k, v in out["paired"].items():
    lines.append(f"paired {k}: {100 * v['diff']:+.2f} pp, CI [{100 * v['ci'][0]:+.2f}, {100 * v['ci'][1]:+.2f}]")
for k, v in out["consistency"].items():
    lines.append(f"consistency {k}: old {100 * v['old_macro_pp']:.3f} new {100 * v['new_macro_pp']:.3f} diff {v['diff_pp_points']:+.3f}")
for k, v in out["rtf"].items():
    lines.append(f"rtf {k}: total {v['rtf_total']:.4f} model {v['rtf_model']:.4f} median latency {v['median_latency_s']} s on {v['hardware']}")
if "paired_C_minus_A" in out["combined"]:
    for part, v in out["combined"]["paired_C_minus_A"].items():
        lines.append(f"combined C-A {part}: {100 * v['diff']:+.2f} pp CI [{100 * v['ci'][0]:+.2f}, {100 * v['ci'][1]:+.2f}]")
(OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
