#!/usr/bin/env python3
"""Build every number, table and figure of the Elsevier manuscript from measured files.

    python make_paper_assets.py OUT_DIR

Inputs (the cluster, assessment_2026-11/):
  results/analysis.json            metrics, intervals, LID, gender, phonology, RTF, combined study
  results_aux/data_audit.json      corpus statistics and split checks
  results_aux/xlsr_feasibility.json  CTC feasibility with and without the XLS-R adapter
  results_aux/validation_curves.json validation WER during training
Values published by Abdullah et al. (2026) are used only for attributed comparisons (REPORTED_*).
Outputs: OUT_DIR/numbers.tex, OUT_DIR/tables/*.tex, OUT_DIR/figures/*.pdf
The XLS-R-1B run without the output adapter (ours_xlsr_noadapter) is added to the tables and figures
only when analysis.json contains it. Sentences in main.tex that depend on the data are checked at the end
(CLAIM CHECK lines); the script exits with status 2 if any of them no longer holds.
ANALYSIS_JSON and CURVES_JSON may point to other input files (used only for layout tests).
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

A = Path("/path/to/Ethio-ASR/assessment_2026-11")
OUT = Path(sys.argv[1])
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

an = json.loads(Path(os.environ.get("ANALYSIS_JSON", A / "results/analysis.json")).read_text())
audit = json.loads((A / "results_aux/data_audit.json").read_text())
feas = json.loads((A / "results_aux/xlsr_feasibility.json").read_text())
curves = json.loads(Path(os.environ.get("CURVES_JSON", A / "results_aux/validation_curves.json")).read_text())["runs"]
MOD = an["models"]
LANGS = ["amh", "orm", "sid", "tir", "wal"]
LN = {"amh": "Amharic", "orm": "Oromo", "sid": "Sidaama", "tir": "Tigrinya", "wal": "Wolaytta"}
CODE = {k: k.upper() for k in LANGS}

# Published values of Abdullah et al. (2026): Table 3 (test WER %), Table 5 (LID %), Table 7 (gender WER %).
REPORTED_WER = {"released_94M": {"amh": 30.95, "tir": 42.42, "orm": 27.57, "wal": 40.44, "sid": 34.02, "macro": 35.08},
                "released_300M": {"amh": 30.19, "tir": 41.62, "orm": 26.41, "wal": 39.10, "sid": 32.66, "macro": 33.99},
                "released_1B": {"amh": 26.14, "tir": 37.63, "orm": 23.69, "wal": 37.51, "sid": 31.02, "macro": 31.20},
                "released_600M": {"amh": 22.92, "tir": 35.22, "orm": 24.44, "wal": 38.19, "sid": 31.65, "macro": 30.48}}
REPORTED_LID = {"released_94M": 99.92, "released_300M": 99.92, "released_1B": 99.91, "released_600M": 99.83}
REPORTED_GAP = {"released_94M": {"amh": 2.06, "orm": 3.23, "sid": -3.61, "tir": 18.54},
                "released_300M": {"amh": 1.95, "orm": 2.53, "sid": -3.14, "tir": 18.74},
                "released_1B": {"amh": 2.89, "orm": 3.76, "sid": -4.55, "tir": 18.20},
                "released_600M": {"amh": 3.81, "orm": 1.68, "sid": -1.39, "tir": 18.64}}

ROWS = [("released_94M", "ours_afrihubert", "AfriHuBERT", "94M"), ("released_300M", "ours_mms300m", "MMS-300M", "300M"),
        ("released_1B", "ours_mms1b", "MMS-1B", "1B"), ("released_600M", "ours_w2vbert", "w2v-BERT-2.0", "600M")]
PREFIX = {"released_94M": "RelA", "released_300M": "RelB", "released_1B": "RelC", "released_600M": "RelD",
          "ours_afrihubert": "OurA", "ours_mms300m": "OurB", "ours_mms1b": "OurC", "ours_w2vbert": "OurD",
          "ours_w2vbert_43k": "OurE", "ours_xlsr_adapter": "OurX", "ours_xlsr_noadapter": "OurXN"}
LABEL = {"released_94M": "Released 94M (AfriHuBERT)", "released_300M": "Released 300M (MMS-300M)",
         "released_1B": "Released 1B (MMS-1B)", "released_600M": "Released 600M (w2v-BERT-2.0)",
         "ours_afrihubert": "Retrained AfriHuBERT (fp32)", "ours_mms300m": "Retrained MMS-300M",
         "ours_mms1b": "Retrained MMS-1B", "ours_w2vbert": "Retrained w2v-BERT-2.0",
         "ours_w2vbert_43k": "Retrained w2v-BERT-2.0, 43.2k steps", "ours_xlsr_adapter": "Retrained XLS-R-1B, adapter on",
         "ours_xlsr_noadapter": "Retrained XLS-R-1B, adapter off"}
XN = "ours_xlsr_noadapter"
HAVE_XN = XN in an["models"]
WORDS = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

macros = []
claim_failures = []


def M(name, value):
    macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def check(condition, text):
    """A sentence in main.tex relies on this; report it loudly if the data no longer support it."""
    if not condition:
        claim_failures.append(text)


def p(x, d=2):
    return f"{100 * x:.{d}f}"


def sgn(x, d=2):
    return f"{100 * x:+.{d}f}".replace("-", "$-$").replace("+", "$+$")


def thousands(n):
    return f"{n:,}".replace(",", "{,}")


def num(x, d=2):
    """Percentage points with a typeset minus sign for negative values, no plus sign."""
    return f"{100 * x:.{d}f}".replace("-", "$-$")


# ------------------------------------------------------------------ dataset
tr = audit["waxal"]["train"]; va = audit["waxal"]["validation"]; te = audit["waxal"]["test"]
M("nTrainUtt", thousands(sum(tr[l]["rows"] for l in LANGS)))
M("nValUtt", thousands(sum(va[l]["rows"] for l in LANGS)))
M("nTestUtt", thousands(sum(te[l]["rows"] for l in LANGS)))
M("trainHours", f"{sum(tr[l]['hours'] for l in LANGS):.1f}")
M("nTrainSpk", thousands(sum(tr[l]["speakers"] for l in LANGS)))
M("nValSpk", thousands(sum(va[l]["speakers"] for l in LANGS)))
M("nTestSpk", thousands(sum(te[l]["speakers"] for l in LANGS)))
ov = audit["waxal"]["text"]["test"]
M("tirTextOverlap", f"{ov['tir']['in_train_text_pct']:.2f}")
M("otherTextOverlapMax", f"{max(ov[l]['in_train_text_pct'] for l in LANGS if l != 'tir'):.2f}")
sn = audit["snapwre"]
M("snpTrainUtt", thousands(sn["train"]["rows"])); M("snpTrainHours", f"{sn['train']['hours']:.1f}")
M("snpTrainSpk", sn["train"]["speakers"]); M("snpTestUtt", thousands(sn["test"]["rows"])); M("snpTestSpk", sn["test"]["speakers"])
M("snpValUtt", thousands(sn["validation"]["rows"]))
M("waxAmhHours", f"{tr['amh']['hours']:.0f}")
oov = [audit["waxal"]["text"]["train"][l]["oov_char_pct"] for l in LANGS]
M("oovMin", f"{min(oov):.2f}"); M("oovMax", f"{max(oov):.2f}")

with open(OUT / "tables/tab_data.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
    f.write("Language & Train h & Train utt. & Train spk. & Test h & Test utt. & Test spk. & Test male h\\\\\n\\midrule\n")
    for l in LANGS:
        f.write(f"{LN[l]} & {tr[l]['hours']:.1f} & {thousands(tr[l]['rows'])} & {tr[l]['speakers']} & {te[l]['hours']:.1f} & "
                f"{thousands(te[l]['rows'])} & {te[l]['speakers']} & {te[l]['gender_hours'].get('male', 0):.2f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ main comparison
for k, pre in PREFIX.items():
    if k not in MOD:
        continue
    m = MOD[k]
    M(f"werPP{pre}", p(m["macro"]["pp"])); M(f"werRaw{pre}", p(m["macro"]["raw"])); M(f"cerPP{pre}", p(m["macro_cer"]["pp"]))
    M(f"werLo{pre}", p(m["macro"]["pp_ci"][0])); M(f"werHi{pre}", p(m["macro"]["pp_ci"][1]))
    M(f"lid{pre}", p(m["lid_overall"]["accuracy"])); M(f"ppGain{pre}", p(m["macro"]["raw"] - m["macro"]["pp"]))
for k, v in REPORTED_WER.items():
    M(f"rep{PREFIX[k]}", f"{v['macro']:.2f}")
    M(f"repGap{PREFIX[k]}", f"{100 * MOD[k]['macro']['pp'] - v['macro']:.2f}")
PAIRKEY = {"A": "ours_afrihubert-minus-released_94M", "B": "ours_mms300m-minus-released_300M",
           "C": "ours_mms1b-minus-released_1B", "D": "ours_w2vbert-minus-released_600M", "E": "ours_w2vbert_43k-minus-ours_w2vbert",
           # XLS-R-1B without the adapter against MMS-1B, and the two best released checkpoints (published ranking)
           "XM": "ours_xlsr_noadapter-minus-ours_mms1b", "Top": "released_600M-minus-released_1B"}
for s, key in PAIRKEY.items():
    if key in an["paired"]:
        d = an["paired"][key]
        M(f"diff{s}", sgn(d["diff"])); M(f"diffLo{s}", sgn(d["ci"][0])); M(f"diffHi{s}", sgn(d["ci"][1]))
        M(f"diffAbs{s}", p(abs(d["diff"])))
rg = [100 * MOD[k]["macro"]["pp"] - v["macro"] for k, v in REPORTED_WER.items()]
M("repGapMin", f"{min(rg):.2f}"); M("repGapMax", f"{max(rg):.2f}")
gains = [MOD[k]["macro"]["raw"] - MOD[k]["macro"]["pp"] for k in PREFIX if k in MOD and k != "ours_xlsr_adapter"]
M("ppGainMin", p(min(gains))); M("ppGainMax", p(max(gains)))
M("pairCDMax", p(max(abs(an["paired"][PAIRKEY[s]]["diff"]) for s in "CD")))
# Agreement between this re-run and the earlier evaluation of the same checkpoints.
M("consistMax", f"{max(abs(v['diff_pp_points']) for v in an['consistency'].values()):.2f}")
if an.get("consistency_combined"):
    M("consistCmbMax", f"{max(abs(v['diff_pp_points']) for v in an['consistency_combined'].values()):.2f}")

with open(OUT / "tables/tab_main.tex", "w") as f:
    f.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
    f.write("Encoder & Model & Reported & WER raw & WER post-proc. [95\\% CI] & CER post-proc. & Retrained $-$ released [95\\% CI]\\\\\n\\midrule\n")
    for rel, ours, enc, size in ROWS:
        r, o = MOD[rel], MOD[ours]
        d = an["paired"].get(f"{ours}-minus-{rel}")
        f.write(f"{enc} ({size}) & Released & {REPORTED_WER[rel]['macro']:.2f} & {p(r['macro']['raw'])} & {p(r['macro']['pp'])} [{p(r['macro']['pp_ci'][0])}, {p(r['macro']['pp_ci'][1])}] & {p(r['macro_cer']['pp'])} & \\\\\n")
        dtxt = f"{sgn(d['diff'])} [{sgn(d['ci'][0])}, {sgn(d['ci'][1])}]" if d else "n/a"
        f.write(f" & Retrained & n/a & {p(o['macro']['raw'])} & {p(o['macro']['pp'])} [{p(o['macro']['pp_ci'][0])}, {p(o['macro']['pp_ci'][1])}] & {p(o['macro_cer']['pp'])} & {dtxt}\\\\\n")
        f.write("\\addlinespace\n")
    if HAVE_XN:
        # XLS-R-1B has no released checkpoint: both runs are listed, with no paired column.
        for k, what, enc in (("ours_xlsr_adapter", "Retrained, adapter on", "XLS-R-1B (1B)"), (XN, "Retrained, adapter off", "")):
            o = MOD[k]
            f.write(f"{enc} & {what} & n/a & {p(o['macro']['raw'])} & {p(o['macro']['pp'])} [{p(o['macro']['pp_ci'][0])}, {p(o['macro']['pp_ci'][1])}] & {p(o['macro_cer']['pp'])} & n/a\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
if HAVE_XN:
    M("xlsrAdapterGain", p(MOD["ours_xlsr_adapter"]["macro"]["pp"] - MOD[XN]["macro"]["pp"]))

# ------------------------------------------------------------------ per-language
order = ["released_94M", "ours_afrihubert", "released_300M", "ours_mms300m", "released_1B", "ours_mms1b",
         "released_600M", "ours_w2vbert", "ours_w2vbert_43k"] + ([XN] if HAVE_XN else [])
with open(OUT / "tables/tab_perlang.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\nModel & AMH & ORM & SID & TIR & WAL & Macro\\\\\n\\midrule\n")
    best = {l: min(MOD[k]["per_language"][l]["pp"]["wer"] for k in order if k in MOD) for l in LANGS}
    for k in order:
        if k not in MOD:
            continue
        cells = []
        for l in LANGS:
            v = MOD[k]["per_language"][l]["pp"]["wer"]
            cells.append(f"\\textbf{{{p(v)}}}" if abs(v - best[l]) < 1e-12 else p(v))
        f.write(f"{LABEL[k]} & " + " & ".join(cells) + f" & {p(MOD[k]['macro']['pp'])}\\\\\n")
    f.write("\\midrule\n")
    for rel in REPORTED_WER:
        f.write(f"{LABEL[rel]}, as reported & " + " & ".join(f"{REPORTED_WER[rel][l]:.2f}" for l in LANGS) + f" & {REPORTED_WER[rel]['macro']:.2f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
d600 = {l: 100 * MOD["released_600M"]["per_language"][l]["pp"]["wer"] - REPORTED_WER["released_600M"][l] for l in LANGS}
for l in LANGS:
    M(f"repLangGap{CODE[l].capitalize()}", f"{d600[l]:+.2f}".replace("-", "$-$").replace("+", "$+$"))
# Re-scored minus published, every released checkpoint and language.
allgaps = [100 * MOD[k]["per_language"][l]["pp"]["wer"] - REPORTED_WER[k][l] for k in REPORTED_WER for l in LANGS]
M("repLangGapMinAll", f"{min(allgaps):+.2f}".replace("-", "$-$").replace("+", "$+$"))
M("repLangGapMaxAll", f"{max(allgaps):+.2f}".replace("-", "$-$").replace("+", "$+$"))
M("repLangGapAbsMax", f"{max(abs(x) for x in allgaps):.1f}")
# Lead of the best model over the runner-up, per language.
margins = []
for l in LANGS:
    v = sorted(MOD[k]["per_language"][l]["pp"]["wer"] for k in order if k in MOD)
    margins.append(v[1] - v[0])
M("bestMarginMin", p(min(margins))); M("bestMarginMax", p(max(margins)))

# Language-specific behaviour over the eight models of the four original encoders (released and retrained).
main8 = [k for k in order[:8] if k in MOD]
for l in LANGS:
    v = [MOD[k]["per_language"][l]["pp"]["wer"] for k in main8]
    M(f"werMin{CODE[l].capitalize()}", p(min(v))); M(f"werMax{CODE[l].capitalize()}", p(max(v)))
for k in order:
    if k in MOD:
        w = {l: MOD[k]["per_language"][l]["pp"]["wer"] for l in LANGS}
        check(max(w, key=w.get) == "tir", f"Tigrinya is the hardest language for every model ({k})")
        check(min(w, key=w.get) in ("orm", "amh"), f"Oromo or Amharic is the easiest language for every model ({k})")
# Encoder-by-language interaction: w2v-BERT-2.0 minus MMS-1B, retrained models (negative: w2v-BERT-2.0 better).
for l in LANGS:
    d = MOD["ours_w2vbert"]["per_language"][l]["pp"]["wer"] - MOD["ours_mms1b"]["per_language"][l]["pp"]["wer"]
    M(f"wvMinusMms{CODE[l].capitalize()}", num(d)); M(f"wvMinusMms{CODE[l].capitalize()}Abs", p(abs(d)))
wv_sid_tir = max(abs(MOD["ours_w2vbert"]["per_language"][l]["pp"]["wer"] - MOD["ours_mms1b"]["per_language"][l]["pp"]["wer"]) for l in ("sid", "tir"))
M("wvMmsSidTirMax", p(wv_sid_tir))
# Spread between the hardest and easiest language for each model, against the spread between encoders.
spreads = [max(MOD[k]["per_language"][l]["pp"]["wer"] for l in LANGS) - min(MOD[k]["per_language"][l]["pp"]["wer"] for l in LANGS) for k in main8]
M("langSpreadMin", p(min(spreads))); M("langSpreadMax", p(max(spreads)))
enc_spread = max(MOD[k]["macro"]["pp"] for k in main8) - min(MOD[k]["macro"]["pp"] for k in main8)
M("encSpread", p(enc_spread))
check(min(spreads) > 2 * enc_spread, "the language spread is several times the encoder spread")
d_amh = MOD["ours_w2vbert"]["per_language"]["amh"]["pp"]["wer"] - MOD["ours_mms1b"]["per_language"]["amh"]["pp"]["wer"]
d_orm = MOD["ours_w2vbert"]["per_language"]["orm"]["pp"]["wer"] - MOD["ours_mms1b"]["per_language"]["orm"]["pp"]["wer"]
check(d_amh < 0 < d_orm, "w2v-BERT-2.0 is better than MMS-1B on Amharic and worse on Oromo (retrained)")
d_amh_r = MOD["released_600M"]["per_language"]["amh"]["pp"]["wer"] - MOD["released_1B"]["per_language"]["amh"]["pp"]["wer"]
d_orm_r = MOD["released_600M"]["per_language"]["orm"]["pp"]["wer"] - MOD["released_1B"]["per_language"]["orm"]["pp"]["wer"]
check(d_amh_r < 0 < d_orm_r, "the same holds for the released 600M and 1B checkpoints")
# CER is much lower than WER everywhere; a Ge'ez character is a syllable, so CER is not comparable across scripts.
cerw = {l: np.mean([MOD[k]["per_language"][l]["pp"]["cer"] for k in main8]) for l in LANGS}
for l in LANGS:
    M(f"cerMean{CODE[l].capitalize()}", p(cerw[l]))
check(cerw["orm"] <= cerw["sid"] <= cerw["wal"], "mean CER of the Latin-script languages runs from Oromo (lowest) to Wolaytta (highest)")
# Post-processing moves macro WER by more than the difference between most pairs of models.
pairs_d = [abs(MOD[a]["macro"]["pp"] - MOD[b]["macro"]["pp"]) for i, a in enumerate(main8) for b in main8[i + 1:]]
check(sum(d < min(gains) for d in pairs_d) > len(pairs_d) / 2,
      "the post-processing effect exceeds the difference between most pairs of models")

# ------------------------------------------------------------------ language identification
lid_models = main8 + ([XN] if HAVE_XN else [])
M("nLidModelsWord", WORDS[len(lid_models)])
lids = [MOD[k]["lid_overall"]["accuracy"] for k in lid_models]
M("lidMin", p(min(lids))); M("lidMax", p(max(lids)))
M("lidMissingMax", max(MOD[k]["lid_overall"]["missing_tag"] for k in lid_models))
relLids = [MOD[k]["lid_overall"]["accuracy"] for k in REPORTED_LID]
M("lidRelMin", p(min(relLids))); M("lidRelMax", p(max(relLids)))
M("lidRepMin", f"{min(REPORTED_LID.values()):.2f}"); M("lidRepMax", f"{max(REPORTED_LID.values()):.2f}")
M("lidRelRepAbsMax", f"{max(abs(100 * MOD[k]['lid_overall']['accuracy'] - v) for k, v in REPORTED_LID.items()):.2f}")
with open(OUT / "tables/tab_lid.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\nModel & AMH & ORM & SID & TIR & WAL & All & Reported\\\\\n\\midrule\n")
    for k in lid_models:
        m = MOD[k]
        cells = [p(m["lid"][l]["correct"] / m["lid"][l]["n"]) for l in LANGS]
        rep = f"{REPORTED_LID[k]:.2f}" if k in REPORTED_LID else "n/a"
        f.write(f"{LABEL[k]} & " + " & ".join(cells) + f" & {p(m['lid_overall']['accuracy'])} & {rep}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ repeated language tokens (sensitivity analysis)
# The primary scoring removes only the leading language token, so a token that a model emits again later in the
# transcript counts as an error, as in the original evaluation code. results_aux/tag_sensitivity/analysis.json repeats
# the whole analysis with every token removed from the hypotheses (same speaker resamples), and
# results_aux/repeated_tags.json counts the repetitions per model (count_repeated_tags.py).
TS_PATH, RT_PATH = A / "results_aux/tag_sensitivity/analysis.json", A / "results_aux/repeated_tags.json"
HAVE_TS = HAVE_XN and TS_PATH.exists() and RT_PATH.exists()
if HAVE_TS:
    ts = json.loads(TS_PATH.read_text()); TM = ts["models"]
    rt = json.loads(RT_PATH.read_text())["models"]
    check(ts.get("tag_scoring") == "all", "the sensitivity analysis removes every language token")
    reps = {k: rt[k]["overall"]["repeated"] / rt[k]["overall"]["n"] for k in lid_models}

    def rdiff(a, b):
        """a - b in points from the two values as printed (two decimals), so the table adds up for the reader."""
        d = round(100 * a, 2) - round(100 * b, 2)
        return 0.0 if abs(d) < 0.005 else d

    RC = ">{\\raggedleft\\arraybackslash}p{%s}"
    with open(OUT / "tables/tab_reptag.tex", "w") as f:
        f.write("\\begin{tabular}{l" + "".join(RC % w for w in ("2.1cm", "2.5cm", "2.5cm", "1.9cm")) + "}\n\\toprule\n"
                "Model & Repeated token (\\%) & WER, leading token removed & WER, every token removed & Difference\\\\\n\\midrule\n")
        for k in lid_models:
            a_, b_ = MOD[k]["macro"]["pp"], TM[k]["macro"]["pp"]
            f.write(f"{LABEL[k]} & {p(reps[k], 1)} & {p(a_)} & {p(b_)} & {num(rdiff(b_, a_) / 100)}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    for k in lid_models:
        M(f"repTag{PREFIX[k]}", p(reps[k], 1))
        M(f"werTagAll{PREFIX[k]}", p(TM[k]["macro"]["pp"]))
        M(f"werTagAllLo{PREFIX[k]}", p(TM[k]["macro"]["pp_ci"][0])); M(f"werTagAllHi{PREFIX[k]}", p(TM[k]["macro"]["pp_ci"][1]))
        M(f"tagGain{PREFIX[k]}", f"{rdiff(MOD[k]['macro']['pp'], TM[k]['macro']['pp']):.2f}")
    for s, key in PAIRKEY.items():
        if key in ts["paired"]:
            d = ts["paired"][key]
            M(f"diffTag{s}", sgn(d["diff"])); M(f"diffTagLo{s}", sgn(d["ci"][0])); M(f"diffTagHi{s}", sgn(d["ci"][1]))
            M(f"diffTagAbs{s}", p(abs(d["diff"])))
    xr = rt[XN]["overall"]
    M("repTagGluedPctOurXN", p(xr["glued"] / xr["repeated"], 1))
    M("tagGainOthersMax", f"{max(rdiff(MOD[k]['macro']['pp'], TM[k]['macro']['pp']) for k in main8):.2f}")
    M("repTagSmallMax", p(max(reps[k] for k in ("ours_afrihubert", "released_94M", "ours_mms300m", "released_300M")), 1))
    M("tagPairShiftMax", p(max(abs(ts["paired"][PAIRKEY[s]]["diff"] - an["paired"][PAIRKEY[s]]["diff"]) for s in "ABCD")))
    # statements in the text about the repeated tokens
    check(reps["ours_w2vbert"] == 0 and reps["released_600M"] == 0, "w2v-BERT-2.0 never repeats the language token")
    check(max(reps, key=reps.get) == XN, "XLS-R-1B without the adapter repeats the token most often")
    check(xr["at_end"] / xr["repeated"] > 0.99 and xr["glued"] / xr["repeated"] > 0.99,
          "XLS-R-1B puts the repeated token at the end, attached to the last word")
    check(all(TM[k]["macro"]["pp"] <= MOD[k]["macro"]["pp"] + 1e-12 for k in lid_models),
          "removing every token never raises macro WER")
    check(all(rt[k]["overall"]["at_end"] / rt[k]["overall"]["repeated"] > 0.9 for k in lid_models if rt[k]["overall"]["repeated"]),
          "a repeated token nearly always ends the transcript")
    check(all(TM[k]["lid_overall"] == MOD[k]["lid_overall"] for k in lid_models), "language identification is the same under both conventions")
    # XLS-R-1B against MMS-1B: worse in every language with the leading token removed, level with every token removed
    xm_lang_pri = [MOD[XN]["per_language"][l]["pp"]["wer"] - MOD["ours_mms1b"]["per_language"][l]["pp"]["wer"] for l in LANGS]
    xm_lang_all = [TM[XN]["per_language"][l]["pp"]["wer"] - TM["ours_mms1b"]["per_language"][l]["pp"]["wer"] for l in LANGS]
    M("xnMmsTagLangMax", p(max(abs(v) for v in xm_lang_all)))
    check(min(xm_lang_pri) > 0 and an["paired"][PAIRKEY["XM"]]["ci"][0] > 0,
          "with the leading token removed, XLS-R-1B is worse than MMS-1B in every language and overall")
    xm_all = ts["paired"][PAIRKEY["XM"]]
    check(xm_all["ci"][0] < 0 < xm_all["ci"][1] and max(abs(v) for v in xm_lang_all) < 0.005,
          "with every token removed, XLS-R-1B and MMS-1B are level overall and within half a point in every language")
    top_all = ts["paired"][PAIRKEY["Top"]]
    check(top_all["ci"][0] < 0 < top_all["ci"][1], "the released 600M and 1B checkpoints remain without a clear order")
    wv_all = {l: TM["ours_w2vbert"]["per_language"][l]["pp"]["wer"] - TM["ours_mms1b"]["per_language"][l]["pp"]["wer"] for l in LANGS}
    check(wv_all["amh"] < 0 < wv_all["orm"], "the w2v-BERT-2.0 and MMS-1B interaction persists with every token removed")

# ------------------------------------------------------------------ gender
tirgaps = [MOD[k]["gender"]["tir"]["gap_male_minus_female"] for k in main8]
M("tirGapMin", num(min(tirgaps))); M("tirGapMax", num(max(tirgaps)))
relgaps = [MOD[k]["gender"]["tir"]["gap_male_minus_female"] for k in REPORTED_GAP]
M("tirGapRelMin", num(min(relgaps), 1)); M("tirGapRelMax", num(max(relgaps), 1))
M("tirGapCiLoMin", num(min(MOD[k]["gender"]["tir"]["gap_ci"][0] for k in main8), 1))
M("tirGapCiHiMax", num(max(MOD[k]["gender"]["tir"]["gap_ci"][1] for k in main8), 1))
sidgaps = [MOD[k]["gender"]["sid"]["gap_male_minus_female"] for k in main8]
M("sidGapMin", num(min(sidgaps))); M("sidGapMax", num(max(sidgaps)))
amhgaps = [MOD[k]["gender"]["amh"]["gap_male_minus_female"] for k in main8]
M("amhGapMin", num(min(amhgaps))); M("amhGapMax", num(max(amhgaps)))
orm = [MOD[k]["gender"]["orm"]["gap_male_minus_female"] for k in main8]
M("ormGapMin", num(min(orm))); M("ormGapMax", num(max(orm)))
g = MOD["ours_w2vbert"]["gender"]["tir"]
M("tirGapD", num(g["gap_male_minus_female"])); M("tirGapDlo", num(g["gap_ci"][0])); M("tirGapDhi", num(g["gap_ci"][1]))
g = MOD["released_600M"]["gender"]["tir"]
M("tirGapRelD", num(g["gap_male_minus_female"])); M("tirGapRelDlo", num(g["gap_ci"][0])); M("tirGapRelDhi", num(g["gap_ci"][1]))
M("tirSpkMale", g["speakers"]["male"]); M("tirSpkFemale", g["speakers"]["female"])
M("tirSpkAll", g["speakers"]["male"] + g["speakers"]["female"])
check(min(tirgaps) > 0, "male speakers have higher Tigrinya WER for every model")
check(max(sidgaps) < 0, "female speakers have higher Sidaama WER for every model")
check(max(MOD[k]["gender"]["tir"]["gap_ci"][1] for k in main8) > 18.2 / 100,
      "the Tigrinya intervals reach the published 18 points (the text says sampling is not the explanation)")
with open(OUT / "tables/tab_gender.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\nModel & AMH & ORM & SID & TIR\\\\\n\\midrule\n")
    for k in main8:
        cells = []
        for l in ("amh", "orm", "sid", "tir"):
            gg = MOD[k]["gender"][l]
            cells.append(f"{sgn(gg['gap_male_minus_female'])} [{sgn(gg['gap_ci'][0])}, {sgn(gg['gap_ci'][1])}]")
        f.write(f"{LABEL[k]} & " + " & ".join(cells) + "\\\\\n")
    f.write("\\midrule\n")
    for k, v in REPORTED_GAP.items():
        f.write(f"{LABEL[k]}, as reported & " + " & ".join(f"{v[l]:+.2f}".replace('-', '$-$').replace('+', '$+$') for l in ("amh", "orm", "sid", "tir")) + "\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ phonology
with open(OUT / "tables/tab_phonology.tex", "w") as f:
    f.write("\\begin{tabular}{llrrrr}\n\\toprule\nModel & Language & Post-proc. & + vowel length & + gemination & + both\\\\\n\\midrule\n")
    for k in ["ours_afrihubert", "ours_mms300m", "ours_mms1b", "ours_w2vbert"]:
        for i, l in enumerate(("orm", "sid", "wal")):
            ph = MOD[k]["phonology"][l]
            name = LABEL[k] if i == 0 else ""
            f.write(f"{name} & {LN[l]} & {p(ph['none'])} & {p(ph['vowel'])} & {p(ph['gemination'])} & {p(ph['both'])}\\\\\n")
        f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
red_both = [MOD[k]["phonology"][l]["none"] - MOD[k]["phonology"][l]["both"] for k in main8 for l in ("orm", "sid", "wal")]
M("phonBothMin", p(min(red_both))); M("phonBothMax", p(max(red_both)))
vow = {l: np.mean([MOD[k]["phonology"][l]["none"] - MOD[k]["phonology"][l]["vowel"] for k in main8]) for l in ("orm", "sid", "wal")}
gem = {l: np.mean([MOD[k]["phonology"][l]["none"] - MOD[k]["phonology"][l]["gemination"] for k in main8]) for l in ("orm", "sid", "wal")}
for l in ("orm", "sid", "wal"):
    M(f"vowGain{CODE[l].capitalize()}", p(vow[l])); M(f"gemGain{CODE[l].capitalize()}", p(gem[l]))

# ------------------------------------------------------------------ real-time factor
rtf = an.get("rtf", {})
have_rtf = any(k.endswith("__cuda") for k in rtf)
rtf_models = main8 + ([XN] if HAVE_XN and f"{XN}__cuda" in rtf else [])
with open(OUT / "tables/tab_rtf.tex", "w") as f:
    f.write("\\begin{tabular}{lrrrrr}\n\\toprule\nModel & Parameters (M) & GPU RTF & GPU median latency (s) & CPU RTF & WER post-proc.\\\\\n\\midrule\n")
    for k in rtf_models:
        g_ = rtf.get(f"{k}__cuda"); c_ = rtf.get(f"{k}__cpu")
        gtxt = f"{g_['rtf_total']:.4f}" if g_ else "n/a"; ltxt = f"{g_['median_latency_s']:.3f}" if g_ else "n/a"
        ctxt = f"{c_['rtf_total']:.3f}" if c_ else "n/a"
        par = f"{g_['params_millions']:.0f}" if g_ else "n/a"
        f.write(f"{LABEL[k]} & {par} & {gtxt} & {ltxt} & {ctxt} & {p(MOD[k]['macro']['pp'])}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
if have_rtf:
    M("nRtfModelsWord", WORDS[len(rtf_models)])
    g_r = [rtf[f"{k}__cuda"]["rtf_total"] for k in rtf_models if f"{k}__cuda" in rtf]
    M("gpuRtfMin", f"{min(g_r):.4f}"); M("gpuRtfMax", f"{max(g_r):.4f}")
    gpuname = next(v["hardware"] for k, v in rtf.items() if k.endswith("__cuda"))
    M("gpuName", gpuname.replace("NVIDIA ", ""))
    c_r = [rtf[f"{k}__cpu"]["rtf_total"] for k in rtf_models if f"{k}__cpu" in rtf]
    if c_r:
        M("cpuRtfMin", f"{min(c_r):.3f}"); M("cpuRtfMax", f"{max(c_r):.3f}")
        M("nCpuModelsWord", WORDS.get(len(c_r), str(len(c_r))))
    # w2v-BERT-2.0 is the slowest model on both devices although it has fewer parameters than MMS-1B
    slow_g = max(rtf_models, key=lambda k: rtf[f"{k}__cuda"]["rtf_total"] if f"{k}__cuda" in rtf else -1)
    cpu_models = [k for k in rtf_models if f"{k}__cpu" in rtf]
    slow_c = max(cpu_models, key=lambda k: rtf[f"{k}__cpu"]["rtf_total"]) if cpu_models else None
    check("w2vbert" in slow_g or "600M" in slow_g, f"w2v-BERT-2.0 is the slowest model on the GPU (slowest: {slow_g})")
    check(slow_c is None or "w2vbert" in slow_c, f"w2v-BERT-2.0 is the slowest model on the CPU (slowest: {slow_c})")
    M("rtfRatioWvMms", f"{rtf['ours_w2vbert__cuda']['rtf_total'] / rtf['ours_mms1b__cuda']['rtf_total']:.1f}")
    # compute time per hour of audio for the slowest model: seconds on the GPU, minutes on the CPU
    M("gpuSecPerHourMax", f"{3600 * max(g_r):.0f}")
    if c_r:
        M("cpuMinPerHourMax", f"{60 * max(c_r):.1f}")

# ------------------------------------------------------------------ combined Amharic study
cb = an.get("combined", {})
PARTS = [("waxal_am", "WAXAL Amharic test"), ("snapwre", "snapwre test"), ("combined", "Combined test")]
CM = [("A_waxal_am", "A: WAXAL Amharic only"), ("B_snapwre", "B: snapwre only"), ("C_combined", "C: WAXAL Amharic + snapwre")]
with open(OUT / "tables/tab_combined.tex", "w") as f:
    f.write("\\begin{tabular}{lrrr}\n\\toprule\nTraining data & " + " & ".join(t for _, t in PARTS) + "\\\\\n\\midrule\n")
    for k, t in CM:
        if k in cb:
            f.write(f"{t} & " + " & ".join(f"{p(cb[k][q]['wer'])} [{p(cb[k][q]['wer_ci'][0])}, {p(cb[k][q]['wer_ci'][1])}]" for q, _ in PARTS) + "\\\\\n")
    if "paired_C_minus_A" in cb:
        f.write("\\midrule\nC $-$ A & " + " & ".join(f"{sgn(cb['paired_C_minus_A'][q]['diff'])} [{sgn(cb['paired_C_minus_A'][q]['ci'][0])}, {sgn(cb['paired_C_minus_A'][q]['ci'][1])}]" for q, _ in PARTS) + "\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
for k, t in CM:
    if k in cb:
        s = {"A_waxal_am": "A", "B_snapwre": "B", "C_combined": "C"}[k]
        for q, _ in PARTS:
            qq = {"waxal_am": "Wax", "snapwre": "Snp", "combined": "All"}[q]
            M(f"cmb{s}{qq}", p(cb[k][q]["wer"]))
if "paired_C_minus_A" in cb:
    for q, _ in PARTS:
        qq = {"waxal_am": "Wax", "snapwre": "Snp", "combined": "All"}[q]
        d = cb["paired_C_minus_A"][q]
        M(f"cmbDiff{qq}", sgn(d["diff"])); M(f"cmbDiffLo{qq}", sgn(d["ci"][0])); M(f"cmbDiffHi{qq}", sgn(d["ci"][1]))
        M(f"cmbDiffAbs{qq}", p(abs(d["diff"])))
        check(d["ci"][1] < 0, f"adding snapwre lowers WER on the {q} test set (interval below zero)")
    # Out-of-domain penalty of each single-corpus model, relative to the in-domain model on that test set.
    M("cmbXferA", p(cb["A_waxal_am"]["snapwre"]["wer"] - cb["B_snapwre"]["snapwre"]["wer"]))
    M("cmbXferB", p(cb["B_snapwre"]["waxal_am"]["wer"] - cb["A_waxal_am"]["waxal_am"]["wer"]))
    check(cb["B_snapwre"]["waxal_am"]["wer"] - cb["A_waxal_am"]["waxal_am"]["wer"]
          > cb["A_waxal_am"]["snapwre"]["wer"] - cb["B_snapwre"]["snapwre"]["wer"],
          "the snapwre-only model loses more on WAXAL than the WAXAL-only model loses on snapwre")
    check(abs(cb["paired_C_minus_A"]["snapwre"]["diff"]) > 10 * abs(cb["paired_C_minus_A"]["waxal_am"]["diff"]),
          "the snapwre gain is more than ten times the WAXAL gain")
    check(all(cb["C_combined"][q]["wer"] < min(cb["A_waxal_am"][q]["wer"], cb["B_snapwre"][q]["wer"]) for q, _ in PARTS),
          "the combined model has the lowest WER on every test set")
# Training exposure of the Amharic-only models: six epochs each, so model C sees every WAXAL utterance as often as A.
n_wax_am = tr["amh"]["rows"]; n_snp = sn["train"]["rows"]
upd_a = n_wax_am * 6 / 32; upd_c = (n_wax_am + n_snp) * 6 / 32
check(abs(upd_a - 7129) < 1 and abs(upd_c - 8250) < 1, f"7,129 and 8,250 updates are six epochs at batch 32 ({upd_a:.1f}, {upd_c:.1f})")
M("cmbExtraUpdPct", f"{100 * (8250 - 7129) / 7129:.1f}")
M("cmbSnpSharePct", f"{100 * n_snp / (n_wax_am + n_snp):.1f}")

# ------------------------------------------------------------------ published findings against this study
# One row per finding of Abdullah et al. (2026) that this study re-measured. "Published" values come from
# REPORTED_*; everything else from analysis.json. The outcome wording is fixed here and checked below.
rel4 = list(REPORTED_WER)
rep_macro = [REPORTED_WER[k]["macro"] for k in rel4]
our_macro = [100 * MOD[k]["macro"]["pp"] for k in rel4]
pd_ = [100 * an["paired"][PAIRKEY[s]]["diff"] for s in "ABCD"]
lid_rel = [100 * MOD[k]["lid_overall"]["accuracy"] for k in rel4]
lid_dev = max(abs(100 * MOD[k]["lid_overall"]["accuracy"] - REPORTED_LID[k]) for k in rel4)
check(lid_dev < 0.2, "measured LID of the released checkpoints is within 0.2 points of the published values")
tir_rep = [REPORTED_GAP[k]["tir"] for k in rel4]; tir_our = [100 * MOD[k]["gender"]["tir"]["gap_male_minus_female"] for k in rel4]
sid_rep = [REPORTED_GAP[k]["sid"] for k in rel4]; sid_our = [100 * MOD[k]["gender"]["sid"]["gap_male_minus_female"] for k in rel4]
ao_rep = [REPORTED_GAP[k][l] for k in rel4 for l in ("amh", "orm")]
ao_our = [100 * MOD[k]["gender"][l]["gap_male_minus_female"] for k in rel4 for l in ("amh", "orm")]
check(max(sid_our) < 0 and max(sid_rep) < 0, "Sidaama gap negative in both studies")
ratios = [o / r for o, r in zip(tir_our, tir_rep)]
check(0.24 < min(ratios) and max(ratios) < 0.34, f"the Tigrinya gap is a quarter to a third of the published value ({min(ratios):.2f} to {max(ratios):.2f})")
check(max(abs(x) for x in ao_our) < 4, "Amharic and Oromo gaps are small (below 4 points) for the released checkpoints")
vow_dom = vow["orm"] > gem["orm"] and vow["wal"] > gem["wal"] and gem["sid"] > vow["sid"]
check(vow_dom, "vowel length dominates in Oromo and Wolaytta, gemination in Sidaama")


def rng(v, d=2, signed=False):
    lo, hi = min(v), max(v)
    f = (lambda x: f"{x:+.{d}f}") if signed else (lambda x: f"{x:.{d}f}")
    return (f(lo) + " to " + f(hi)).replace("-", "$-$").replace("+", "$+$")


rows_c = [("Macro WER of the four released checkpoints", rng(rep_macro), rng(our_macro),
           f"{min(o - r for o, r in zip(our_macro, rep_macro)):.2f} to {max(o - r for o, r in zip(our_macro, rep_macro)):.2f} points higher under our scoring")]
top = an["paired"].get(PAIRKEY["Top"])
if top:
    lo, hi = 100 * top["ci"][0], 100 * top["ci"][1]
    outcome = "Ranking not resolved: interval includes zero" if lo < 0 < hi else ("Ranking reproduced" if hi < 0 else "Ranking reversed")
    rows_c.append(("w2v-BERT-2.0 (600M) is the best released model",
                   f"{REPORTED_WER['released_600M']['macro']:.2f} against {REPORTED_WER['released_1B']['macro']:.2f} (MMS-1B)",
                   f"{p(MOD['released_600M']['macro']['pp'])} against {p(MOD['released_1B']['macro']['pp'])}; paired difference {sgn(top['diff'])} [{sgn(top['ci'][0])}, {sgn(top['ci'][1])}]",
                   outcome))
    M("diffTopOutcome", outcome[0].lower() + outcome[1:])
rows_c += [
    ("The public recipe reproduces the released models", "Code, data and configurations released",
     f"Retrained minus released: {rng(pd_, signed=True)} points",
     "Matched for MMS-1B and w2v-BERT-2.0; MMS-300M better and AfriHuBERT worse"),
    ("Language identification is almost perfect", rng(list(REPORTED_LID.values())) + " percent", rng(lid_rel) + " percent",
     f"Reproduced, within {lid_dev:.2f} points"),
    ("Tigrinya: male speakers have about 18 points higher WER", rng(tir_rep, signed=True), rng(tir_our, signed=True),
     "Not reproduced; direction agrees, size does not"),
    ("Sidaama: female speakers have higher WER", rng(sid_rep, signed=True), rng(sid_our, signed=True), "Reproduced in direction"),
    ("Amharic and Oromo gaps are small", rng(ao_rep, signed=True), rng(ao_our, signed=True), "Reproduced; sign varies in Oromo"),
    ("Vowel length matters most in Oromo and Wolaytta, gemination in Sidaama", "Reported as a pattern",
     f"Ignoring vowel length: {p(vow['orm'])}, {p(vow['wal'])}, {p(vow['sid'])}; ignoring gemination: {p(gem['orm'])}, {p(gem['wal'])}, {p(gem['sid'])} points (ORM, WAL, SID)",
     "Reproduced"),
]
with open(OUT / "tables/tab_claims.tex", "w") as f:
    f.write("\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}p{0.27\\textwidth}>{\\raggedright\\arraybackslash}p{0.15\\textwidth}"
            ">{\\raggedright\\arraybackslash}X>{\\raggedright\\arraybackslash}p{0.22\\textwidth}}\n\\toprule\n")
    f.write("Finding & Published & Measured here & Outcome\\\\\n\\midrule\n")
    for r in rows_c:
        f.write(" & ".join(r) + "\\\\\n\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabularx}\n")

# ------------------------------------------------------------------ XLS-R feasibility and training facts
M("xlsrInfeasTrain", f"{feas['train']['pct_infeasible_adapter']:.1f}")
M("xlsrInfeasTrainRound", f"{feas['train']['pct_infeasible_adapter']:.0f}")
M("xlsrInfeasTest", f"{feas['test']['pct_infeasible_adapter']:.1f}")
M("xlsrInfeasTrainNoAd", f"{100 * feas['train']['infeasible_no_adapter'] / feas['train']['n']:.3f}")
M("xlsrInfeasTrainNoAdN", feas["train"]["infeasible_no_adapter"])
for l in LANGS:
    M(f"xlsrInfeas{CODE[l].capitalize()}", f"{feas['train']['per_language'][l]['pct_infeasible_adapter']:.1f}")
bf = curves["AfriHuBERT bf16"]
M("bfSteps", thousands(bf["global_step"])); M("bfEvals", len(bf["eval"]))

# ------------------------------------------------------------------ figures
SURF, INK, INK2, GRID, AX = "#ffffff", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
# Omit the creation date so that regenerated PDFs are byte-identical when nothing changed.
PDFMETA = {"CreationDate": None}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.edgecolor": AX, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})

# Figure: released vs retrained per language
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=True, sharex=True)
rows = [LN[l] for l in LANGS] + ["Macro"]
yp = list(range(len(rows)))[::-1]; yp[-1] -= 0.4
for ax, (rel, ours, enc, size) in zip(axes, ROWS):
    rv = [100 * MOD[rel]["per_language"][l]["pp"]["wer"] for l in LANGS] + [100 * MOD[rel]["macro"]["pp"]]
    ov_ = [100 * MOD[ours]["per_language"][l]["pp"]["wer"] for l in LANGS] + [100 * MOD[ours]["macro"]["pp"]]
    for y, a, b in zip(yp, rv, ov_):
        ax.plot([a, b], [y, y], color=AX, lw=1.5, zorder=1)
    ax.scatter(rv, yp, s=34, facecolor="white", edgecolor=S1, linewidth=1.5, zorder=3, label="Released checkpoint")
    ax.scatter(ov_, yp, s=12, color=S2, zorder=4, label="Retrained here")
    ax.set_title(f"{enc} ({size})", loc="left"); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Test WER (%)")
axes[0].set_yticks(yp); axes[0].set_yticklabels(rows)
h, lab = axes[0].get_legend_handles_labels(); fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07))
fig.savefig(OUT / "figures/fig_released_vs_retrained.pdf", metadata=PDFMETA); plt.close(fig)

# Figure: validation curves
XPAN = (("XLS-R-1B", [("XLS-R-1B no adapter", "adapter off", S1), ("XLS-R-1B", "adapter on", S2)])
        if "XLS-R-1B no adapter" in curves and HAVE_XN else ("XLS-R-1B, adapter on", [("XLS-R-1B", None, S1)]))
PAN = [("AfriHuBERT (94M)", [("AfriHuBERT fp32", "fp32, LR 5e-5", S1), ("AfriHuBERT bf16", "bf16, LR 3e-4", S2)]),
       ("MMS-300M", [("MMS-300M", None, S1)]), ("MMS-1B", [("MMS-1B", None, S1)]),
       ("w2v-BERT-2.0 (600M)", [("w2v-BERT-2.0", "36.8k-step schedule", S1), ("w2v-BERT-2.0 extended", "60k-step schedule (stopped)", S2)]),
       XPAN,
       ("Amharic-only models", [("A WAXAL-am", "A", S1), ("B snapwre", "B", S2), ("C combined", "C", S3)])]
if HAVE_XN and "XLS-R-1B no adapter" in curves:
    xc = curves["XLS-R-1B no adapter"]
    # best checkpoint: highest score = 100 (1 - (WER + CER) / 2), the trainer's selection metric
    best = max(xc["eval"], key=lambda q: 1 - 0.5 * (q[1] + q[2]))
    M("xlsrNoAdSteps", thousands(xc["global_step"])); M("xlsrNoAdBestStep", thousands(best[0]))
    M("xlsrNoAdBestValWer", p(best[1])); M("xlsrNoAdEvals", len(xc["eval"]))
    check(abs(100 * (1 - 0.5 * (best[1] + best[2])) - xc["best_metric"]) < 0.01, "best step matches the trainer's best_metric")
    check(best[0] == xc["global_step"], "the selected XLS-R-1B checkpoint is the last one (validation WER still falling)")
    # same selection rule for MMS-1B, for the validation comparison in the text
    mc = curves["MMS-1B"]; mb = max(mc["eval"], key=lambda q: 1 - 0.5 * (q[1] + q[2]))
    M("mmsBestValWer", p(mb[1]))
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharey=True)
for ax, (title, items) in zip(axes.flat, PAN):
    for key, lab, col in items:
        e = curves[key]["eval"]
        ax.plot([q[0] / 1000 for q in e], [100 * q[1] for q in e], color=col, lw=1.4, label=lab)
    ax.set_title(title, loc="left"); ax.set_ylim(0, 105); ax.set_xlim(left=0)
    if len(items) > 1:
        # panels with a curve stuck near 100 percent get the legend below it
        ax.legend(fontsize=6.5, loc="center right" if ("AfriHuBERT" in title or "XLS-R" in title) else "upper right")
for ax in axes[1]:
    ax.set_xlabel("Training step (thousands)")
for ax in axes[:, 0]:
    ax.set_ylabel("Validation WER (%)")
fig.tight_layout(); fig.savefig(OUT / "figures/fig_validation_curves.pdf", metadata=PDFMETA); plt.close(fig)

# Figure: gender gap, measured with intervals, against the values reported by the original paper
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
for ax, l in zip(axes, ("amh", "orm", "sid", "tir")):
    for i, k in enumerate(main8):
        gg = MOD[k]["gender"][l]; y = len(main8) - i
        col = S1 if k.startswith("released") else S2
        ax.plot([100 * gg["gap_ci"][0], 100 * gg["gap_ci"][1]], [y, y], color=col, lw=1.2)
        ax.scatter([100 * gg["gap_male_minus_female"]], [y], color=col, s=12, zorder=3)
        if k in REPORTED_GAP:
            ax.scatter([REPORTED_GAP[k][l]], [y], marker="D", s=14, facecolor="white", edgecolor=INK, linewidth=0.9, zorder=4)
    ax.axvline(0, color=AX, lw=0.8); ax.set_title(LN[l], loc="left"); ax.set_xlabel("Male $-$ female WER (pp)")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(main8), 0, -1)); axes[0].set_yticklabels([LABEL[k].replace(" (", "\n(") for k in main8], fontsize=6)
from matplotlib.lines import Line2D  # noqa: E402
fig.legend([Line2D([], [], color=S1, marker="o", lw=1.2, ms=3), Line2D([], [], color=S2, marker="o", lw=1.2, ms=3),
            Line2D([], [], color=INK, marker="D", lw=0, ms=4, markerfacecolor="white")],
           ["Released, measured here (95% CI)", "Retrained, measured here (95% CI)", "Reported by Abdullah et al. (2026)"],
           loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.1))
fig.savefig(OUT / "figures/fig_gender_gap.pdf", metadata=PDFMETA); plt.close(fig)

# Figure: CTC feasibility with and without the adapter
fig, ax = plt.subplots(figsize=(3.5, 2.1))
x = np.arange(len(LANGS)); w = 0.36
ad = [feas["train"]["per_language"][l]["pct_infeasible_adapter"] for l in LANGS]
noad = [100 * feas["train"]["per_language"][l]["infeasible_no_adapter"] / feas["train"]["per_language"][l]["n"] for l in LANGS]
ax.bar(x - w / 2, ad, w * 0.92, color=S2, label="Adapter on (8$\\times$ shorter output)")
ax.bar(x + w / 2, noad, w * 0.92, color=S1, label="Adapter off")
for xi, v in zip(x, ad):
    ax.text(xi - w / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([LN[l] for l in LANGS], fontsize=7); ax.set_ylim(0, 112)
ax.set_ylabel("Training utterances CTC\ncannot align (%)"); ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0, 1.18), ncol=2)
fig.savefig(OUT / "figures/fig_xlsr_ctc.pdf", metadata=PDFMETA); plt.close(fig)

# Figure: accuracy against real-time factor
if have_rtf:
    # One label per encoder: the released and retrained points of an encoder share its name.
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for rel, ours, enc, size in ROWS:
        pts = []
        for k in (rel, ours):
            if f"{k}__cuda" not in rtf:
                continue
            r = rtf[f"{k}__cuda"]["rtf_total"]; wv = 100 * MOD[k]["macro"]["pp"]
            pts.append((r, wv))
            if k.startswith("released"):
                ax.scatter([r], [wv], s=30, facecolor="white", edgecolor=S1, linewidth=1.3, zorder=3)
            else:
                ax.scatter([r], [wv], s=12, color=S2, zorder=4)
        if pts:
            ax.annotate(enc, (max(q[0] for q in pts), np.mean([q[1] for q in pts])), fontsize=6.5,
                        xytext=(6, -2), textcoords="offset points", color=INK2)
    if HAVE_XN and f"{XN}__cuda" in rtf:
        r = rtf[f"{XN}__cuda"]["rtf_total"]; wv = 100 * MOD[XN]["macro"]["pp"]
        ax.scatter([r], [wv], s=12, color=S2, zorder=4)
        ax.annotate("XLS-R-1B", (r, wv), fontsize=6.5, xytext=(6, -2), textcoords="offset points", color=INK2)
    ax.set_xlim(0, 1.25 * max(v["rtf_total"] for k, v in rtf.items() if k.endswith("__cuda")))
    ax.scatter([], [], s=30, facecolor="white", edgecolor=S1, linewidth=1.3, label="Released checkpoint")
    ax.scatter([], [], s=12, color=S2, label="Retrained here")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_xlabel("Real-time factor on GPU (batch size 1)"); ax.set_ylabel("Test WER, post-proc. (%)")
    fig.savefig(OUT / "figures/fig_wer_rtf.pdf", metadata=PDFMETA); plt.close(fig)

(OUT / "numbers.tex").write_text("% Generated by make_paper_assets.py from measured files. Do not edit by hand.\n" + "\n".join(macros) + "\n")
print(f"wrote {len(macros)} macros, tables and figures to {OUT} (XLS-R-1B without adapter included: {HAVE_XN})")
for c in claim_failures:
    print("CLAIM CHECK FAILED:", c)
if claim_failures:
    sys.exit(2)
print("all claim checks passed")
