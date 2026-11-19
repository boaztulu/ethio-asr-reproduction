#!/usr/bin/env python
"""
Aggregate the 9 cross-eval JSON files into:
  - a WER matrix (train X × eval Y)
  - a delta table (combined - baseline)
  - a written verdict paragraph

Writes analysis/results_table.md and analysis/report.md.
"""

import json
import os
from pathlib import Path

EXP = Path("/path/to/Ethio-ASR/combined_dataset_experiment")
RESULTS = EXP / "results"
OUT_DIR = EXP / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["A_waxal_am_baseline", "B_snapwre_only", "C_combined_am"]
TESTS  = ["waxal_am", "snapwre_am", "combined_am"]

LABEL = {
    "A_waxal_am_baseline": "A: WAXAL-am (baseline)",
    "B_snapwre_only":       "B: snapwre only",
    "C_combined_am":        "C: WAXAL-am + snapwre",
    "waxal_am":             "WAXAL-am test",
    "snapwre_am":           "snapwre test",
    "combined_am":          "combined test",
}


def load(model, test):
    p = RESULTS / f"{model}__on__{test}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def pct(x):
    return f"{x*100:.2f}"


def build_table(metric):
    """metric ∈ {'wer','cer'}; returns markdown table string."""
    lines = []
    header = "| Trained on \\ Eval on | " + " | ".join(LABEL[t] for t in TESTS) + " |"
    sep    = "|" + "|".join(["---"] * (len(TESTS) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for m in MODELS:
        row = [LABEL[m]]
        for t in TESTS:
            r = load(m, t)
            if r is None:
                row.append("—")
                continue
            per_lang = r.get("per_language", {})
            # single-language, so read amharic if present, else averages micro
            if "amharic" in per_lang:
                v = per_lang["amharic"].get(metric)
            else:
                v = r.get("averages", {}).get(f"micro_{metric}")
            row.append(pct(v) + "%" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_delta_table(metric):
    """Delta = C_combined - A_baseline (negative = improvement)."""
    lines = ["| Eval set | A baseline | C combined | Δ (C-A) | Improved? |",
             "|---|---|---|---|---|"]
    for t in TESTS:
        a = load("A_waxal_am_baseline", t)
        c = load("C_combined_am", t)
        if a is None or c is None:
            lines.append(f"| {LABEL[t]} | — | — | — | — |")
            continue
        av = a.get("per_language", {}).get("amharic", {}).get(metric) or \
             a.get("averages", {}).get(f"micro_{metric}")
        cv = c.get("per_language", {}).get("amharic", {}).get(metric) or \
             c.get("averages", {}).get(f"micro_{metric}")
        if av is None or cv is None:
            lines.append(f"| {LABEL[t]} | — | — | — | — |")
            continue
        d = cv - av
        arrow = "yes ↓" if d < 0 else "no ↑" if d > 0 else "="
        lines.append(f"| {LABEL[t]} | {pct(av)}% | {pct(cv)}% | {d*100:+.2f}pp | {arrow} |")
    return "\n".join(lines)


def main():
    wer_matrix = build_table("wer")
    cer_matrix = build_table("cer")
    wer_delta  = build_delta_table("wer")
    cer_delta  = build_delta_table("cer")

    md = f"""# Combined-Dataset Amharic ASR — Results

## WER matrix
{wer_matrix}

## CER matrix
{cer_matrix}

## Δ table — did adding snapwre help? (C vs. A)

### WER
{wer_delta}

### CER
{cer_delta}

_All numbers are post-processed (Ge'ez homophone normalization + punctuation removal), per the paper's Section 5.2._
"""
    (OUT_DIR / "results_table.md").write_text(md)
    print(md)

    # Determine verdict from combined_am test set
    a = load("A_waxal_am_baseline", "combined_am")
    c = load("C_combined_am", "combined_am")
    verdict_lines = ["# Verdict — did combined training improve Amharic ASR?", ""]
    if a and c:
        def _v(r, metric):
            return r.get("per_language", {}).get("amharic", {}).get(metric) or \
                   r.get("averages", {}).get(f"micro_{metric}")
        wa, wc = _v(a, "wer"), _v(c, "wer")
        ca, cc = _v(a, "cer"), _v(c, "cer")
        if wa is not None and wc is not None:
            dw = (wc - wa) * 100
            dc = (cc - ca) * 100 if ca is not None and cc is not None else None
            verdict_lines.append(f"- Baseline (WAXAL-am only) WER on held-out combined test: **{pct(wa)}%**")
            verdict_lines.append(f"- Combined (WAXAL-am + snapwre) WER on same test:          **{pct(wc)}%**")
            verdict_lines.append(f"- Δ WER: **{dw:+.2f}pp** (negative = better)")
            if dc is not None:
                verdict_lines.append(f"- Δ CER: **{dc:+.2f}pp**")
            if dw < -0.5:
                verdict_lines.append("\n**Verdict: adding snapwre/amharic-speech improved Amharic ASR.**")
            elif dw > 0.5:
                verdict_lines.append("\n**Verdict: adding snapwre HURT performance — data mixing may not be aligned.**")
            else:
                verdict_lines.append("\n**Verdict: no meaningful change (within noise).**")
    else:
        verdict_lines.append("_(missing results — some runs did not finish; re-run once training completes.)_")

    verdict_lines.append("")
    verdict_lines.append("## See also")
    verdict_lines.append("- Full WER/CER cross-matrix: [results_table.md](results_table.md)")
    verdict_lines.append("- Per-run JSON: `../results/`")

    (OUT_DIR / "report.md").write_text("\n".join(verdict_lines))
    print("\nWrote", OUT_DIR / "results_table.md")
    print("Wrote", OUT_DIR / "report.md")


if __name__ == "__main__":
    main()
