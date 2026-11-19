"""Per-language WER with the leading token removed (primary) and with every token removed (sensitivity)."""
import json
A = "/path/to/Ethio-ASR/assessment_2026-11/"
pri = json.load(open(A + "results/analysis.json"))["models"]
sen = json.load(open(A + "results_aux/tag_sensitivity/analysis.json"))
rt = json.load(open(A + "results_aux/repeated_tags.json"))["models"]
L = ["amh", "orm", "sid", "tir", "wal"]
print("tag_scoring", sen.get("tag_scoring"))
S = sen["models"]
for k in ["ours_xlsr_noadapter", "ours_mms1b", "released_1B", "ours_w2vbert", "released_600M", "ours_mms300m", "released_94M"]:
    a = " ".join(f"{100 * pri[k]['per_language'][l]['pp']['wer']:6.2f}" for l in L)
    b = " ".join(f"{100 * S[k]['per_language'][l]['pp']['wer']:6.2f}" for l in L)
    print(f"{k:22s} primary {a} | all removed {b}")
print("XN minus MMS-1B (retrained), all removed:", " ".join(f"{l} {100 * (S['ours_xlsr_noadapter']['per_language'][l]['pp']['wer'] - S['ours_mms1b']['per_language'][l]['pp']['wer']):+.2f}" for l in L))
print("XN minus MMS-1B (retrained), primary:    ", " ".join(f"{l} {100 * (pri['ours_xlsr_noadapter']['per_language'][l]['pp']['wer'] - pri['ours_mms1b']['per_language'][l]['pp']['wer']):+.2f}" for l in L))
print("w2v-BERT minus MMS-1B (retrained), all removed:", " ".join(f"{l} {100 * (S['ours_w2vbert']['per_language'][l]['pp']['wer'] - S['ours_mms1b']['per_language'][l]['pp']['wer']):+.2f}" for l in L))
print("600M minus 1B (released), all removed:", " ".join(f"{l} {100 * (S['released_600M']['per_language'][l]['pp']['wer'] - S['released_1B']['per_language'][l]['pp']['wer']):+.2f}" for l in L))
print("LID identical in both files:", all(S[k]["lid_overall"] == pri[k]["lid_overall"] for k in pri))
print("gender gaps, Tigrinya, all removed (released):", {k: round(100 * S[k]["gender"]["tir"]["gap_male_minus_female"], 2) for k in ["released_94M", "released_300M", "released_1B", "released_600M"]})
print("paired, all removed:", {k: (round(100 * v["diff"], 3), [round(100 * x, 3) for x in v["ci"]]) for k, v in sen["paired"].items()})
for k, v in rt.items():
    o = v["overall"]
    if o["repeated"]:
        print(f"{k:22s} repeated {100 * o['repeated'] / o['n']:5.2f}%  at_end/repeated {o['at_end'] / o['repeated']:.3f}  glued/repeated {o['glued'] / o['repeated']:.3f}")
    else:
        print(f"{k:22s} repeated 0")
