#!/usr/bin/env python3
"""Count language tags repeated inside the WAXAL test hypotheses (CPU only, added 2026-11-08).

For every assessment_2026-11/predictions/*__waxal.jsonl file, per language and overall:
  n          utterances
  leading    hypotheses that start with a language tag
  repeated   hypotheses with a further tag after the leading one
  at_end     hypotheses whose last token is such a further tag
  glued      hypotheses where a further tag is attached to the preceding word, as in "baala.[SID]"
The tag pattern is LANG_TOKEN_RE from slurm_scripts/evaluate_ctc.py, the one used in scoring.
Prints JSON to stdout (tag_sensitivity.sbatch writes it to results_aux/repeated_tags.json).
"""
import glob
import json
import re
import sys
from pathlib import Path

P = Path("/path/to/Ethio-ASR")
A = P / "assessment_2026-11"
sys.path.insert(0, str(P / "slurm_scripts"))
from evaluate_ctc import LANG_TOKEN_RE  # noqa: E402

LEAD = re.compile(r"^\s*(?:" + LANG_TOKEN_RE.pattern + r")\s*", re.IGNORECASE)
LANGS = ["amh", "orm", "sid", "tir", "wal"]
KEYS = ("n", "leading", "repeated", "at_end", "glued")

out = {"pattern": LANG_TOKEN_RE.pattern, "models": {}}
for f in sorted(glob.glob(str(A / "predictions" / "*__waxal.jsonl"))):
    name = Path(f).name.split("__")[0]
    c = {L: dict.fromkeys(KEYS, 0) for L in LANGS}
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        h = r["hyp"]
        d = c[r["language"]]
        m = LEAD.match(h)
        rest = h[m.end():] if m else h
        tags = list(LANG_TOKEN_RE.finditer(rest))
        d["n"] += 1
        d["leading"] += bool(m)
        if tags:
            d["repeated"] += 1
            d["at_end"] += tags[-1].end() == len(rest.rstrip())
            d["glued"] += any(t.start() > 0 and not rest[t.start() - 1].isspace() for t in tags)
    out["models"][name] = {"overall": {k: sum(c[L][k] for L in LANGS) for k in KEYS}, "per_language": c}
print(json.dumps(out, indent=1))
