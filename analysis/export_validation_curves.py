"""Read-only: export validation-WER histories of the Ethio-ASR training runs as one JSON on stdout.

Sources (relative to the project root):
  trainer_state.json of the newest checkpoint of each run, and, for the bf16 AfriHuBERT run whose
  checkpoints were not kept, the evaluation lines in its Slurm log (one evaluation every 800 steps).
"""
import glob
import json
import re

P = "/path/to/Ethio-ASR/"


def newest_state(run_dir):
    """trainer_state.json of the highest-numbered checkpoint in a run folder (relative to P)."""
    found = glob.glob(P + run_dir + "/checkpoint-*/trainer_state.json")
    if not found:
        return None
    best = max(found, key=lambda f: int(f.split("checkpoint-")[1].split("/")[0]))
    return best[len(P):]
RUNS = {
    "AfriHuBERT fp32": "models/ajesujoba/AfriHuBERT-11062026-234348/checkpoint-36800/trainer_state.json",
    "MMS-300M": "models/facebook/mms-300m-09062026-205411/checkpoint-36800/trainer_state.json",
    "MMS-1B": "models/facebook/mms-1b-09062026-203848/checkpoint-36000/trainer_state.json",
    "w2v-BERT-2.0": "models/facebook/w2v-bert-2.0-09062026-212456/checkpoint-36000/trainer_state.json",
    "w2v-BERT-2.0 extended": "models/facebook/w2v-bert-2.0-10062026-003217/checkpoint-43200/trainer_state.json",
    "XLS-R-1B": "models/facebook/wav2vec2-xls-r-1b-09062026-230639/checkpoint-34400/trainer_state.json",
    "A WAXAL-am": "combined_dataset_experiment/models/facebook/w2v-bert-2.0-26082026-181418/checkpoint-7000/trainer_state.json",
    "B snapwre": "combined_dataset_experiment/models/facebook/w2v-bert-2.0-26082026-181420/checkpoint-1800/trainer_state.json",
    "C combined": "combined_dataset_experiment/models/facebook/w2v-bert-2.0-26082026-181421/checkpoint-8000/trainer_state.json",
}
# Added 2026-11-07: the XLS-R-1B run without the output adapter (job 50160141), read from its newest checkpoint.
_noad = newest_state("models/facebook/wav2vec2-xls-r-1b-07112026-162109")
if _noad:
    RUNS["XLS-R-1B no adapter"] = _noad
# Corrected 2026-11-07: training validated on the full validation split (sample: false), not a 2,000-utterance subset.
out = {"source_note": "eval_wer on the full WAXAL validation split, 15,949 utterances (combined runs: their own validation split)", "runs": {}}
for name, rel in RUNS.items():
    s = json.load(open(P + rel))
    ev = [[e["step"], round(e["eval_wer"], 5), round(e.get("eval_cer", float("nan")), 5), round(e["eval_loss"], 4)]
          for e in s["log_history"] if "eval_wer" in e]
    tr = [[e["step"], round(e["loss"], 4)] for e in s["log_history"] if "loss" in e]
    out["runs"][name] = {"source": rel, "max_steps": s.get("max_steps"), "global_step": s.get("global_step"),
                         "eval": ev, "train_loss_first_last": [tr[0], tr[-1]] if tr else None,
                         # checkpoint the trainer kept as best (load_best_model_at_end, metric "score")
                         "best_model_checkpoint": s.get("best_model_checkpoint"), "best_global_step": s.get("best_global_step"),
                         "best_metric": s.get("best_metric")}

txt = open(P + "logs/ethio-paper-afrihubert_34206548.out", errors="replace").read()
wers = re.findall(r"'eval_wer': ([0-9.]+)", txt)
losses = re.findall(r"'eval_loss': ([0-9.]+)", txt)
out["runs"]["AfriHuBERT bf16"] = {
    "source": "logs/ethio-paper-afrihubert_34206548.out (job cancelled at step 32,609 of 36,800)",
    "max_steps": 36800, "global_step": 32609,
    "eval": [[800 * (i + 1), float(w), None, float(l) if i < len(losses) else None] for i, (w, l) in enumerate(zip(wers, losses))],
}
print(json.dumps(out))
