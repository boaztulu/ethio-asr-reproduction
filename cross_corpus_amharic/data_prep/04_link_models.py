#!/usr/bin/env python3
"""Create stable symlinks A/B/C -> actual timestamped model dirs.

train_model.py names outputs by datetime (e.g. models/facebook/w2v-bert-2.0-<ts>/),
which the eval matrix can't hard-code. This script parses each training run's
.err/.out log for the "Vocabulary saved to <path>/ctc_tokenizer" line, resolves
the parent dir, and creates a stable symlink at models/<STABLE>/.

Idempotent: existing symlinks are replaced; existing regular directories are
left alone (with a clear error) so we don't destroy training output.

Exits non-zero on any failure so the caller (evaluate_matrix.sbatch) aborts.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

EXP = Path("/path/to/Ethio-ASR/combined_dataset_experiment")
LOGS = EXP / "logs"
MODELS = EXP / "models"

STABLE_NAMES = ["A_waxal_am_baseline", "B_snapwre_only", "C_combined_am"]
VOCAB_LINE_RE = re.compile(r"Vocabulary saved to (\S+)/ctc_tokenizer")


def find_model_dir(stable: str) -> Path:
    log_glob = list(LOGS.glob(f"am-{stable}_*.err")) + list(LOGS.glob(f"am-{stable}_*.out"))
    if not log_glob:
        raise FileNotFoundError(f"no training logs found for {stable} in {LOGS}")
    for log in sorted(log_glob):
        for line in log.read_text(errors="ignore").splitlines():
            m = VOCAB_LINE_RE.search(line)
            if m:
                d = Path(m.group(1))
                if d.is_dir():
                    return d
                raise FileNotFoundError(
                    f"{stable}: log points to {d} but that dir doesn't exist"
                )
    raise LookupError(
        f"{stable}: no 'Vocabulary saved to ...' line in any of "
        f"{[str(p) for p in log_glob]}"
    )


def link(stable: str, target: Path) -> None:
    link_path = MODELS / stable
    if link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(
            f"{link_path} exists and is not a symlink; refusing to overwrite"
        )
    os.symlink(target, link_path)
    print(f"  {stable} -> {target}")


def main() -> int:
    print(f"=== linking stable model names in {MODELS} ===")
    errors: list[str] = []
    for stable in STABLE_NAMES:
        try:
            target = find_model_dir(stable)
            link(stable, target)
        except Exception as e:
            errors.append(f"{stable}: {e}")
            print(f"  [ERROR] {stable}: {e}", file=sys.stderr)
    if errors:
        print(f"=== FAILED ({len(errors)} error(s)) ===", file=sys.stderr)
        return 1
    print("=== all stable symlinks created ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
