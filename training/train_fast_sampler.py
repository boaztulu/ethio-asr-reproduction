"""Run Ethio-ASR's scripts/train_model.py with one data-loading fix.

transformers 4.55 drops every column the model does not take as input, including the
precomputed `length` column, before it builds the length-grouped sampler
(Trainer._get_dataloader). The sampler then measures each utterance again by reading
its whole input array. On the WAXAL training set this took about 2 h 50 min before
the first training step and about 11 min before each validation pass (June 2026 logs).
the cluster cancels B200 jobs whose GPU stays idle for an hour, which is what happened
to job 50146121 on 2026-11-07.

This wrapper gives the sampler the `length` column of the dataset it was built from.
The encoder stored len(input_values) (or len(input_features)) in that column, which is
exactly what the sampler would have computed, and dropping columns does not reorder
rows, so the batches and their order are unchanged. The first rows are checked against
the inputs before the column is used. The repository code is not modified.

Run it from the Ethio-ASR repository directory, with the same arguments as
scripts/train_model.py:
    python3 -u /path/to/train_fast_sampler.py --config CONFIG.yaml
"""
import logging
import os
import runpy
import sys

import datasets
from transformers import Trainer
from transformers.trainer_pt_utils import LengthGroupedSampler

_original_train_sampler = Trainer._get_train_sampler
_original_eval_sampler = Trainer._get_eval_sampler
_checked = set()


def _stored_lengths(trainer, reduced, full, tag):
    """Lengths from the `length` column of `full`, when `reduced` lost that column."""
    col = trainer.args.length_column_name
    if not (trainer.args.group_by_length
            and isinstance(reduced, datasets.Dataset) and isinstance(full, datasets.Dataset)
            and col not in reduced.column_names and col in full.column_names
            and len(reduced) == len(full)):
        return None
    lengths = list(full[col])
    name = (trainer.processing_class.model_input_names[0]
            if trainer.processing_class is not None else "input_ids")
    if tag not in _checked:
        n = min(64, len(reduced))
        measured = [len(reduced[i][name]) for i in range(n)]
        if measured != lengths[:n]:
            raise RuntimeError(f"'{col}' column differs from len({name}) in the first {n} {tag} rows")
        _checked.add(tag)
        logging.info("Sampler fix: using the stored '%s' column for %d %s rows (first %d checked)",
                     col, len(lengths), tag, n)
    return lengths, name


def _train_sampler(self, train_dataset=None):
    found = None
    if train_dataset is not None:
        found = _stored_lengths(self, train_dataset, self.train_dataset, "training")
    if found is None:
        return _original_train_sampler(self, train_dataset)
    lengths, name = found
    return LengthGroupedSampler(
        self.args.train_batch_size * self.args.gradient_accumulation_steps,
        dataset=train_dataset, lengths=lengths, model_input_name=name)


def _eval_sampler(self, eval_dataset):
    found = None
    if eval_dataset is not None and not self.args.use_legacy_prediction_loop:
        found = _stored_lengths(self, eval_dataset, self.eval_dataset, "validation")
    if found is None:
        return _original_eval_sampler(self, eval_dataset)
    lengths, name = found
    return LengthGroupedSampler(
        self.args.eval_batch_size, dataset=eval_dataset, lengths=lengths, model_input_name=name)


if __name__ == "__main__":
    Trainer._get_train_sampler = _train_sampler
    Trainer._get_eval_sampler = _eval_sampler
    script = os.path.join(os.getcwd(), "scripts", "train_model.py")
    if not os.path.isfile(script):
        sys.exit(f"run this from the Ethio-ASR repository directory ({script} not found)")
    sys.path[0] = os.path.dirname(script)
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
