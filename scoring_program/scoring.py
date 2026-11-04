"""Scoring program for the competition.

The ingestion run cross-validates the submission, one fold per repetition.
Each row of ``results.parquet`` holds the fold's metrics (``objective_*``)
and its predictions on each hidden test set (``objective_<test set>_pred``).
This program scores these predictions against the labels of the phase
``reference_data`` (one ``<test set>.csv`` with a ``target`` column per hidden
test set), and averages every score over the folds into ``scores.json``:
``accuracy_cv``, ``accuracy_public``, ``accuracy_private``, ``train_time``
and ``duration``.

The metric is the benchmark's ``benchmark_utils/metric.py``, which
``tools/create_bundle.py`` copies next to this program.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_utils.metric import NAME, score


def main(prediction_dir, reference_dir, output_dir):
    from benchopt.results import read_results

    df = read_results(prediction_dir / "results.parquet")

    scores = {}
    for ref in sorted(reference_dir.glob("*.csv")):
        y_true = pd.read_csv(ref)["target"]
        scores[f"{NAME}_{ref.stem}"] = np.mean(
            [score(y_true, pred)
             for pred in df[f"objective_{ref.stem}_pred"]])
    metrics = df.select_dtypes("number").filter(regex="^objective_")
    for col, value in metrics.mean().items():
        scores[col.removeprefix("objective_")] = value
    if not scores or np.isnan(list(scores.values())).any():
        raise SystemExit(f"[scoring] missing or NaN scores: {list(scores)}")

    # benchopt times the solver's run (training), ingestion the whole run.
    scores["train_time"] = df["time"].mean()
    scores["duration"] = json.loads(
        (prediction_dir / "metadata.json").read_text())["duration"]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scores.json").write_text(
        json.dumps({k: float(v) for k, v in scores.items()}))
    # Keys only: the log is shown to the submitter, values include hidden
    # private metrics.
    print(f"[scoring] wrote {', '.join(scores)} to scores.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Competition scoring")
    parser.add_argument("--prediction-dir", default="/app/input/res")
    parser.add_argument("--reference-dir", default="/app/input/ref")
    parser.add_argument("--output-dir", default="/app/output")
    args = parser.parse_args()

    main(Path(args.prediction_dir), Path(args.reference_dir),
         Path(args.output_dir))
