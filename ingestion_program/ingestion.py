"""Ingestion program for the competition.

A thin wrapper around ``benchopt run``. A submission is a benchopt solver:
a single ``submission.py`` (a ``CompetSolver``) holding all its Python
code, plus optional non-code files (weights, configs) it reads from
``meta["submission_dir"]``. This program runs the benchmark shipped in the
phase ``input_data/benchmark`` on that file, loaded by path::

    benchopt run <input_data>/benchmark --config <input_data>/config.yaml \
        -s <submission_dir>/submission.py --no-cache --no-plot \
        --output <output_dir>/results.parquet

The phase ``config.yaml`` is a native benchopt run config (``dataset``,
``seed``, ``n_repetitions``: one cross-validation fold per repetition). The
scoring program scores the test predictions of ``results.parquet``. The
prepared data comes with the phase, in ``input_data/data``. Without it, the
run uses the image's $BENCHOPT_DATA_HOME (/app/data, where a self-hosted
worker mounts the staged data read-only). $BENCHOPT_DEBUG ensures any solver
error aborts the run.
"""

import os
from pathlib import Path

os.environ["BENCHOPT_DEBUG"] = "true"

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402


def main(submission_dir, output_dir, input_dir):
    submission = submission_dir / "submission.py"
    if not submission.is_file():
        raise SystemExit(f"No submission.py found in {submission_dir}.")
    benchmark_dir = input_dir / "benchmark"
    if not (benchmark_dir / "objective.py").is_file():
        raise SystemExit(f"No benchopt benchmark found in {benchmark_dir}.")

    if (input_dir / "data").is_dir():
        os.environ["BENCHOPT_DATA_HOME"] = str((input_dir / "data").resolve())

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "benchopt", "run", str(benchmark_dir), "-s", str(submission),
        "--no-cache", "--no-plot", "--no-html", "--no-display",
        "--output", str(output_dir / "results.parquet"),
        "--config", str(input_dir / "config.yaml"),
    ]
    print(f"[ingestion] evaluating {submission}", flush=True)
    start = time.time()
    subprocess.run(cmd, check=True)

    (output_dir / "metadata.json").write_text(
        json.dumps({"duration": time.time() - start}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Competition ingestion")
    parser.add_argument("--submission-dir", default="/app/ingested_program")
    parser.add_argument("--output-dir", default="/app/output")
    parser.add_argument("--input-data", default="/app/input_data",
                        help="Phase dir: config.yaml, benchmark/ and data/")
    args = parser.parse_args()

    main(
        Path(args.submission_dir),
        Path(args.output_dir),
        Path(args.input_data),
    )
