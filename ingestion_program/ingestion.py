"""Ingestion program for the competition.

A thin wrapper around ``benchopt run``. A submission is a benchopt solver:
a single ``submission.py`` (a ``CompetSolver``) holding all its Python
code, plus optional non-code files (weights, configs) it reads from
``meta["submission_dir"]``. This program runs the benchmark shipped in the
phase ``input_data/benchmark`` on that file, loaded by path::

    benchopt run <input_data>/benchmark --config <run_config> \
        -s <submission_dir>/submission.py -r 1 --no-cache --no-plot \
        --output <output_dir>/results.parquet

The phase ``config.yaml`` is a **native benchopt run config** (``dataset``,
``seed``, ``no_timeout``, ...) plus one competition-only key stripped before
the run (benchopt rejects unknown options): ``scoring`` (parsed by scoring.py
— the config is forwarded with the raw results parquet; all evaluation happens
in the objective, which always trains the solver). The data location is the
image's $BENCHOPT_DATA_HOME (/app/data, where the compute worker mounts the
staged data read-only). $BENCHOPT_DEBUG ensures any solver error aborts the
run.
"""

import os
from pathlib import Path

os.environ["BENCHOPT_DEBUG"] = "true"

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import yaml  # noqa: E402


def main(submission_dir, output_dir, input_dir):
    submission = submission_dir / "submission.py"
    if not submission.is_file():
        raise SystemExit(f"No submission.py found in {submission_dir}.")
    benchmark_dir = input_dir / "benchmark"
    if not (benchmark_dir / "objective.py").is_file():
        raise SystemExit(f"No benchopt benchmark found in {benchmark_dir}.")

    config = input_dir / "config.yaml"
    cfg = yaml.safe_load(config.read_text())
    # benchopt rejects unknown config keys: strip the competition-only one.
    cfg = {k: v for k, v in cfg.items() if k != "scoring"}
    run_config = Path(tempfile.mkdtemp(prefix="compet_run_")) / "config.yml"
    run_config.write_text(yaml.safe_dump(cfg))

    # Raw results + config for the scoring program, which only parses them.
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "benchopt", "run", str(benchmark_dir), "-s", str(submission),
        "-r", "1", "--no-cache", "--no-plot", "--no-html", "--no-display",
        "--output", str(output_dir / "results.parquet"),
        "--config", str(run_config),
    ]
    print(f"[ingestion] evaluating {submission}", flush=True)
    start = time.time()
    subprocess.run(cmd, check=True)

    (output_dir / "metadata.json").write_text(
        json.dumps({"duration": time.time() - start}))
    shutil.copyfile(config, output_dir / "config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Competition ingestion")
    parser.add_argument("--submission-dir", default="/app/ingested_program")
    parser.add_argument("--output-dir", default="/app/output")
    parser.add_argument("--input-data", default="/app/input_data",
                        help="Phase dir holding config.yaml and benchmark/")
    args = parser.parse_args()

    main(
        Path(args.submission_dir),
        Path(args.output_dir),
        Path(args.input_data),
    )
