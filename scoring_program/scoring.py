"""Scoring program for the competition.

All evaluation happens in the ingestion run; this only parses the raw
benchopt results (``results.parquet``) into the flat ``scores.json`` the
leaderboard consumes. The forwarded phase config's ``scoring.columns`` maps
leaderboard keys to dataframe columns; a listed column that is missing or
NaN fails loudly.
"""

import json
import math
from pathlib import Path

import yaml


def main(prediction_dir, output_dir):
    from benchopt.results import read_results

    df = read_results(prediction_dir / "results.parquet")
    config = yaml.safe_load((prediction_dir / "config.yaml").read_text())
    columns = config["scoring"]["columns"]

    scores = {}
    for key, col in columns.items():
        value = float(df[col].iloc[0]) if col in df.columns else math.nan
        if math.isnan(value):
            raise SystemExit(f"[scoring] column {col!r} (leaderboard "
                             f"key {key!r}) is missing or NaN.")
        scores[key] = value

    scores["duration"] = json.loads(
        (prediction_dir / "metadata.json").read_text())["duration"]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scores.json").write_text(json.dumps(scores))
    # Keys only: the log is shown to the submitter, values include hidden
    # private metrics.
    print(f"[scoring] wrote {', '.join(scores)} to scores.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Competition scoring")
    parser.add_argument("--prediction-dir", default="/app/input/res")
    parser.add_argument("--output-dir", default="/app/output")
    args = parser.parse_args()

    main(Path(args.prediction_dir), Path(args.output_dir))
