"""Build the image and run ingestion + scoring in it (local platform test).

Mirrors the platform layout: the phase directory (run config) and the
benchmark are mounted as ``/app/input_data``, a submission as
``/app/ingested_program``, and the prepared data as ``/app/data``
(read-only). Prepare the data first with
``benchopt prepare benchmark -d Real`` (writes ``benchmark/data``).

Usage
-----
    python tools/run_docker.py [--data <host-data-dir>] [--submission <dir>]
"""

import argparse
from pathlib import Path

try:
    import docker
except ImportError:
    raise ImportError(
        "The 'docker' package is required: pip install docker."
    )

REPO = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docker ingestion test")
    parser.add_argument("--data",
                        default=str(REPO / "benchmark/data"),
                        help="Host data dir mounted read-only as /app/data")
    parser.add_argument("--submission",
                        help="Folder holding submission.py (default: the "
                        "sample benchmark/solvers/random-forest.py)")
    args = parser.parse_args()

    client = docker.from_env()
    image = "codabench-compet:dev"
    print(f"Building {image}...")
    client.images.build(path=str(REPO), dockerfile="tools/Dockerfile",
                        tag=image)

    phase = REPO / "phases/main/input_data"
    # Without --submission, mount the sample solver as submission.py.
    submission = (
        f"{args.submission}:/app/ingested_program" if args.submission
        else f"{REPO / 'benchmark/solvers/random-forest.py'}:"
        "/app/ingested_program/submission.py"
    )
    out = REPO / "_codabench/ingestion"
    out.mkdir(parents=True, exist_ok=True)
    print("Running ingestion...")
    logs = client.containers.run(
        image=image, remove=True, name="ingestion", user="root",
        command="python3 /compet/ingestion_program/ingestion.py",
        volumes=[
            f"{phase}:/app/input_data",
            f"{REPO / 'benchmark'}:/app/input_data/benchmark:ro",
            submission,
            f"{out}:/app/output",
            f"{args.data}:/app/data:ro",
        ],
    )
    print(logs.decode("utf-8"))

    score = REPO / "_codabench/scoring"
    score.mkdir(parents=True, exist_ok=True)
    print("Running scoring...")
    logs = client.containers.run(
        image=image, remove=True, name="scoring", user="root",
        command="python3 /compet/scoring_program/scoring.py",
        volumes=[
            f"{out}:/app/input/res",
            f"{score}:/app/output",
        ],
    )
    print(logs.decode("utf-8"))
    print("Docker container ran successfully.")
