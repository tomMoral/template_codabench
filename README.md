# Template for a benchopt-powered Codabench competition

This repository is a template to create a machine-learning competition on
[Codabench](https://www.codabench.org) whose evaluation is a
[benchopt](https://benchopt.github.io) benchmark. It sets up a dummy binary
classification task, scored by accuracy on a public and a private test set.

**The benchmark is the competition.** Participants run locally, with
`benchopt run`, exactly the code the server runs. The Codabench ingestion and
scoring programs are thin wrappers around it.

## Layout

| Path | Content |
|---|---|
| `benchmark/objective.py` | the task: what the solver receives, and the metric |
| `benchmark/datasets/real.py` | the real data: `prepare()` builds the splits, `get_data()` loads them |
| `benchmark/datasets/simulated.py` | synthetic data with the same keys, for smoke tests and CI |
| `benchmark/benchmark_utils/base_solver.py` | the submission contract (`CompetSolver`) |
| `benchmark/solvers/` | built-in baselines (here a `Dummy` floor) |
| `solution/submission.py` | the sample submission participants start from |
| `ingestion_program/` | runs `benchopt run` on a submission |
| `scoring_program/` | turns the benchopt results into `scores.json` |
| `phases/main/input_data/config.yaml` | the phase: benchopt run config + leaderboard column map |
| `competition.yaml`, `pages/` | the Codabench competition and its pages |
| `tools/` | bundle builder, Docker image, local Docker run |
| `template_starting_kit.ipynb` | skeleton of the participants' starting kit |

## Adapting the template

In this order, since each step builds on the previous one:

1. **Evaluation protocol.** Decide how the data is split (random, by group,
   by time...) and what separates the public from the private test set.
2. **Data.** Rewrite `Real.prepare()` to fetch or convert your raw data and
   cut the splits, and `Real.get_data()` to load them. Make `Simulated`
   return the same keys and test subsets.
3. **Task and metric.** Adapt `Objective.set_data`, `get_objective` and
   `evaluate_result`. Metrics come out as `objective_<metric>_<subset>`.
4. **Submission contract.** Adapt `CompetSolver` to what `get_objective`
   returns, then the `Dummy` floor and `solution/submission.py`.
5. **Phase and leaderboard.** Map the leaderboard keys to the objective's
   columns in `phases/main/input_data/config.yaml`, and declare the same keys
   in `competition.yaml` (`hidden: True` for the private ones).
6. **Environment.** List everything submissions may import in
   `requirements.txt`, build and push the image from `tools/Dockerfile`, and
   set `docker_image` in `competition.yaml`.
7. **Pages, dates and quotas** in `pages/` and `competition.yaml`.

## Running locally

```bash
pip install -r requirements.txt
benchopt test benchmark --skip-install                  # design checks
benchopt run benchmark -d Simulated -s Dummy -s solution/submission.py

benchopt prepare benchmark -d Real
benchopt run benchmark -d Real -s solution/submission.py
```

The data goes to `$BENCHOPT_DATA_HOME` (default: `benchmark/data`).
benchopt caches `prepare`: if you delete the data or change
`BENCHOPT_DATA_HOME`, re-run it with `--force`.

## Building and testing the bundle

```bash
python tools/create_bundle.py        # -> bundle.zip, to upload on Codabench
```

The bundle carries the competition config, the pages, the programs, the
sample submission and the phase `input_data`: its config and the benchmark,
so a benchmark change is a push of the phase, not an image rebuild. The data
is staged on the compute worker, so it is not in the bundle.

To run the platform path locally, extract the bundle and run its programs.
Point `BENCHOPT_DATA_HOME` at the prepared data: the bundled benchmark
otherwise looks for it in its own `data/` folder.

```bash
python -c "import zipfile; zipfile.ZipFile('bundle.zip').extractall('_bundle')"
export BENCHOPT_DATA_HOME=$PWD/benchmark/data
python _bundle/ingestion_program/ingestion.py \
    --submission-dir _bundle/solution \
    --input-data _bundle/main_phase/input_data --output-dir ingestion_res
python _bundle/scoring_program/scoring.py \
    --prediction-dir ingestion_res --output-dir scoring_res
cat scoring_res/scores.json
```

`python tools/run_docker.py` does the same inside the Docker image (needs
`pip install docker`).

## Staging the data on a compute worker

The worker mounts the prepared data read-only at `/app/data`. Prepare it
once, with the image, from the phase `input_data` of the bundle: what is
staged and what submissions run then come from the same files, and the
datasets are those of the phase config.

```bash
docker run -e BENCHOPT_CACHE=/tmp/benchopt -v <host-data>:/app/data \
    -v <bundle>/main_phase/input_data:/app/input_data:ro <image> \
    benchopt prepare /app/input_data/benchmark \
    --config /app/input_data/config.yaml
```

`BENCHOPT_CACHE` keeps the prepare cache off the read-only benchmark, and the
container drops it, so a new staging prepares for real. Run it a second time
with `<host-data>:/app/data:ro`, as submissions see the data: a dataset that
still writes fails there rather than on every submission.

## CI

`.github/workflows/test.yml` lints, runs `benchopt test`, smoke-runs the
baselines on `Simulated`, then builds the bundle and runs its ingestion and
scoring end to end on `Simulated` (CI has no real data), and builds the
Docker image.
