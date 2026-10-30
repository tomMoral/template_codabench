# Template for a benchopt-powered Codabench competition

This repository is a template to create a machine-learning competition on
[Codabench](https://www.codabench.org) whose evaluation is a
[benchopt](https://benchopt.github.io) benchmark. It sets up a dummy binary
classification task, scored by accuracy with 5-fold cross-validation (CV) and
on a public and a private test set.

**The benchmark is the competition.** Participants run locally, with
`benchopt run`, exactly the code the server runs. The Codabench ingestion and
scoring programs are thin wrappers around it:

- **Ingestion** runs the benchmark on a submission. Each repetition trains it
  on one CV fold, scores it on the held-out part of the fold and on the
  public test set, and returns its predictions on the private test set.
- **Scoring** scores these predictions against the private labels, which
  only it can read (the phase `reference_data`), and averages every score
  over the folds.

The metric lives in one place, `benchmark/benchmark_utils/metric.py`, which
both the objective and the scoring program import.

<!-- BEGIN DataCamp course section. Delete up to END for a general release. -->
## DataCamp course: instructions and grading

Your challenge is graded on the three points of
[What the starting kit must show](#what-the-starting-kit-must-show): the data
is interesting, the question is interesting and this data can answer it, and
the evaluation can answer the question. Only what the starting kit shows
counts.

To hand in:

- [ ] Everything in [Before opening the competition](#before-opening-the-competition)
      is checked.
- [ ] The starting kit is executed and committed with its outputs.
- [ ] The repository link and the Codabench competition link are sent to the
      instructors.
<!-- END DataCamp course section. -->

## Documentation and AI skills

- **benchopt documentation**: <https://benchopt.github.io>. Start with the
  [user guide](https://benchopt.github.io/stable/user_guide/) and the
  [seeding guide](https://benchopt.github.io/stable/user_guide/controlling_randomness.html)
  for cross-validation.
- **Codabench documentation**: <https://docs.codabench.org>, in particular the
  [competition bundle](https://docs.codabench.org/latest/Organizers/Benchmark_Creation/Competition-Bundle-Structure/)
  page.
- **AI agent skills**, in `.agents/skills/` (mirrored in `.claude/skills/`
  for Claude Code). Coding agents (Claude Code, Codex, Copilot, Cursor,
  Gemini CLI...) load them when you open the repository. They are plain
  Markdown, so you can also read them yourself.
  - `using-benchopt`: writing and running benchopt datasets, objectives and
    solvers. It ships with benchopt and matches your installed version:
    install it with `benchopt sync-skills` from the repository root, and
    re-run it after a benchopt upgrade.
  - [`codabench-benchopt`](.agents/skills/codabench-benchopt/SKILL.md):
    designing a benchopt-powered Codabench competition. Read
    [`data-and-splits.md`](.agents/skills/codabench-benchopt/data-and-splits.md)
    before choosing your split: it covers cross-validation versus a sealed
    split, leakage, baselines and the public/private shift.

## Layout

| Path | Content |
|---|---|
| `template_starting_kit.ipynb` | skeleton of the participants' starting kit, with the EDA and the evaluation |
| `benchmark/objective.py` | the task: what the solver receives, the CV scheme and the metric |
| `benchmark/datasets/real.py` | the real data: `prepare()` builds the splits, `get_data()` loads them |
| `benchmark/datasets/simulated.py` | synthetic data with the same keys, for smoke tests and CI |
| `benchmark/benchmark_utils/base_solver.py` | the submission contract (`CompetSolver`) |
| `benchmark/solvers/` | built-in baselines: a `Dummy` floor, and `random-forest.py`, the sample submission participants start from |
| `ingestion_program/` | runs `benchopt run` on a submission |
| `scoring_program/` | scores the test predictions against the labels, averages over the folds into `scores.json` |
| `phases/main/input_data/config.yaml` | the phase: benchopt run config (dataset, seed, one CV fold per repetition) |
| `competition.yaml`, `pages/` | the Codabench competition and its pages |
| `tools/` | bundle builder, base and competition Docker images, local Docker run |
| `.agents/skills/` | the AI agent skills above |

## What the starting kit must show

A good challenge makes three points, each backed by evidence in the starting
kit notebook and tied to its exploratory data analysis (EDA).

| Point | Evidence expected in the starting kit |
|---|---|
| **The data is interesting.** | Where it comes from, its license, its size in independent units (patients, sites, sessions, not rows). It is not an already-solved toy dataset. |
| **The question is interesting and this data can answer it.** | Who would use the model and for what decision. A simple baseline clearly beats the `Dummy` floor, and leaves room above it. |
| **The evaluation can answer the question.** | Why this metric and not the obvious alternative. Why this split, which unit it holds out and why it matches deployment. Checks that it does not leak. Public and private scores of the baseline agree, and the test set is big enough for leaderboard gaps to mean something. |

The starting kit carries the full argument. The Participate page is a short
summary of it that links to the notebook. Keep both easy to read: short code
cells, every figure with a title, axis labels and a one-line takeaway, and the
notebook committed **with its outputs** so it reads on GitHub without running
it.

## Adapting the template

In this order, since each step builds on the previous one:

1. **Data and question.** Pick the data and the question. Rename the
   notebook to `<challenge>_starting_kit.ipynb` and fill in its introduction.
2. **EDA.** Explore the data in the notebook. Look for the structure that
   decides the split: groups (patients, sites, subjects), time, duplicates.
3. **Evaluation protocol.** From the EDA, choose the metric, the CV scheme
   (stratified, by group, by time...), and how the test sets are cut and what
   separates the public from the private one.
4. **Data in the benchmark.** Rewrite `Real.prepare()` to fetch or convert
   your raw data, cut the test sets and write the private labels apart (they
   become the phase `reference_data`), and `Real.get_data()` to load the
   rest. Make `Simulated` return the same keys and types.
5. **Task, CV and metric.** Write the metric in
   `benchmark/benchmark_utils/metric.py`. Adapt `Objective.set_data` (data
   and `cv`), `get_objective` and `evaluate_result`, and set `n_repetitions`
   in `phases/main/input_data/config.yaml` to the number of folds.
6. **Submission contract.** Adapt `CompetSolver` to what `get_objective`
   returns, then the `Dummy` floor and the sample `random-forest.py`.
7. **Evaluation evidence.** Fill the notebook's evaluation section: leakage
   checks, and baseline scores in CV, public and private, with their spread
   over the folds.
8. **Leaderboard.** Declare the keys of `scores.json` in `competition.yaml`
   (`hidden: True` for the private ones).
9. **Environment.** The default image,
   `ghcr.io/x-datascience-datacamp/codabench-benchopt`, has benchopt,
   pandas, scikit-learn and torch (CPU). If submissions need more, list it in
   `requirements.txt`, build and push your image from `tools/Dockerfile`, and
   set `docker_image` in `competition.yaml`. `codabench-benchopt-gpu` is the
   CUDA variant, for a self-hosted GPU worker.
10. **Pages, dates and quotas** in `pages/` and `competition.yaml`. The
    Participate page summarizes the notebook.

## Before opening the competition

- [ ] The starting kit covers the three points of
      [What the starting kit must show](#what-the-starting-kit-must-show).
- [ ] `Dummy` and the sample submission are scored in CV, public and
      private, and the sample submission clearly beats `Dummy`.
- [ ] The split has a leakage check in the notebook.
- [ ] CI is green.
- [ ] `bundle.zip` uploads on Codabench and scores the sample submission.
- [ ] No `TODO` is left in `pages/`, and no `<placeholder>` in
      `competition.yaml`.

## Running locally

```bash
pip install -r requirements.txt
benchopt test benchmark --skip-install                  # design checks
benchopt run benchmark -d Simulated -s Dummy -s RandomForest -r 5

benchopt prepare benchmark -d Real
benchopt run benchmark -d Real -s RandomForest -r 5        # 5 CV folds
```

The data goes to `$BENCHOPT_DATA_HOME` (default: `benchmark/data`).
benchopt caches `prepare`: if you delete the data or change
`BENCHOPT_DATA_HOME`, re-run it with `--force`.

## Building and testing the bundle

```bash
benchopt prepare benchmark -d Real   # the bundle ships the prepared data
python tools/create_bundle.py        # -> bundle.zip, to upload on Codabench
```

The bundle carries the competition config, the pages, the programs, the
sample submission and the phase data, from `$BENCHOPT_DATA_HOME` (default
`benchmark/data`):

- `input_data`: the config, the benchmark and the prepared data, without the
  private labels. Submissions run with it.
- `reference_data`: the private labels (the `reference/` folder written by
  `prepare`). Only the scoring program reads it, with the metric of
  `benchmark_utils`, which the bundle copies next to it.

A benchmark or data change is a new upload of the phase, not an image
rebuild. It runs on Codabench's default compute workers, so you need no
server of your own. Keep the prepared data small (tens of MB), in a compact
format (Parquet, `.npz`) rather than raw files.

To run the platform path locally, extract the bundle and run its programs:

```bash
python -c "import zipfile; zipfile.ZipFile('bundle.zip').extractall('_codabench/bundle')"
python _codabench/bundle/ingestion_program/ingestion.py \
    --submission-dir _codabench/bundle/solution \
    --input-data _codabench/bundle/main_phase/input_data \
    --output-dir _codabench/ingestion
python _codabench/bundle/scoring_program/scoring.py \
    --prediction-dir _codabench/ingestion \
    --reference-dir _codabench/bundle/main_phase/reference_data \
    --output-dir _codabench/scoring
cat _codabench/scoring/scores.json
```

`python tools/run_docker.py` does the same inside the Docker image (needs
`pip install docker`).

## Large data: staging it on a self-hosted worker

Data too large for the bundle needs a
[self-hosted compute worker](https://docs.codabench.org/latest/Organizers/Running_a_benchmark/Compute-Worker-Management---Setup/)
attached to a private queue. Delete `main_phase/input_data/data/` from the
bundle and keep its `reference_data`: ingestion then reads the data from
`/app/data`, where the worker mounts it read-only. Prepare it once, with the
image, from the phase `input_data` of the bundle, so that what is staged and
what submissions run come from the same files. Then delete the staged
`reference/` folder, which submissions could read:

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
scoring end to end on `Real`, and builds the Docker image on the CPU base
image. If your data cannot be prepared in CI, run the end-to-end step on
`Simulated`, which has no private labels.

`.github/workflows/images.yml` builds and pushes the base images
`codabench-benchopt` and `codabench-benchopt-gpu` from
`tools/Dockerfile.base` to the GitHub container registry.
