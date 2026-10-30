---
name: codabench-benchopt
description: >
  Designing and operating a machine-learning competition on Codabench whose
  evaluation is a benchopt benchmark — where each decision belongs (benchmark
  vs phase config vs ingestion vs scoring vs competition.yaml vs worker), the
  design forks that are genuinely open (training on the platform or not,
  metric in the objective or in the scoring program, hosted or self-hosted
  workers, how many phases), and how much to hide from participants at what
  cost. Also the evaluation protocol the platform design rests on: single
  sealed split vs cross-validation, leakage in the split, and the distribution
  shift between public and ranked data. Consult when setting up or reviewing a
  competition bundle, an ingestion/scoring program, a phase input_data
  directory, a hidden/sealed test phase, a self-hosted compute worker, or when
  deciding what goes in the public starting kit. Trigger words: Codabench,
  competition bundle, competition.yaml, ingestion program, scoring program,
  input_data, reference_data, phase, sealed phase, hidden test set,
  leaderboard, starting kit, submission contract, compute worker,
  hide_prediction_output, leakage, train/test split, cross-validation, group
  split, distribution shift, benchopt competition.
---

# Benchopt-powered Codabench competitions

**The benchmark is the competition.** The platform is a scheduler and a
leaderboard; the programs it runs are adapters. Everything here serves one
invariant — participants run the same code the server runs.

## Start here: protocol, then forks, then build

**Step 0 is the evaluation protocol**, not the platform: cross-validation or a
single sealed split, how the split is cut, what shift sits between what
participants develop on and what ranks them
([data-and-splits.md](./data-and-splits.md)). The forks below inherit it.

Then settle these, in order — each narrows the ones after it. Ask if they are
not on record rather than picking silently
([design-decisions.md](./design-decisions.md)):

1. Does the platform train, or only run inference on a shipped model?
2. How many phases — one, a development/final pair, or more?
3. Where is the metric computed — objective, or scoring program?
4. Hosted Codabench workers, or a self-hosted queue?
5. Where does the benchmark live — phase bundle, or baked into the image?
6. Is the submission environment frozen, or resolved per submission?

## Layer map

| Layer | Artifact | Owns |
|---|---|---|
| Benchmark | `tracks/<t>/{objective,datasets,solvers}`, or `benchmark/` for a single track | task, loaders, submission contract, and the metric *if* you score there |
| Submission | `submission.py` + weights and/or training code | the model, nothing else |
| Phase | `input_data/` = `config.yaml` (+ benchmark, + sealed `datasets/*.py`) | which data, which split, seed, training on or off, leaderboard column map |
| Ingestion | `ingestion_program/` | filesystem wiring; exec `benchopt run` |
| Scoring | `scoring_program/` | results → `scores.json`, parsing or computing the metric |
| Competition | `competition.yaml` | windows, quotas, visibility flags, pages, image |
| Infra | image + worker + staging | environment, data residency, mounts, queue |

Per-layer boundaries and traps: [responsibilities.md](./responsibilities.md).

## Invariants, whichever fork you take

- **One code path.** The benchmark participants run locally is the benchmark the
  server runs. No platform detection, no "server mode" branch in the objective.
- **Task in the benchmark, configuration in the phase.** Whatever a phase may
  change *without changing the task* — dataset and parameters, seed, training
  on/off, leaderboard column map — lives in the phase config, not the objective
  and not the image.
- **Fail loudly at every boundary.** The characteristic failure here is not a
  crash, it is a plausible wrong number on a leaderboard nobody can re-derive
  three weeks later.
- **Placement is per artifact, not per repo.** Everything shipped in a phase is
  readable by every submission container; everything in the public repo is
  readable by everyone ([sensitivity.md](./sensitivity.md)).
- **Minimize what differs between phases.** Each difference is a code path first
  exercised where mistakes are unfixable. Some designs legitimately change more
  than the data — pay that knowingly, and rehearse before the phase opens.

## Starting point: the template

Start a new single-track competition from
[`x-datascience-datacamp/template_codabench`](https://github.com/x-datascience-datacamp/template_codabench),
a GitHub template repo. It encodes the defaults: training on the platform,
one phase with hidden private columns, metric in the objective, benchmark
shipped in the phase `input_data`, frozen environment, data built by
`Dataset.prepare()`, and CI end to end on `Simulated`. Its README lists the adaptation steps in
order. Change the template itself rather than diverging copies, so fixes
reach the next competition.

## Reference implementation

For multi-track and sealed-phase designs: the NeurIPS 2026 neural-interfaces
competition — `2026-neurips_compet-eeg`
(public benchmark + bundles) and its private ops repo (Terraform workers, phase
publishing). Four tracks, one image, warm-up on public proxy data and a sealed
final phase on a held-out cohort; metric in the objective, inference-only,
self-hosted workers.

**Its code encodes its forks. Re-check each piece against yours before copying
it.** Copying it verbatim into a single-track port means deleting, unless your
fork needs them:

- The objective `training` toggle. Only needed when phases differ (fork 1).
- Symlink dereferencing and the dangling-link guard. These come from
  multiple tracks sharing code through symlinks.
- The sealed `datasets/*.py` drop-in. Only for a sealed phase that ships its
  split as code.
- The no-config auto-detect fallback in scoring. With one phase config, read
  `config["scoring"]["columns"]` and fail otherwise.
- The workdir copy and solver install in ingestion. They predate benchopt path
  selectors ([responsibilities.md](./responsibilities.md)).

For a single track, put the benchmark in its own `benchmark/` folder rather
than the repo root, so it is clear what is benchopt and what is the Codabench
wrapper.
