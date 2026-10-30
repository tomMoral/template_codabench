# Who owns what

Per layer: what it decides, what it must not, and the trap when the boundary is
crossed. Boundaries that depend on an unsettled fork point at
[design-decisions.md](./design-decisions.md).

## Benchmark (the benchopt repo)

**Owns** data loading, the `meta` keys handed to a submission, batch shapes, the
base solver class that spells out the contract — and the metric, if you score in
the objective (fork 3). **Not** which dataset instance runs, the seed,
leaderboard names, or anything with a date or a quota in it.

- The benchmark *is* the starting kit and *is* what the server runs: a plain
  `benchopt run <benchmark>` must reproduce a platform evaluation. Any
  competition-only knob in the objective (a `track` parameter, an env var read,
  a branch on "am I on the platform") forfeits that.
- Whether the platform trains is phase-level. When phases differ, carry it on
  the **dataset** the phase config selects (fork 1), not two benchmarks, not a
  server-side flag. Keeping it implicit in the solver's arguments (no train
  loader rather than `training=False`) means a phase that staged only a test
  split cannot accidentally train. When every phase trains, there is no toggle
  at all.
- Make `benchopt test` exercise paths the phases you run today never take —
  otherwise the untaken half is first tested by participants, or on the day you
  open a phase that uses it.

## Submission

**Owns** the model, and the training recipe if the platform trains.

Ingestion loads it by path (`-s <dir>/submission.py`), so the base class finds
its shipped files from `inspect.getfile(type(self))` with no env var. The same
file then works locally and on the platform. Loading by path has one
consequence to state on the participation page: **all Python code lives in that
single file**, because sibling `.py` files are not importable. The folder may
still hold non-code files (weights, configs). For the inference-only shape it
ships as code + weights. A `save_model`
hook that zips solver + weights at the end of a training run lets participants
*produce* the upload artifact from a local run — it removes an entire class of
"my zip had the wrong layout" tickets.

## Phase (`input_data/`)

**Owns** dataset name and parameters, seed, whether this phase trains, the
leaderboard-key → results-column map, and for a sealed phase the dataset module
defining the split.

- Keep it a **native `benchopt run --config` file** plus the minimum of
  competition-only keys, stripped by ingestion before the run (benchopt rejects
  unknown options) and forwarded to scoring. The same file then drives a local
  reproduction.
- If the benchmark travels in the phase (fork 5), this directory is the unit
  published *and* staged, and symlinked shared code must be dereferenced at
  every copy — see fork 5.

## Ingestion program

**Owns** wiring only. benchopt takes paths, so ingestion needs no workdir copy
and no solver install:

    benchopt run <benchmark> -s <submission_dir>/submission.py \
        --config <run_config> --output <output_dir>/results.parquet \
        -r 1 --no-cache --no-plot --no-html --no-display

What remains is to write the run config (the phase config minus its
competition-only keys), exec, and forward the config to scoring. Path loading
also means ingestion never imports the submission itself. A sealed phase that
ships `datasets/*.py` passes them the same way (`-d <path>`). Copying the
benchmark to a workdir and dropping the submission into `solvers/` is only the
fallback for a benchopt without path selectors. Run this way, benchopt writes
nothing into the benchmark, so it runs as is from a read-only `input_data`.

- **Refuse to run when the phase config is missing.** Without `-d`, benchopt
  runs *every* dataset in the benchmark — on a sealed phase that is a runaway
  bill and an evaluation on data the phase never meant to touch.
- Debug flag on, so a broken submission aborts with its traceback instead of an
  empty results frame that scoring reports as a missing column. Caching off: two
  submissions can carry the same solver name, and a cache outliving one of them
  is both a wrong score and a channel between participants.
- If the benchmark travels in the phase through a symlink (fork 5 A), check the
  *unresolved* path when guarding against a dangling link. `.resolve()` flattens
  a broken symlink into an ordinary missing path and the error loses its cause.
- `benchopt run` does not call `Dataset.prepare()` unless passed `--prepare`,
  and ingestion runs `--no-cache` without it — so at eval time `get_data` is the
  only thing that readies data. Have it load already-prepared data and fail
  loudly if the staged data is missing (prepare is the explicit staging step),
  not silently re-download or repeat the extraction.
- **Data preparation is `Dataset.prepare()`, not a `tools/setup_data.py`.**
  Fetching, converting and splitting the raw data belongs to the dataset that
  loads it, so participants and the staging step both run
  `benchopt prepare <benchmark> -d <Dataset>` and get the same files. A side
  script duplicates the split logic outside the benchmark and drifts from the
  loader. Prepare writes the sealed split only when its raw source is present
  (the worker), and the public kit prepares the public part only. Make that
  a dataset parameter the phase config sets (`mode=full` versus the default
  `mode=public`), with `full` failing loudly without the raw source. The raw
  source sits in a private bucket prefix that staging mirrors for prepare and
  deletes before publishing the data to the workers. Synthetic
  data for CI goes in the `Simulated` dataset, not in a `--demo` flag.
  Trap: benchopt caches `prepare` in `<benchmark>/__cache__` without checking
  its output. After deleting the data or changing `BENCHOPT_DATA_HOME` it
  reports "cached" and writes nothing, so pass `--force`, and say so in the
  `get_data` error for missing data.

## Scoring program

**Owns** results dataframe → flat `scores.json`, and the metric if you score
here (fork 3). Either way, a score the phase declared but that is missing or NaN
must fail loudly — a silent zero reaches the leaderboard and is found weeks
later.

`reference_data/` follows the same fork: scoring in the objective means no
labels reach this step, so it is a documented placeholder and the scoring
program stays public even for a sealed phase; scoring here means it holds the
ground truth and inherits the phase's confidentiality.

## `competition.yaml`

**Owns** everything with a date, a quota or a visibility flag: windows,
submissions per day and total, execution time limit, `hide_*`, pages, the Docker
image, the leaderboard columns.

- Leaderboard column keys must match the scoring output keys exactly — the most
  common silent failure, and it shows as an empty column, not an error.
- **Phase windows must not touch**: an end equal to the next start is rejected
  as a conflict.
- **Leaderboard column titles are capped at 36 characters** (keys too). A longer
  one fails the bundle upload with a garbled message (`'int' object has no
  attribute 'items'`): Codabench's error formatter crashes on the nested
  validation error and hides which field it was.
- The yaml is authoritative for dates and prose pages drift from it immediately
  — keep the prose thin and point at the platform's own phase display.
- Once live, the yaml is no longer the deployed state: the bundle is an
  initialiser and later edits go through the REST API. Keep the repo yaml as the
  intended state and reconcile deliberately.
- **Leaderboard columns change only through the competition PATCH, with the
  phases in the same request.** `/api/leaderboards/` is read-only, and the
  competition view saves `leaderboards[0]` only when `phases` is present (else
  a 500, nothing applied). It rewrites each sent phase's `public_data` and
  `starting_kit`, so send them. Columns sent with their id keep their scores.
- **Hidden columns are absent from `/api/submissions/<id>/` scores**, even for
  an organizer. To check a private column fills, read `scores.json` from the
  `scoring_result` zip of `get_details`.

## Infra: image, worker, staging

**Owns** the environment and where the data physically lives. The staging points
apply to a self-hosted queue (fork 4); on hosted workers they are someone else's
problem, and so is the guarantee they provide.

- **One directory is the source of truth for a phase**, pushed both to the
  platform and to the worker — otherwise what a submission runs and what was
  staged can diverge, which is quiet, since both halves work on their own. But
  the two halves deploy differently: the **code** (config + ingestion/scoring
  programs) is *published* to the platform and delivered per-job, while the
  heavy **data** is *staged* worker-local and mounted. A config or program
  change is a re-publish; a data change is a re-stage — confuse them and new
  code runs on old data, or the reverse.
- **Stage from the phase directory itself.** Ship the phase `input_data` (the
  same bytes the platform gets) to the worker, then run, in the competition
  image, `benchopt prepare /app/input_data/benchmark --config
  /app/input_data/config.yaml`. Mount `input_data` read-only and set
  `BENCHOPT_CACHE` to a path inside the container. Otherwise the prepare cache
  lands in the read-only benchmark, or outlives the data and reports "cached"
  on the next staging. `prepare --config` reads only the `dataset` key, so the
  datasets come from the phase config, never from a second list in the infra
  config. Ship the programs and the sample submission with it for the
  read-only check below. Staging is then a script run from the checkout, not
  a re-provisioning, and it never needs the benchmark inside the image.
- **Provisioning and staging are separate.** Provisioning leaves the worker
  *stopped*; staging pushes the phase, prepares the data, then starts it. A
  worker booting into a queue with no data fails every submission it picks up.
- **Mount the data read-only** if your datasets tolerate it: runs then hit warm
  caches or fail loudly rather than downloading inside the execution limit.
  Validate by running the sample submission through the ingestion and scoring
  programs during staging, with the data read-only. That is what catches a
  dataset that writes at load time, and a missing leaderboard column. Replaying
  `prepare` read-only does not: submissions never call it, and a `prepare`
  that rebuilds its outputs every time fails that replay for nothing. If `prepare` fans out across
  processes, its workers can race a shared on-disk cache and corrupt it on a
  large study; serialize the extractor for those datasets.
- **A read-only data mount cannot double as a writable cache.** Pointing a model
  cache (HF, torch) at it breaks a participant's own `from_pretrained`, which
  must write. Pre-stage the shared weights once, then have ingestion
  symlink-mirror them into each container's writable default cache — reuse with
  no download inside the limit, and it works offline for a sealed phase. Declare
  the pre-staged models in the *public* repo so participants can rely on and PR
  the list.
- **Submission containers run as root and inherit the docker daemon's defaults**
  — a self-hosted worker spawns them over the socket with no `--user`, no
  `--shm-size`. The 64 MB default `/dev/shm` then kills any DataLoader with
  `num_workers>0` on large tensors; raise the daemon-wide `default-shm-size` in
  provisioning, the only lever you have.
- **One worker serves one phase.** Bounds the disk, and bounds what a submission
  reaches if the container boundary fails.
- Publish the environment contract early, whichever fork 6 branch you took:
  participants need to know what they may import before they train.

## CI for the competition repo

Test the same path the platform runs: `benchopt test <benchmark>`, a smoke
`benchopt run` on the synthetic dataset, then build the bundle and run its
ingestion and scoring programs end to end, asserting that `scores.json` holds
every leaderboard key. CI has no real data and needs no data step: swap the
dataset in the bundled phase config for `Simulated` (`sed -i
's/<Dataset>/Simulated/' _bundle/<phase>/input_data/config.yaml`). This works
because `Simulated` emits the same test subsets as the real dataset. Traps
that each cost a red run:

- `benchopt test` needs `pytest` installed explicitly.
- `benchopt prepare` caches into `<benchmark>/__cache__`, which flake8 then
  fails on. Exclude `__cache__` in the lint step.
- A CUDA torch image runs out of disk on a GitHub runner when exported. CI only
  needs to know the image builds, so build a CPU variant (a `TORCH_CHANNEL`
  build arg selecting the wheel index) with `load: false` and free runner disk
  first. Production keeps the CUDA channel.
- The sample submission lives outside `solvers/`, so `-s <name>` cannot select
  it in a smoke run. Smoke-run the benchmark's own solvers and exercise the
  submission through the end-to-end ingestion step.
