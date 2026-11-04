# The open design decisions

Six forks with no universally right answer. **Do not start from the benchmark**
— what the objective returns, what a submission is, what the image contains and
whether a staging pipeline exists at all descend from these. If you are asked
to scaffold a competition and they are not on record, ask; if the user has no
opinion, propose a default *with its cost* rather than picking silently. Record
the answers in the repo.

Two preconditions, neither a platform decision: the evaluation protocol
([data-and-splits.md](./data-and-splits.md), which may already decide fork 1)
and what must stay hidden ([sensitivity.md](./sensitivity.md), which constrains
forks 1, 2 and 4).

Listed in dependency order — each narrows the ones below:

| # | Decision | Determines |
|---|---|---|
| 1 | Does the platform train? | the submission contract, the objective's parameters, the time limit and the compute bill |
| 2 | How many phases? | how many data configurations exist, the windows, whether a sealed split exists |
| 3 | Where the metric is computed? | what the objective returns, what scoring does, what `reference_data/` holds, what `hide_*` means |
| 4 | Hosted or self-hosted workers? | whether you own staging, mounts and a queue — half of [responsibilities.md](./responsibilities.md) |
| 5 | Where the benchmark lives? | the bundle/build tooling and the release cadence |
| 6 | Frozen or resolved environment? | the Dockerfile, the dependency contract, whether the container needs network |

## 1. Does the platform train?

**A — inference-only.** The submission is a trained model (code + weights); the
server runs `predict`. Compute per submission is small and predictable. Forces a
public or proxy training split, and forfeits adaptation to the evaluation cohort.

**B — training on the platform.** The submission is a training recipe; the phase
hands over a train loader. Everyone trains on identical hardware under an
identical budget — a comparison of methods rather than of compute access — and
the training data can stay hidden. Costs: the execution time limit must cover
training (× the quota = the GPU bill), scores become seed- and budget-sensitive,
and a training crash on the server is far harder to debug than a local one.

First because it defines the submission contract, which the benchmark, the
sample submission and every participant-facing page are written against. It may
also be forced: a protocol cross-validating over data participants never hold
can only be B.

Wire the fork so one benchmark serves both and nothing branches on "am I on the
platform", and so the *absence* of training reaches the solver as
`train_loader=None` rather than a boolean it might ignore (a phase that staged
only a test split then cannot accidentally train). This competition put
`training` on the **objective** (`-o "Obj[training=True]"`, selected by the phase
config); in hindsight that is probably the wrong axis — it is unrelated to the
data yet participants must set it alongside `-d`, and hiding it took a dedicated
training-config file. Prefer carrying it on the **dataset** instead: a training
dataset variant supplies the `train_loader`, the evaluation variant supplies
none, so whether-to-train falls out of the `-d` selection participants already
make. (Not worth retrofitting the 2026 competition — note it for the next.)

The toggle only earns its place when phases differ. If every phase trains, drop
it entirely: the objective always hands over the training data and the base
solver's `run` always calls `fit`, locally and on the server. A `training`
parameter added "for flexibility" in a train-only competition is dead
configuration.

Mixed designs (a warm-up that trains, a final that does not) are legitimate, but
that is a second variable changing between phases — see the invariants in
[SKILL.md](./SKILL.md).

## 2. How many phases

**A — one.** No staging switchover, no deadline to miss. The leaderboard *is*
the ranking, so every submission probes the data that decides prizes, bounded
only by the quota — and the first time the stack runs under real load is the
competition itself.

**B — development + final.** Long development phase on public or proxy data
with a generous quota, short final phase on the held-out cohort with a tight one
and hidden outputs. Bounds probing of the ranking data, puts the support load
where mistakes are still fixable, and makes the development phase the rehearsal
for the final. Costs: two data configurations, a mid-competition switchover, and
a resubmission step participants reliably forget.

**C — more than two.** For genuinely multi-round designs. Each phase is another
staging event and another deadline, and the overhead multiplies by tracks.

Whatever the count:

- **Submissions do not carry over.** Entering the final phase is a fresh upload;
  say so on the participation page and again when the phase opens.
- **Phase windows must not touch** — an end equal to the next start is rejected
  as a conflict; end at `23:59:59`. The ranked phase must be unambiguous in the
  rules *and* in the platform's phase flags.
- **Stage before the window opens**, while the worker still serves the previous
  phase: either a second worker or disk sized for two phases. A switchover
  discovered at the boundary is downtime on the phase that matters most.
- If all you want is bounded leaderboard overfitting, one phase scored on two
  subsets — one surfaced, one held back for the ranking — gets most of it for a
  fraction of the ops. Verify your Codabench version can hide a leaderboard
  column before building on it.

## 3. Where the metric is computed

**A — in the objective; scoring parses.** `evaluate_result` returns the metrics,
they land as `objective_<name>` columns, scoring maps columns to leaderboard
keys. One implementation of the metric, participants get the exact leaderboard
number from a local `benchopt run`, and `reference_data/` stays empty because no
labels reach the scoring step.

**B — the objective returns raw predictions; scoring scores them.** The classic
Codabench shape. Benchopt packs non-scalar objective columns at write time, so
scoring must rematerialize with `benchopt.results.read_results` — a plain
`pd.read_parquet` returns the packed blob, not the array.

- B keeps the metric code and the ground truth out of the container the
  submission runs in, and lets you re-score stored predictions after the fact (a
  metric bug, an added ranking). Under A the number is all you kept.
- A is exercised by every local run; under B the metric has no local code path,
  so it needs its own tests and documentation.
- B moves prediction-sized arrays through the results file — check the size for
  your largest phase; embeddings per test window are not class labels.

## 4. Hosted workers or self-hosted

**A — the public Codabench queue.** No infrastructure. Requires the evaluation
data to be fetchable by a worker you do not control, and gives no say over
hardware, disk or isolation.

**B — self-hosted on your own queue.** Required when the data may not leave your
control, or for specific GPUs, large disk or a guaranteed slot. You inherit
provisioning, staging, mount policy, queue credentials and the bill; everything
in [responsibilities.md](./responsibilities.md) about staging and read-only
mounts applies to this branch only.

Mostly decided by the two above: hidden data, or training on the platform at any
scale, rules out A.

## 5. Where the benchmark lives

**A — in the phase bundle**, the image carrying the environment only. One image
serves every track and phase, and a sealed phase can ship a modified benchmark
with no rebuild. Costs an assembly step — and if shared code is symlinked into
the phase directories, every copy must dereference (`cp -rL`, `tar -czh`,
`copytree`, `os.walk(followlinks=True)`); `Path.rglob` does not descend into a
symlinked directory and silently yields a benchmark with no shared code.

**B — baked into the image.** No assembly, and the image tag pins the evaluation
exactly. Costs a build and push per benchmark change, an image variant per track
or phase that differs, and a sealed split living in the image — so a private
registry and pull credentials on the worker.

**Default: A, even for one track and one phase.** The benchmark is a few tens
of KB of Python, so shipping it costs nothing, and a fix to a loader or the
metric is a phase push (`push.py`) rather than an image build and push. The
image then changes only when a dependency does. For a single track the
assembly step is one more `_add_dir` in the bundle builder, with no symlinks.
Staging then prepares the data from that same phase directory, not from a
benchmark copy baked into an older image ([responsibilities.md](./responsibilities.md),
Infra).

## 6. Submission environment: frozen or resolved

**A — frozen image.** A pinned requirements file is the contract; nothing
installs at submission time. Participants are restricted to that stack, and a
new dependency means an image release mid-competition.

**B — resolved per submission**, solvers declaring their own requirements. Costs
network egress from the evaluation container (which undercuts a hidden-data
story), install time inside the execution limit, and environments that differ
between two runs of the same submission.

Middle path: freeze the image, publish the requirements file early, and run a
documented process for adding to it — so the constraint is negotiable before
people have trained against the wrong stack.
