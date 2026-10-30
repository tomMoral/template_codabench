# Step 0: the evaluation protocol

Settle before the forks: the protocol decides fork 1 (a protocol needing
server-side cross-validation can only be training-on-platform) and fork 2
(whether a sealed split exists at all). Deciding it after the benchmark is
written means rewriting the benchmark.

## Cross-validation or a single sealed split

Rule of thumb: **scarce data needs CV; expensive or confidential data needs a
privacy design that CV does not fit.** Most competitions sit between and must
choose.

- Few independent units (subjects, sessions, sites — not rows) means a single
  split's variance can exceed the gap between the top entries. If you rank on
  one split anyway, estimate that variance first and use it to decide whether
  the top-k are distinguishable before awarding anything on the difference.
- CV puts every sample in a test fold, so you cannot cross-validate *and* keep
  a fold hidden. The only way to have both is server-side CV under
  training-on-platform (fork 1 B), at k times the compute.
- Under inference-only submissions, server-side CV means the submission ships
  k models. If you get there, the ranking wanted a single sealed split.

Common compromise: rank on a single sealed split, and ship a CV configuration
over the public data so participants model-select against something other than
the leaderboard.

## Leakage in the split

The split is part of the task definition, and in a competition its failure mode
is specific: a leak ships inside the starting kit, every participant inherits
it, a participant finds it publicly mid-phase, and it cannot be fixed without
invalidating the leaderboard.

Train and test must be independent for the generalization the leaderboard
claims to measure. Beyond the usual group, temporal and duplicate leaks, one
case is easy to get wrong:

- **Several axes at once.** When an observation is a combination (gene ×
  condition, user × item, subject × stimulus), holding out "new A × new B" is an
  L-shaped split: A-test units still appear in train with B-train, and the
  reverse. That is by design, so do not assert A disjointness across sides.
  Assert instead that each axis label is constant per unit (a unit is test on
  that axis everywhere), and that no *cell* is shared. An axis held out
  completely (a new species, a new site) must be disjoint.

What follows from it:

- Split on the coarsest unit that differs at deployment — by subject for "works
  on a new subject", by site for "works at a new site". Recording-level for
  windowed signals, plus a gap at the boundary when windows overlap.
- The *target* leaks even when the input does not: one stimulus seen by many
  subjects lets a model retrieve instead of predict. Deduplicate on the target.
- Anything fitted on the full dataset (normalization, PCA, channel selection)
  belongs inside the training fold — in a competition this one is not a private
  mistake, it is shipped code.
- Ids, file names and row order are readable to every submission, since a phase
  bundle is ([sensitivity.md](./sensitivity.md)). Strip what should not be.

Test it: assert the independence above in the split script (no unit on both
sides of a fully held-out axis, no shared cell otherwise), and run a **cheating
baseline** on ids and metadata alone. Above chance means the split leaks.

## Check the metric against sensible baselines

Before opening, score a trivial floor and a simple model on every ranked subset:
the simple model must clearly beat the floor (else the task is broken or
impossible) and leave room above it (else it is trivial). Seed them so the
numbers are reproducible. Three less obvious traps:

- **Probabilistic metrics.** Under a likelihood, a constant with a wide,
  well-calibrated spread can be a hard floor a decent point predictor loses to.
- **Heavy-tailed per-observation losses.** A count likelihood (NB, Poisson)
  grows like `y · log(1/mu)` when `mu` is near 0, so one bad prediction on a
  highly expressed observation can outweigh thousands of rows. Look at the
  per-observation loss of the reference (max, top-10 share) and cap it (e.g.
  at about 15× a typical row) rather than ranking on a mean one row decides.
  Clip the baselines' `mu` away from 0, too: `expm1` of a negative log
  prediction is negative, and a clip at 0 then predicts zero counts.
- **The synthetic smoke-test dataset needs the same check**, with enough signal
  that the reference beats the floor, or the smoke test passes on a pipeline
  that cannot learn.

When a public half shows on the leaderboard and a hidden half ranks, halve each
held-out region by the unit that is new in it (gene, run, subject), with a hash
of its key, so the same unit falls in the same half everywhere. A region with no
new unit can be halved by observation. Then score the reference on both halves:
they should agree within noise.

## Name the shift you are measuring

Any gap between what participants develop on and what ranks them — public proxy
vs hidden cohort, development vs final phase — is what the leaderboard actually
measures.

- Separate shift you **chose** (different site, hardware or population, because
  generalizing across it is the research question) from shift you **inherited**
  (the proxy is an older, dirtier dataset). Announce the first; reduce the
  second.
- **Measure it before opening**: train a baseline on the public part, evaluate
  on the sealed part. If two baselines swap order, that is what participants
  will experience as randomness. This doubles as the end-to-end rehearsal of the
  sealed phase.
- Declare it on the participation page — "developed on cohort X, ranked on
  cohort Y recorded with the same hardware" is also the honest reading of the
  numbers when they are later cited.

What to publish of the data: dial 1 of [sensitivity.md](./sensitivity.md).
