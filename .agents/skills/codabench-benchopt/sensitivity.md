# Information sensitivity: what to hide, and what it costs

Decide first whether you are hiding anything. A competition on fully public
data, ranked on a reproducible split, is legitimate and by far the cheapest to
run. Everything below is the price of the alternative — spend it where a hidden
cohort buys something concrete: a licence that forbids redistribution, a
generalization claim a public test set cannot support, or prize money large
enough to pay for cheating.

## Three tiers, and where each artifact goes

**Public — the starting kit.** Benchmark code, the submission contract, proxy
dataset modules, the ingestion and scoring programs, the Dockerfile and
requirements, the pages, a sample submission. Plus the metric, unless you
deliberately moved it out of the container (fork 3) — hiding it buys little,
since participants infer it from the leaderboard, and costs the one-code-path
property.

**Published to a phase — readable by every submission container.** The phase
`config.yaml` and a sealed phase's `datasets/*.py`. Tempting to treat as private
because they are not in the public repo; they are not. **Treat everything in a
phase bundle as disclosed the moment the phase opens** — what leaks is the split
logic, dataset identifiers and the seed. If you mind, the split has to live in
the staged data, not in a module shipped beside it.

**Never published.** Credentials always — queue broker URL, infrastructure state
files, API tokens. And, with a hidden cohort, the raw held-out data: staged
out-of-band onto the worker, never in a bundle, never in an API-visible dataset.
This tier is what forces a self-hosted queue (fork 4).

> Make the publisher refuse anything outside an explicit allowlist of names and
> directories. One stray state file published as `input_data` is a credential
> rotation — the mistake a hand-rolled `zip -r` makes on its first run. Related:
> a competition's REST dump embeds the queue credential in clear text. Match
> skip and allow patterns against the path *relative to the bundled root*. A
> skip name like `data` matched against absolute path parts silently drops every
> file once the checkout lives under a `/data/...` mount.

## What the visibility flags actually do

`hide_output`, `hide_prediction_output` and `hide_score_output` close the
**feedback channel**, not the data channel: they make iterative probing of a
hidden split — reconstructing labels a few bits per submission — expensive
rather than free.

They do nothing about exfiltration, since a submission is arbitrary code sitting
next to the evaluation data. That is bounded by the read-only mount, no network
egress, one phase per worker, the quota, and a **reproducibility audit of the
top-k before prizes**. Announce the audit: it is the only one of those that does
not depend on the sandbox holding.

What the flags hide depends on fork 3. Score in the objective and "prediction
output" is the results dataframe. It holds every metric the objective returns,
not only the ones the leaderboard shows. With `hidden: True` leaderboard columns
(one phase scored on a public and a private subset), the results dataframe and
`scores.json` both carry the private scores. A hidden column only hides the
leaderboard cell, so set both `hide_prediction_output` and `hide_score_output`.
Without hidden columns the flag buys almost nothing. Score in the scoring
program and it is per-sample predictions, a real channel. Hiding the *scoring*
output is the expensive one either way: it removes the error traces participants
debug with, so make the development phase generous enough that nobody needs to
debug in the final one.

## The dials, and the cost of each

**1 — How much data is public.** From everything public, through a public proxy
beside a hidden cohort, to nothing public. The middle is the usual compromise
and you pay in distribution shift — a leaderboard that does not rank the way the
proxy did. Pick the proxy that matches the hidden cohort on the axis that
matters to the model's input (acquisition hardware, montage, sampling rate)
rather than the one with the most samples. Releasing nothing is defensible under
a licence, but then you owe participants a simulated dataset they can at least
run the contract against.

**2 — Hidden dataset identity.** Blocks targeted pretraining and leakage through
public annotations of the same recordings; costs participants any domain
reasoning and costs you questions you cannot answer. Commit upfront to
disclosing at phase close — late disclosure is what makes the competition
reproducible without being leaky beforehand.

**3 — Who holds the training data** (the confidentiality half of fork 1). If the
platform trains, participants never hold any of it. If submissions arrive
pre-trained, some training data must be public or proxied and only the
evaluation cohort stays hidden. Decide the fork on its merits first; this is its
consequence, not a separate knob.

**4 — Submission quota = leakage budget.** Submissions-per-day prices
information extraction. A generous warm-up on public data plus a tight sealed
phase (order 1/day, small lifetime cap) puts iteration where it is free and
bounds probing where it is not.

**5 — Asymmetric phases** (if you run more than one — fork 2). The asymmetry
belongs in the data and the visibility flags, not in the code that runs:
anything first executed in the final phase executes where you cannot fix it. If
the phases must differ structurally (training in one, not the other), rehearse
the final configuration on throwaway data before it opens.

**6 — Isolation versus deterrence.** Any submission runs beside the staged
data, so it can read the hidden files and print them in its logs or send them
over the network. A hidden leaderboard column stops none of that. Technical
isolation (hiding logs, blocking egress, running the visible part as a user
that cannot read the sealed files) costs participants their debugging output
and costs you ingestion complexity. When submissions are code the organizers
read, as in a course, a stated sanction (a zero grade, failing the course)
plus a final code review is the proportionate choice. Reserve isolation for
open competitions with anonymous entrants and prizes.

## Tell participants the map

Publish the public/hidden split as a table on the participation page: what
differs between phases (the data, and the dataset identity until disclosure) and
what is identical (image, programs, objective, `meta` keys, batch shapes, their
own submission). Then the rule that follows: **use only `meta` and the batches
you are given** — a solver reading a dataset name, a file path, a subject id or
a hard-coded channel count may break on data it has never seen.

That reframing is the part worth copying: it turns a confidentiality property
participants have no reason to care about into a portability rule that protects
their own score.
