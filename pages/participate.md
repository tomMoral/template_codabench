# How to participate

## The task

<Describe the data, the task and why it matters.> In this template: a binary
classification of tabular samples, scored by **accuracy** (higher is better).

## The split

- **Public column** (live feedback): a public test set.
- **Private / final ranking** (hidden): a private test set. Do not overfit
  the public column.

## What you submit

A single **`submission.py`** defining `class Solver(CompetSolver)`. All your
Python code must live in this one file: the platform loads it on its own, so
it cannot import other `.py` files you ship. You may add non-code files next
to it (model weights, configs) and read them from `meta["submission_dir"]`.

```python
from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):
    name = "submission"

    def get_model(self, meta):
        # build your (untrained) model
        ...

    def fit(self, model, X_train, y_train):
        # optional: train it (default: model.fit(X_train, y_train))
        ...
```

Your trained model must expose `predict(X) -> ndarray (n_samples,)`. The
platform **trains your code**, then evaluates it. Copy
`benchmark/solvers/random-forest.py`, a working baseline, to
`my_submission/submission.py`.

## Run it locally, same code as the server

```bash
pip install -r requirements.txt
benchopt prepare benchmark -d Real
benchopt run benchmark -d Real -s my_submission/submission.py
```

The number you get locally on the public split is the number the leaderboard
shows. The benchmark you run *is* the benchmark the server runs.
