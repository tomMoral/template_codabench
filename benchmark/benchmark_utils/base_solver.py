"""Submission contract.

A submission is a **valid benchopt solver**: a folder with a ``submission.py``
defining ``class Solver(CompetSolver)``. All its Python code lives in that
single file (benchopt loads it by path, so sibling ``.py`` files cannot be
imported); the folder may also hold non-code files (weights, configs) read
from ``meta["submission_dir"]``. Participants implement:

- ``get_model(self, meta)`` -> model (required): build the untrained model.
- ``fit(self, model, X_train, y_train)`` (optional): train it. Defaults to
  ``model.fit(X_train, y_train)``. It runs on the platform.

The trained model must expose ``model.predict(X) -> ndarray (n_samples,)``.

Adapt: the arguments of ``set_objective`` / ``fit`` follow
``Objective.get_objective``, and the ``predict`` contract follows
``Objective.evaluate_result``.
"""

import inspect
from pathlib import Path

from benchopt import BaseSolver


class CompetSolver(BaseSolver):
    """Base class for all submissions (and the built-in baselines).

    The competition image fixes the available packages: submissions cannot
    install extra ones.
    """

    requirements = []

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        submission_dir = Path(inspect.getfile(type(self))).parent
        self.model = self.get_model({"submission_dir": submission_dir})

    def run(self, _):
        self.fit(self.model, self.X_train, self.y_train)

    def get_result(self):
        return dict(model=self.model)

    # --- to be implemented by the submission ---------------------------

    def get_model(self, meta):
        """Build and return the (untrained) model."""
        raise NotImplementedError

    def fit(self, model, X_train, y_train):
        """Train the model in place."""
        model.fit(X_train, y_train)
