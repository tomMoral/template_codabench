"""Objective: what a submission is evaluated on, and how.

This template's task is a binary classification scored by accuracy. The
solver gets the training data, trains its model (on the platform), and the
objective scores the trained model on each held-out subset: ``public``
(shown live on the leaderboard) and ``private`` (hidden, final ranking).
Metrics are emitted per subset as ``objective_<metric>_<subset>``; the phase
config maps them to leaderboard keys.

Adapt: ``set_data`` / ``get_objective`` to your data, ``evaluate_result`` to
your metric.
"""

from benchopt import BaseObjective
from sklearn.metrics import accuracy_score


class Objective(BaseObjective):

    name = "classification-accuracy"
    url = "https://github.com/x-datascience-datacamp/template_codabench"

    requirements = ["scikit-learn"]

    min_benchopt_version = "1.10.0"

    # Each solver trains once (no convergence curve).
    sampling_strategy = "run_once"

    def set_data(self, X_train, y_train, test_sets):
        # test_sets: ``name -> (X, y)``; the test labels stay here so the
        # objective owns evaluation.
        self.X_train = X_train
        self.y_train = y_train
        self.test_sets = test_sets

    def get_objective(self):
        return dict(X_train=self.X_train, y_train=self.y_train)

    def evaluate_result(self, model):
        return {
            f"accuracy_{name}": float(accuracy_score(y, model.predict(X)))
            for name, (X, y) in self.test_sets.items()
        }

    def get_one_result(self):
        # A trained trivial model, to validate the metric in ``benchopt test``.
        from sklearn.dummy import DummyClassifier
        return dict(model=DummyClassifier().fit(self.X_train, self.y_train))
