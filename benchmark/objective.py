"""Objective: what a submission is evaluated on, and how.

This template's task is a binary classification scored by accuracy. The
objective cross-validates the solver on the training data, one fold per
repetition (run with ``n_repetitions`` equal to the number of folds): the
solver trains on the fold's training part, and ``evaluate_result`` scores it
on the held-out part (``objective_accuracy_cv``).

The test sets come without labels. ``evaluate_result`` also returns the fold
model's predictions on each of them (``objective_<test set>_pred``), and the
scoring program scores these against the labels of the phase
``reference_data``, then averages over the folds.

Adapt: ``set_data`` / ``get_objective`` to your data, ``cv`` to your
evaluation protocol (e.g. ``GroupKFold`` for a group split), and
``evaluate_result`` to your metric. Keep the scoring program's metric the
same.
"""

from benchopt import BaseObjective
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


class Objective(BaseObjective):

    name = "classification-accuracy"
    url = "https://github.com/x-datascience-datacamp/template_codabench"

    requirements = ["scikit-learn"]

    min_benchopt_version = "1.10.0"

    # Each solver trains once per fold (no convergence curve).
    sampling_strategy = "run_once"

    def set_data(self, X, y, X_test):
        # X_test: ``name -> X``, the test sets without their labels.
        self.X, self.y, self.X_test = X, y, X_test
        self.cv = StratifiedKFold(n_splits=5, shuffle=True,
                                  random_state=self.get_seed())

    def split(self, cv_fold, X, y):
        train, val = cv_fold
        return X.iloc[train], X.iloc[val], y.iloc[train], y.iloc[val]

    def get_objective(self):
        # The fold follows the repetition index.
        X_train, self.X_val, y_train, self.y_val = self.get_split(self.X,
                                                                  self.y)
        return dict(X_train=X_train, y_train=y_train)

    def evaluate_result(self, model):
        # Test predictions, scored by the scoring program.
        preds = {f"{name}_pred": model.predict(X)
                 for name, X in self.X_test.items()}
        return dict(accuracy_cv=float(
            accuracy_score(self.y_val, model.predict(self.X_val))), **preds)

    def get_one_result(self):
        # A trained trivial model, to validate the metric in ``benchopt test``.
        from sklearn.dummy import DummyClassifier
        return dict(model=DummyClassifier().fit(self.X, self.y))
