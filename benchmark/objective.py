"""Objective: what a submission is evaluated on, and how.

This template's task is a binary classification, scored by the metric of
``benchmark_utils/metric.py`` (accuracy). The objective cross-validates the
solver on the training data, one fold per repetition (run with
``n_repetitions`` equal to the number of folds): the solver trains on the
fold's training part, and ``evaluate_result`` scores it on the held-out part
(``objective_accuracy_cv``) and on the public test set
(``objective_accuracy_public``).

The private test set comes without labels: ``evaluate_result`` returns the
fold model's predictions on it (``objective_private_pred``), and the scoring
program scores them against the labels of the phase ``reference_data``. The
scoring program averages every score over the folds.

Adapt: ``set_data`` / ``get_objective`` to your data, and ``cv`` to your
evaluation protocol (e.g. ``GroupKFold`` for a group split).
"""

from benchopt import BaseObjective
from sklearn.model_selection import StratifiedKFold

from benchmark_utils.metric import NAME, score


class Objective(BaseObjective):

    name = "classification-accuracy"
    url = "https://github.com/x-datascience-datacamp/template_codabench"

    requirements = ["scikit-learn"]

    min_benchopt_version = "1.10.0"

    # Each solver trains once per fold (no convergence curve).
    sampling_strategy = "run_once"

    def set_data(self, X, y, X_public, y_public, X_private):
        self.X, self.y = X, y
        self.X_public, self.y_public = X_public, y_public
        self.X_private = X_private
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
        return {
            f"{NAME}_cv": score(self.y_val, model.predict(self.X_val)),
            f"{NAME}_public": score(self.y_public,
                                    model.predict(self.X_public)),
            # Scored by the scoring program against the hidden labels.
            "private_pred": model.predict(self.X_private),
        }

    def get_one_result(self):
        # A trained trivial model, to validate the metric in ``benchopt test``.
        from sklearn.dummy import DummyClassifier
        return dict(model=DummyClassifier().fit(self.X, self.y))
