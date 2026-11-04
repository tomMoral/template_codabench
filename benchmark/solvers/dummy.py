"""Floor baseline: predicts the most frequent class.

Any real model should beat it. Keep one such floor in every competition.
"""

from sklearn.dummy import DummyClassifier

from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):

    name = "Dummy"

    requirements = ["scikit-learn"]

    def get_model(self, meta):
        return DummyClassifier()
