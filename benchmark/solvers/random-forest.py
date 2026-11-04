"""Sample submission: the starting point participants copy and improve.

A submission is a benchopt solver: a ``submission.py`` defining
``class Solver(CompetSolver)`` with a ``get_model`` method (and optionally
``fit``). See ``benchmark/benchmark_utils/base_solver.py`` for the contract.
The bundle ships this file as the Codabench sample submission.

Copy it to a folder of your own as ``submission.py`` and run it exactly as
the platform does, with the dataset, seed and folds of ``local_config.yml``::

    benchopt prepare benchmark --config local_config.yml
    benchopt run benchmark --config local_config.yml \
        -s my_submission/submission.py
"""

from sklearn.ensemble import RandomForestClassifier

from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):

    name = "RandomForest"

    def get_model(self, meta):
        return RandomForestClassifier(random_state=0)
