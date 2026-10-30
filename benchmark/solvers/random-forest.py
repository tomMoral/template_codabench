"""Sample submission: the starting point participants copy and improve.

A submission is a benchopt solver: a ``submission.py`` defining
``class Solver(CompetSolver)`` with a ``get_model`` method (and optionally
``fit``). See ``benchmark/benchmark_utils/base_solver.py`` for the contract.
The bundle ships this file as the Codabench sample submission.

Copy it to a folder of your own as ``submission.py`` and run it exactly as
the platform does::

    benchopt run benchmark -d Simulated -s my_submission/submission.py
    # real data
    benchopt prepare benchmark -d Real
    benchopt run benchmark -d Real -s my_submission/submission.py
"""

from sklearn.ensemble import RandomForestClassifier

from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):

    name = "RandomForest"

    def get_model(self, meta):
        return RandomForestClassifier(random_state=0)
