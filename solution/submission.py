"""Sample submission: the starting point participants copy and improve.

A submission is a benchopt solver: a ``submission.py`` defining
``class Solver(CompetSolver)`` with a ``get_model`` method (and optionally
``fit``). See ``benchmark/benchmark_utils/base_solver.py`` for the contract.

Run it locally exactly as the platform does::

    benchopt run benchmark -d Simulated -s solution/submission.py
    # real data
    benchopt prepare benchmark -d Real
    benchopt run benchmark -d Real -s solution/submission.py
"""

from sklearn.ensemble import RandomForestClassifier

from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):

    name = "submission"

    def get_model(self, meta):
        return RandomForestClassifier(random_state=0)
