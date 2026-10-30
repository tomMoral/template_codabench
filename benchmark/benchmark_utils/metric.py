"""The competition metric, the single source of truth for the objective and
the scoring program (``tools/create_bundle.py`` ships ``benchmark_utils``
with the scoring program).

Adapt: ``NAME`` and ``score`` to your metric. Higher must be better, or flip
``sorting`` in ``competition.yaml``.
"""

from sklearn.metrics import accuracy_score

NAME = "accuracy"


def score(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))
