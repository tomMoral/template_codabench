"""Real dataset: ``prepare`` builds the splits, ``get_data`` only loads them.

    benchopt prepare benchmark -d Real

``prepare`` writes one CSV per split (features + a ``target`` column) under
``get_data_path("real")``. The bundle ships the prepared data in the phase
``input_data``, and ingestion points ``BENCHOPT_DATA_HOME`` at it.

Adapt: replace the scikit-learn stand-in data with a download or conversion
of your raw data, and cut the split the way your evaluation protocol needs
(groups, time, held-out sites...). Participants who can run ``prepare`` on
the full raw data can rebuild the ``private`` split: keep its raw source out
of what you publish, or rely on the terms to protect it.

For a cross-validated protocol, cut the folds in ``get_data`` from
``self.get_seed(use_repetition=True)``, as ``Simulated`` does, set
``n_repetitions`` in the phase config and average the repetitions in the
scoring program. See benchopt's seeding guide:
https://benchopt.github.io/stable/user_guide/controlling_randomness.html
"""

import pandas as pd
from benchopt import BaseDataset
from benchopt.config import get_data_path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

SPLITS = ("train", "public", "private")


class Dataset(BaseDataset):

    name = "Real"

    requirements = ["scikit-learn"]

    def prepare(self):
        df = load_breast_cancer(as_frame=True).frame
        train, test = train_test_split(df, test_size=0.4, random_state=0)
        public, private = train_test_split(test, test_size=0.5,
                                           random_state=0)
        root = get_data_path("real")
        root.mkdir(parents=True, exist_ok=True)
        for name, split in zip(SPLITS, (train, public, private)):
            split.to_csv(root / f"{name}.csv", index=False)

    def get_data(self):
        root = get_data_path("real")
        if not (root / "train.csv").exists():
            # Fail if the data has not been prepared. Use --force to bypass
            # an eventual cache.
            raise SystemExit(f"[Real] no data at {root}. Run "
                             "`benchopt prepare benchmark -d Real --force`.")

        test_sets = {}
        for name in SPLITS:
            if (root / f"{name}.csv").exists():
                df = pd.read_csv(root / f"{name}.csv")
                test_sets[name] = (df.drop(columns="target"), df["target"])
        X_train, y_train = test_sets.pop("train")
        return dict(X_train=X_train, y_train=y_train, test_sets=test_sets)
