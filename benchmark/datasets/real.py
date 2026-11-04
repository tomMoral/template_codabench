"""Real dataset: ``prepare`` builds the splits, ``get_data`` only loads them.

    benchopt prepare benchmark -d Real

``prepare`` writes, under ``get_data_path("real")``, the training data and
the public test set (features + a ``target`` column), and the private test
set (features only). The private labels go to ``get_data_path("reference")``:
the bundle ships them in the phase ``reference_data``, which only the scoring
program reads, and the rest in the phase ``input_data``, where ingestion
points ``BENCHOPT_DATA_HOME``.

Adapt: replace the scikit-learn stand-in data with a download or conversion
of your raw data, and cut the test sets the way your evaluation protocol
needs (groups, time, held-out sites...), consistently with the objective's
``cv``. Participants who can run ``prepare`` on the full raw data can rebuild
the private labels: keep their raw source out of what you publish, or rely on
the terms to protect them.
"""

import pandas as pd
from benchopt import BaseDataset
from benchopt.config import get_data_path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class Dataset(BaseDataset):

    name = "Real"

    requirements = ["scikit-learn"]

    def prepare(self):
        df = load_breast_cancer(as_frame=True).frame
        train, test = train_test_split(df, test_size=0.4, random_state=0)
        public, private = train_test_split(test, test_size=0.5,
                                           random_state=0)
        root, ref = get_data_path("real"), get_data_path("reference")
        root.mkdir(parents=True, exist_ok=True)
        ref.mkdir(parents=True, exist_ok=True)
        train.to_csv(root / "train.csv", index=False)
        public.to_csv(root / "public.csv", index=False)
        private.drop(columns="target").to_csv(root / "private.csv",
                                              index=False)
        private[["target"]].to_csv(ref / "private.csv", index=False)

    def get_data(self):
        root = get_data_path("real")
        if not (root / "train.csv").exists():
            # Fail if the data has not been prepared. Use --force to bypass
            # an eventual cache.
            raise SystemExit(f"[Real] no data at {root}. Run "
                             "`benchopt prepare benchmark -d Real --force`.")

        train = pd.read_csv(root / "train.csv")
        public = pd.read_csv(root / "public.csv")
        return dict(
            X=train.drop(columns="target"), y=train["target"],
            X_public=public.drop(columns="target"), y_public=public["target"],
            X_private=pd.read_csv(root / "private.csv"),
        )
