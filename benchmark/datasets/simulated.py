"""Synthetic dataset: no data needed, runs anywhere (smoke tests and CI).

It must return the same keys, types and test subsets as the real dataset, so
that CI can run the full competition pipeline on it.
"""

import pandas as pd
from benchopt import BaseDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class Dataset(BaseDataset):

    name = "Simulated"

    requirements = ["scikit-learn"]

    parameters = {"n_samples": [500], "n_features": [5]}
    test_config = {"n_samples": 100}

    def get_data(self):
        seed = self.get_seed()
        X, y = make_classification(n_samples=self.n_samples,
                                   n_features=self.n_features,
                                   random_state=seed)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.4, random_state=seed)
        X_pub, X_priv = train_test_split(X_test, test_size=0.5,
                                         random_state=seed)
        return dict(X=X_train, y=y_train,
                    X_test={"public": X_pub, "private": X_priv})
