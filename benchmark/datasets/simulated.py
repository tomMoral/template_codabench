"""Synthetic dataset: no data needed, runs anywhere (smoke tests and CI).

It must return the same keys and test subsets as the real dataset, so that
CI can run the full competition pipeline on it.
"""

from benchopt import BaseDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class Dataset(BaseDataset):

    name = "Simulated"

    requirements = ["scikit-learn"]

    parameters = {"n_samples": [500], "n_features": [5]}
    test_config = {"n_samples": 100}

    def get_data(self):
        seed = self.get_seed(use_repetition=True)
        X, y = make_classification(n_samples=self.n_samples,
                                   n_features=self.n_features,
                                   random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=seed)
        X_pub, X_priv, y_pub, y_priv = train_test_split(
            X_test, y_test, test_size=0.5, random_state=seed)
        return dict(
            X_train=X_train, y_train=y_train,
            test_sets={"public": (X_pub, y_pub), "private": (X_priv, y_priv)},
        )
