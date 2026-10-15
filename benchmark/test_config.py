"""Hooks for ``benchopt test`` (see benchopt's test_config mechanism)."""

import pytest

# The real dataset needs prepared data files, which CI does not carry;
# ``Simulated`` covers the no-data smoke test.
REAL_DATASETS = ("real",)


def check_test_dataset_install(dataset_class):
    if dataset_class.name.lower() in REAL_DATASETS:
        pytest.skip("real data is staged separately, not available in CI")


def check_test_dataset_get_data(dataset_class):
    if dataset_class.name.lower() in REAL_DATASETS:
        pytest.skip("real data is staged separately, not available in CI")
