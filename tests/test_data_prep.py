"""
Unit tests for data preparation and feature engineering.
"""

import numpy as np
import pytest
from pathlib import Path
import config


def test_processed_files_exist():
    """Verify that full features and PCA files exist on disk."""
    assert (config.PROCESSED_DATA_DIR / "full_features.npz").exists()
    assert (config.PROCESSED_DATA_DIR / "pca_4_features.npz").exists()
    assert (config.PROCESSED_DATA_DIR / "pca_6_features.npz").exists()


def test_quantum_subset_shapes_and_balance():
    """Verify quantum subsets have exact train/test sizes and 10% fraud ratio."""
    for n in [4, 6]:
        for rep in range(config.N_REPEATS):
            file_path = config.PROCESSED_DATA_DIR / f"quantum_subset_n{n}_rep{rep}.npz"
            assert file_path.exists(), f"Missing file: {file_path}"

            data = np.load(file_path)
            X_train, y_train = data["X_train"], data["y_train"]
            X_test, y_test = data["X_test"], data["y_test"]

            assert X_train.shape == (config.QUANTUM_TRAIN, n)
            assert X_test.shape == (config.QUANTUM_TEST, n)
            assert len(y_train) == config.QUANTUM_TRAIN
            assert len(y_test) == config.QUANTUM_TEST

            # Check 10% fraud ratio: exactly 30 fraud in train, 20 fraud in test
            expected_train_fraud = int(config.QUANTUM_TRAIN * config.FRAUD_RATIO_QUANTUM)
            expected_test_fraud = int(config.QUANTUM_TEST * config.FRAUD_RATIO_QUANTUM)

            assert int(np.sum(y_train)) == expected_train_fraud
            assert int(np.sum(y_test)) == expected_test_fraud

            # Check values lie in [0, pi]
            assert np.all(X_train >= 0.0) and np.all(X_train <= np.pi + 1e-5)
            assert np.all(X_test >= 0.0) and np.all(X_test <= np.pi + 1e-5)

            # Check no NaNs or Infs
            assert not np.isnan(X_train).any()
            assert not np.isnan(X_test).any()
