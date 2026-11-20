"""
Unit tests for Stage 5 noisy simulation results.
"""

import pandas as pd
import pytest
import config


def test_noisy_results_csv_exists():
    """Verify noisy results CSV exists and contains expected columns."""
    res_path = config.RESULTS_DIR / "noisy_results.csv"
    assert res_path.exists(), f"Missing file: {res_path}"

    df = pd.read_csv(res_path)
    assert len(df) == 9  # 3 configurations * 3 repeats = 9 runs
    assert "frobenius_rel_error" in df.columns
    assert "pearson_correlation" in df.columns
    assert "variance_retention_ratio" in df.columns
    assert "pr_auc" in df.columns


def test_noise_distortion_trends():
    """Verify physical properties of noisy simulation: higher qubits have higher error."""
    df = pd.read_csv(config.RESULTS_DIR / "noisy_results.csv")
    err_4q = df[df["label"] == "Top-1 (Angle Ring 4Q)"]["frobenius_rel_error"].mean()
    err_6q = df[df["label"] == "Top-2 (Angle Ring 6Q)"]["frobenius_rel_error"].mean()

    # 6-qubit system experiences more cumulative decoherence and readout error than 4-qubit system
    assert err_6q > err_4q
    assert df["pearson_correlation"].min() > 0.90
