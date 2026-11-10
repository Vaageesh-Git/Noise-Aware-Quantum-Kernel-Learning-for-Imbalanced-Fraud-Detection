"""
Unit tests for classical baseline benchmarking.
"""

import pandas as pd
import pytest
import config


def test_classical_results_csv_exists():
    """Verify that classical results CSV is generated and non-empty."""
    results_path = config.RESULTS_DIR / "classical_results.csv"
    assert results_path.exists(), f"Missing file: {results_path}"

    df = pd.read_csv(results_path)
    assert len(df) > 0
    assert "pr_auc" in df.columns
    assert "roc_auc" in df.columns
    assert "f1" in df.columns


def test_classical_results_coverage():
    """Verify all models and regimes are evaluated across all repeats."""
    df = pd.read_csv(config.RESULTS_DIR / "classical_results.csv")

    regimes = df["regime"].unique()
    assert "full_features" in regimes
    assert "fair_quantum_subset" in regimes

    models = df["model"].unique()
    for expected_model in ["Dummy", "SVM_RBF", "RandomForest", "XGBoost"]:
        assert expected_model in models

    # Check 5 repeats for each model and each n_qubits in fair regime
    fair_df = df[df["regime"] == "fair_quantum_subset"]
    for n in config.N_QUBITS_LIST:
        for model in ["Dummy", "SVM_RBF", "RandomForest", "XGBoost"]:
            subset = fair_df[(fair_df["n_qubits"] == n) & (fair_df["model"] == model)]
            assert len(subset) == config.N_REPEATS, f"Expected {config.N_REPEATS} repeats for {model} at n={n}"
