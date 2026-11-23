"""
Unit tests for Stage 7 Imbalance study module.
"""

import pandas as pd
import pytest
import config


def test_imbalance_results_csv_exists():
    """Verify imbalance results CSV exists and contains expected columns."""
    res_path = config.RESULTS_DIR / "imbalance_results.csv"
    assert res_path.exists(), f"Missing file: {res_path}"

    df = pd.read_csv(res_path)
    assert len(df) == 80  # 4 strategies * 5 reps * 2 models + 4 ratios * 5 reps * 2 models = 80
    assert "experiment" in df.columns
    assert "strategy" in df.columns
    assert "model_type" in df.columns
    assert "pr_auc" in df.columns
    assert "recall" in df.columns


def test_imbalance_strategies_coverage():
    """Verify all 4 strategies and all 4 fraud ratios were benchmarked."""
    df = pd.read_csv(config.RESULTS_DIR / "imbalance_results.csv")
    strategies = df[df["experiment"] == "part1_strategies"]["strategy"].unique()
    for s in ["none", "class_weights", "undersampling", "smote"]:
        assert s in strategies

    ratios = df[df["experiment"] == "part2_fraud_ratios"]["fraud_ratio"].unique()
    for r in [0.50, 0.20, 0.10, 0.05]:
        assert any(np.isclose(r, ratios))
import numpy as np
