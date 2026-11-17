"""
Unit tests for Stage 4 ideal simulation results.
"""

import pandas as pd
import pytest
import config


def test_ideal_results_csv_exists():
    """Verify ideal results CSV exists and contains expected columns."""
    res_path = config.RESULTS_DIR / "ideal_results.csv"
    assert res_path.exists(), f"Missing file: {res_path}"

    df = pd.read_csv(res_path)
    assert len(df) == 120  # 12 configs * 5 repeats * 2 settings = 120
    assert "setting" in df.columns
    assert "pr_auc" in df.columns
    assert "kernel_target_alignment" in df.columns
    assert "off_diagonal_var" in df.columns


def test_top_configurations_file():
    """Verify top configurations file exists and ranks CustomAngleRingMap."""
    top_path = config.RESULTS_DIR / "top_configurations.csv"
    assert top_path.exists(), f"Missing file: {top_path}"

    df = pd.read_csv(top_path)
    assert len(df) >= 2
    assert "CustomAngleRingMap" in df["feature_map"].values
