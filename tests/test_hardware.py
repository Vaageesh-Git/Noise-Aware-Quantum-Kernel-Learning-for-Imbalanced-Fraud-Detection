"""
Unit tests for Stage 6 Hardware benchmarking module.
"""

import pandas as pd
import pytest
import config
from src.run_hardware import project_to_nearest_psd
import numpy as np


def test_hardware_results_csv_exists():
    """Verify hardware results CSV exists and contains expected columns."""
    res_path = config.RESULTS_DIR / "hardware_results.csv"
    assert res_path.exists(), f"Missing file: {res_path}"

    df = pd.read_csv(res_path)
    assert len(df) == 3
    settings = df["setting"].tolist()
    assert "D1_raw_hardware" in settings
    assert "D2_mitigated_hardware" in settings
    assert "D3_mitigated_plus_classical_psd" in settings
    assert "frobenius_rel_error" in df.columns
    assert "pearson_correlation" in df.columns


def test_psd_projection_properties():
    """Verify classical PSD projection guarantees positive semi-definiteness and unit diagonal."""
    # Construct an indefinite symmetric matrix with negative eigenvalue
    K_indef = np.array([
        [1.0, 0.9, 0.9],
        [0.9, 1.0, -0.6],
        [0.9, -0.6, 1.0],
    ], dtype=np.float32)

    min_eig_before = np.min(np.linalg.eigvalsh(K_indef))
    assert min_eig_before < 0.0

    K_psd = project_to_nearest_psd(K_indef)

    # Diagonal must be identically 1.0
    assert np.allclose(np.diag(K_psd), 1.0)

    # Minimum eigenvalue must be non-negative
    min_eig_after = np.min(np.linalg.eigvalsh(K_psd))
    assert min_eig_after >= -1e-6

    # Symmetric
    assert np.allclose(K_psd, K_psd.T)
